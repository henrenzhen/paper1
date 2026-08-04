import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent          # ~/project/llm
PROJECT_ROOT = BASE_DIR.parent                      # ~/project
DATA_DIR = PROJECT_ROOT / "data"                   # ~/project/data

# 输入
STAGE1_PRED_CSV = DATA_DIR / "llm_kg_context_test_micro_state_details.csv"
GOLD_CSV = DATA_DIR / "micro_state_gold_v2_plus_v3.csv"

# 输出
OUTPUT_DETAIL_CSV = DATA_DIR / "stage2_llm_kg_context_test_predictions.csv"
OUTPUT_METRIC_TXT = DATA_DIR / "stage2_llm_kg_context_test_metrics.txt"

# vLLM
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

TEMPERATURE = 0.0
MAX_TOKENS = 128

# 调试先跑前100条；全量改成 None
N_DEBUG = 100

MICRO_COLS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def build_candidate_labels(gold_df):
    labels = sorted(set(normalize_str(x) for x in gold_df["true_label"].tolist() if normalize_str(x)))
    return labels


def build_system_prompt():
    return """
You are a cyber operation next-step predictor.

You will be given a structured micro-state and a candidate set of ATT&CK parent-technique labels.
Choose the 5 most likely next ATT&CK parent-techniques from the candidate set.

Return a JSON object with exactly one field:
- top5_next_ttp: an array of exactly 5 labels

Rules:
- Only choose labels from the provided candidate set.
- Do not output duplicates.
- Rank from most likely to less likely.
""".strip()


def build_user_prompt(row, candidate_labels):
    micro_state_text = " ; ".join([
        f"context={normalize_str(row['pred__current_access_context'])}",
        f"constraint={normalize_str(row['pred__exhausted_action_constraint'])}",
        f"target={normalize_str(row['pred__next_target_artifact'])}",
        f"verb={normalize_str(row['pred__next_micro_verb'])}",
    ])

    candidate_text = ", ".join(candidate_labels)

    return f"""
Predicted micro-state:
{micro_state_text}

Candidate ATT&CK parent-technique labels:
{candidate_text}
""".strip()


def build_response_format(candidate_labels):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "stage2_top5_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "top5_next_ttp": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": candidate_labels,
                        },
                        "minItems": 5,
                        "maxItems": 5,
                    }
                },
                "required": ["top5_next_ttp"],
                "additionalProperties": False,
            },
        },
    }


def compute_metrics_from_top5(df):
    top1 = 0
    top5 = 0
    mrr = 0.0

    for _, row in df.iterrows():
        true_label = normalize_str(row["true_label"])
        preds = [normalize_str(x) for x in str(row["top5_labels"]).split("||") if normalize_str(x)]

        if preds and preds[0] == true_label:
            top1 += 1

        if true_label in preds:
            top5 += 1
            rank = preds.index(true_label) + 1
            mrr += 1.0 / rank
            df.loc[row.name, "true_rank"] = rank
        else:
            df.loc[row.name, "true_rank"] = -1

    n = len(df)
    return {
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": mrr / n,
    }


def print_progress(stage_name, i, total):
    pct = 100.0 * i / total if total else 0.0
    print(f"\r[{stage_name}] {i}/{total} | {pct:5.1f}%", end="", flush=True)


def main():
    if not STAGE1_PRED_CSV.exists():
        raise FileNotFoundError(f"Missing Stage1 prediction CSV: {STAGE1_PRED_CSV}")
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing gold CSV: {GOLD_CSV}")

    stage1 = pd.read_csv(STAGE1_PRED_CSV, encoding="utf-8-sig")
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")

    if N_DEBUG is not None:
        stage1 = stage1.head(N_DEBUG).copy()

    needed_stage1 = ["sequence_id", "true_label"] + [f"pred__{c}" for c in MICRO_COLS]
    for c in needed_stage1:
        if c not in stage1.columns:
            raise ValueError(f"Missing Stage1 column: {c}")

    for c in needed_stage1:
        stage1[c] = stage1[c].fillna("").astype(str).str.strip()

    candidate_labels = build_candidate_labels(gold)
    response_format = build_response_format(candidate_labels)

    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",
    )

    total_n = len(stage1)
    rows = []

    print(f"[INFO] Total samples: {total_n}")
    print(f"[INFO] Candidate labels: {len(candidate_labels)}")
    print("[INFO] Starting Stage2: LLM micro-state -> Top5 nextTTP")

    for i, (_, row) in enumerate(stage1.iterrows(), start=1):
        user_prompt = build_user_prompt(row, candidate_labels)

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )

            raw_text = normalize_str(resp.choices[0].message.content)

            try:
                parsed = json.loads(raw_text)
                parse_ok = 1
                preds = parsed.get("top5_next_ttp", [])
                preds = [normalize_str(x) for x in preds if normalize_str(x)]
            except Exception:
                parse_ok = 0
                preds = []

        except Exception as e:
            raw_text = f"[ERROR] {type(e).__name__}: {e}"
            parse_ok = 0
            preds = []

        # 保底：长度不够时补空
        preds = preds[:5]
        while len(preds) < 5:
            preds.append("")

        rows.append({
            "sequence_id": row["sequence_id"],
            "true_label": row["true_label"],
            "pred__current_access_context": row["pred__current_access_context"],
            "pred__exhausted_action_constraint": row["pred__exhausted_action_constraint"],
            "pred__next_target_artifact": row["pred__next_target_artifact"],
            "pred__next_micro_verb": row["pred__next_micro_verb"],
            "json_parse_ok": parse_ok,
            "raw_output": raw_text,
            "pred_top1": preds[0],
            "top5_labels": " || ".join(preds),
        })

        print_progress("Stage2-LLM", i, total_n)

    print()
    out_df = pd.DataFrame(rows)
    out_df["true_rank"] = -1

    metrics = compute_metrics_from_top5(out_df)
    out_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    parse_ok_rate = out_df["json_parse_ok"].mean()

    lines = []
    lines.append("=== STAGE2 LLM ON KG-STAGE1 OUTPUTS ===")
    lines.append(f"samples={len(out_df)}")
    lines.append(f"candidate_labels={len(candidate_labels)}")
    lines.append(f"stage2_json_parse_ok_rate={parse_ok_rate:.4f}")
    lines.append(f"top1={metrics['top1']:.4f}")
    lines.append(f"top5={metrics['top5']:.4f}")
    lines.append(f"mrr={metrics['mrr']:.4f}")

    OUTPUT_METRIC_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved -> {OUTPUT_DETAIL_CSV}")
    print(f"Saved -> {OUTPUT_METRIC_TXT}")


if __name__ == "__main__":
    main()