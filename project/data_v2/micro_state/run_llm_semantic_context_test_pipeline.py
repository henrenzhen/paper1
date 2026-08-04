import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

# 输入
SIM_TEST_CSV = BASE_DIR / "sim_test_parent_min3_semantic_context.csv"
GOLD_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"

# 输出
STAGE1_DETAIL_CSV = BASE_DIR / "llm_semantic_context_test_micro_state_details.csv"
FINAL_DETAIL_CSV = BASE_DIR / "llm_semantic_context_test_predictions.csv"
FINAL_METRIC_TXT = BASE_DIR / "llm_semantic_context_test_metrics.txt"

# vLLM
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

TEMPERATURE = 0.0
MAX_TOKENS = 96

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


def truncate_semantic_context(text: str, max_chars: int = 500):
    text = normalize_str(text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip() + " ..."


def load_schema(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    for k in MICRO_COLS:
        if k not in schema or not isinstance(schema[k], list):
            raise ValueError(f"Invalid schema field: {k}")
    return schema


def build_response_format(schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "micro_state_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "current_access_context": {
                        "type": "string",
                        "enum": schema["current_access_context"],
                    },
                    "exhausted_action_constraint": {
                        "type": "string",
                        "enum": schema["exhausted_action_constraint"],
                    },
                    "next_target_artifact": {
                        "type": "string",
                        "enum": schema["next_target_artifact"],
                    },
                    "next_micro_verb": {
                        "type": "string",
                        "enum": schema["next_micro_verb"],
                    },
                },
                "required": MICRO_COLS,
                "additionalProperties": False,
            },
        },
    }


def build_system_prompt():
    return """
You infer the next-step micro-state from an ATT&CK prefix and short semantic context.

Return exactly four fields:
- current_access_context
- exhausted_action_constraint
- next_target_artifact
- next_micro_verb

Use only allowed labels from the schema.
Be concise and conservative.
""".strip()


def build_user_prompt(row, schema):
    prefix = normalize_str(row["prefix_technique_ids_parent"])
    recent_ids = normalize_str(row.get("recent_prefix_ids", ""))
    recent_comments = normalize_str(row.get("recent_prefix_comments", ""))
    semantic_context = normalize_str(row.get("semantic_context_text", ""))

    short_context = recent_comments if recent_comments else semantic_context
    short_context = truncate_semantic_context(short_context, max_chars=500)

    return f"""
Infer the next-step micro-state.

Prefix ids:
{prefix}

Recent prefix ids:
{recent_ids}

Short semantic context:
{short_context}
""".strip()


def build_gold_micro_text(row):
    return " ; ".join([
        f"context={row['current_access_context']}",
        f"constraint={row['exhausted_action_constraint']}",
        f"target={row['next_target_artifact']}",
        f"verb={row['next_micro_verb']}",
    ])


def build_pred_micro_text(row):
    return " ; ".join([
        f"context={row['pred__current_access_context']}",
        f"constraint={row['pred__exhausted_action_constraint']}",
        f"target={row['pred__next_target_artifact']}",
        f"verb={row['pred__next_micro_verb']}",
    ])


def evaluate_one(true_label, probs, classes_):
    order = probs.argsort()[::-1]
    ranked_labels = [classes_[j] for j in order]

    top1_hit = int(ranked_labels[0] == true_label)
    top5_hit = int(true_label in ranked_labels[:5])

    if true_label in ranked_labels:
        rank = ranked_labels.index(true_label) + 1
        rr = 1.0 / rank
    else:
        rank = -1
        rr = 0.0

    return {
        "pred_top1": ranked_labels[0],
        "true_rank": rank,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "rr": rr,
        "top5_labels": " || ".join(ranked_labels[:5]),
    }


def print_progress(stage_name, i, total):
    pct = 100.0 * i / total if total else 0.0
    print(f"\r[{stage_name}] {i}/{total} | {pct:5.1f}%", end="", flush=True)


def main():
    if not SIM_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {SIM_TEST_CSV}")
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing gold CSV: {GOLD_CSV}")
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Missing schema JSON: {SCHEMA_JSON}")

    sim_df = pd.read_csv(SIM_TEST_CSV, encoding="utf-8-sig")
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    schema = load_schema(SCHEMA_JSON)
    response_format = build_response_format(schema)

    required_sim_cols = [
        "sequence_id",
        "prefix_technique_ids_parent",
        "next_technique_id_parent",
    ]
    for c in required_sim_cols:
        if c not in sim_df.columns:
            raise ValueError(f"Missing sim column: {c}")

    for c in ["annotation_id", "true_label"] + MICRO_COLS:
        if c not in gold.columns:
            raise ValueError(f"Missing gold column: {c}")

    gold = gold[["annotation_id", "true_label"] + MICRO_COLS].copy()
    for c in MICRO_COLS:
        gold[c] = gold[c].fillna("").astype(str).str.strip()
    gold["gold_micro_state_text"] = gold.apply(build_gold_micro_text, axis=1)

    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",
    )

    total_n = len(sim_df)
    stage1_rows = []

    print(f"[INFO] Total samples: {total_n}")
    print("[INFO] Starting Stage1: LLM semantic-context -> micro-state")

    for i, (_, row) in enumerate(sim_df.iterrows(), start=1):
        user_prompt = build_user_prompt(row, schema)

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
            except Exception:
                parsed = {}
                parse_ok = 0

        except Exception as e:
            raw_text = f"[ERROR] {type(e).__name__}: {e}"
            parsed = {}
            parse_ok = 0

        out = {
            "sequence_id": normalize_str(row["sequence_id"]),
            "state": normalize_str(row["prefix_technique_ids_parent"]),
            "true_label": normalize_str(row["next_technique_id_parent"]),
            "json_parse_ok": parse_ok,
            "raw_output": raw_text,
        }

        for c in MICRO_COLS:
            pred = normalize_str(parsed.get(c, ""))
            out[f"pred__{c}"] = pred
            out[f"schema_ok__{c}"] = int(pred in set(schema[c]))

        stage1_rows.append(out)
        print_progress("Stage1", i, total_n)

    print()
    stage1_df = pd.DataFrame(stage1_rows)
    stage1_df.to_csv(STAGE1_DETAIL_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved Stage1 detail -> {STAGE1_DETAIL_CSV}")

    print("[INFO] Starting Stage2: predicted micro-state -> nextTTP")
    stage1_df["pred_micro_state_text"] = stage1_df.apply(build_pred_micro_text, axis=1)

    eval_rows = []

    X_train = gold["gold_micro_state_text"].tolist()
    y_train = gold["true_label"].tolist()

    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(Xtr, y_train)

    for i, (_, test_row) in enumerate(stage1_df.iterrows(), start=1):
        X_test = [test_row["pred_micro_state_text"]]
        Xte = vec.transform(X_test)

        probs = clf.predict_proba(Xte)[0]
        classes_ = clf.classes_

        pred_eval = evaluate_one(
            true_label=test_row["true_label"],
            probs=probs,
            classes_=classes_,
        )

        eval_rows.append({
            "sequence_id": test_row["sequence_id"],
            "state": test_row["state"],
            "true_label": test_row["true_label"],
            "pred_micro_state_text": test_row["pred_micro_state_text"],
            **pred_eval,
        })

        print_progress("Stage2", i, total_n)

    print()
    final_df = pd.DataFrame(eval_rows)
    final_df.to_csv(FINAL_DETAIL_CSV, index=False, encoding="utf-8-sig")

    top1 = final_df["top1_hit"].mean()
    top5 = final_df["top5_hit"].mean()
    mrr = final_df["rr"].mean()
    parse_ok_rate = stage1_df["json_parse_ok"].mean()

    lines = []
    lines.append("=== LLM SEMANTIC-CONTEXT TEST PIPELINE ===")
    lines.append(f"samples={len(final_df)}")
    lines.append(f"num_labels={final_df['true_label'].nunique()}")
    lines.append(f"stage1_json_parse_ok_rate={parse_ok_rate:.4f}")
    lines.append(f"top1={top1:.4f}")
    lines.append(f"top5={top5:.4f}")
    lines.append(f"mrr={mrr:.4f}")

    FINAL_METRIC_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved -> {FINAL_DETAIL_CSV}")
    print(f"Saved -> {FINAL_METRIC_TXT}")


if __name__ == "__main__":
    main()