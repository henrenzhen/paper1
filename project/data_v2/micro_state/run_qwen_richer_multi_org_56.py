import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    BASE_DIR / "apt29_aligned_richer_input.csv",
    BASE_DIR / "oilrig_aligned_richer_input.csv",
    BASE_DIR / "sandworm_aligned_richer_input.csv",
    BASE_DIR / "turla_carbon_aligned_richer_input.csv",
]
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"

OUTPUT_DETAIL_CSV = BASE_DIR / "qwen_richer_multi_org_56_details.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "qwen_richer_multi_org_56_summary.csv"
OUTPUT_SUMMARY_BY_ORG_CSV = BASE_DIR / "qwen_richer_multi_org_56_summary_by_org.csv"
OUTPUT_JSONL = BASE_DIR / "qwen_richer_multi_org_56_raw.jsonl"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

TEMPERATURE = 0.0
MAX_TOKENS = 128

FIELDS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def load_schema(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    for k in FIELDS:
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
                "required": FIELDS,
                "additionalProperties": False,
            },
        },
    }


def build_system_prompt():
    return """
You are a structured cyber operation state annotator.

Infer the NEXT-STEP micro-state from:
1) an ATT&CK parent-technique prefix
2) local procedural context for the matched next step

Predict exactly four fields:

1. current_access_context
- The current operational access stage implied by the prefix and context.

2. exhausted_action_constraint
- The main unmet prerequisite still constraining the operator.

3. next_target_artifact
- The immediate object/entity/channel/mechanism that will be acted on next.

4. next_micro_verb
- The immediate action applied to next_target_artifact.

Use only the allowed labels.
Be precise and conservative.
""".strip()


def build_user_prompt(row, schema):
    prefix = normalize_str(row["state"])
    tech_name = normalize_str(row.get("matched_technique_name", ""))
    description = normalize_str(row.get("matched_description", ""))
    command_summary = normalize_str(row.get("matched_command_summary", ""))

    allowed_block = "\n".join(
        f"{f}: [{', '.join(schema[f])}]"
        for f in FIELDS
    )

    return f"""
Task:
Infer the next-step micro-state from the ATT&CK parent-technique prefix and local procedural context.

Prefix:
{prefix}

Matched next-step technique name:
{tech_name}

Matched next-step description:
{description}

Matched next-step command summary:
{command_summary}

Allowed labels:
{allowed_block}
""".strip()


def load_input_frames():
    frames = []
    for p in INPUT_FILES:
        if not p.exists():
            print(f"[WARN] Missing input file, skipped: {p}")
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "alignment_ok" not in df.columns:
            raise ValueError(f"Missing alignment_ok in {p}")
        df = df[df["alignment_ok"] == 1].copy()
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError("No non-empty aligned input files found.")
    merged = pd.concat(frames, ignore_index=True)
    merged["source_org"] = merged["source_org"].astype(str).str.strip().str.lower()
    return merged


def run_one(client, schema, response_format, row):
    user_prompt = build_user_prompt(row, schema)

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
        json_parse_ok = 1
    except Exception:
        parsed = None
        json_parse_ok = 0

    out = {
        "annotation_id": normalize_str(row["annotation_id"]),
        "source_org": normalize_str(row["source_org"]),
        "true_label": normalize_str(row["true_label"]),
        "json_parse_ok": json_parse_ok,
        "raw_output": raw_text,
    }

    for f in FIELDS:
        gold = normalize_str(row[f])
        pred = normalize_str(parsed.get(f, "")) if isinstance(parsed, dict) else ""
        allowed = set(schema[f])

        out[f"gold__{f}"] = gold
        out[f"pred__{f}"] = pred
        out[f"schema_ok__{f}"] = int(pred in allowed)
        out[f"field_hit__{f}"] = int(pred == gold)

    out["exact_match"] = int(all(out[f"field_hit__{f}"] == 1 for f in FIELDS))
    out["any_schema_violation"] = int(any(out[f"schema_ok__{f}"] != 1 for f in FIELDS))

    raw = {
        "annotation_id": out["annotation_id"],
        "source_org": out["source_org"],
        "raw_output": raw_text,
        "parsed": parsed,
    }
    return out, raw


def summarize(df: pd.DataFrame, group_col=None):
    rows = []
    grouped = df.groupby(group_col) if group_col else [("all", df)]
    for key, g in grouped:
        row = {
            group_col if group_col else "scope": key,
            "samples": len(g),
            "json_parse_ok_rate": g["json_parse_ok"].mean(),
            "exact_match_rate": g["exact_match"].mean(),
            "schema_violation_rate": g["any_schema_violation"].mean(),
        }
        for f in FIELDS:
            row[f"{f}_acc"] = g[f"field_hit__{f}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Missing schema JSON: {SCHEMA_JSON}")

    df = load_input_frames()
    schema = load_schema(SCHEMA_JSON)
    response_format = build_response_format(schema)

    print(f"[INFO] Total aligned samples to run: {len(df)}")
    print(df["source_org"].value_counts().to_string())

    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",
    )

    detail_rows = []
    raw_rows = []

    for _, row in df.iterrows():
        out, raw = run_one(client, schema, response_format, row)
        detail_rows.append(out)
        raw_rows.append(raw)
        print(
            f"[DONE] {out['source_org']} | {out['annotation_id']} | "
            f"parse_ok={out['json_parse_ok']} | exact={out['exact_match']} | "
            f"schema_violation={out['any_schema_violation']}"
        )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize(detail_df)
    summary_by_org_df = summarize(detail_df, group_col="source_org")

    detail_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    summary_by_org_df.to_csv(OUTPUT_SUMMARY_BY_ORG_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Saved details      : {OUTPUT_DETAIL_CSV}")
    print(f"[INFO] Saved summary      : {OUTPUT_SUMMARY_CSV}")
    print(f"[INFO] Saved summary_by_org: {OUTPUT_SUMMARY_BY_ORG_CSV}")
    print(f"[INFO] Saved raw          : {OUTPUT_JSONL}")

    print("\n=== OVERALL SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== BY ORG SUMMARY ===")
    print(summary_by_org_df.to_string(index=False))


if __name__ == "__main__":
    main()