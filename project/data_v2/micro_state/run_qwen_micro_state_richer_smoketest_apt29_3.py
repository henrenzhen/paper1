import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "APT29_richer_smoketest_10_aligned.csv"
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"
OUTPUT_CSV = BASE_DIR / "qwen_micro_state_richer_smoketest_apt29_3.csv"
OUTPUT_JSONL = BASE_DIR / "qwen_micro_state_richer_smoketest_apt29_3_raw.jsonl"

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


def load_schema(schema_path: Path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    for k in FIELDS:
        if k not in schema or not isinstance(schema[k], list):
            raise ValueError(f"Invalid schema field: {k}")
    return schema


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def build_system_prompt():
    return """
You are a structured cyber operation state annotator.

Infer the NEXT-STEP micro-state from:
1) an ATT&CK parent-technique prefix
2) local procedural context for the matched next step

Predict exactly four fields:

1. current_access_context
- The CURRENT operational access stage implied by the prefix and local context.
- This is the current situation, not the next action.

2. exhausted_action_constraint
- The main unmet prerequisite that still constrains the operator.
- This is what is still missing before effective next progress.

3. next_target_artifact
- The immediate object, entity, store, channel, or mechanism that will be acted on next.
- This is the direct target of the next action, not a broad goal.

4. next_micro_verb
- The immediate action applied to next_target_artifact.
- This is not the ATT&CK technique name.

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
Infer the next-step micro-state from the ATT&CK parent-technique prefix and the matched local procedural context.

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


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Missing schema JSON: {SCHEMA_JSON}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    schema = load_schema(SCHEMA_JSON)
    response_format = build_response_format(schema)

    # 只取 alignment_ok=1 的干净样本
    if "alignment_ok" not in df.columns:
        raise ValueError("Missing column: alignment_ok")

    df = df[df["alignment_ok"] == 1].copy()

    required_cols = ["annotation_id", "state"] + FIELDS + [
        "matched_technique_name",
        "matched_description",
        "matched_command_summary",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",
    )

    records = []
    raw_lines = []

    for _, row in df.iterrows():
        ann_id = normalize_str(row["annotation_id"])
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
            "annotation_id": ann_id,
            "raw_output": raw_text,
            "json_parse_ok": json_parse_ok,
        }

        for f in FIELDS:
            gold = normalize_str(row[f])
            pred = normalize_str(parsed.get(f, "")) if isinstance(parsed, dict) else ""

            allowed = set(schema[f])
            schema_ok = int(pred in allowed)
            field_hit = int(pred == gold)

            out[f"gold__{f}"] = gold
            out[f"pred__{f}"] = pred
            out[f"schema_ok__{f}"] = schema_ok
            out[f"field_hit__{f}"] = field_hit

        exact_match = int(all(out[f"field_hit__{f}"] == 1 for f in FIELDS))
        any_schema_violation = int(any(out[f"schema_ok__{f}"] != 1 for f in FIELDS))

        out["exact_match"] = exact_match
        out["any_schema_violation"] = any_schema_violation

        records.append(out)
        raw_lines.append(json.dumps({
            "annotation_id": ann_id,
            "raw_output": raw_text,
            "parsed": parsed,
        }, ensure_ascii=False))

        print(
            f"[DONE] {ann_id} | "
            f"parse_ok={json_parse_ok} | "
            f"exact={exact_match} | "
            f"schema_violation={any_schema_violation}"
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for line in raw_lines:
            f.write(line + "\n")

    print(f"\n[INFO] Saved CSV : {OUTPUT_CSV}")
    print(f"[INFO] Saved JSONL: {OUTPUT_JSONL}")

    if len(out_df) > 0:
        print("\n=== SUMMARY ===")
        print(f"json_parse_ok_rate      = {out_df['json_parse_ok'].mean():.4f}")
        print(f"exact_match_rate        = {out_df['exact_match'].mean():.4f}")
        print(f"schema_violation_rate   = {out_df['any_schema_violation'].mean():.4f}")
        for f in FIELDS:
            print(f"{f}_acc = {out_df[f'field_hit__{f}'].mean():.4f}")
    else:
        print("\n[WARN] No aligned samples found.")


if __name__ == "__main__":
    main()