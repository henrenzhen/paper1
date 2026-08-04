import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"
OUTPUT_CSV = BASE_DIR / "qwen_target_verb_smoketest_10.csv"
OUTPUT_JSONL = BASE_DIR / "qwen_target_verb_smoketest_10_raw.jsonl"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

N_SAMPLES = 10
TEMPERATURE = 0.0
MAX_TOKENS = 96

FIELDS = [
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


def build_system_prompt():
    return """
You are a structured cyber operation state annotator.

Your task is to infer only the immediate next-step target and action
from an ATT&CK parent-technique prefix.

Predict exactly two fields:

1. next_target_artifact
- The immediate object/entity/channel that the attacker will act on next.
- This is the direct next target, not the broad goal.

2. next_micro_verb
- The action applied to that target artifact.
- This is the immediate next operation, not the ATT&CK technique name.

Important distinctions:
- Choose the immediate next object, not a broad semantic category.
- Do not infer current stage or operational constraint.
- Focus only on: what object is acted on next, and what action is applied to it.
- network_configuration / system_information are different from process_inventory
- credential_material is different from access_token_context
- collect is different from enumerate
- modify is different from execute
- exfiltrate should be used only when the next action is actual exfiltration

Use only the allowed labels.
Be precise and conservative.
""".strip()


def build_user_prompt(row, schema):
    prefix = str(row["state"]).strip()

    target_options = ", ".join(schema["next_target_artifact"])
    verb_options = ", ".join(schema["next_micro_verb"])

    few_shot = """
Examples:

Example 1
Prefix:
T1082 T1033 T1016
Reasoning target/action:
- The attacker is continuing environment discovery
- The next direct object is network/system environment, not process inventory
Output:
{
  "next_target_artifact": "network_configuration",
  "next_micro_verb": "enumerate"
}

Example 2
Prefix:
T1003
Reasoning target/action:
- The next direct object is credential material
- The action is dumping it
Output:
{
  "next_target_artifact": "credential_material",
  "next_micro_verb": "dump"
}

Example 3
Prefix:
T1547
Reasoning target/action:
- The next direct object is a persistence mechanism
- The action is to modify/set it
Output:
{
  "next_target_artifact": "persistence_mechanism",
  "next_micro_verb": "modify"
}

Example 4
Prefix:
T1018 T1021
Reasoning target/action:
- The attacker is moving toward remote execution
- The next direct object is a command/service execution channel
Output:
{
  "next_target_artifact": "command_execution_channel",
  "next_micro_verb": "execute"
}
""".strip()

    return f"""
Task:
Infer only the immediate next target artifact and next micro verb from the ATT&CK parent-technique prefix.

{few_shot}

Now annotate this sample.

Prefix:
{prefix}

Allowed labels:
next_target_artifact: [{target_options}]
next_micro_verb: [{verb_options}]
""".strip()


def build_response_format(schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "target_verb_schema",
            "schema": {
                "type": "object",
                "properties": {
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


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Missing schema JSON: {SCHEMA_JSON}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    schema = load_schema(SCHEMA_JSON)
    response_format = build_response_format(schema)

    for col in ["annotation_id", "state"] + FIELDS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.head(N_SAMPLES).copy()

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


if __name__ == "__main__":
    main()