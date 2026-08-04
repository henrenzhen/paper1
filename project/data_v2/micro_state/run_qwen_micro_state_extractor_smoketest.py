import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"
OUTPUT_CSV = BASE_DIR / "qwen_micro_state_smoketest_10.csv"
OUTPUT_JSONL = BASE_DIR / "qwen_micro_state_smoketest_10_raw.jsonl"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

N_SAMPLES = 10
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


def build_system_prompt():
    return """
You are a structured cyber operation state annotator.

Your task is to infer the NEXT-STEP micro-state from an ATT&CK parent-technique prefix.

You must predict exactly four fields:

1. current_access_context
- What operational access stage the attacker is CURRENTLY in.
- This is the current situation implied by the prefix.
- It is NOT the next action.

2. exhausted_action_constraint
- What prerequisite is still missing before the attacker can proceed effectively.
- This is the main unmet operational constraint.
- It is NOT the target object.

3. next_target_artifact
- The immediate object/entity/channel that the attacker will act on next.
- This is the direct target of the next micro action.
- It is NOT the final goal.

4. next_micro_verb
- The action applied to the next_target_artifact.
- This should describe the immediate next operation, not the ATT&CK technique name.

Important distinctions:
- environment_not_enumerated is different from running_processes_not_enumerated
- network_configuration / system_information are different from process_inventory
- credential_material is different from access_token_context
- collect is different from enumerate
- current_access_context should not default to post_compromise_discovery unless the prefix really indicates that stage
- next_target_artifact must be the immediate next object, not a broad semantic guess

Choose the best labels only from the allowed label sets.
Be precise and conservative.
""".strip()

def build_user_prompt(row, schema):
    prefix = str(row["state"]).strip()

    def fmt(name):
        return f"{name}: [{', '.join(schema[name])}]"

    allowed_block = "\n".join([
        fmt("current_access_context"),
        fmt("exhausted_action_constraint"),
        fmt("next_target_artifact"),
        fmt("next_micro_verb"),
    ])

    few_shot_block = """
Examples:

Example 1
Prefix:
T1082 T1033 T1016
Interpretation:
- The attacker is already executing discovery on a host
- The missing prerequisite is broader environment enumeration
- The next object is network/system environment, not process inventory
Output:
{
  "current_access_context": "post_compromise_discovery",
  "exhausted_action_constraint": "environment_not_enumerated",
  "next_target_artifact": "network_configuration",
  "next_micro_verb": "enumerate"
}

Example 2
Prefix:
T1078 T1021
Interpretation:
- The attacker already has valid user / remote access context
- The missing prerequisite is a usable remote execution path
- The next object is the remote execution/service channel
Output:
{
  "current_access_context": "internal_host_access",
  "exhausted_action_constraint": "remote_execution_path_not_established",
  "next_target_artifact": "command_execution_channel",
  "next_micro_verb": "execute"
}

Example 3
Prefix:
T1003
Interpretation:
- The attacker is executing on a compromised host
- The missing prerequisite is credential acquisition
- The next object is credential material itself
Output:
{
  "current_access_context": "execution_on_host",
  "exhausted_action_constraint": "credential_material_not_collected",
  "next_target_artifact": "credential_material",
  "next_micro_verb": "dump"
}

Example 4
Prefix:
T1547
Interpretation:
- The attacker is already executing on host
- The missing prerequisite is a persistence path
- The next object is the persistence mechanism
Output:
{
  "current_access_context": "execution_on_host",
  "exhausted_action_constraint": "persistence_path_not_established",
  "next_target_artifact": "persistence_mechanism",
  "next_micro_verb": "modify"
}
""".strip()

    prompt = f"""
Task:
Infer the next-step micro-state from the ATT&CK parent-technique prefix.

{few_shot_block}

Now annotate this sample.

Prefix:
{prefix}

Allowed labels:
{allowed_block}

Return the best structured answer using only the allowed labels.
""".strip()

    return prompt


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