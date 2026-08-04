import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "micro_state_label_schema.json"
OUTPUT_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"

TARGET_KEYS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]

def extract_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in ["labels", "values", "options", "enum", "allowed_values"]:
            if k in value and isinstance(value[k], list):
                return value[k]
    raise ValueError(f"Unsupported schema field format: {type(value)}")

def dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Schema file not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    cleaned = {}
    for key in TARGET_KEYS:
        if key not in schema:
            raise KeyError(f"Missing key in schema: {key}")
        values = extract_list(schema[key])
        cleaned[key] = dedupe_keep_order(values)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote cleaned schema to: {OUTPUT_JSON}")
    for key in TARGET_KEYS:
        print(f"[FIELD] {key}: {len(cleaned[key])} unique labels")

if __name__ == "__main__":
    main()