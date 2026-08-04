import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.json"

TARGET_KEYS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]

def normalize_to_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        # 兼容 schema 里用 {"labels": [...]} / {"values": [...]} 之类的写法
        for k in ["labels", "values", "options", "enum", "allowed_values"]:
            if k in value and isinstance(value[k], list):
                return value[k]
    return None

def main():
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_JSON}")

    with open(SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    print(f"[INFO] Loaded schema: {SCHEMA_JSON}\n")

    missing = [k for k in TARGET_KEYS if k not in schema]
    if missing:
        raise KeyError(f"Missing schema keys: {missing}")

    for key in TARGET_KEYS:
        values = normalize_to_list(schema[key])
        if values is None:
            raise ValueError(f"Schema key '{key}' is not a list-like label set.")

        values = [str(v).strip() for v in values]
        counter = Counter(values)
        duplicates = [v for v, c in counter.items() if c > 1]
        unique_values = list(dict.fromkeys(values))

        print("=" * 80)
        print(f"[FIELD] {key}")
        print(f"[COUNT] total={len(values)}, unique={len(unique_values)}")

        if duplicates:
            print(f"[WARN] duplicates={duplicates}")
        else:
            print("[WARN] duplicates=None")

        print("[VALUES]")
        for i, v in enumerate(unique_values, 1):
            print(f"  {i:02d}. {v}")
        print()

if __name__ == "__main__":
    main()
