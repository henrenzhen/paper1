import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "micro_state_gold_v3_prefill.csv"
SCHEMA_JSON = BASE_DIR / "micro_state_label_schema.cleaned.json"
OUTPUT_CSV = BASE_DIR / "micro_state_gold_v3_prefill.v1.csv"

MICRO_COLS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]

def load_schema(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    allowed = {}
    for k in MICRO_COLS:
        if k not in schema:
            raise KeyError(f"Missing schema key: {k}")
        allowed[k] = set(schema[k])
    return allowed

# 第一版只做“保守预填”：
# - 只填我有把握的 true_label
# - 不能稳定映射的标签先留空
# - 不碰 label_confidence / annotation_notes
LABEL_TO_MICROSTATE = {
    "T1003": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "credential_material_not_collected",
        "next_target_artifact": "credential_material",
        "next_micro_verb": "dump",
    },
    "T1007": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "host_service_inventory_not_enumerated",
        "next_target_artifact": "service_inventory",
        "next_micro_verb": "enumerate",
    },
    "T1016": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "environment_not_enumerated",
        "next_target_artifact": "network_configuration",
        "next_micro_verb": "query",
    },
    "T1018": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "remote_targets_not_identified",
        "next_target_artifact": "remote_host_list",
        "next_micro_verb": "enumerate",
    },
    "T1033": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "current_user_context_not_confirmed",
        "next_target_artifact": "user_account_context",
        "next_micro_verb": "query",
    },
    "T1041": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "exfiltration_path_not_established",
        "next_target_artifact": "exfiltration_channel",
        "next_micro_verb": "exfiltrate",
    },
    "T1049": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "environment_not_enumerated",
        "next_target_artifact": "network_configuration",
        "next_micro_verb": "enumerate",
    },
    "T1069": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "privilege_context_not_enumerated",
        "next_target_artifact": "permission_group_context",
        "next_micro_verb": "enumerate",
    },
    "T1078": {
        "current_access_context": "valid_user_context",
        "exhausted_action_constraint": "current_user_context_not_confirmed",
        "next_target_artifact": "user_account_context",
        "next_micro_verb": "impersonate",
    },
    "T1082": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "environment_not_enumerated",
        "next_target_artifact": "system_information",
        "next_micro_verb": "enumerate",
    },
    "T1083": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "collection_target_not_identified",
        "next_target_artifact": "local_file_data",
        "next_micro_verb": "enumerate",
    },
    "T1087": {
        "current_access_context": "post_compromise_discovery",
        "exhausted_action_constraint": "current_user_context_not_confirmed",
        "next_target_artifact": "user_account_context",
        "next_micro_verb": "enumerate",
    },
    "T1189": {
        "current_access_context": "initial_foothold",
        "exhausted_action_constraint": "user_execution_path_not_established",
        "next_target_artifact": "user_execution_channel",
        "next_micro_verb": "trigger",
    },
    "T1489": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "impact_path_not_established",
        "next_target_artifact": "host_availability",
        "next_micro_verb": "disrupt",
    },
    "T1490": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "impact_path_not_established",
        "next_target_artifact": "system_configuration_store",
        "next_micro_verb": "modify",
    },
    "T1505": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "persistence_path_not_established",
        "next_target_artifact": "persistence_mechanism",
        "next_micro_verb": "modify",
    },
    "T1547": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "persistence_path_not_established",
        "next_target_artifact": "persistence_mechanism",
        "next_micro_verb": "modify",
    },
    "T1548": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "privilege_context_insufficient",
        "next_target_artifact": "access_token_context",
        "next_micro_verb": "elevate",
    },
    "T1552": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "credential_material_not_collected",
        "next_target_artifact": "stored_credentials",
        "next_micro_verb": "collect",
    },
    "T1555": {
        "current_access_context": "execution_on_host",
        "exhausted_action_constraint": "credential_material_not_collected",
        "next_target_artifact": "stored_credentials",
        "next_micro_verb": "collect",
    },
}

def is_empty(series: pd.Series):
    return series.fillna("").astype(str).str.strip() == ""

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"Schema JSON not found: {SCHEMA_JSON}")

    df = pd.read_csv(INPUT_CSV)
    allowed = load_schema(SCHEMA_JSON)

    # schema 校验
    for label, mapping in LABEL_TO_MICROSTATE.items():
        for col, val in mapping.items():
            if val not in allowed[col]:
                raise ValueError(f"[{label}] invalid schema value for {col}: {val}")

    filled_rows = 0
    unmapped_labels = set()
    touched_by_label = {}

    for idx, row in df.iterrows():
        true_label = str(row.get("true_label", "")).strip()
        if true_label not in LABEL_TO_MICROSTATE:
            if true_label:
                unmapped_labels.add(true_label)
            continue

        mapping = LABEL_TO_MICROSTATE[true_label]
        row_touched = False

        for col in MICRO_COLS:
            current_val = "" if pd.isna(row[col]) else str(row[col]).strip()
            if current_val == "":
                df.at[idx, col] = mapping[col]
                row_touched = True

        if row_touched:
            filled_rows += 1
            touched_by_label[true_label] = touched_by_label.get(true_label, 0) + 1

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Input : {INPUT_CSV}")
    print(f"[INFO] Output: {OUTPUT_CSV}")
    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Rows prefilled: {filled_rows}")

    print("\n[INFO] Prefilled rows by true_label:")
    for label in sorted(touched_by_label.keys()):
        print(f"  - {label}: {touched_by_label[label]}")

    print("\n[INFO] Unmapped true_label left blank:")
    for label in sorted(unmapped_labels):
        print(f"  - {label}")

if __name__ == "__main__":
    main()