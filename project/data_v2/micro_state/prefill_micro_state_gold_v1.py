from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_V2_ROOT = PROJECT_ROOT / "data_v2"

IN_PATH = DATA_V2_ROOT / "micro_state" / "micro_state_gold_v1_seed.csv"
OUT_PATH = DATA_V2_ROOT / "micro_state" / "micro_state_gold_v1.csv"


DISCOVERY_SET = {
    "T1016", "T1018", "T1033", "T1049", "T1069", "T1082", "T1087", "T1518", "T1007"
}
EXECUTION_SET = {
    "T1059", "T1204", "T1569", "T1047", "T1218"
}
CRED_OR_PRIV_SET = {
    "T1003", "T1134", "T1548", "T1552", "T1055"
}
PERSISTENCE_SET = {
    "T1547", "T1546", "T1053", "T1037"
}
COLLECTION_SET = {
    "T1114", "T1115", "T1119", "T1005", "T1560"
}
EXFIL_SET = {
    "T1041", "T1567", "T1537"
}


TECHNIQUE_HINTS = {
    "T1016": {
        "next_target_artifact": "network_configuration",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "environment_not_enumerated",
    },
    "T1018": {
        "next_target_artifact": "remote_host_list",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "remote_targets_not_identified",
    },
    "T1033": {
        "next_target_artifact": "user_account_context",
        "next_micro_verb": "query",
        "exhausted_action_constraint": "current_user_context_not_confirmed",
    },
    "T1047": {
        "next_target_artifact": "wmi_execution_channel",
        "next_micro_verb": "execute",
        "exhausted_action_constraint": "remote_execution_path_not_established",
    },
    "T1049": {
        "next_target_artifact": "network_connection_state",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "network_relationships_not_enumerated",
    },
    "T1053": {
        "next_target_artifact": "scheduled_task",
        "next_micro_verb": "schedule",
        "exhausted_action_constraint": "persistence_path_not_established",
    },
    "T1059": {
        "next_target_artifact": "command_execution_channel",
        "next_micro_verb": "execute",
        "exhausted_action_constraint": "execution_capability_not_established",
    },
    "T1069": {
        "next_target_artifact": "permission_group_context",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "privilege_structure_not_enumerated",
    },
    "T1082": {
        "next_target_artifact": "system_information",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "host_environment_not_profiled",
    },
    "T1087": {
        "next_target_artifact": "account_inventory",
        "next_micro_verb": "enumerate",
        "exhausted_action_constraint": "account_landscape_not_enumerated",
    },
    "T1114": {
        "next_target_artifact": "mailbox_data",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "collection_target_not_identified",
    },
    "T1115": {
        "next_target_artifact": "clipboard_data",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "user_activity_data_not_collected",
    },
    "T1119": {
        "next_target_artifact": "local_file_data",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "collection_target_not_identified",
    },
    "T1134": {
        "next_target_artifact": "access_token_context",
        "next_micro_verb": "impersonate",
        "exhausted_action_constraint": "privilege_context_insufficient",
    },
    "T1204": {
        "next_target_artifact": "user_execution_channel",
        "next_micro_verb": "trigger",
        "exhausted_action_constraint": "user_execution_path_not_established",
    },
    "T1547": {
        "next_target_artifact": "persistence_mechanism",
        "next_micro_verb": "modify",
        "exhausted_action_constraint": "persistence_path_not_established",
    },
    "T1548": {
        "next_target_artifact": "privilege_boundary",
        "next_micro_verb": "elevate",
        "exhausted_action_constraint": "privilege_context_insufficient",
    },
    "T1552": {
        "next_target_artifact": "stored_credentials",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "credential_material_not_collected",
    },
    "T1560": {
        "next_target_artifact": "staged_collection",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "data_not_staged",
    },
    "T1567": {
        "next_target_artifact": "external_transfer_channel",
        "next_micro_verb": "exfiltrate",
        "exhausted_action_constraint": "staged_data_not_exfiltrated",
    },
    "T1569": {
        "next_target_artifact": "service_execution_channel",
        "next_micro_verb": "execute",
        "exhausted_action_constraint": "remote_execution_path_not_established",
    },
    "T1003": {
        "next_target_artifact": "credential_material",
        "next_micro_verb": "dump",
        "exhausted_action_constraint": "credential_material_not_collected",
    },
    "T1005": {
        "next_target_artifact": "local_file_data",
        "next_micro_verb": "collect",
        "exhausted_action_constraint": "collection_target_not_identified",
    },
}


def parse_state(state: str) -> list[str]:
    return [x.strip() for x in str(state).split() if x.strip()]


def infer_current_access_context(tokens: list[str], prefix_len: int) -> str:
    token_set = set(tokens)

    if token_set & CRED_OR_PRIV_SET:
        return "valid_user_or_privileged_context"

    if token_set & EXECUTION_SET:
        return "execution_on_host"

    discovery_count = len(token_set & DISCOVERY_SET)
    if discovery_count >= 2:
        return "post_compromise_discovery"

    if prefix_len <= 2:
        return "initial_foothold"

    if token_set & COLLECTION_SET:
        return "internal_host_access"

    return "execution_on_host"


def infer_constraint(true_label: str, tokens: list[str]) -> str:
    if true_label in TECHNIQUE_HINTS:
        return TECHNIQUE_HINTS[true_label]["exhausted_action_constraint"]

    if true_label in DISCOVERY_SET:
        return "environment_not_enumerated"
    if true_label in CRED_OR_PRIV_SET:
        return "privilege_context_insufficient"
    if true_label in PERSISTENCE_SET:
        return "persistence_path_not_established"
    if true_label in COLLECTION_SET:
        return "collection_target_not_identified"
    if true_label in EXFIL_SET:
        return "staged_data_not_exfiltrated"

    return "operational_context_not_sufficiently_prepared"


def infer_target_and_verb(true_label: str) -> tuple[str, str]:
    if true_label in TECHNIQUE_HINTS:
        return (
            TECHNIQUE_HINTS[true_label]["next_target_artifact"],
            TECHNIQUE_HINTS[true_label]["next_micro_verb"],
        )
    return ("unknown_artifact", "unknown_action")


def main():
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

    prefill_contexts = []
    prefill_constraints = []
    prefill_targets = []
    prefill_verbs = []
    prefill_notes = []

    for _, row in df.iterrows():
        tokens = parse_state(row["state"])
        prefix_len = int(row["prefix_len"])
        true_label = str(row["true_label"]).strip()

        context = infer_current_access_context(tokens, prefix_len)
        constraint = infer_constraint(true_label, tokens)
        target, verb = infer_target_and_verb(true_label)

        prefill_contexts.append(context)
        prefill_constraints.append(constraint)
        prefill_targets.append(target)
        prefill_verbs.append(verb)
        prefill_notes.append("auto_prefill_v1")

    df["current_access_context"] = prefill_contexts
    df["exhausted_action_constraint"] = prefill_constraints
    df["next_target_artifact"] = prefill_targets
    df["next_micro_verb"] = prefill_verbs

    # 只给预填置信度，不冒充人工 gold
    if "label_confidence" in df.columns:
        df["label_confidence"] = "auto_heuristic"
    else:
        df["label_confidence"] = "auto_heuristic"

    if "annotation_notes" in df.columns:
        df["annotation_notes"] = prefill_notes
    else:
        df["annotation_notes"] = prefill_notes

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {OUT_PATH}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()