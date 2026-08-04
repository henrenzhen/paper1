import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

GOLD_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
STEP_CSV = BASE_DIR / "APT29_step_context_table.csv"
OUTPUT_CSV = BASE_DIR / "APT29_richer_smoketest_10_aligned.csv"


def to_parent_tid(x: str) -> str:
    x = str(x).strip().upper()
    m = re.match(r"^(T\d{4})", x)
    return m.group(1) if m else x


def norm_state_list(state: str):
    state = str(state).strip()
    if not state:
        return []
    return [to_parent_tid(t) for t in state.split() if str(t).strip()]


def find_alignment(yaml_seq, prefix_seq, true_label):
    """
    在 yaml parent-technique 序列中寻找：
    prefix_seq 后面紧跟 true_label 的位置。
    返回匹配到的 next-step 下标（0-based），否则 None。
    """
    n = len(prefix_seq)
    true_label = to_parent_tid(true_label)

    for start in range(0, len(yaml_seq) - n):
        if yaml_seq[start:start + n] == prefix_seq:
            next_idx = start + n
            if next_idx < len(yaml_seq) and yaml_seq[next_idx] == true_label:
                return next_idx
    return None


def main():
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    step = pd.read_csv(STEP_CSV, encoding="utf-8-sig")

    gold["source_org"] = gold["source_org"].astype(str).str.strip().str.lower()
    gold = gold[gold["source_org"] == "apt29"].copy()
    gold["ann_num"] = (
        gold["annotation_id"].astype(str).str.extract(r"ms_(\d+)", expand=False).astype(int)
    )
    gold = gold.sort_values("ann_num").head(10).copy()

    step["source_org"] = step["source_org"].astype(str).str.strip().str.lower()
    step = step[step["source_org"] == "apt29"].copy()
    step["step_idx"] = step["step_idx"].astype(int)
    step["parent_tid"] = step["technique_attack_id"].astype(str).map(to_parent_tid)

    yaml_seq = step.sort_values("step_idx")["parent_tid"].tolist()
    step_sorted = step.sort_values("step_idx").reset_index(drop=True)

    rows = []

    for _, row in gold.iterrows():
        prefix_seq = norm_state_list(row["state"])
        true_label = to_parent_tid(row["true_label"])

        matched_yaml_pos = find_alignment(yaml_seq, prefix_seq, true_label)

        out = row.to_dict()
        out["matched_yaml_pos_0based"] = matched_yaml_pos if matched_yaml_pos is not None else ""
        out["matched_step_idx"] = ""
        out["matched_technique_attack_id"] = ""
        out["matched_parent_tid"] = ""
        out["matched_technique_name"] = ""
        out["matched_description"] = ""
        out["matched_command_summary"] = ""
        out["alignment_ok"] = 0

        if matched_yaml_pos is not None:
            s = step_sorted.iloc[matched_yaml_pos]
            out["matched_step_idx"] = int(s["step_idx"])
            out["matched_technique_attack_id"] = s["technique_attack_id"]
            out["matched_parent_tid"] = s["parent_tid"]
            out["matched_technique_name"] = s["technique_name"]
            out["matched_description"] = s["description"]
            out["matched_command_summary"] = s["command_summary"]
            out["alignment_ok"] = int(s["parent_tid"] == true_label)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote: {OUTPUT_CSV}")
    print(f"[INFO] Rows : {len(out_df)}")
    print(f"[INFO] alignment_ok sum: {int(out_df['alignment_ok'].sum())}/{len(out_df)}")

    preview_cols = [
        "annotation_id",
        "true_label",
        "matched_step_idx",
        "matched_technique_attack_id",
        "matched_parent_tid",
        "matched_technique_name",
        "alignment_ok",
    ]
    print("\n[PREVIEW]")
    print(out_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()