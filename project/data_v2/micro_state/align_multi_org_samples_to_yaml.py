import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

GOLD_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"

STEP_TABLES = {
    "apt29": BASE_DIR / "APT29_step_context_table.csv",
    "oilrig": BASE_DIR / "oilrig_step_context_table.csv",
    "sandworm": BASE_DIR / "sandworm_step_context_table.csv",
    "turla_carbon": BASE_DIR / "turla_carbon_step_context_table.csv",
}


def to_parent_tid(x: str) -> str:
    x = str(x).strip().upper()
    m = re.match(r"^(T\d{4})", x)
    return m.group(1) if m else x


def norm_state_list(state: str):
    state = str(state).strip()
    if not state:
        return []
    return [to_parent_tid(t) for t in state.split() if str(t).strip()]


def suffix_match_len(yaml_prefix, sample_prefix):
    """
    返回 sample_prefix 与 yaml_prefix 的最长后缀匹配长度
    """
    max_len = min(len(yaml_prefix), len(sample_prefix))
    for k in range(max_len, 0, -1):
        if yaml_prefix[-k:] == sample_prefix[-k:]:
            return k
    return 0


def find_best_alignment(yaml_seq, prefix_seq, true_label):
    """
    在 yaml parent-technique 序列中：
    - 找到所有等于 true_label 的位置
    - 对每个候选位置，计算其前缀与 sample prefix 的最长后缀匹配长度
    - 选后缀匹配长度最大的候选
    返回: (best_pos_0based, best_suffix_len)
    """
    true_label = to_parent_tid(true_label)
    best_pos = None
    best_len = -1

    for pos, tid in enumerate(yaml_seq):
        if tid != true_label:
            continue
        yaml_prefix = yaml_seq[:pos]
        match_len = suffix_match_len(yaml_prefix, prefix_seq)
        if match_len > best_len:
            best_len = match_len
            best_pos = pos

    return best_pos, best_len


def align_one_org(gold_df: pd.DataFrame, step_df: pd.DataFrame, org: str):
    org_gold = gold_df[gold_df["source_org"] == org].copy()
    if org_gold.empty:
        return pd.DataFrame()

    org_gold["ann_num"] = (
        org_gold["annotation_id"].astype(str).str.extract(r"ms_(\d+)", expand=False).astype(int)
    )
    org_gold = org_gold.sort_values("ann_num").copy()

    step_df = step_df.copy()
    step_df["step_idx"] = step_df["step_idx"].astype(int)
    step_df["parent_tid"] = step_df["technique_attack_id"].astype(str).map(to_parent_tid)
    step_sorted = step_df.sort_values("step_idx").reset_index(drop=True)
    yaml_seq = step_sorted["parent_tid"].tolist()

    rows = []
    for _, row in org_gold.iterrows():
        prefix_seq = norm_state_list(row["state"])
        true_label = to_parent_tid(row["true_label"])

        best_pos, best_len = find_best_alignment(yaml_seq, prefix_seq, true_label)

        out = row.to_dict()
        out["matched_yaml_pos_0based"] = best_pos if best_pos is not None else ""
        out["matched_suffix_len"] = best_len if best_pos is not None else 0
        out["prefix_seq_len"] = len(prefix_seq)

        out["matched_step_idx"] = ""
        out["matched_technique_attack_id"] = ""
        out["matched_parent_tid"] = ""
        out["matched_technique_name"] = ""
        out["matched_description"] = ""
        out["matched_command_summary"] = ""
        out["alignment_ok"] = 0

        if best_pos is not None:
            s = step_sorted.iloc[best_pos]
            out["matched_step_idx"] = int(s["step_idx"])
            out["matched_technique_attack_id"] = s["technique_attack_id"]
            out["matched_parent_tid"] = s["parent_tid"]
            out["matched_technique_name"] = s["technique_name"]
            out["matched_description"] = s["description"]
            out["matched_command_summary"] = s["command_summary"]

            # 这里把“parent tid 一致”视为基本对齐成功
            out["alignment_ok"] = int(s["parent_tid"] == true_label)

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing gold CSV: {GOLD_CSV}")

    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    gold["source_org"] = gold["source_org"].astype(str).str.strip().str.lower()

    total_samples = 0
    total_aligned = 0

    for org, step_path in STEP_TABLES.items():
        if not step_path.exists():
            print(f"[WARN] Missing step table for {org}: {step_path}")
            continue

        step_df = pd.read_csv(step_path, encoding="utf-8-sig")
        aligned_df = align_one_org(gold, step_df, org)

        if aligned_df.empty:
            print(f"[WARN] No samples found for {org}")
            continue

        out_path = BASE_DIR / f"{org}_aligned_richer_input.csv"
        aligned_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        n = len(aligned_df)
        ok = int(aligned_df["alignment_ok"].sum())
        total_samples += n
        total_aligned += ok

        print(f"[OK] {org}")
        print(f"     samples      : {n}")
        print(f"     alignment_ok : {ok}")
        print(f"     output       : {out_path}")

    print(f"\n[INFO] Total samples      : {total_samples}")
    print(f"[INFO] Total alignment_ok : {total_aligned}")


if __name__ == "__main__":
    main()