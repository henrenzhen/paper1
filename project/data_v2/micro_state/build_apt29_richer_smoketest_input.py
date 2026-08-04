import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

GOLD_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
STEP_CSV = BASE_DIR / "APT29_step_context_table.csv"
OUTPUT_CSV = BASE_DIR / "APT29_richer_smoketest_10.csv"

def main():
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    step = pd.read_csv(STEP_CSV, encoding="utf-8-sig")

    gold = gold.copy()
    gold["source_org"] = gold["source_org"].astype(str).str.strip().str.lower()
    gold["plan_id"] = gold["plan_id"].astype(str).str.strip().str.lower()
    gold["scenario_id"] = gold["scenario_id"].astype(str).str.strip()
    gold["prefix_len"] = gold["prefix_len"].astype(int)

    step = step.copy()
    step["source_org"] = step["source_org"].astype(str).str.strip().str.lower()
    step["plan_id"] = step["plan_id"].astype(str).str.strip().str.lower()
    step["scenario_id"] = step["scenario_id"].astype(str).str.strip()
    step["step_idx"] = step["step_idx"].astype(int)

    # 只取前 10 条 APT29 smoke test
    subset = gold[gold["source_org"] == "apt29"].copy()
    subset["ann_num"] = (
        subset["annotation_id"].astype(str).str.extract(r"ms_(\d+)", expand=False).astype(int)
    )
    subset = subset.sort_values("ann_num").head(10).copy()

    # 下一步所在 YAML step
    subset["target_step_idx"] = subset["prefix_len"] + 1

    merged = subset.merge(
    step[
        [
            "source_org",
            "plan_id",
            "step_idx",
            "technique_attack_id",
            "technique_name",
            "description",
            "command_summary",
        ]
    ],
    left_on=["source_org", "plan_id", "target_step_idx"],
    right_on=["source_org", "plan_id", "step_idx"],
    how="left",
)

    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote: {OUTPUT_CSV}")
    print(f"[INFO] Rows : {len(merged)}")
    print("\n[PREVIEW]")
    preview_cols = [
        "annotation_id",
        "prefix_len",
        "target_step_idx",
        "true_label",
        "technique_attack_id",
        "technique_name",
        "description",
        "command_summary",
    ]
    print(merged[preview_cols].to_string(index=False))

if __name__ == "__main__":
    main()