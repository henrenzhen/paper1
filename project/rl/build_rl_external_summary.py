from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "rl" / "apt29_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你当前已确认的 RL 主分布结果
SIM_TEST_METRICS = {
    "dataset": "sim_test_parent_min3",
    "samples": None,
    "top1": 0.5496,
    "top5": 0.8771,
    "mrr": 0.6868,
    "note": "in-distribution"
}

# 从刚才分析结果读取外部分布结果
summary_path = OUT_DIR / "summary_metrics.csv"
wrong_path = OUT_DIR / "wrong_top1_pred_frequency.csv"

summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
wrong_df = pd.read_csv(wrong_path, encoding="utf-8-sig")

apt29_metrics = {
    "dataset": "APT29_external_in184",
    "samples": int(summary_df.loc[0, "samples"]),
    "top1": float(summary_df.loc[0, "top1"]),
    "top5": float(summary_df.loc[0, "top5"]),
    "mrr": float(summary_df.loc[0, "mrr"]),
    "note": "external / OOD"
}

compare_df = pd.DataFrame([SIM_TEST_METRICS, apt29_metrics])
compare_df["top1_drop_vs_sim"] = compare_df["top1"].apply(
    lambda x: round((SIM_TEST_METRICS["top1"] - x), 4) if pd.notna(x) else None
)
compare_df["top5_drop_vs_sim"] = compare_df["top5"].apply(
    lambda x: round((SIM_TEST_METRICS["top5"] - x), 4) if pd.notna(x) else None
)
compare_df["mrr_drop_vs_sim"] = compare_df["mrr"].apply(
    lambda x: round((SIM_TEST_METRICS["mrr"] - x), 4) if pd.notna(x) else None
)

compare_out = OUT_DIR / "rl_sim_vs_apt29_comparison.csv"
compare_df.to_csv(compare_out, index=False, encoding="utf-8-sig")

top_wrong = wrong_df.head(10).copy()
top_wrong_out = OUT_DIR / "rl_apt29_top_wrong_labels_top10.csv"
top_wrong.to_csv(top_wrong_out, index=False, encoding="utf-8-sig")

print("=== RL IN-DIST vs EXTERNAL ===")
print(compare_df.to_string(index=False))

print("\n=== TOP WRONG TOP1 LABELS ON APT29 ===")
print(top_wrong.to_string(index=False))

print(f"\nSaved: {compare_out}")
print(f"Saved: {top_wrong_out}")
