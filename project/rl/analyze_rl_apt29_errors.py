import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "rl" / "rl_apt29_predictions_top5.csv"
OUT_DIR = PROJECT_ROOT / "rl" / "apt29_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prefix_bucket(x: int) -> str:
    if x <= 3:
        return "1-3"
    if x <= 5:
        return "4-5"
    if x <= 10:
        return "6-10"
    if x <= 20:
        return "11-20"
    return "21+"


def main():
    df = pd.read_csv(PRED_PATH, encoding="utf-8-sig")

    # 基础派生列
    df["is_top1_correct"] = (df["top1_pred"].astype(str) == df["true_label"].astype(str)).astype(int)
    df["is_top5_correct"] = (df["true_rank"].astype(int) <= 5).astype(int)
    df["rr"] = 1.0 / df["true_rank"].astype(int)
    df["prefix_bucket"] = df["prefix_len"].astype(int).apply(prefix_bucket)

    # 1) 总体指标
    summary = pd.DataFrame([{
        "samples": len(df),
        "top1": df["is_top1_correct"].mean(),
        "top5": df["is_top5_correct"].mean(),
        "mrr": df["rr"].mean(),
        "avg_true_rank": df["true_rank"].mean(),
        "median_true_rank": df["true_rank"].median(),
        "max_true_rank": df["true_rank"].max(),
    }])
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    # 2) true rank 分布
    rank_bins = pd.DataFrame({
        "bucket": ["1", "2-5", "6-10", "11-20", "21-50", "51+"],
        "count": [
            (df["true_rank"] == 1).sum(),
            df["true_rank"].between(2, 5).sum(),
            df["true_rank"].between(6, 10).sum(),
            df["true_rank"].between(11, 20).sum(),
            df["true_rank"].between(21, 50).sum(),
            (df["true_rank"] >= 51).sum(),
        ]
    })
    rank_bins["ratio"] = rank_bins["count"] / len(df)
    rank_bins.to_csv(OUT_DIR / "true_rank_distribution.csv", index=False, encoding="utf-8-sig")

    # 3) 最常见 top1 预测标签
    top1_pred_freq = (
        df["top1_pred"]
        .value_counts(dropna=False)
        .rename_axis("top1_pred")
        .reset_index(name="count")
    )
    top1_pred_freq["ratio"] = top1_pred_freq["count"] / len(df)
    top1_pred_freq.to_csv(OUT_DIR / "top1_pred_frequency.csv", index=False, encoding="utf-8-sig")

    # 4) Top1 错误时，哪些标签最常被预测出来
    wrong_top1_freq = (
        df[df["is_top1_correct"] == 0]["top1_pred"]
        .value_counts(dropna=False)
        .rename_axis("wrong_top1_pred")
        .reset_index(name="count")
    )
    wrong_top1_freq["ratio_among_errors"] = wrong_top1_freq["count"] / max((df["is_top1_correct"] == 0).sum(), 1)
    wrong_top1_freq.to_csv(OUT_DIR / "wrong_top1_pred_frequency.csv", index=False, encoding="utf-8-sig")

    # 5) 各真实标签表现
    per_true_label = (
        df.groupby("true_label", as_index=False)
        .agg(
            samples=("true_label", "size"),
            top1=("is_top1_correct", "mean"),
            top5=("is_top5_correct", "mean"),
            mrr=("rr", "mean"),
            avg_true_rank=("true_rank", "mean"),
            most_common_top1_pred=("top1_pred", lambda s: s.value_counts().index[0]),
        )
        .sort_values(["samples", "mrr"], ascending=[False, False])
    )
    per_true_label.to_csv(OUT_DIR / "per_true_label_metrics.csv", index=False, encoding="utf-8-sig")

    # 6) prefix_len 分桶表现
    per_prefix_bucket = (
        df.groupby("prefix_bucket", as_index=False)
        .agg(
            samples=("prefix_bucket", "size"),
            top1=("is_top1_correct", "mean"),
            top5=("is_top5_correct", "mean"),
            mrr=("rr", "mean"),
            avg_true_rank=("true_rank", "mean"),
        )
    )
    bucket_order = {"1-3": 0, "4-5": 1, "6-10": 2, "11-20": 3, "21+": 4}
    per_prefix_bucket["_ord"] = per_prefix_bucket["prefix_bucket"].map(bucket_order)
    per_prefix_bucket = per_prefix_bucket.sort_values("_ord").drop(columns="_ord")
    per_prefix_bucket.to_csv(OUT_DIR / "per_prefix_bucket_metrics.csv", index=False, encoding="utf-8-sig")

    # 7) 最难样本：真实 rank 最大
    hardest = df.sort_values(["true_rank", "prefix_len"], ascending=[False, False]).head(50)
    hardest.to_csv(OUT_DIR / "hardest_cases_top50.csv", index=False, encoding="utf-8-sig")

    # 8) 最容易样本：top1 命中
    easiest = df[df["is_top1_correct"] == 1].sort_values(["prefix_len", "true_rank"], ascending=[True, True])
    easiest.to_csv(OUT_DIR / "top1_correct_cases.csv", index=False, encoding="utf-8-sig")

    print("=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== TRUE RANK DISTRIBUTION ===")
    print(rank_bins.to_string(index=False))
    print("\n=== TOP WRONG TOP1 PREDICTIONS ===")
    print(wrong_top1_freq.head(15).to_string(index=False))
    print("\n=== PREFIX BUCKET METRICS ===")
    print(per_prefix_bucket.to_string(index=False))
    print(f"\nSaved analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()