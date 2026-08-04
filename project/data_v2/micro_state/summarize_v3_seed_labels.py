import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "micro_state_gold_v3_prefill.csv"
OUTPUT_CSV = BASE_DIR / "v3_seed_label_summary.csv"

REQUIRED_COLS = [
    "annotation_id",
    "source_org",
    "true_label",
]

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[INFO] Loaded: {INPUT_CSV}")
    print(f"[INFO] Rows: {len(df)}")

    # 1) 按组织统计
    print("\n[BY source_org]")
    org_counts = df["source_org"].fillna("").astype(str).str.strip().value_counts()
    for org, cnt in org_counts.items():
        print(f"  - {org}: {cnt}")

    # 2) 全局 true_label 统计
    print("\n[BY true_label]")
    label_counts = (
        df["true_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_index()
    )
    for label, cnt in label_counts.items():
        print(f"  - {label}: {cnt}")

    # 3) 组织 x true_label 交叉汇总
    summary = (
        df.groupby(["source_org", "true_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source_org", "true_label"])
    )

    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Wrote summary to: {OUTPUT_CSV}")

    # 4) 打印 unique label 数
    unique_labels = sorted(df["true_label"].fillna("").astype(str).str.strip().unique())
    print(f"\n[INFO] Unique true_label count: {len(unique_labels)}")
    print("[INFO] Unique true_label list:")
    for i, label in enumerate(unique_labels, 1):
        print(f"  {i:02d}. {label}")

if __name__ == "__main__":
    main()