import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OLD_GOLD = BASE_DIR / "micro_state_gold_v2.csv"
NEW_V3 = BASE_DIR / "micro_state_gold_v3_prefill.v1.csv"
OUTPUT = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"

ID_COL = "annotation_id"

def main():
    if not OLD_GOLD.exists():
        raise FileNotFoundError(f"Old gold not found: {OLD_GOLD}")
    if not NEW_V3.exists():
        raise FileNotFoundError(f"New v3 file not found: {NEW_V3}")

    old_df = pd.read_csv(OLD_GOLD)
    new_df = pd.read_csv(NEW_V3)

    print(f"[INFO] Loaded old gold: {OLD_GOLD} -> {len(old_df)} rows")
    print(f"[INFO] Loaded new v3 : {NEW_V3} -> {len(new_df)} rows")

    # 列对齐检查
    old_cols = list(old_df.columns)
    new_cols = list(new_df.columns)

    missing_in_new = [c for c in old_cols if c not in new_cols]
    extra_in_new = [c for c in new_cols if c not in old_cols]

    if missing_in_new:
        raise ValueError(f"New file is missing columns from old gold: {missing_in_new}")

    if extra_in_new:
        print(f"[WARN] New file has extra columns not in old gold: {extra_in_new}")
        # 这里只保留 old gold 的列顺序，避免合并后列漂移
        new_df = new_df[old_cols]
    else:
        new_df = new_df[old_cols]

    # ID 检查
    if ID_COL not in old_df.columns or ID_COL not in new_df.columns:
        raise ValueError(f"Missing required id column: {ID_COL}")

    old_ids = set(old_df[ID_COL].fillna("").astype(str).str.strip())
    new_ids = set(new_df[ID_COL].fillna("").astype(str).str.strip())

    overlap = sorted(x for x in (old_ids & new_ids) if x)
    if overlap:
        raise ValueError(f"Found overlapping annotation_id values: {overlap[:10]} ... total={len(overlap)}")

    merged = pd.concat([old_df, new_df], ignore_index=True)

    # 再做一次重复检查
    dup_mask = merged[ID_COL].fillna("").astype(str).str.strip().duplicated(keep=False)
    dup_ids = merged.loc[dup_mask, ID_COL].fillna("").astype(str).str.strip().unique().tolist()
    dup_ids = [x for x in dup_ids if x]
    if dup_ids:
        raise ValueError(f"Duplicate annotation_id found after merge: {dup_ids[:10]} ... total={len(dup_ids)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote merged file: {OUTPUT}")
    print(f"[INFO] Total merged rows: {len(merged)}")
    print(f"[INFO] Old rows: {len(old_df)} | New rows: {len(new_df)}")

    # 简单统计一下新旧编号范围
    merged_ids = merged[ID_COL].fillna("").astype(str).str.strip().tolist()
    print(f"[INFO] Non-empty annotation_id count: {sum(1 for x in merged_ids if x)}")

if __name__ == "__main__":
    main()