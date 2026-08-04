import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "micro_state_gold_v3_seed.csv"
OUTPUT_CSV = BASE_DIR / "micro_state_gold_v3_prefill.csv"

MICRO_STATE_COLS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded: {INPUT_CSV}")
    print(f"[INFO] Shape: {df.shape[0]} rows x {df.shape[1]} cols")

    missing_cols = [c for c in MICRO_STATE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("\n[CHECK] Micro-state columns status:")
    all_empty = True
    for col in MICRO_STATE_COLS:
        non_empty_mask = df[col].fillna("").astype(str).str.strip() != ""
        non_empty_count = int(non_empty_mask.sum())
        empty_count = int((~non_empty_mask).sum())

        print(f"  - {col}: non_empty={non_empty_count}, empty={empty_count}")

        if non_empty_count > 0:
            all_empty = False

    if all_empty:
        print("\n[RESULT] 这 4 个四元组列当前全空，可以安全开始做 prefill。")
    else:
        print("\n[RESULT] 这 4 个四元组列不是全空；后续 prefill 前要注意不要覆盖已有内容。")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Copied to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()