from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MS_DIR = PROJECT_ROOT / "data_v2" / "micro_state"

v1_path = MS_DIR / "micro_state_gold_v1.csv"
v2_path = MS_DIR / "micro_state_gold_v2_seed.csv"
out_path = MS_DIR / "micro_state_gold_v2.csv"

df1 = pd.read_csv(v1_path, encoding="utf-8-sig")
df2 = pd.read_csv(v2_path, encoding="utf-8-sig")

merged = pd.concat([df1, df2], ignore_index=True)

# 按 annotation_id 排序
merged["ann_num"] = merged["annotation_id"].astype(str).str.extract(r"ms_(\d+)").astype(int)
merged = merged.sort_values("ann_num").drop(columns="ann_num").reset_index(drop=True)

merged.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"[OK] wrote: {out_path}")
print(f"[INFO] total={len(merged)}")
print(merged[["annotation_id", "sample_id", "true_label"]].tail(15).to_string(index=False))