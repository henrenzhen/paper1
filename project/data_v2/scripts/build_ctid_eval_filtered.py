from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]

ctid_path = project_root / "data_v2" / "external_ctid" / "parsed" / "ctid_eval_parent.csv"
label_path = project_root / "data" / "rl_label_vocab.csv"
out_path = project_root / "data_v2" / "external_ctid" / "parsed" / "ctid_eval_parent_in184.csv"

df = pd.read_csv(ctid_path, encoding="utf-8-sig")
labels = pd.read_csv(label_path, encoding="utf-8-sig")

label_set = set(labels["technique_id_parent"].astype(str).str.strip())

df["next_technique_id_parent"] = df["next_technique_id_parent"].astype(str).str.strip()
keep_df = df[df["next_technique_id_parent"].isin(label_set)].copy()
drop_df = df[~df["next_technique_id_parent"].isin(label_set)].copy()

keep_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("original samples:", len(df))
print("kept samples:", len(keep_df))
print("dropped samples:", len(drop_df))
print("dropped labels:", sorted(drop_df['next_technique_id_parent'].unique().tolist()))
print("saved to:", out_path)