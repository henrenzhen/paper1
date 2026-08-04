from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]

ctid_path = project_root / "data_v2" / "external_ctid" / "parsed" / "ctid_eval_parent.csv"
label_path = project_root / "data" / "rl_label_vocab.csv"

df = pd.read_csv(ctid_path, encoding="utf-8-sig")
labels = pd.read_csv(label_path, encoding="utf-8-sig")

label_set = set(labels["technique_id_parent"].astype(str).str.strip())
ctid_set = set(df["next_technique_id_parent"].astype(str).str.strip())

missing = sorted(ctid_set - label_set)
covered = sorted(ctid_set & label_set)

print("CTID unique next-parent count:", len(ctid_set))
print("Covered by RL label vocab:", len(covered))
print("Missing from RL label vocab:", len(missing))
print("Missing labels:", missing)
print("Coverage ratio:", f"{len(covered)}/{len(ctid_set)} = {len(covered)/max(len(ctid_set),1):.4f}")