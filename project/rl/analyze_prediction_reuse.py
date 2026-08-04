from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    PROJECT_ROOT / "rl" / "rl_apt29_predictions_top5.csv",
    PROJECT_ROOT / "rl" / "rl_fin6_predictions_top5.csv",
]

def parse_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]

for path in FILES:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["prefix_set"] = df["state"].astype(str).apply(lambda s: set(parse_state(s)))
    df["top1_in_prefix"] = df.apply(lambda r: r["top1_pred"] in r["prefix_set"], axis=1)
    df["is_top1_correct"] = (df["top1_pred"].astype(str) == df["true_label"].astype(str))

    print("=" * 80)
    print(path.name)
    print("samples:", len(df))
    print("top1_in_prefix ratio:", df["top1_in_prefix"].mean())
    print("among wrong predictions:", df.loc[~df["is_top1_correct"], "top1_in_prefix"].mean())
    print("among correct predictions:", df.loc[df["is_top1_correct"], "top1_in_prefix"].mean() if df["is_top1_correct"].any() else 0.0)

    reused = df[df["top1_in_prefix"]][["sample_id", "state", "true_label", "top1_pred", "true_rank"]]
    print("\nexamples:")
    print(reused.head(10).to_string(index=False))