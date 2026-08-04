from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    PROJECT_ROOT / "rl" / "rl_apt29_predictions_top5.csv",
    PROJECT_ROOT / "rl" / "rl_fin6_predictions_top5.csv",
]

def parse_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]

def parse_topk_labels(s: str):
    return [x.strip() for x in str(s).split("||") if x.strip()]

def parse_topk_probs(s: str):
    return [float(x.strip()) for x in str(s).split("||") if x.strip()]

for path in FILES:
    df = pd.read_csv(path, encoding="utf-8-sig")

    masked_top1 = 0
    masked_top5 = 0
    masked_mrr = 0.0
    kept = 0

    for _, row in df.iterrows():
        prefix = set(parse_state(row["state"]))
        true_label = str(row["true_label"]).strip()
        labels = parse_topk_labels(row["top5_labels"])
        probs = parse_topk_probs(row["top5_probs"])

        filtered = [(lab, pr) for lab, pr in zip(labels, probs) if lab not in prefix]

        if not filtered:
            continue

        kept += 1
        filtered_labels = [x[0] for x in filtered]

        if filtered_labels[0] == true_label:
            masked_top1 += 1
        if true_label in filtered_labels[:5]:
            masked_top5 += 1
            rank = filtered_labels.index(true_label) + 1
            masked_mrr += 1.0 / rank
        else:
            # 只在 filtered top5 内算近似 MRR；没有命中则记 0
            masked_mrr += 0.0

    print("=" * 80)
    print(path.name)
    print("kept_samples:", kept)
    print("masked_top1:", masked_top1 / kept if kept else 0.0)
    print("masked_top5:", masked_top5 / kept if kept else 0.0)
    print("masked_mrr_approx:", masked_mrr / kept if kept else 0.0)