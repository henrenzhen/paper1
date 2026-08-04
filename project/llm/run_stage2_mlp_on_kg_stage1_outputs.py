from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent          # ~/project/llm
PROJECT_ROOT = BASE_DIR.parent                      # ~/project
DATA_DIR = PROJECT_ROOT / "data"                   # ~/project/data

# 输入
STAGE1_PRED_CSV = DATA_DIR / "llm_kg_context_test_micro_state_details.csv"
GOLD_CSV = DATA_DIR / "micro_state_gold_v2_plus_v3.csv"

# 输出
OUTPUT_DETAIL_CSV = DATA_DIR / "stage2_mlp_kg_context_test_predictions.csv"
OUTPUT_METRIC_TXT = DATA_DIR / "stage2_mlp_kg_context_test_metrics.txt"

MICRO_COLS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]


def normalize_str(x):
    if x is None:
        return ""
    return str(x).strip()


def compute_metrics(y_true_text, probas, classes_text):
    top1_hit = 0
    top5_hit = 0
    mrr = 0.0
    rows = []

    for i, true_label in enumerate(y_true_text):
        probs = probas[i]
        order = np.argsort(probs)[::-1]
        ranked_labels = [classes_text[j] for j in order]

        pred_top1 = ranked_labels[0]
        rank = ranked_labels.index(true_label) + 1 if true_label in ranked_labels else -1

        if pred_top1 == true_label:
            top1_hit += 1
        if true_label in ranked_labels[:5]:
            top5_hit += 1
        if rank > 0:
            mrr += 1.0 / rank

        rows.append({
            "true_label": true_label,
            "pred_top1": pred_top1,
            "true_rank": rank,
            "top5_labels": " || ".join(ranked_labels[:5]),
            "top1_hit": int(pred_top1 == true_label),
            "top5_hit": int(true_label in ranked_labels[:5]),
            "rr": 1.0 / rank if rank > 0 else 0.0,
        })

    n = len(y_true_text)
    metrics = {
        "top1": top1_hit / n,
        "top5": top5_hit / n,
        "mrr": mrr / n,
    }
    return metrics, pd.DataFrame(rows)


def build_slot_vocab(gold_df):
    slot_vocab = {}
    for col in MICRO_COLS:
        vals = sorted(set(normalize_str(v) for v in gold_df[col].fillna("").tolist()))
        slot_vocab[col] = vals
    return slot_vocab


def encode_row_from_slots(row, prefix, slot_vocab):
    feats = []
    for col in MICRO_COLS:
        val = normalize_str(row[f"{prefix}__{col}"])
        vocab = slot_vocab[col]
        onehot = [0] * len(vocab)
        if val in vocab:
            onehot[vocab.index(val)] = 1
        feats.extend(onehot)
    return feats


def main():
    if not STAGE1_PRED_CSV.exists():
        raise FileNotFoundError(f"Missing Stage1 prediction CSV: {STAGE1_PRED_CSV}")
    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing gold CSV: {GOLD_CSV}")

    stage1 = pd.read_csv(STAGE1_PRED_CSV, encoding="utf-8-sig")
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")

    needed_gold = ["annotation_id", "true_label"] + MICRO_COLS
    for c in needed_gold:
        if c not in gold.columns:
            raise ValueError(f"Missing gold column: {c}")

    gold = gold[needed_gold].copy()
    for c in MICRO_COLS + ["annotation_id", "true_label"]:
        gold[c] = gold[c].fillna("").astype(str).str.strip()

    needed_stage1 = ["sequence_id", "true_label"] + [f"pred__{c}" for c in MICRO_COLS]
    for c in needed_stage1:
        if c not in stage1.columns:
            raise ValueError(f"Missing Stage1 column: {c}")

    for c in ["sequence_id", "true_label"] + [f"pred__{x}" for x in MICRO_COLS]:
        stage1[c] = stage1[c].fillna("").astype(str).str.strip()

    slot_vocab = build_slot_vocab(gold)

    # gold -> structured one-hot
    gold_tmp = gold.rename(columns={c: f"__{c}" for c in MICRO_COLS}).copy()
    X_train = np.array(
        [encode_row_from_slots(r, "", slot_vocab) for _, r in gold_tmp.iterrows()],
        dtype=np.float32
    )
    y_train_text = gold["true_label"].tolist()

    # stage1 predicted micro-state -> structured one-hot
    X_test = np.array(
        [encode_row_from_slots(r, "pred", slot_vocab) for _, r in stage1.iterrows()],
        dtype=np.float32
    )
    y_test_text = stage1["true_label"].tolist()

    # 标签编码，避免字符串类别引发 MLP 内部问题
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=16,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=False,
    )
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)
    classes_encoded = clf.classes_
    classes_text = label_encoder.inverse_transform(classes_encoded)

    metrics, pred_df = compute_metrics(y_test_text, probas, classes_text)

    out_df = pd.concat(
        [
            stage1[["sequence_id", "true_label"] + [f"pred__{c}" for c in MICRO_COLS]].reset_index(drop=True),
            pred_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=== STAGE2 MLP ON KG-STAGE1 OUTPUTS ===")
    lines.append(f"samples={len(stage1)}")
    lines.append(f"num_labels={len(set(y_test_text))}")
    lines.append(f"top1={metrics['top1']:.4f}")
    lines.append(f"top5={metrics['top5']:.4f}")
    lines.append(f"mrr={metrics['mrr']:.4f}")

    OUTPUT_METRIC_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved -> {OUTPUT_DETAIL_CSV}")
    print(f"Saved -> {OUTPUT_METRIC_TXT}")


if __name__ == "__main__":
    main()