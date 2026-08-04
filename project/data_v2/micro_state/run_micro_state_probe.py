from pathlib import Path
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data_v2" / "micro_state" / "micro_state_gold_v2_plus_v3.csv"
OUT_DIR = PROJECT_ROOT / "data_v2" / "micro_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # 只使用人工确认样本
    if "label_confidence" in df.columns:
        df["label_confidence"] = df["label_confidence"].astype(str).str.strip()
        df = df[df["label_confidence"] == "human_verified"].copy()

    # 去掉关键字段缺失
    df = df.dropna(subset=[
        "current_access_context",
        "exhausted_action_constraint",
        "next_target_artifact",
        "next_micro_verb",
        "true_label",
    ]).copy()

    # 清理空字符串
    for col in [
        "current_access_context",
        "exhausted_action_constraint",
        "next_target_artifact",
        "next_micro_verb",
        "true_label",
    ]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["current_access_context"] != "") &
        (df["exhausted_action_constraint"] != "") &
        (df["next_target_artifact"] != "") &
        (df["next_micro_verb"] != "") &
        (df["true_label"] != "")
    ].copy()

    if len(df) < 2:
        raise RuntimeError("Not enough valid samples for probe.")

    df["micro_state_text"] = df.apply(
        lambda r: " ; ".join([
            f"context={r['current_access_context']}",
            f"constraint={r['exhausted_action_constraint']}",
            f"target={r['next_target_artifact']}",
            f"verb={r['next_micro_verb']}",
        ]),
        axis=1
    )

    labels = sorted(df["true_label"].astype(str).unique().tolist())
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}

    y = df["true_label"].astype(str).map(label2id).values
    X_text = df["micro_state_text"].tolist()

    vec = CountVectorizer(token_pattern=r"(?u)\b[\w=]+\b")
    X = vec.fit_transform(X_text)

    loo = LeaveOneOut()

    top1_hits = 0
    top3_hits = 0
    records = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            multi_class="ovr",
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[0]
        true_id = y_test[0]

        present_classes = clf.classes_
        order = probs.argsort()[::-1]
        ranked_class_ids = present_classes[order]

        if true_id in ranked_class_ids:
            rank = list(ranked_class_ids).index(true_id) + 1
            label_seen_in_train = 1
        else:
            rank = -1
            label_seen_in_train = 0

        top1_hits += int(rank == 1)
        top3_hits += int(rank != -1 and rank <= 3)

        top1_pred = id2label[ranked_class_ids[0]]
        top3_preds = " || ".join(id2label[i] for i in ranked_class_ids[:3])

        records.append({
            "annotation_id": df.iloc[test_idx[0]]["annotation_id"],
            "sample_id": df.iloc[test_idx[0]]["sample_id"],
            "true_label": id2label[true_id],
            "micro_state_text": df.iloc[test_idx[0]]["micro_state_text"],
            "rank": rank,
            "top1_pred": top1_pred,
            "top3_preds": top3_preds,
            "label_seen_in_train": label_seen_in_train,
        })

    pred_df = pd.DataFrame(records)
    pred_df["rr"] = pred_df["rank"].apply(lambda x: 0.0 if x == -1 else 1.0 / x)

    n = len(pred_df)
    mrr = pred_df["rr"].mean()
    seen_ratio = pred_df["label_seen_in_train"].mean()

    out_path = OUT_DIR / "micro_state_probe_predictions.csv"
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== MICRO-STATE PROBE ===")
    print(f"samples={n}")
    print(f"num_labels={len(labels)}")
    print(f"label_seen_in_train_ratio={seen_ratio:.4f}")
    print(f"LOO_top1={top1_hits / n:.4f}")
    print(f"LOO_top3={top3_hits / n:.4f}")
    print(f"LOO_MRR={mrr:.4f}")

    print("\n=== LABEL DISTRIBUTION ===")
    print(df["true_label"].value_counts().to_string())

    print("\n=== SAMPLE PREDICTIONS ===")
    print(pred_df.head(10).to_string(index=False))

    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()