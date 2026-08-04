from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data_v2" / "micro_state" / "micro_state_gold_v2_plus_v3.csv"
OUT_DIR = PROJECT_ROOT / "data_v2" / "micro_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_micro_state_text(row):
    return " ; ".join([
        f"context={row['current_access_context']}",
        f"constraint={row['exhausted_action_constraint']}",
        f"target={row['next_target_artifact']}",
        f"verb={row['next_micro_verb']}",
    ])


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if "label_confidence" in df.columns:
        df["label_confidence"] = df["label_confidence"].astype(str).str.strip()
        df = df[df["label_confidence"] == "human_verified"].copy()

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

    df["micro_state_text"] = df.apply(build_micro_state_text, axis=1)

    labels = sorted(df["true_label"].unique().tolist())
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}

    X_text = df["micro_state_text"].tolist()
    y = df["true_label"].map(label2id).values

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w=]+\b")
    X = vectorizer.fit_transform(X_text)

    loo = LeaveOneOut()
    rows = []
    top1 = 0
    top3 = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            multi_class="ovr",
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[0]
        present_classes = clf.classes_
        ranked = present_classes[probs.argsort()[::-1]]

        true_id = y_test[0]
        if true_id in ranked:
            rank = list(ranked).index(true_id) + 1
            seen = 1
        else:
            rank = -1
            seen = 0

        top1 += int(rank == 1)
        top3 += int(rank != -1 and rank <= 3)

        rows.append({
            "annotation_id": df.iloc[test_idx[0]]["annotation_id"],
            "sample_id": df.iloc[test_idx[0]]["sample_id"],
            "true_label": id2label[true_id],
            "micro_state_text": df.iloc[test_idx[0]]["micro_state_text"],
            "rank": rank,
            "top1_pred": id2label[ranked[0]],
            "top3_preds": " || ".join(id2label[i] for i in ranked[:3]),
            "label_seen_in_train": seen,
        })

    pred_df = pd.DataFrame(rows)
    pred_df["rr"] = pred_df["rank"].apply(lambda x: 0.0 if x == -1 else 1.0 / x)

    out_path = OUT_DIR / "micro_state_predictor_predictions.csv"
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== MICRO-STATE PREDICTOR ===")
    print(f"samples={len(pred_df)}")
    print(f"num_labels={len(labels)}")
    print(f"label_seen_in_train_ratio={pred_df['label_seen_in_train'].mean():.4f}")
    print(f"top1={top1 / len(pred_df):.4f}")
    print(f"top3={top3 / len(pred_df):.4f}")
    print(f"mrr={pred_df['rr'].mean():.4f}")
    print("\n=== LABEL DISTRIBUTION ===")
    print(df['true_label'].value_counts().to_string())
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()