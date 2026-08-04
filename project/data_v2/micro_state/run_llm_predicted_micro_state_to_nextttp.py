from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

GOLD_CSV = BASE_DIR / "micro_state_gold_v2_plus_v3.csv"
LLM_DETAIL_CSV = BASE_DIR / "qwen_richer_multi_org_56_details.csv"

OUTPUT_PRED_CSV = BASE_DIR / "llm_predicted_micro_state_to_nextttp_details.csv"
OUTPUT_METRIC_TXT = BASE_DIR / "llm_predicted_micro_state_to_nextttp_metrics.txt"

MICRO_COLS = [
    "current_access_context",
    "exhausted_action_constraint",
    "next_target_artifact",
    "next_micro_verb",
]


def build_gold_text(row):
    return " ; ".join([
        f"context={row['current_access_context']}",
        f"constraint={row['exhausted_action_constraint']}",
        f"target={row['next_target_artifact']}",
        f"verb={row['next_micro_verb']}",
    ])


def build_pred_text(row):
    return " ; ".join([
        f"context={row['pred__current_access_context']}",
        f"constraint={row['pred__exhausted_action_constraint']}",
        f"target={row['pred__next_target_artifact']}",
        f"verb={row['pred__next_micro_verb']}",
    ])


def evaluate_one(true_label, probs, classes_):
    order = probs.argsort()[::-1]
    ranked_labels = [classes_[j] for j in order]

    top1_hit = int(ranked_labels[0] == true_label)
    top5_hit = int(true_label in ranked_labels[:5])

    if true_label in ranked_labels:
        rank = ranked_labels.index(true_label) + 1
        rr = 1.0 / rank
    else:
        rank = -1
        rr = 0.0

    return {
        "true_label": true_label,
        "pred_top1": ranked_labels[0],
        "true_rank": rank,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "rr": rr,
        "top5_labels": ",".join(ranked_labels[:5]),
    }


def main():
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    llm = pd.read_csv(LLM_DETAIL_CSV, encoding="utf-8-sig")

    # 全 80 条 gold，用作 Stage 2 训练池
    keep_cols = ["annotation_id", "true_label"] + MICRO_COLS
    gold = gold[keep_cols].copy()

    for c in MICRO_COLS:
        gold[c] = gold[c].fillna("").astype(str).str.strip()

    gold["gold_micro_state_text"] = gold.apply(build_gold_text, axis=1)

    # 56 条 LLM richer-input 预测结果
    needed = ["annotation_id", "source_org", "true_label"] + [f"pred__{c}" for c in MICRO_COLS]
    llm = llm[needed].copy()

    for c in MICRO_COLS:
        llm[f"pred__{c}"] = llm[f"pred__{c}"].fillna("").astype(str).str.strip()

    llm["pred_micro_state_text"] = llm.apply(build_pred_text, axis=1)

    # 只保留 annotation_id 能在 gold 中找到的样本
    df = llm.merge(
        gold[["annotation_id"]],
        on="annotation_id",
        how="inner",
    )

    if df.empty:
        raise ValueError("No overlapping annotation_id between LLM detail file and gold file.")

    eval_rows = []

    for _, test_row in df.iterrows():
        test_ann = test_row["annotation_id"]

        # 训练集：gold micro-state -> label，排除当前测试样本
        train_gold = gold[gold["annotation_id"] != test_ann].copy()
        X_train = train_gold["gold_micro_state_text"].tolist()
        y_train = train_gold["true_label"].tolist()

        # 测试集：LLM predicted micro-state
        X_test = [test_row["pred_micro_state_text"]]

        vec = TfidfVectorizer()
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)

        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )
        clf.fit(Xtr, y_train)

        probs = clf.predict_proba(Xte)[0]
        classes_ = clf.classes_

        row_eval = evaluate_one(
            true_label=test_row["true_label"],
            probs=probs,
            classes_=classes_,
        )

        eval_rows.append({
            "annotation_id": test_row["annotation_id"],
            "source_org": test_row["source_org"],
            "pred_micro_state_text": test_row["pred_micro_state_text"],
            **row_eval,
        })

    rank_df = pd.DataFrame(eval_rows)

    metrics = {
        "top1": rank_df["top1_hit"].mean(),
        "top5": rank_df["top5_hit"].mean(),
        "mrr": rank_df["rr"].mean(),
    }

    rank_df.to_csv(OUTPUT_PRED_CSV, index=False, encoding="utf-8-sig")

    label_seen_ratio = 0.0
    seen_flags = []
    for _, test_row in df.iterrows():
        other_labels = set(gold[gold["annotation_id"] != test_row["annotation_id"]]["true_label"])
        seen_flags.append(int(test_row["true_label"] in other_labels))
    if seen_flags:
        label_seen_ratio = sum(seen_flags) / len(seen_flags)

    text = []
    text.append("=== LLM PREDICTED MICRO-STATE -> NEXTTTP ===")
    text.append(f"samples={len(df)}")
    text.append(f"num_labels={df['true_label'].nunique()}")
    text.append(f"label_seen_in_train_ratio={label_seen_ratio:.4f}")
    text.append(f"top1={metrics['top1']:.4f}")
    text.append(f"top5={metrics['top5']:.4f}")
    text.append(f"mrr={metrics['mrr']:.4f}")

    OUTPUT_METRIC_TXT.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"\nSaved -> {OUTPUT_PRED_CSV}")
    print(f"Saved -> {OUTPUT_METRIC_TXT}")


if __name__ == "__main__":
    main()