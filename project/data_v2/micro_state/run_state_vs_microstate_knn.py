from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def evaluate_representation(df: pd.DataFrame, text_col: str, name: str):
    texts = df[text_col].astype(str).tolist()

    vec = CountVectorizer(token_pattern=r"(?u)\b[\w=]+\b")
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)

    records = []
    top1_hits = 0
    mrr_sum = 0.0

    for i in range(len(df)):
        sims = sim[i].copy()
        sims[i] = -1.0
        ranked_idx = sims.argsort()[::-1]

        true_label = df.iloc[i]["true_label"]
        rank = -1
        for r, j in enumerate(ranked_idx, start=1):
            if df.iloc[j]["true_label"] == true_label:
                rank = r
                break

        nn_idx = ranked_idx[0]
        nn_label = df.iloc[nn_idx]["true_label"]

        top1_hits += int(nn_label == true_label)
        mrr_sum += 0.0 if rank == -1 else 1.0 / rank

        records.append({
            "representation": name,
            "annotation_id": df.iloc[i]["annotation_id"],
            "sample_id": df.iloc[i]["sample_id"],
            "true_label": true_label,
            "query_text": df.iloc[i][text_col],
            "nn1_sample_id": df.iloc[nn_idx]["sample_id"],
            "nn1_label": nn_label,
            "match_at_1": int(nn_label == true_label),
            "rank_of_first_same_label": rank,
        })

    out_df = pd.DataFrame(records)
    metrics = {
        "representation": name,
        "samples": len(df),
        "top1_nn_accuracy": top1_hits / len(df),
        "mrr_same_label": mrr_sum / len(df),
    }
    return metrics, out_df


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    df["ann_num"] = (
        df["annotation_id"]
        .astype(str)
        .str.extract(r"ms_(\d+)", expand=False)
        .astype(int)
    )

    # 不再限制只看前 20 条，直接使用当前文件中的全部样本

    for col in [
        "state",
        "true_label",
        "current_access_context",
        "exhausted_action_constraint",
        "next_target_artifact",
        "next_micro_verb",
    ]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["state"] != "") &
        (df["true_label"] != "") &
        (df["current_access_context"] != "") &
        (df["exhausted_action_constraint"] != "") &
        (df["next_target_artifact"] != "") &
        (df["next_micro_verb"] != "")
    ].copy()

    df["micro_state_text"] = df.apply(build_micro_state_text, axis=1)
    df["state_plus_micro_state"] = df["state"] + " || " + df["micro_state_text"]

    all_metrics = []
    all_details = []

    for text_col, name in [
        ("state", "state"),
        ("micro_state_text", "micro_state"),
        ("state_plus_micro_state", "state_plus_micro_state"),
    ]:
        metrics, details = evaluate_representation(df, text_col, name)
        all_metrics.append(metrics)
        all_details.append(details)

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.concat(all_details, ignore_index=True)

    metrics_path = OUT_DIR / "state_vs_microstate_knn_metrics.csv"
    details_path = OUT_DIR / "state_vs_microstate_knn_details.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    print("=== 1-NN RETRIEVAL COMPARISON ===")
    print(metrics_df.to_string(index=False))

    print(f"\nsaved -> {metrics_path}")
    print(f"saved -> {details_path}")


if __name__ == "__main__":
    main()