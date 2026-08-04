from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/root/project")
FUSION_DIR = PROJECT_ROOT / "fusion"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_ranking_metrics(df: pd.DataFrame, score_col: str) -> dict:
    y_true: List[str] = []
    ranked_preds: List[List[str]] = []

    for instance_id, g in df.groupby("instance_id", sort=False):
        g = g.sort_values(score_col, ascending=False)
        gold = str(g["gold_label"].iloc[0]).strip()
        preds = g["candidate_tid"].astype(str).tolist()
        y_true.append(gold)
        ranked_preds.append(preds)

    def topk_acc(y_true, ranked_preds, k):
        hit = 0
        for y, preds in zip(y_true, ranked_preds):
            if y in preds[:k]:
                hit += 1
        return hit / len(y_true) if y_true else 0.0

    def mrr(y_true, ranked_preds):
        s = 0.0
        for y, preds in zip(y_true, ranked_preds):
            rr = 0.0
            for i, p in enumerate(preds, start=1):
                if p == y:
                    rr = 1.0 / i
                    break
            s += rr
        return s / len(y_true) if y_true else 0.0

    return {
        "num_instances": len(y_true),
        "top1": topk_acc(y_true, ranked_preds, 1),
        "top5": topk_acc(y_true, ranked_preds, 5),
        "mrr": mrr(y_true, ranked_preds),
    }


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    num_steps: int = 5000,
    l2: float = 1e-4,
):
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for step in range(num_steps):
        logits = X @ w + b
        probs = sigmoid(logits)

        error = probs - y
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = error.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    wins = 0.0
    ties = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        ties += np.sum(p == neg)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_csv",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_features_top5.csv"),
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_lr_top5_metrics.json"),
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_lr_top5_predictions.csv"),
    )
    args = parser.parse_args()

    ensure_dir(Path(args.metrics_json).parent)
    ensure_dir(Path(args.pred_csv).parent)

    df = pd.read_csv(args.feature_csv).copy()

    feature_cols = [
    "rl_rank",
    "rl_prob",
    "rl_top1_prob",
    "rl_margin_top1_top2",
    "rl_is_top1",
    "llm_is_top1",
    "artifact_semantic_match",
    "artifact_match_rank",
]

    missing = [c for c in feature_cols + ["label", "instance_id", "gold_label", "candidate_tid"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    instance_ids = df["instance_id"].drop_duplicates().tolist()
    split_idx = int(len(instance_ids) * 0.7)
    train_ids = set(instance_ids[:split_idx])
    test_ids = set(instance_ids[split_idx:])

    train_df = df[df["instance_id"].isin(train_ids)].copy()
    test_df = df[df["instance_id"].isin(test_ids)].copy()

    X_train = train_df[feature_cols].astype(float).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()

    X_test = test_df[feature_cols].astype(float).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    mean, std = standardize_fit(X_train)
    X_train_std = standardize_apply(X_train, mean, std)
    X_test_std = standardize_apply(X_test, mean, std)

    w, b = fit_logistic_regression(
        X_train_std,
        y_train,
        lr=0.05,
        num_steps=5000,
        l2=1e-4,
    )

    train_df["fusion_score"] = predict_proba(X_train_std, w, b)
    test_df["fusion_score"] = predict_proba(X_test_std, w, b)

    train_df["rl_score"] = train_df["rl_prob"]
    test_df["rl_score"] = test_df["rl_prob"]

    fusion_metrics = compute_ranking_metrics(test_df, "fusion_score")
    rl_metrics = compute_ranking_metrics(test_df, "rl_score")
    auc = binary_auc(y_test, test_df["fusion_score"].to_numpy())

    metrics = {
        "num_train_instances": len(train_ids),
        "num_test_instances": len(test_ids),
        "feature_cols": feature_cols,
        "fusion_metrics": fusion_metrics,
        "rl_metrics_same_split": rl_metrics,
        "candidate_level_auc": auc,
        "coef": {c: float(v) for c, v in zip(feature_cols, w)},
        "intercept": float(b),
    }

    out_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    out_df.to_csv(args.pred_csv, index=False)

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Saved:", args.pred_csv)
    print("Saved:", args.metrics_json)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()