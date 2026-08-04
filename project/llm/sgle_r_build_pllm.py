from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sgle_r_common import PROJECT_ROOT

LLM_DIR = PROJECT_ROOT / "llm"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def softmax_np(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 1:
        raise ValueError("softmax_np expects a 1D array")
    if xs.size == 0:
        return xs

    xs = xs - np.max(xs)
    exps = np.exp(xs)
    denom = np.sum(exps)
    if denom <= 0:
        raise ValueError("softmax denominator is non-positive")
    return exps / denom


def validate_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_test_sequence_scores.csv"),
    )
    parser.add_argument(
        "--mask_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_label_mask_test.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_test_pllm.csv"),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature applied to Stage 2 sequence scores before softmax.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Exponent applied to mask_score.",
    )
    args = parser.parse_args()

    if args.tau <= 0:
        raise ValueError("--tau must be > 0")
    if args.gamma < 0:
        raise ValueError("--gamma must be >= 0")

    ensure_dir(Path(args.output_csv).parent)

    seq_df = pd.read_csv(args.seq_csv)
    mask_df = pd.read_csv(args.mask_csv)

    validate_required_columns(
        seq_df,
        [
            "sample_id",
            "instance_id",
            "gold_label",
            "candidate_tid",
            "length_norm_score",
        ],
        "seq_df",
    )
    validate_required_columns(
        mask_df,
        [
            "sample_id",
            "instance_id",
            "gold_label",
            "candidate_tid",
            "mask_score",
        ],
        "mask_df",
    )

    seq_df = seq_df.copy()
    mask_df = mask_df.copy()

    seq_df["length_norm_score"] = seq_df["length_norm_score"].astype(float)
    mask_df["mask_score"] = mask_df["mask_score"].astype(float)

    # 防止 mask_score 为 0 或负数导致后续幂运算/归一化不稳定
    mask_df["mask_score"] = mask_df["mask_score"].clip(lower=1e-12)

    merged = seq_df.merge(
        mask_df[
            [
                "instance_id",
                "candidate_tid",
                "mask_score",
            ]
        ],
        on=["instance_id", "candidate_tid"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("Merged seq_df and mask_df is empty.")

    out_parts = []

    for instance_id, g in merged.groupby("instance_id", sort=False):
        g = g.copy()

        seq_scores = g["length_norm_score"].to_numpy(dtype=float)
        mask_scores = g["mask_score"].to_numpy(dtype=float)

        # Stage 2 score -> probability
        seq_probs = softmax_np(seq_scores / args.tau)

        # Stage 3 mask reweight
        weighted = seq_probs * np.power(mask_scores, args.gamma)
        weighted_sum = np.sum(weighted)
        if weighted_sum <= 0:
            raise ValueError(f"Weighted probability sum <= 0 for instance_id={instance_id}")
        weighted = weighted / weighted_sum

        g["seq_prob"] = seq_probs
        g["pllm"] = weighted
        g["rank_pllm"] = g["pllm"].rank(method="first", ascending=False).astype(int)

        g = g.sort_values(["rank_pllm", "candidate_tid"], ascending=[True, True])
        out_parts.append(g)

    out_df = pd.concat(out_parts, axis=0, ignore_index=True)
    out_df.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)

    for instance_id, g in out_df.groupby("instance_id", sort=False):
        gold = g["gold_label"].iloc[0]
        top5 = g.sort_values("rank_pllm").head(5)[["candidate_tid", "pllm"]]
        print("=" * 100)
        print(f"instance_id={instance_id} gold_label={gold}")
        for i, (_, row) in enumerate(top5.iterrows(), start=1):
            print(f"rank={i} candidate={row['candidate_tid']} pllm={row['pllm']:.6f}")


if __name__ == "__main__":
    main()