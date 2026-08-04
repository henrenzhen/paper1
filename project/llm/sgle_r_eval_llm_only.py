from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from sgle_r_common import PROJECT_ROOT, compute_basic_metrics

LLM_DIR = PROJECT_ROOT / "llm"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pllm_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_test_pllm.csv"),
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=str(LLM_DIR / "sgle_r_llm_only_metrics.json"),
    )
    parser.add_argument(
        "--top5_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_llm_only_top5.csv"),
    )
    args = parser.parse_args()

    ensure_dir(Path(args.metrics_json).parent)
    ensure_dir(Path(args.top5_csv).parent)

    df = pd.read_csv(args.pllm_csv)
    required = ["instance_id", "gold_label", "candidate_tid", "rank_pllm", "pllm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    y_true: List[str] = []
    ranked_preds: List[List[str]] = []
    top5_rows: List[Dict] = []

    for instance_id, g in df.groupby("instance_id", sort=False):
        g = g.sort_values("rank_pllm", ascending=True).copy()
        gold = str(g["gold_label"].iloc[0]).strip()
        preds = g["candidate_tid"].astype(str).tolist()

        y_true.append(gold)
        ranked_preds.append(preds)

        for _, row in g.head(5).iterrows():
            top5_rows.append(
                {
                    "instance_id": instance_id,
                    "gold_label": gold,
                    "rank_pllm": int(row["rank_pllm"]),
                    "candidate_tid": str(row["candidate_tid"]),
                    "pllm": float(row["pllm"]),
                }
            )

    metrics = compute_basic_metrics(y_true=y_true, ranked_preds=ranked_preds)
    metrics["num_instances"] = len(y_true)

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(top5_rows).to_csv(args.top5_csv, index=False)

    print("Saved:", args.metrics_json)
    print("Saved:", args.top5_csv)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()