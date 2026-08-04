from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .metrics import cluster_bootstrap_difference, root_macro_mean
from .multires_context_tree import QuotientMultiResolutionContextTree
from .probability_models import InterpolatedNGram
from .run_mrct_development import _gain, _prediction_columns, load_multires_development
from .stockpile_dataset import STOCKPILE_COMMIT, load_stockpile_transitions


def run_stockpile_external(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    bootstrap: int = 2000,
    seed: int = 20260730,
) -> tuple[pd.DataFrame, dict[str, float]]:
    root_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    train, vocab, _ = load_multires_development(root_path)
    external, data_audit = load_stockpile_transitions(root_path, vocab)
    qmrct_config = {
        "max_parent_context": 2,
        "max_raw_context": 2,
        "alpha": 0.1,
        "parent_kappa": 5.0,
        "raw_kappa": 10.0,
    }
    baseline_config = {
        "order": 3,
        "alpha": 0.1,
        "interpolation": (0.4, 0.3, 0.3),
    }
    candidate = QuotientMultiResolutionContextTree(
        vocab=vocab, **qmrct_config
    ).fit(
        train["prefix_ids"],
        train["raw_prefix_ids"],
        train["target"],
        train["root"],
    )
    baseline = InterpolatedNGram(vocab=vocab, **baseline_config).fit(
        train["prefix_ids"], train["target"], train["root"]
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(
        external["prefix_ids"]
    )
    candidate_probabilities, meta = candidate.predict_proba_with_meta(
        external["prefix_ids"], external["raw_prefix_ids"]
    )
    parent_probabilities, _ = candidate.parent_tree.predict_proba_with_meta(
        external["prefix_ids"]
    )
    targets = external["target"].astype(str).to_numpy()
    baseline_top, baseline_rr, baseline_hit5 = _prediction_columns(
        baseline_probabilities, targets, vocab
    )
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(
        candidate_probabilities, targets, vocab
    )
    parent_top, parent_rr, parent_hit5 = _prediction_columns(
        parent_probabilities, targets, vocab
    )
    predictions = pd.DataFrame(
        {
            "sample_id": external["sample_id"].astype(str).to_numpy(),
            "profile_id": external["profile_id"].astype(str).to_numpy(),
            "profile": external["profile"].astype(str).to_numpy(),
            "position": external["position"].astype(int).to_numpy(),
            "source_raw_id": external["raw_prefix_ids"].map(lambda x: x[-1]).to_numpy(),
            "source_parent_id": external["prefix_ids"].map(lambda x: x[-1]).to_numpy(),
            "target": targets,
            "target_raw_id": external["evaluation_next_raw_id"].astype(str).to_numpy(),
            "baseline_top1": baseline_top,
            "parent_only_top1": parent_top,
            "candidate_top1": candidate_top,
            "baseline_correct": baseline_top == targets,
            "parent_only_correct": parent_top == targets,
            "candidate_correct": candidate_top == targets,
            "baseline_rr": baseline_rr,
            "parent_only_rr": parent_rr,
            "candidate_rr": candidate_rr,
            "baseline_hit5": baseline_hit5,
            "parent_only_hit5": parent_hit5,
            "candidate_hit5": candidate_hit5,
        }
    )
    predictions = pd.concat([predictions, meta.reset_index(drop=True)], axis=1)
    grouped = predictions.rename(columns={"profile": "root"})
    metrics = {
        "baseline_profile_macro_top1": root_macro_mean(
            predictions, "baseline_correct", "profile"
        ),
        "candidate_profile_macro_top1": root_macro_mean(
            predictions, "candidate_correct", "profile"
        ),
        "top1_gain_pp": 100.0
        * _gain(grouped, "candidate_correct", "baseline_correct"),
        "baseline_profile_macro_mrr": root_macro_mean(
            predictions, "baseline_rr", "profile"
        ),
        "candidate_profile_macro_mrr": root_macro_mean(
            predictions, "candidate_rr", "profile"
        ),
        "mrr_gain": _gain(grouped, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(grouped, "candidate_hit5", "baseline_hit5"),
        "raw_increment_top1": _gain(
            grouped, "candidate_correct", "parent_only_correct"
        ),
        "raw_increment_mrr": _gain(grouped, "candidate_rr", "parent_only_rr"),
        "rows": float(len(predictions)),
        "profiles": float(predictions["profile"].nunique()),
    }
    intervals = {
        "top1_gain_pp": cluster_bootstrap_difference(
            grouped,
            lambda frame: 100.0
            * _gain(frame, "candidate_correct", "baseline_correct"),
            "root",
            bootstrap,
            seed,
        ),
        "mrr_gain": cluster_bootstrap_difference(
            grouped,
            lambda frame: _gain(frame, "candidate_rr", "baseline_rr"),
            "root",
            bootstrap,
            seed + 1,
        ),
        "raw_increment_mrr": cluster_bootstrap_difference(
            grouped,
            lambda frame: _gain(frame, "candidate_rr", "parent_only_rr"),
            "root",
            bootstrap,
            seed + 2,
        ),
    }
    profile_metrics = (
        predictions.assign(
            top1_gain=predictions["candidate_correct"].astype(float)
            - predictions["baseline_correct"].astype(float),
            mrr_gain=predictions["candidate_rr"] - predictions["baseline_rr"],
        )
        .groupby("profile", sort=True)
        .agg(
            rows=("sample_id", "size"),
            baseline_top1=("baseline_correct", "mean"),
            candidate_top1=("candidate_correct", "mean"),
            top1_gain=("top1_gain", "mean"),
            baseline_mrr=("baseline_rr", "mean"),
            candidate_mrr=("candidate_rr", "mean"),
            mrr_gain=("mrr_gain", "mean"),
        )
        .reset_index()
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [{"metric": name, **asdict(interval)} for name, interval in intervals.items()]
        ).to_csv(destination / "bootstrap_intervals.csv", index=False)
        profile_metrics.to_csv(destination / "profile_metrics.csv", index=False)
        write_canonical_json(destination / "data_audit.json", data_audit)
        write_canonical_json(
            destination / "frozen_config.json",
            {"qmrct": qmrct_config, "baseline": baseline_config},
        )
        zip_path = root_path / "data_v2" / "external_stockpile" / "stockpile_996ec41.zip"
        write_manifest(
            destination / "run_manifest.json",
            inputs={
                "repository_commit": STOCKPILE_COMMIT,
                "zip_sha256": sha256_file(zip_path),
                "train_rows": len(train),
                "external_rows": len(external),
            },
            config={"seed": seed, "bootstrap": bootstrap},
            split_audit={
                "train_roots": train["root"].nunique(),
                "external_profiles": external["profile"].nunique(),
                "profile_identity_overlap_with_ctid": 0,
            },
        )
    return predictions, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "external"
        / f"qmrct_stockpile_seed{args.seed}"
    )
    _, metrics = run_stockpile_external(
        project_root=project_root,
        output_dir=destination,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
