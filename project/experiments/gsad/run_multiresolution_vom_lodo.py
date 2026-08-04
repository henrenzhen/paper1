from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .metrics import (
    cluster_bootstrap_difference,
    domain_root_bootstrap_difference,
    root_macro_mean,
)
from .probability_models import InterpolatedNGram
from .run_mrct_development import _gain, _prediction_columns, load_multires_development
from .run_qsmr_multisource_sensitivity import load_multisource_domains


def mix_raw_residual(
    parent_probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    parent_prefixes: Sequence[Sequence[str]],
    raw_prefixes: Sequence[Sequence[str]],
    raw_metadata: pd.DataFrame,
    maximum_weight: float = 0.5,
    kappa: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent_probabilities, dtype=float)
    raw = np.asarray(raw_probabilities, dtype=float)
    if parent.shape != raw.shape or parent.ndim != 2:
        raise ValueError("parent and raw probabilities must have the same matrix shape")
    if not (
        len(parent)
        == len(parent_prefixes)
        == len(raw_prefixes)
        == len(raw_metadata)
    ):
        raise ValueError("probabilities, prefixes, and metadata must align")
    if not 0 <= float(maximum_weight) <= 1 or float(kappa) < 0:
        raise ValueError("maximum_weight must be in [0,1] and kappa nonnegative")
    support = pd.to_numeric(
        raw_metadata["context_root_support"], errors="raise"
    ).to_numpy(dtype=float)
    is_subtechnique = np.asarray(
        [
            bool(parent_prefix)
            and bool(raw_prefix)
            and str(parent_prefix[-1]) != str(raw_prefix[-1])
            for parent_prefix, raw_prefix in zip(
                parent_prefixes, raw_prefixes, strict=True
            )
        ],
        dtype=bool,
    )
    reliability = (
        np.ones_like(support)
        if float(kappa) == 0
        else support / (support + float(kappa))
    )
    weights = float(maximum_weight) * reliability * is_subtechnique.astype(float)
    output = (1.0 - weights[:, None]) * parent + weights[:, None] * raw
    output /= output.sum(axis=1, keepdims=True)
    return output, weights


def _model(vocab: Sequence[str]) -> InterpolatedNGram:
    return InterpolatedNGram(
        vocab,
        order=3,
        alpha=0.1,
        interpolation=(0.2, 0.3, 0.5),
        domain_power=0.0,
    )


def _evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Sequence[str],
    maximum_weight: float,
    kappa: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    parent_model = _model(vocab).fit(
        train["prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"],
    )
    raw_model = _model(vocab).fit(
        train["raw_prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"],
    )
    parent, _ = parent_model.predict_proba_with_meta(test["prefix_ids"])
    raw, raw_metadata = raw_model.predict_proba_with_meta(test["raw_prefix_ids"])
    candidate, weights = mix_raw_residual(
        parent,
        raw,
        test["prefix_ids"],
        test["raw_prefix_ids"],
        raw_metadata,
        maximum_weight,
        kappa,
    )
    targets = test["target"].astype(str).to_numpy()
    parent_top, parent_rr, parent_hit5 = _prediction_columns(parent, targets, vocab)
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(
        candidate, targets, vocab
    )
    predictions = pd.DataFrame(
        {
            "root": test["root"].astype(str).to_numpy(),
            "target": targets,
            "is_subtechnique": [
                bool(parent_prefix)
                and bool(raw_prefix)
                and str(parent_prefix[-1]) != str(raw_prefix[-1])
                for parent_prefix, raw_prefix in zip(
                    test["prefix_ids"], test["raw_prefix_ids"], strict=True
                )
            ],
            "raw_weight": weights,
            "baseline_correct": parent_top == targets,
            "candidate_correct": candidate_top == targets,
            "baseline_rr": parent_rr,
            "candidate_rr": candidate_rr,
            "baseline_hit5": parent_hit5,
            "candidate_hit5": candidate_hit5,
        }
    )
    subtech = predictions.loc[predictions["is_subtechnique"].astype(bool)]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "candidate_top1": root_macro_mean(predictions, "candidate_correct"),
        "top1_gain_pp": 100.0
        * _gain(predictions, "candidate_correct", "baseline_correct"),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "subtech_top1_gain_pp": 100.0
        * _gain(subtech, "candidate_correct", "baseline_correct"),
        "subtech_mrr_gain": _gain(subtech, "candidate_rr", "baseline_rr"),
        "subtech_fraction": float(predictions["is_subtechnique"].mean()),
        "mean_raw_weight": float(predictions["raw_weight"].mean()),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    return predictions, metrics


def run_lodo(
    project_root: Path,
    output_dir: Path | None = None,
    maximum_weight: float = 0.5,
    kappa: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(project_root)
    _, vocab, _ = load_multires_development(root_path)
    domains = load_multisource_domains(root_path, vocab)
    heldout_domains = ("ctid", "attack_flow", "stockpile")
    rows: list[dict[str, Any]] = []
    prediction_outputs: list[pd.DataFrame] = []
    interval_rows: list[dict[str, Any]] = []
    for index, heldout in enumerate(heldout_domains):
        train = pd.concat(
            [frame for name, frame in domains.items() if name != heldout],
            ignore_index=True,
        )
        predictions, metrics = _evaluate(
            train, domains[heldout], vocab, maximum_weight, kappa
        )
        predictions["heldout_domain"] = heldout
        prediction_outputs.append(predictions)
        rows.append({"heldout_domain": heldout, **metrics})
        for offset, (metric, function) in enumerate(
            {
                "top1_gain_pp": lambda frame: 100.0
                * _gain(frame, "candidate_correct", "baseline_correct"),
                "mrr_gain": lambda frame: _gain(
                    frame, "candidate_rr", "baseline_rr"
                ),
            }.items()
        ):
            interval = cluster_bootstrap_difference(
                predictions,
                function,
                "root",
                2000,
                20261100 + index * 100 + offset,
            )
            interval_rows.append(
                {"heldout_domain": heldout, "metric": metric, **asdict(interval)}
            )
    metrics_frame = pd.DataFrame(rows)
    predictions = pd.concat(prediction_outputs, ignore_index=True)
    aggregate_rows = []
    for offset, (metric, candidate, baseline, scale) in enumerate(
        (
            ("top1_gain_pp", "candidate_correct", "baseline_correct", 100.0),
            ("mrr_gain", "candidate_rr", "baseline_rr", 1.0),
        )
    ):
        work = predictions.assign(
            _candidate=scale * predictions[candidate].astype(float),
            _baseline=scale * predictions[baseline].astype(float),
        )
        interval = domain_root_bootstrap_difference(
            work,
            "_candidate",
            "_baseline",
            "heldout_domain",
            "root",
            10000,
            20261200 + offset,
        )
        aggregate_rows.append({"metric": metric, **asdict(interval)})
    aggregate = pd.DataFrame(aggregate_rows)
    lookup = aggregate.set_index("metric")
    summary = {
        "all_domain_top1_nonnegative": bool(
            (metrics_frame["top1_gain_pp"] >= 0).all()
        ),
        "all_domain_mrr_nonnegative": bool((metrics_frame["mrr_gain"] >= 0).all()),
        "mean_top1_gain_pp": float(metrics_frame["top1_gain_pp"].mean()),
        "mean_mrr_gain": float(metrics_frame["mrr_gain"].mean()),
        "aggregate_top1_ci_lower_pp": float(lookup.loc["top1_gain_pp", "lower"]),
        "aggregate_mrr_ci_lower": float(lookup.loc["mrr_gain", "lower"]),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        metrics_frame.to_csv(destination / "domain_metrics.csv", index=False)
        pd.DataFrame(interval_rows).to_csv(
            destination / "bootstrap_intervals.csv", index=False
        )
        aggregate.to_csv(destination / "aggregate_bootstrap_intervals.csv", index=False)
        predictions.to_csv(destination / "predictions.csv", index=False)
        write_canonical_json(destination / "summary.json", summary)
        write_manifest(
            destination / "run_manifest.json",
            inputs={
                "domain_rows": {name: len(frame) for name, frame in domains.items()},
                "development_cache_sha256": sha256_file(
                    root_path / "data_v2" / "core" / "sim_development_multires_min3.csv"
                ),
            },
            config={
                "maximum_weight": maximum_weight,
                "kappa": kappa,
                "heldout_domains": list(heldout_domains),
                "baseline": "equal-domain parent-context trigram",
            },
            split_audit={
                "heldout_excluded_from_training": True,
                "attack_flow_ctid_overlap_excluded": True,
            },
        )
    return metrics_frame, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-weight", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "external"
        / "multiresolution_vom_lodo_seed20260730"
    )
    _, summary = run_lodo(
        project_root, destination, args.maximum_weight, args.kappa
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
