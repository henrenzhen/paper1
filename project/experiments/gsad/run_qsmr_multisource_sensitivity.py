from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .attack_flow_dataset import load_attack_flow_transitions
from .metrics import (
    cluster_bootstrap_difference,
    domain_root_bootstrap_difference,
    root_macro_mean,
)
from .probability_models import InterpolatedNGram
from .run_mrct_development import _gain, _prediction_columns, load_multires_development
from .run_qmrct_external import load_ctid_external
from .semimarkov_router import QuotientSemiMarkovRouter
from .stockpile_dataset import load_stockpile_transitions


def _domain_frame(frame: pd.DataFrame, domain: str, root_column: str) -> pd.DataFrame:
    output = frame.loc[:, ["prefix_ids", "raw_prefix_ids", "target"]].copy()
    output["domain"] = str(domain)
    output["root"] = str(domain) + ":" + frame[root_column].astype(str)
    return output


def load_multisource_domains(
    project_root: Path, vocab: Sequence[str]
) -> dict[str, pd.DataFrame]:
    sim, _, _ = load_multires_development(project_root)
    ctid = load_ctid_external(project_root)
    attack_flow, _ = load_attack_flow_transitions(
        project_root, vocab, exclude_overlapping_ctid=True
    )
    stockpile, _ = load_stockpile_transitions(project_root, vocab)
    return {
        "sim": _domain_frame(sim, "sim", "root"),
        "ctid": _domain_frame(ctid, "ctid", "actor"),
        "attack_flow": _domain_frame(attack_flow, "attack_flow", "flow"),
        "stockpile": _domain_frame(stockpile, "stockpile", "profile"),
    }


def _evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Sequence[str],
    domain_balanced: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline = InterpolatedNGram(
        vocab, order=3, alpha=0.1, interpolation=(0.2, 0.3, 0.5)
    ).fit(
        train["prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"] if domain_balanced else None,
    )
    candidate = QuotientSemiMarkovRouter(
        vocab, kappa=2.0, destination_kappa=5.0
    ).fit(
        train["prefix_ids"],
        train["raw_prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"] if domain_balanced else None,
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(test["prefix_ids"])
    candidate_probabilities, _ = candidate.predict_proba_with_meta(
        baseline_probabilities,
        test["prefix_ids"],
        test["raw_prefix_ids"],
        destination_fraction=1.0,
    )
    targets = test["target"].astype(str).to_numpy()
    baseline_top, baseline_rr, baseline_hit5 = _prediction_columns(
        baseline_probabilities, targets, vocab
    )
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(
        candidate_probabilities, targets, vocab
    )
    predictions = pd.DataFrame(
        {
            "root": test["root"].astype(str).to_numpy(),
            "target": targets,
            "is_self": [
                bool(prefix) and str(prefix[-1]) == target
                for prefix, target in zip(test["prefix_ids"], targets, strict=True)
            ],
            "baseline_correct": baseline_top == targets,
            "candidate_correct": candidate_top == targets,
            "baseline_rr": baseline_rr,
            "candidate_rr": candidate_rr,
            "baseline_hit5": baseline_hit5,
            "candidate_hit5": candidate_hit5,
        }
    )
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "candidate_top1": root_macro_mean(predictions, "candidate_correct"),
        "baseline_mrr": root_macro_mean(predictions, "baseline_rr"),
        "candidate_mrr": root_macro_mean(predictions, "candidate_rr"),
        "top1_gain_pp": 100.0 * _gain(
            predictions, "candidate_correct", "baseline_correct"
        ),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "nonself_top1_gain_pp": 100.0 * _gain(
            nonself, "candidate_correct", "baseline_correct"
        ),
        "nonself_mrr_gain": _gain(nonself, "candidate_rr", "baseline_rr"),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    return predictions, metrics


def run_sensitivity(
    project_root: Path,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(project_root)
    _, vocab, _ = load_multires_development(root_path)
    domains = load_multisource_domains(root_path, vocab)
    rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    prediction_outputs: list[pd.DataFrame] = []
    heldout_domains = ("ctid", "attack_flow", "stockpile")
    for heldout in heldout_domains:
        train = pd.concat(
            [frame for name, frame in domains.items() if name != heldout],
            ignore_index=True,
        )
        test = domains[heldout]
        balanced_predictions, balanced_metrics = _evaluate(
            train, test, vocab, domain_balanced=True
        )
        root_predictions, root_metrics = _evaluate(
            train, test, vocab, domain_balanced=False
        )
        comparison = balanced_predictions.copy()
        comparison["root_only_correct"] = root_predictions["candidate_correct"].to_numpy()
        comparison["root_only_rr"] = root_predictions["candidate_rr"].to_numpy()
        comparison["root_baseline_correct"] = root_predictions[
            "baseline_correct"
        ].to_numpy()
        comparison["root_baseline_rr"] = root_predictions["baseline_rr"].to_numpy()
        interval_specs = {
            "top1_gain_pp": lambda frame: 100.0
            * _gain(frame, "candidate_correct", "baseline_correct"),
            "mrr_gain": lambda frame: _gain(
                frame, "candidate_rr", "baseline_rr"
            ),
            "domain_balance_top1_increment_pp": lambda frame: 100.0
            * _gain(frame, "candidate_correct", "root_only_correct"),
            "domain_balance_mrr_increment": lambda frame: _gain(
                frame, "candidate_rr", "root_only_rr"
            ),
            "baseline_balance_top1_increment_pp": lambda frame: 100.0
            * _gain(frame, "baseline_correct", "root_baseline_correct"),
            "baseline_balance_mrr_increment": lambda frame: _gain(
                frame, "baseline_rr", "root_baseline_rr"
            ),
        }
        for offset, (metric, function) in enumerate(interval_specs.items()):
            interval = cluster_bootstrap_difference(
                comparison,
                function,
                "root",
                2000,
                20260730 + 100 * len(rows) + offset,
            )
            interval_rows.append(
                {"heldout_domain": heldout, "metric": metric, **asdict(interval)}
            )
        comparison["heldout_domain"] = heldout
        prediction_outputs.append(comparison)
        rows.append(
            {
                "heldout_domain": heldout,
                **{f"balanced_{key}": value for key, value in balanced_metrics.items()},
                **{f"root_only_{key}": value for key, value in root_metrics.items()},
                "domain_balance_increment_top1_pp": 100.0
                * (
                    balanced_metrics["candidate_top1"]
                    - root_metrics["candidate_top1"]
                ),
                "domain_balance_increment_mrr": balanced_metrics["candidate_mrr"]
                - root_metrics["candidate_mrr"],
                "domain_balance_baseline_increment_top1_pp": 100.0
                * (
                    balanced_metrics["baseline_top1"]
                    - root_metrics["baseline_top1"]
                ),
                "domain_balance_baseline_increment_mrr": balanced_metrics[
                    "baseline_mrr"
                ]
                - root_metrics["baseline_mrr"],
            }
        )
    metrics = pd.DataFrame(rows)
    intervals = pd.DataFrame(interval_rows)
    combined_predictions = pd.concat(prediction_outputs, ignore_index=True)
    aggregate_frame = combined_predictions.assign(
        candidate_correct_pp=100.0
        * combined_predictions["candidate_correct"].astype(float),
        baseline_correct_pp=100.0
        * combined_predictions["baseline_correct"].astype(float),
    )
    aggregate_intervals = pd.DataFrame(
        [
            {
                "metric": "equal_domain_root_macro_top1_gain_pp",
                **asdict(
                    domain_root_bootstrap_difference(
                        aggregate_frame,
                        "candidate_correct_pp",
                        "baseline_correct_pp",
                        "heldout_domain",
                        "root",
                        10000,
                        20260730,
                    )
                ),
            },
            {
                "metric": "equal_domain_root_macro_mrr_gain",
                **asdict(
                    domain_root_bootstrap_difference(
                        combined_predictions,
                        "candidate_rr",
                        "baseline_rr",
                        "heldout_domain",
                        "root",
                        10000,
                        20260731,
                    )
                ),
            },
            {
                "metric": "equal_domain_root_macro_balance_mrr_increment",
                **asdict(
                    domain_root_bootstrap_difference(
                        combined_predictions,
                        "candidate_rr",
                        "root_only_rr",
                        "heldout_domain",
                        "root",
                        10000,
                        20260732,
                    )
                ),
            },
            {
                "metric": "equal_domain_root_macro_baseline_balance_mrr_increment",
                **asdict(
                    domain_root_bootstrap_difference(
                        combined_predictions,
                        "baseline_rr",
                        "root_baseline_rr",
                        "heldout_domain",
                        "root",
                        10000,
                        20260733,
                    )
                ),
            },
        ]
    )
    aggregate_lookup = aggregate_intervals.set_index("metric")
    summary = {
        "all_domains_top1_nonnegative": bool(
            (metrics["balanced_top1_gain_pp"] >= 0).all()
        ),
        "all_domains_mrr_nonnegative": bool(
            (metrics["balanced_mrr_gain"] >= 0).all()
        ),
        "mean_top1_gain_pp": float(metrics["balanced_top1_gain_pp"].mean()),
        "mean_mrr_gain": float(metrics["balanced_mrr_gain"].mean()),
        "domain_balance_mean_top1_increment_pp": float(
            metrics["domain_balance_increment_top1_pp"].mean()
        ),
        "domain_balance_mean_mrr_increment": float(
            metrics["domain_balance_increment_mrr"].mean()
        ),
        "top1_ci_positive_domains": int(
            (
                intervals.loc[intervals["metric"] == "top1_gain_pp", "lower"]
                > 0
            ).sum()
        ),
        "mrr_ci_positive_domains": int(
            (
                intervals.loc[intervals["metric"] == "mrr_gain", "lower"] > 0
            ).sum()
        ),
        "domain_balance_mrr_ci_positive_domains": int(
            (
                intervals.loc[
                    intervals["metric"] == "domain_balance_mrr_increment", "lower"
                ]
                > 0
            ).sum()
        ),
        "equal_domain_top1_ci_lower_pp": float(
            aggregate_lookup.loc[
                "equal_domain_root_macro_top1_gain_pp", "lower"
            ]
        ),
        "equal_domain_mrr_ci_lower": float(
            aggregate_lookup.loc["equal_domain_root_macro_mrr_gain", "lower"]
        ),
        "equal_domain_balance_mrr_ci_lower": float(
            aggregate_lookup.loc[
                "equal_domain_root_macro_balance_mrr_increment", "lower"
            ]
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        metrics.to_csv(destination / "domain_metrics.csv", index=False)
        intervals.to_csv(destination / "bootstrap_intervals.csv", index=False)
        aggregate_intervals.to_csv(
            destination / "aggregate_bootstrap_intervals.csv", index=False
        )
        combined_predictions.to_csv(
            destination / "predictions.csv", index=False, encoding="utf-8"
        )
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
                "heldout_domains": list(heldout_domains),
                "kappa": 2.0,
                "destination_kappa": 5.0,
                "destination_fraction": 1.0,
                "baseline": {"order": 3, "alpha": 0.1, "weights": [0.2, 0.3, 0.5]},
            },
            split_audit={
                "heldout_excluded_from_training": True,
                "attack_flow_ctid_overlap_excluded": True,
            },
        )
    return metrics, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "external"
        / "qsmr_multisource_lodo_seed20260730"
    )
    _, summary = run_sensitivity(project_root, destination)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
