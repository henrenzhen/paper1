from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .metrics import cluster_bootstrap_difference, root_macro_mean
from .probability_models import InterpolatedNGram
from .run_mrct_development import _gain, _prediction_columns, load_multires_development
from .run_qsmr_multisource_sensitivity import load_multisource_domains
from .semimarkov_router import QuotientSemiMarkovRouter


@dataclass(frozen=True)
class RouterConfig:
    kappa: float
    destination_kappa: float
    hazard_fraction: float
    destination_fraction: float

    @property
    def config_id(self) -> str:
        return (
            f"h{self.kappa:g}_d{self.destination_kappa:g}"
            f"_hf{self.hazard_fraction:g}_df{self.destination_fraction:g}"
        )


def _grid() -> tuple[RouterConfig, ...]:
    return tuple(
        RouterConfig(kappa, destination_kappa, hazard_fraction, destination_fraction)
        for kappa in (1.0, 2.0, 5.0)
        for destination_kappa in (2.0, 5.0, 10.0)
        for hazard_fraction in (0.5, 1.0)
        for destination_fraction in (0.5, 1.0)
    )


def _baseline_probabilities(
    train: pd.DataFrame, test: pd.DataFrame, vocab: Sequence[str]
) -> Any:
    model = InterpolatedNGram(
        vocab, order=3, alpha=0.1, interpolation=(0.2, 0.3, 0.5)
    ).fit(
        train["prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"],
    )
    probabilities, _ = model.predict_proba_with_meta(test["prefix_ids"])
    return probabilities


def _score(
    test: pd.DataFrame,
    vocab: Sequence[str],
    baseline_probabilities: Any,
    candidate_probabilities: Any,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
        "top1_gain_pp": 100.0
        * _gain(predictions, "candidate_correct", "baseline_correct"),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "nonself_top1_gain_pp": 100.0
        * _gain(nonself, "candidate_correct", "baseline_correct"),
        "nonself_mrr_gain": _gain(nonself, "candidate_rr", "baseline_rr"),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    return predictions, metrics


def _fit_router(
    train: pd.DataFrame,
    vocab: Sequence[str],
    kappa: float,
    destination_kappa: float,
) -> QuotientSemiMarkovRouter:
    return QuotientSemiMarkovRouter(
        vocab, kappa=kappa, destination_kappa=destination_kappa
    ).fit(
        train["prefix_ids"],
        train["raw_prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"],
    )


def _inner_grid_scores(
    domains: dict[str, pd.DataFrame],
    outer_heldout: str,
    vocab: Sequence[str],
) -> pd.DataFrame:
    available = tuple(name for name in domains if name != outer_heldout)
    configs = _grid()
    rows: list[dict[str, Any]] = []
    for inner_heldout in available:
        train = pd.concat(
            [
                domains[name]
                for name in available
                if name != inner_heldout
            ],
            ignore_index=True,
        )
        test = domains[inner_heldout]
        baseline = _baseline_probabilities(train, test, vocab)
        by_fit: dict[tuple[float, float], QuotientSemiMarkovRouter] = {}
        for config in configs:
            fit_key = (config.kappa, config.destination_kappa)
            if fit_key not in by_fit:
                by_fit[fit_key] = _fit_router(train, vocab, *fit_key)
            candidate, _ = by_fit[fit_key].predict_proba_with_meta(
                baseline,
                test["prefix_ids"],
                test["raw_prefix_ids"],
                destination_fraction=config.destination_fraction,
                hazard_fraction=config.hazard_fraction,
            )
            _, metrics = _score(test, vocab, baseline, candidate)
            rows.append(
                {
                    "outer_heldout": outer_heldout,
                    "domain": inner_heldout,
                    "config_id": config.config_id,
                    **asdict(config),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_robust_config(inner_scores: pd.DataFrame) -> str:
    required = {"config_id", "domain", "top1_gain_pp", "mrr_gain"}
    if not required.issubset(inner_scores.columns) or inner_scores.empty:
        raise ValueError(f"inner scores must contain {sorted(required)}")
    grouped = inner_scores.groupby("config_id", sort=True).agg(
        min_top1=("top1_gain_pp", "min"),
        min_mrr=("mrr_gain", "min"),
        mean_top1=("top1_gain_pp", "mean"),
        mean_mrr=("mrr_gain", "mean"),
        domains=("domain", "nunique"),
    )
    grouped["all_nonnegative"] = (
        (grouped["min_top1"] >= 0) & (grouped["min_mrr"] >= 0)
    )
    ranked = grouped.reset_index().sort_values(
        [
            "all_nonnegative",
            "min_mrr",
            "min_top1",
            "mean_mrr",
            "mean_top1",
            "config_id",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["config_id"])


def _config_from_id(config_id: str) -> RouterConfig:
    matches = [config for config in _grid() if config.config_id == config_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous config: {config_id}")
    return matches[0]


def run_nested_lodo(
    project_root: Path, output_dir: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(project_root)
    _, vocab, _ = load_multires_development(root_path)
    domains = load_multisource_domains(root_path, vocab)
    outer_rows: list[dict[str, Any]] = []
    all_inner: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    intervals: list[dict[str, Any]] = []
    heldout_domains = ("ctid", "attack_flow", "stockpile")
    fixed = RouterConfig(2.0, 5.0, 1.0, 1.0)
    for outer_index, heldout in enumerate(heldout_domains):
        inner = _inner_grid_scores(domains, heldout, vocab)
        all_inner.append(inner)
        selected_id = select_robust_config(inner)
        selected = _config_from_id(selected_id)
        train = pd.concat(
            [frame for name, frame in domains.items() if name != heldout],
            ignore_index=True,
        )
        test = domains[heldout]
        baseline = _baseline_probabilities(train, test, vocab)
        selected_router = _fit_router(
            train, vocab, selected.kappa, selected.destination_kappa
        )
        candidate, _ = selected_router.predict_proba_with_meta(
            baseline,
            test["prefix_ids"],
            test["raw_prefix_ids"],
            destination_fraction=selected.destination_fraction,
            hazard_fraction=selected.hazard_fraction,
        )
        predictions, metrics = _score(test, vocab, baseline, candidate)
        if (
            selected.kappa == fixed.kappa
            and selected.destination_kappa == fixed.destination_kappa
        ):
            fixed_router = selected_router
        else:
            fixed_router = _fit_router(
                train, vocab, fixed.kappa, fixed.destination_kappa
            )
        fixed_probabilities, _ = fixed_router.predict_proba_with_meta(
            baseline,
            test["prefix_ids"],
            test["raw_prefix_ids"],
            destination_fraction=fixed.destination_fraction,
            hazard_fraction=fixed.hazard_fraction,
        )
        fixed_predictions, _ = _score(test, vocab, baseline, fixed_probabilities)
        predictions["fixed_correct"] = fixed_predictions["candidate_correct"].to_numpy()
        predictions["fixed_rr"] = fixed_predictions["candidate_rr"].to_numpy()
        interval_specs = {
            "top1_gain_pp": lambda frame: 100.0
            * _gain(frame, "candidate_correct", "baseline_correct"),
            "mrr_gain": lambda frame: _gain(
                frame, "candidate_rr", "baseline_rr"
            ),
            "selection_top1_increment_pp": lambda frame: 100.0
            * _gain(frame, "candidate_correct", "fixed_correct"),
            "selection_mrr_increment": lambda frame: _gain(
                frame, "candidate_rr", "fixed_rr"
            ),
        }
        for offset, (metric, function) in enumerate(interval_specs.items()):
            interval = cluster_bootstrap_difference(
                predictions,
                function,
                "root",
                2000,
                20260730 + outer_index * 100 + offset,
            )
            intervals.append(
                {"heldout_domain": heldout, "metric": metric, **asdict(interval)}
            )
        predictions["heldout_domain"] = heldout
        predictions["selected_config"] = selected_id
        all_predictions.append(predictions)
        selected_inner = inner.loc[inner["config_id"] == selected_id]
        outer_rows.append(
            {
                "heldout_domain": heldout,
                "selected_config": selected_id,
                **asdict(selected),
                "inner_min_top1_gain_pp": float(
                    selected_inner["top1_gain_pp"].min()
                ),
                "inner_min_mrr_gain": float(selected_inner["mrr_gain"].min()),
                **metrics,
                "selection_increment_top1_pp": 100.0
                * _gain(predictions, "candidate_correct", "fixed_correct"),
                "selection_increment_mrr": _gain(
                    predictions, "candidate_rr", "fixed_rr"
                ),
            }
        )
    metrics_frame = pd.DataFrame(outer_rows)
    interval_frame = pd.DataFrame(intervals)
    summary = {
        "all_outer_top1_nonnegative": bool(
            (metrics_frame["top1_gain_pp"] >= 0).all()
        ),
        "all_outer_mrr_nonnegative": bool((metrics_frame["mrr_gain"] >= 0).all()),
        "mean_top1_gain_pp": float(metrics_frame["top1_gain_pp"].mean()),
        "mean_mrr_gain": float(metrics_frame["mrr_gain"].mean()),
        "top1_ci_positive_domains": int(
            (
                interval_frame.loc[
                    interval_frame["metric"] == "top1_gain_pp", "lower"
                ]
                > 0
            ).sum()
        ),
        "mrr_ci_positive_domains": int(
            (
                interval_frame.loc[
                    interval_frame["metric"] == "mrr_gain", "lower"
                ]
                > 0
            ).sum()
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        metrics_frame.to_csv(destination / "outer_metrics.csv", index=False)
        pd.concat(all_inner, ignore_index=True).to_csv(
            destination / "inner_grid_metrics.csv", index=False
        )
        interval_frame.to_csv(destination / "bootstrap_intervals.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(
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
                "outer_heldout_domains": list(heldout_domains),
                "inner_selection": "lexicographic worst-domain nonnegative",
                "grid": [asdict(config) for config in _grid()],
                "fixed_comparator": asdict(fixed),
                "baseline": {
                    "order": 3,
                    "alpha": 0.1,
                    "weights": [0.2, 0.3, 0.5],
                },
            },
            split_audit={
                "outer_target_excluded_from_inner_selection": True,
                "outer_target_excluded_from_training": True,
                "attack_flow_ctid_overlap_excluded": True,
            },
        )
    return metrics_frame, summary


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
        / "dbqsmr_nested_lodo_seed20260730"
    )
    _, summary = run_nested_lodo(project_root, destination)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
