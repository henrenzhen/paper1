from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .metrics import (
    cluster_bootstrap_difference,
    domain_root_bootstrap_difference,
)
from .run_mrct_development import load_multires_development
from .run_qsmr_multisource_sensitivity import load_multisource_domains
from .run_support_tempered_ngram_lodo import _metric_rows, _model, _predictions


POWERS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Sequence[str],
    power: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_model = _model(vocab, 1.0).fit(
        train["prefix_ids"], train["target"], train["root"]
    )
    candidate_model = _model(vocab, power).fit(
        train["prefix_ids"],
        train["target"],
        train["root"],
        domains=train["domain"],
    )
    baseline, _ = baseline_model.predict_proba_with_meta(test["prefix_ids"])
    candidate, _ = candidate_model.predict_proba_with_meta(test["prefix_ids"])
    frame = _predictions(test, vocab, baseline, candidate, candidate)
    return frame, _metric_rows(frame)


def select_power(inner_scores: pd.DataFrame) -> float:
    required = {"power", "domain", "top1_gain_pp", "mrr_gain"}
    if inner_scores.empty or not required.issubset(inner_scores.columns):
        raise ValueError(f"inner scores must contain {sorted(required)}")
    grouped = inner_scores.groupby("power", sort=True).agg(
        min_top1=("top1_gain_pp", "min"),
        min_mrr=("mrr_gain", "min"),
        mean_top1=("top1_gain_pp", "mean"),
        mean_mrr=("mrr_gain", "mean"),
    )
    grouped["all_nonnegative"] = (
        (grouped["min_top1"] >= 0) & (grouped["min_mrr"] >= 0)
    )
    ranked = grouped.reset_index().sort_values(
        ["all_nonnegative", "mean_mrr", "min_mrr", "mean_top1", "min_top1", "power"],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["power"])


def _inner_scores(
    domains: dict[str, pd.DataFrame],
    outer_heldout: str,
    vocab: Sequence[str],
) -> pd.DataFrame:
    available = tuple(name for name in domains if name != outer_heldout)
    rows: list[dict[str, Any]] = []
    for inner_heldout in available:
        train = pd.concat(
            [domains[name] for name in available if name != inner_heldout],
            ignore_index=True,
        )
        test = domains[inner_heldout]
        for power in POWERS:
            _, metrics = _evaluate(train, test, vocab, power)
            rows.append(
                {
                    "outer_heldout": outer_heldout,
                    "domain": inner_heldout,
                    "power": power,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_lodo(
    project_root: Path, output_dir: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(project_root)
    _, vocab, _ = load_multires_development(root_path)
    domains = load_multisource_domains(root_path, vocab)
    heldout_domains = ("ctid", "attack_flow", "stockpile")
    outer_rows: list[dict[str, Any]] = []
    inner_outputs: list[pd.DataFrame] = []
    prediction_outputs: list[pd.DataFrame] = []
    interval_rows: list[dict[str, Any]] = []
    for index, heldout in enumerate(heldout_domains):
        inner = _inner_scores(domains, heldout, vocab)
        inner_outputs.append(inner)
        selected = select_power(inner)
        train = pd.concat(
            [frame for name, frame in domains.items() if name != heldout],
            ignore_index=True,
        )
        predictions, metrics = _evaluate(train, domains[heldout], vocab, selected)
        predictions["heldout_domain"] = heldout
        predictions["selected_power"] = selected
        prediction_outputs.append(predictions)
        outer_rows.append(
            {"heldout_domain": heldout, "selected_power": selected, **metrics}
        )
        for offset, (metric, function) in enumerate(
            {
                "top1_gain_pp": lambda sample: 100.0
                * (
                    sample["candidate_correct"].astype(float)
                    - sample["baseline_correct"].astype(float)
                ).groupby(sample["root"]).mean().mean(),
                "mrr_gain": lambda sample: (
                    sample["candidate_rr"] - sample["baseline_rr"]
                ).groupby(sample["root"]).mean().mean(),
            }.items()
        ):
            interval = cluster_bootstrap_difference(
                predictions,
                function,
                "root",
                2000,
                20260900 + index * 100 + offset,
            )
            interval_rows.append(
                {"heldout_domain": heldout, "metric": metric, **asdict(interval)}
            )
    metrics_frame = pd.DataFrame(outer_rows)
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
            20261000 + offset,
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
        "selected_powers": {
            row["heldout_domain"]: row["selected_power"] for row in outer_rows
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        metrics_frame.to_csv(destination / "outer_metrics.csv", index=False)
        pd.concat(inner_outputs, ignore_index=True).to_csv(
            destination / "inner_metrics.csv", index=False
        )
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
                "powers": list(POWERS),
                "selection": "all-inner-domain nonnegative then mean MRR",
                "heldout_domains": list(heldout_domains),
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
        / "adaptive_power_ngram_lodo_seed20260730"
    )
    _, summary = run_lodo(project_root, destination)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
