from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

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


def _frame_digest(frame: pd.DataFrame) -> str:
    normalized = frame.loc[
        :, ["domain", "root", "prefix_ids", "raw_prefix_ids", "target"]
    ].copy()
    for column in ("prefix_ids", "raw_prefix_ids"):
        normalized[column] = normalized[column].map(
            lambda values: json.dumps(list(values), separators=(",", ":"))
        )
    normalized = normalized.sort_values(
        ["domain", "root", "prefix_ids", "raw_prefix_ids", "target"],
        kind="mergesort",
    )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model(
    vocab: Sequence[str],
    power: float,
    domain_kappa: float = 0.0,
    leave_one_domain_prior: bool = False,
) -> InterpolatedNGram:
    return InterpolatedNGram(
        vocab,
        order=3,
        alpha=0.1,
        interpolation=(0.2, 0.3, 0.5),
        domain_power=power,
        domain_kappa=domain_kappa,
        leave_one_domain_prior=leave_one_domain_prior,
    )


def _predictions(
    test: pd.DataFrame,
    vocab: Sequence[str],
    baseline: Any,
    candidate: Any,
    equal_domain: Any,
) -> pd.DataFrame:
    targets = test["target"].astype(str).to_numpy()
    output: dict[str, Any] = {
        "root": test["root"].astype(str).to_numpy(),
        "target": targets,
    }
    for name, probabilities in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("equal_domain", equal_domain),
    ):
        top, rr, hit5 = _prediction_columns(probabilities, targets, vocab)
        output[f"{name}_correct"] = top == targets
        output[f"{name}_rr"] = rr
        output[f"{name}_hit5"] = hit5
    return pd.DataFrame(output)


def _metric_rows(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "baseline_top1": root_macro_mean(frame, "baseline_correct"),
        "candidate_top1": root_macro_mean(frame, "candidate_correct"),
        "equal_domain_top1": root_macro_mean(frame, "equal_domain_correct"),
        "top1_gain_pp": 100.0
        * _gain(frame, "candidate_correct", "baseline_correct"),
        "mrr_gain": _gain(frame, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(frame, "candidate_hit5", "baseline_hit5"),
        "over_equal_top1_gain_pp": 100.0
        * _gain(frame, "candidate_correct", "equal_domain_correct"),
        "over_equal_mrr_gain": _gain(frame, "candidate_rr", "equal_domain_rr"),
        "rows": float(len(frame)),
        "roots": float(frame["root"].nunique()),
    }


def run_lodo(
    project_root: Path,
    output_dir: Path | None = None,
    candidate_power: float = 0.5,
    candidate_kappa: float = 0.0,
    include_sim: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(project_root)
    _, vocab, _ = load_multires_development(root_path)
    domains = load_multisource_domains(root_path, vocab)
    rows: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    heldout_domains = (
        ("sim", "ctid", "attack_flow", "stockpile")
        if include_sim
        else ("ctid", "attack_flow", "stockpile")
    )
    for domain_index, heldout in enumerate(heldout_domains):
        train = pd.concat(
            [frame for name, frame in domains.items() if name != heldout],
            ignore_index=True,
        )
        test = domains[heldout]
        baseline_model = _model(vocab, 1.0).fit(
            train["prefix_ids"], train["target"], train["root"]
        )
        candidate_model = _model(vocab, candidate_power, candidate_kappa).fit(
            train["prefix_ids"],
            train["target"],
            train["root"],
            domains=train["domain"],
        )
        equal_model = _model(vocab, 0.0).fit(
            train["prefix_ids"],
            train["target"],
            train["root"],
            domains=train["domain"],
        )
        baseline, _ = baseline_model.predict_proba_with_meta(test["prefix_ids"])
        candidate, _ = candidate_model.predict_proba_with_meta(test["prefix_ids"])
        equal_domain, _ = equal_model.predict_proba_with_meta(test["prefix_ids"])
        frame = _predictions(test, vocab, baseline, candidate, equal_domain)
        frame["heldout_domain"] = heldout
        predictions.append(frame)
        rows.append({"heldout_domain": heldout, **_metric_rows(frame)})
        specs = {
            "top1_gain_pp": lambda sample: 100.0
            * _gain(sample, "candidate_correct", "baseline_correct"),
            "mrr_gain": lambda sample: _gain(
                sample, "candidate_rr", "baseline_rr"
            ),
            "over_equal_top1_gain_pp": lambda sample: 100.0
            * _gain(sample, "candidate_correct", "equal_domain_correct"),
            "over_equal_mrr_gain": lambda sample: _gain(
                sample, "candidate_rr", "equal_domain_rr"
            ),
        }
        for offset, (metric, function) in enumerate(specs.items()):
            interval = cluster_bootstrap_difference(
                frame,
                function,
                "root",
                2000,
                20260730 + domain_index * 100 + offset,
            )
            intervals.append(
                {"heldout_domain": heldout, "metric": metric, **asdict(interval)}
            )
    metrics = pd.DataFrame(rows)
    interval_frame = pd.DataFrame(intervals)
    combined = pd.concat(predictions, ignore_index=True)
    aggregate_specs = (
        ("top1_gain_pp", "candidate_correct", "baseline_correct", 100.0),
        ("mrr_gain", "candidate_rr", "baseline_rr", 1.0),
        (
            "over_equal_top1_gain_pp",
            "candidate_correct",
            "equal_domain_correct",
            100.0,
        ),
        ("over_equal_mrr_gain", "candidate_rr", "equal_domain_rr", 1.0),
    )
    aggregate_rows = []
    for offset, (name, candidate_col, baseline_col, scale) in enumerate(
        aggregate_specs
    ):
        aggregate = combined.assign(
            _candidate=scale * combined[candidate_col].astype(float),
            _baseline=scale * combined[baseline_col].astype(float),
        )
        interval = domain_root_bootstrap_difference(
            aggregate,
            "_candidate",
            "_baseline",
            "heldout_domain",
            "root",
            10000,
            20260800 + offset,
        )
        aggregate_rows.append({"metric": name, **asdict(interval)})
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_lookup = aggregate_frame.set_index("metric")
    summary = {
        "candidate_power": float(candidate_power),
        "candidate_kappa": float(candidate_kappa),
        "all_domain_top1_nonnegative": bool((metrics["top1_gain_pp"] >= 0).all()),
        "all_domain_mrr_nonnegative": bool((metrics["mrr_gain"] >= 0).all()),
        "mean_top1_gain_pp": float(metrics["top1_gain_pp"].mean()),
        "mean_mrr_gain": float(metrics["mrr_gain"].mean()),
        "aggregate_top1_ci_lower_pp": float(
            aggregate_lookup.loc["top1_gain_pp", "lower"]
        ),
        "aggregate_mrr_ci_lower": float(aggregate_lookup.loc["mrr_gain", "lower"]),
        "mean_over_equal_top1_gain_pp": float(
            metrics["over_equal_top1_gain_pp"].mean()
        ),
        "mean_over_equal_mrr_gain": float(metrics["over_equal_mrr_gain"].mean()),
        "aggregate_over_equal_top1_ci_lower_pp": float(
            aggregate_lookup.loc["over_equal_top1_gain_pp", "lower"]
        ),
        "aggregate_over_equal_mrr_ci_lower": float(
            aggregate_lookup.loc["over_equal_mrr_gain", "lower"]
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        metrics.to_csv(destination / "domain_metrics.csv", index=False)
        interval_frame.to_csv(destination / "bootstrap_intervals.csv", index=False)
        aggregate_frame.to_csv(
            destination / "aggregate_bootstrap_intervals.csv", index=False
        )
        combined.to_csv(destination / "predictions.csv", index=False)
        write_canonical_json(destination / "summary.json", summary)
        write_manifest(
            destination / "run_manifest.json",
            inputs={
                "domain_rows": {name: len(frame) for name, frame in domains.items()},
                "domain_frame_sha256": {
                    name: _frame_digest(frame) for name, frame in domains.items()
                },
                "development_cache_sha256": sha256_file(
                    root_path / "data_v2" / "core" / "sim_development_multires_min3.csv"
                ),
                "runner_sha256": sha256_file(Path(__file__)),
                "probability_models_sha256": sha256_file(
                    Path(__file__).with_name("probability_models.py")
                ),
                "multisource_loader_sha256": sha256_file(
                    Path(__file__).with_name("run_qsmr_multisource_sensitivity.py")
                ),
                "attack_flow_loader_sha256": sha256_file(
                    Path(__file__).with_name("attack_flow_dataset.py")
                ),
                "stockpile_loader_sha256": sha256_file(
                    Path(__file__).with_name("stockpile_dataset.py")
                ),
                "ctid_loader_sha256": sha256_file(
                    Path(__file__).with_name("run_qmrct_external.py")
                ),
            },
            config={
                "heldout_domains": list(heldout_domains),
                "candidate_power": candidate_power,
                "candidate_kappa": candidate_kappa,
                "baseline_power": 1.0,
                "equal_domain_ablation_power": 0.0,
                "order": 3,
                "alpha": 0.1,
                "weights": [0.2, 0.3, 0.5],
            },
            split_audit={
                "heldout_excluded_from_training": True,
                "attack_flow_ctid_overlap_excluded": True,
            },
        )
    return metrics, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-power", type=float, default=0.5)
    parser.add_argument("--candidate-kappa", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-sim", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "external"
        / "support_tempered_ngram_lodo_seed20260730"
    )
    _, summary = run_lodo(
        project_root,
        destination,
        args.candidate_power,
        args.candidate_kappa,
        args.include_sim,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
