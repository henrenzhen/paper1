from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .dr_vom_validation import (
    expected_calibration_error,
    prediction_diagnostics,
    root_macro_diagnostics,
)
from .probability_models import InterpolatedNGram
from .run_mrct_development import load_multires_development
from .run_qsmr_multisource_sensitivity import load_multisource_domains


MODEL_NAMES = (
    "row_pooled",
    "root_balanced",
    "domain_row_balanced",
    "dr_vom",
)

COMPARISON_METRICS = (
    ("top1_gain_pp", "hit1", 100.0, False),
    ("mrr_gain", "rr", 1.0, False),
    ("hit5_gain_pp", "hit5", 100.0, False),
    ("nll_improvement", "nll", 1.0, True),
    ("brier_improvement", "brier", 1.0, True),
)


@dataclass(frozen=True)
class ValidationResult:
    domain_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    aggregate_intervals: pd.DataFrame
    predictions: pd.DataFrame
    stress_metrics: pd.DataFrame
    summary: dict[str, Any]


def _model(vocabulary: Sequence[str]) -> InterpolatedNGram:
    return InterpolatedNGram(
        vocabulary,
        order=3,
        alpha=0.1,
        interpolation=(0.2, 0.3, 0.5),
        domain_power=0.0,
        domain_kappa=0.0,
    )


def _validated_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"prefix_ids", "target", "root", "domain"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training frame missing columns: {missing}")
    if frame.empty:
        raise ValueError("training frame must be nonempty")
    return frame.reset_index(drop=True).copy()


def fit_validation_models(
    train: pd.DataFrame,
    vocabulary: Sequence[str],
) -> dict[str, InterpolatedNGram]:
    work = _validated_training_frame(train)
    row_ids = [f"row:{index}" for index in range(len(work))]
    prefixes = work["prefix_ids"]
    targets = work["target"].astype(str)
    roots = work["root"].astype(str)
    domains = work["domain"].astype(str)
    models = {name: _model(vocabulary) for name in MODEL_NAMES}
    models["row_pooled"].fit(prefixes, targets, row_ids)
    models["root_balanced"].fit(prefixes, targets, roots)
    models["domain_row_balanced"].fit(
        prefixes,
        targets,
        row_ids,
        domains=domains,
    )
    models["dr_vom"].fit(prefixes, targets, roots, domains=domains)
    return models


def _validated_perturbation(
    frame: pd.DataFrame,
    domain: str,
    factor: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _validated_training_frame(frame)
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("factor must be a positive integer")
    clean_domain = str(domain)
    selected = work.loc[work["domain"].astype(str) == clean_domain].copy()
    if selected.empty:
        raise ValueError(f"unknown domain: {clean_domain}")
    return work, selected


def duplicate_domain_rows(
    frame: pd.DataFrame,
    domain: str,
    factor: int,
) -> pd.DataFrame:
    work, selected = _validated_perturbation(frame, domain, factor)
    copies = [work] + [selected.copy() for _ in range(factor - 1)]
    return pd.concat(copies, ignore_index=True)


def duplicate_root_rows(
    frame: pd.DataFrame,
    root: str,
    factor: int,
) -> pd.DataFrame:
    work = _validated_training_frame(frame)
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("factor must be a positive integer")
    clean_root = str(root)
    selected = work.loc[work["root"].astype(str) == clean_root].copy()
    if selected.empty:
        raise ValueError(f"unknown root: {clean_root}")
    copies = [work] + [selected.copy() for _ in range(factor - 1)]
    return pd.concat(copies, ignore_index=True)


def clone_domain_roots(
    frame: pd.DataFrame,
    domain: str,
    factor: int,
) -> pd.DataFrame:
    work, selected = _validated_perturbation(frame, domain, factor)
    copies = [work]
    for clone_index in range(1, factor):
        clone = selected.copy()
        clone["root"] = clone["root"].astype(str).map(
            lambda root: f"{root}::domain-clone:{clone_index}"
        )
        copies.append(clone)
    return pd.concat(copies, ignore_index=True)


def _validated_domains(domains: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if len(domains) < 2:
        raise ValueError("validation requires at least two source domains")
    validated: dict[str, pd.DataFrame] = {}
    for raw_name, raw_frame in domains.items():
        name = str(raw_name)
        frame = _validated_training_frame(raw_frame)
        observed = set(frame["domain"].astype(str))
        if observed != {name}:
            raise ValueError(
                f"domain frame {name} contains mismatched labels: {sorted(observed)}"
            )
        validated[name] = frame
    return validated


def _prediction_frame(
    test: pd.DataFrame,
    models: Mapping[str, InterpolatedNGram],
    vocabulary: Sequence[str],
    heldout_domain: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    output = pd.DataFrame(
        {
            "heldout_domain": str(heldout_domain),
            "root": test["root"].astype(str).to_numpy(),
            "target": test["target"].astype(str).to_numpy(),
        }
    )
    probabilities: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        probability, metadata = models[name].predict_proba_with_meta(test["prefix_ids"])
        diagnostics = prediction_diagnostics(
            probability,
            output["target"].tolist(),
            vocabulary,
        ).add_prefix(f"{name}_")
        output = pd.concat([output, diagnostics], axis=1)
        output[f"{name}_used_order"] = metadata["used_order"].to_numpy()
        output[f"{name}_context_root_support"] = metadata[
            "context_root_support"
        ].to_numpy()
        probabilities[name] = probability
    return output, probabilities


def _root_difference_frame(
    frame: pd.DataFrame,
    reference: str,
    field: str,
    scale: float,
    lower_is_better: bool,
) -> pd.DataFrame:
    candidate_values = frame[f"dr_vom_{field}"].astype(float)
    reference_values = frame[f"{reference}_{field}"].astype(float)
    difference = (
        reference_values - candidate_values
        if lower_is_better
        else candidate_values - reference_values
    )
    work = frame.loc[:, ["heldout_domain", "root"]].assign(
        difference=float(scale) * difference
    )
    return (
        work.groupby(["heldout_domain", "root"], sort=False, as_index=False)[
            "difference"
        ]
        .mean()
        .reset_index(drop=True)
    )


def _bootstrap_root_values(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    if clean.ndim != 1 or not len(clean) or not np.isfinite(clean).all():
        raise ValueError("root bootstrap values must be finite and nonempty")
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(clean, size=(int(replicates), len(clean)), replace=True).mean(
        axis=1
    )
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _bootstrap_equal_domain_roots(
    roots: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    domain_values = [
        group["difference"].to_numpy(dtype=float)
        for _, group in roots.groupby("heldout_domain", sort=False)
    ]
    if not domain_values:
        raise ValueError("aggregate bootstrap requires source domains")
    rng = np.random.default_rng(int(seed))
    draws = np.zeros(int(replicates), dtype=float)
    for values in domain_values:
        draws += rng.choice(
            values,
            size=(int(replicates), len(values)),
            replace=True,
        ).mean(axis=1)
    draws /= len(domain_values)
    point = float(np.mean([values.mean() for values in domain_values]))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _comparison_frames(
    predictions: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    references = [name for name in MODEL_NAMES if name != "dr_vom"]
    for reference_index, reference in enumerate(references):
        for metric_index, (metric, field, scale, lower_is_better) in enumerate(
            COMPARISON_METRICS
        ):
            roots = _root_difference_frame(
                predictions,
                reference,
                field,
                scale,
                lower_is_better,
            )
            for domain_index, (domain, group) in enumerate(
                roots.groupby("heldout_domain", sort=False)
            ):
                point, lower, upper = _bootstrap_root_values(
                    group["difference"].to_numpy(dtype=float),
                    bootstrap_replicates,
                    seed + reference_index * 1000 + metric_index * 100 + domain_index,
                )
                domain_rows.append(
                    {
                        "heldout_domain": domain,
                        "reference": reference,
                        "metric": metric,
                        "point": point,
                        "lower": lower,
                        "upper": upper,
                        "valid_replicates": int(bootstrap_replicates),
                    }
                )
            point, lower, upper = _bootstrap_equal_domain_roots(
                roots,
                bootstrap_replicates,
                seed + 10000 + reference_index * 1000 + metric_index,
            )
            aggregate_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "point": point,
                    "lower": lower,
                    "upper": upper,
                    "valid_replicates": int(bootstrap_replicates),
                }
            )
    return pd.DataFrame(domain_rows), pd.DataFrame(aggregate_rows)


def _stress_validation(
    domains: Mapping[str, pd.DataFrame],
    vocabulary: Sequence[str],
    original_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    factors: Sequence[int],
) -> pd.DataFrame:
    columns = [
        "heldout_domain",
        "perturbed_domain",
        "perturbation",
        "factor",
        "model",
        "max_abs_probability_drift",
        "mean_l1_probability_drift",
        "top1_change_rate",
        "expected_invariant",
    ]
    if not factors:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for heldout, test in domains.items():
        train_domains = [name for name in domains if name != heldout]
        train = pd.concat([domains[name] for name in train_domains], ignore_index=True)
        for perturbed_domain in train_domains:
            domain_rows = train.loc[
                train["domain"].astype(str) == str(perturbed_domain)
            ]
            root_sizes = (
                domain_rows.groupby(domain_rows["root"].astype(str), sort=True)
                .size()
                .sort_values(ascending=False, kind="mergesort")
            )
            longest_root = str(root_sizes.index[0])
            for factor in factors:
                for perturbation in ("within_root_rows", "source_root_clones"):
                    perturbed = (
                        duplicate_root_rows(train, longest_root, int(factor))
                        if perturbation == "within_root_rows"
                        else clone_domain_roots(train, perturbed_domain, int(factor))
                    )
                    perturbed_models = fit_validation_models(perturbed, vocabulary)
                    for model_name in MODEL_NAMES:
                        new_probability, _ = perturbed_models[
                            model_name
                        ].predict_proba_with_meta(test["prefix_ids"])
                        original = original_probabilities[heldout][model_name]
                        expected_invariant = (
                            model_name in {"root_balanced", "dr_vom"}
                            if perturbation == "within_root_rows"
                            else model_name in {"domain_row_balanced", "dr_vom"}
                        )
                        rows.append(
                            {
                                "heldout_domain": heldout,
                                "perturbed_domain": perturbed_domain,
                                "perturbation": perturbation,
                                "factor": int(factor),
                                "model": model_name,
                                "max_abs_probability_drift": float(
                                    np.max(np.abs(new_probability - original))
                                ),
                                "mean_l1_probability_drift": float(
                                    np.abs(new_probability - original).sum(axis=1).mean()
                                ),
                                "top1_change_rate": float(
                                    np.mean(
                                        np.argmax(new_probability, axis=1)
                                        != np.argmax(original, axis=1)
                                    )
                                ),
                                "expected_invariant": bool(expected_invariant),
                            }
                        )
    return pd.DataFrame(rows, columns=columns)


def _lookup(
    frame: pd.DataFrame,
    reference: str,
    metric: str,
    column: str = "point",
) -> float:
    selected = frame.loc[
        (frame["reference"] == reference) & (frame["metric"] == metric),
        column,
    ]
    if len(selected) != 1:
        raise ValueError(f"missing aggregate comparison: {reference}/{metric}")
    return float(selected.iloc[0])


def evaluate_domains(
    domains: Mapping[str, pd.DataFrame],
    vocabulary: Sequence[str],
    bootstrap_replicates: int = 2000,
    stress_factors: Sequence[int] = (2, 5, 10),
    seed: int = 20260730,
) -> ValidationResult:
    validated = _validated_domains(domains)
    vocabulary = tuple(str(label) for label in vocabulary)
    domain_metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    original_probabilities: dict[str, dict[str, np.ndarray]] = {}
    for heldout, test in validated.items():
        train = pd.concat(
            [frame for name, frame in validated.items() if name != heldout],
            ignore_index=True,
        )
        models = fit_validation_models(train, vocabulary)
        predictions, probabilities = _prediction_frame(
            test,
            models,
            vocabulary,
            heldout,
        )
        prediction_frames.append(predictions)
        original_probabilities[heldout] = probabilities
        for model_name in MODEL_NAMES:
            metrics = root_macro_diagnostics(predictions, model_name)
            domain_metric_rows.append(
                {
                    "heldout_domain": heldout,
                    "model": model_name,
                    **metrics,
                    "ece": expected_calibration_error(predictions, model_name, 10),
                    "rows": int(len(test)),
                    "roots": int(test["root"].nunique()),
                }
            )
    combined = pd.concat(prediction_frames, ignore_index=True)
    domain_metrics = pd.DataFrame(domain_metric_rows)
    comparisons, aggregate = _comparison_frames(
        combined,
        bootstrap_replicates,
        seed,
    )
    stress = _stress_validation(
        validated,
        vocabulary,
        original_probabilities,
        tuple(int(factor) for factor in stress_factors),
    )
    root_comparisons = comparisons.loc[comparisons["reference"] == "root_balanced"]
    all_source_top1 = bool(
        (root_comparisons.loc[root_comparisons["metric"] == "top1_gain_pp", "point"] >= 0).all()
    )
    all_source_mrr = bool(
        (root_comparisons.loc[root_comparisons["metric"] == "mrr_gain", "point"] >= 0).all()
    )
    macro_hit5 = _lookup(aggregate, "root_balanced", "hit5_gain_pp")
    worst_hit5 = float(
        root_comparisons.loc[
            root_comparisons["metric"] == "hit5_gain_pp", "point"
        ].min()
    )
    mechanism_pass = all(
        _lookup(aggregate, reference, "mrr_gain") > 0.0
        for reference in ("row_pooled", "root_balanced", "domain_row_balanced")
    )
    proper_score_pass = (
        _lookup(aggregate, "root_balanced", "nll_improvement") > 0.0
        or _lookup(aggregate, "root_balanced", "brier_improvement") > 0.0
    )
    if stress.empty:
        stress_invariance_pass: bool | None = None
        stress_sensitivity_detected: bool | None = None
    else:
        invariant = stress.loc[stress["expected_invariant"]]
        sensitive = stress.loc[~stress["expected_invariant"]]
        stress_invariance_pass = bool(
            (invariant["max_abs_probability_drift"] <= 1e-10).all()
        )
        stress_sensitivity_detected = bool(
            (sensitive["max_abs_probability_drift"] > 1e-6).any()
        )
    hit5_pass = bool(macro_hit5 > -1.0 and worst_hit5 > -1.0)
    summary = {
        "heldout_domains": list(validated),
        "bootstrap_replicates": int(bootstrap_replicates),
        "stress_factors": [int(factor) for factor in stress_factors],
        "all_source_top1_nonnegative_vs_root_balanced": all_source_top1,
        "all_source_mrr_nonnegative_vs_root_balanced": all_source_mrr,
        "macro_hit5_gain_pp_vs_root_balanced": macro_hit5,
        "worst_source_hit5_gain_pp_vs_root_balanced": worst_hit5,
        "mechanism_ablation_mrr_pass": bool(mechanism_pass),
        "proper_score_pass_vs_root_balanced": bool(proper_score_pass),
        "hit5_noninferiority_pass": hit5_pass,
        "stress_invariance_pass": stress_invariance_pass,
        "stress_sensitivity_detected": stress_sensitivity_detected,
        "primary_pass": bool(
            all_source_top1
            and all_source_mrr
            and mechanism_pass
            and proper_score_pass
            and hit5_pass
            and stress_invariance_pass is True
            and stress_sensitivity_detected is True
        ),
    }
    return ValidationResult(
        domain_metrics=domain_metrics,
        comparison_metrics=comparisons,
        aggregate_intervals=aggregate,
        predictions=combined,
        stress_metrics=stress,
        summary=summary,
    )


def _frame_digest(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, ["domain", "root", "prefix_ids", "target"]].copy()
    normalized["prefix_ids"] = normalized["prefix_ids"].map(
        lambda values: json.dumps(list(values), separators=(",", ":"))
    )
    normalized = normalized.sort_values(
        ["domain", "root", "prefix_ids", "target"], kind="mergesort"
    )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_validation(
    project_root: Path,
    output_dir: Path,
    domains: Mapping[str, pd.DataFrame] | None = None,
    vocabulary: Sequence[str] | None = None,
    bootstrap_replicates: int = 2000,
    stress_factors: Sequence[int] = (2, 5, 10),
    seed: int = 20260730,
) -> ValidationResult:
    root_path = Path(project_root)
    if domains is None or vocabulary is None:
        _, loaded_vocabulary, _ = load_multires_development(root_path)
        vocabulary = tuple(loaded_vocabulary)
        domains = load_multisource_domains(root_path, vocabulary)
    result = evaluate_domains(
        domains,
        vocabulary,
        bootstrap_replicates,
        stress_factors,
        seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    result.domain_metrics.to_csv(
        destination / "domain_ablation_metrics.csv", index=False
    )
    result.domain_metrics.loc[
        :, ["heldout_domain", "model", "nll", "brier", "ece", "rows", "roots"]
    ].to_csv(destination / "probability_metrics.csv", index=False)
    result.comparison_metrics.to_csv(
        destination / "domain_comparison_intervals.csv", index=False
    )
    result.aggregate_intervals.to_csv(
        destination / "aggregate_ablation_intervals.csv", index=False
    )
    result.stress_metrics.to_csv(destination / "stress_metrics.csv", index=False)
    result.predictions.to_csv(destination / "predictions.csv", index=False)
    write_canonical_json(destination / "summary.json", result.summary)
    write_manifest(
        destination / "run_manifest.json",
        inputs={
            "domain_rows": {name: int(len(frame)) for name, frame in domains.items()},
            "domain_frame_sha256": {
                name: _frame_digest(frame) for name, frame in domains.items()
            },
            "runner_sha256": sha256_file(Path(__file__)),
            "diagnostics_sha256": sha256_file(
                Path(__file__).with_name("dr_vom_validation.py")
            ),
            "probability_models_sha256": sha256_file(
                Path(__file__).with_name("probability_models.py")
            ),
        },
        config={
            "models": list(MODEL_NAMES),
            "order": 3,
            "max_context_length": 2,
            "alpha": 0.1,
            "interpolation": [0.2, 0.3, 0.5],
            "bootstrap_replicates": int(bootstrap_replicates),
            "stress_factors": [int(factor) for factor in stress_factors],
            "seed": int(seed),
        },
        split_audit={
            "complete_leave_one_source_domain_out": True,
            "heldout_rows_used_for_training": False,
            "fresh_source_consumed": False,
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--stress-factors", type=int, nargs="*", default=[2, 5, 10])
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    result = run_validation(
        project_root,
        args.output_dir,
        bootstrap_replicates=args.bootstrap,
        stress_factors=args.stress_factors,
        seed=args.seed,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
