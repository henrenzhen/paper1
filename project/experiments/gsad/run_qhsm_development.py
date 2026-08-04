from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import write_canonical_json, write_manifest
from .data_protocol import TEST_ROOTS
from .metrics import Interval, cluster_bootstrap_difference, root_macro_mean
from .run_development import (
    _select_probability_model,
    assign_balanced_root_folds,
    json_key,
    make_inner_roles,
)
from .run_mrct_development import _gain, _prediction_columns, load_multires_development
from .run_qsmr_development import _score
from .semimarkov_router import QuotientSemiMarkovRouter


@dataclass(frozen=True)
class QHSMConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    max_dwell: int = 5
    kappas: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
    hazard_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        if self.bootstrap < 1 or self.n_splits < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if self.max_dwell < 1 or not self.kappas or not self.hazard_fractions:
            raise ValueError("invalid fixed structure or empty selection grid")
        if min(self.kappas) < 0:
            raise ValueError("kappas must be nonnegative")
        if min(self.hazard_fractions) < 0 or max(self.hazard_fractions) > 1:
            raise ValueError("hazard fractions must be in [0, 1]")


def select_qhsm_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    base_probabilities: np.ndarray,
    vocab: Sequence[str],
    config: QHSMConfig,
    hazard_levels: Sequence[str] = ("parent", "parent_dwell", "parent_dwell_raw"),
) -> tuple[QuotientSemiMarkovRouter, dict[str, float], np.ndarray]:
    scored: list[tuple[Any, ...]] = []
    for kappa in config.kappas:
        model = QuotientSemiMarkovRouter(
            vocab,
            kappa=kappa,
            max_dwell=config.max_dwell,
            hazard_levels=hazard_levels,
        ).fit(
            fit["prefix_ids"], fit["raw_prefix_ids"], fit["target"], fit["root"]
        )
        for fraction in config.hazard_fractions:
            probabilities, _ = model.predict_proba_with_meta(
                base_probabilities,
                validation["prefix_ids"],
                validation["raw_prefix_ids"],
                destination_fraction=0.0,
                hazard_fraction=fraction,
            )
            nonself_mrr, nonself_top1, overall_mrr, overall_top1 = _score(
                probabilities, validation, vocab
            )
            selected = {
                "kappa": float(kappa),
                "hazard_fraction": float(fraction),
            }
            scored.append(
                (
                    -nonself_mrr,
                    -nonself_top1,
                    -overall_mrr,
                    -overall_top1,
                    json_key(selected),
                    model,
                    selected,
                    probabilities,
                )
            )
    _, _, _, _, _, model, selected, probabilities = min(
        scored, key=lambda row: row[:5]
    )
    return model, selected, probabilities


def _evaluate_fold(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: object,
    config: QHSMConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, baseline_config, _, _ = _select_probability_model(
        fit, validation, vocab, dag
    )
    validation_base, _ = baseline.predict_proba_with_meta(validation["prefix_ids"])
    model, selected, _ = select_qhsm_model(
        fit, validation, validation_base, vocab, config
    )
    no_raw_model, no_raw_selected, _ = select_qhsm_model(
        fit,
        validation,
        validation_base,
        vocab,
        config,
        hazard_levels=("parent", "parent_dwell"),
    )
    base, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    candidate, meta = model.predict_proba_with_meta(
        base,
        outer["prefix_ids"],
        outer["raw_prefix_ids"],
        destination_fraction=0.0,
        hazard_fraction=selected["hazard_fraction"],
    )
    no_raw, _ = no_raw_model.predict_proba_with_meta(
        base,
        outer["prefix_ids"],
        outer["raw_prefix_ids"],
        destination_fraction=0.0,
        hazard_fraction=no_raw_selected["hazard_fraction"],
    )
    targets = outer["target"].astype(str).to_numpy()
    data: dict[str, Any] = {
        "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
        "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
        "root": outer["root"].astype(str).to_numpy(),
        "target": targets,
        "is_self": [
            bool(prefix) and str(prefix[-1]) == target
            for prefix, target in zip(outer["prefix_ids"], targets, strict=True)
        ],
    }
    for name, probabilities in {
        "baseline": base,
        "candidate": candidate,
        "no_raw": no_raw,
    }.items():
        top, rr, hit5 = _prediction_columns(probabilities, targets, vocab)
        data[f"{name}_top1"] = top
        data[f"{name}_correct"] = top == targets
        data[f"{name}_rr"] = rr
        if name in {"baseline", "candidate"}:
            data[f"{name}_hit5"] = hit5
    predictions = pd.DataFrame(data)
    return pd.concat([predictions, meta.reset_index(drop=True)], axis=1), {
        "baseline": baseline_config,
        "qhsm": selected,
        "no_raw": no_raw_selected,
    }


def summarize_qhsm(
    predictions: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, Interval], dict[str, bool]]:
    predictions = predictions.copy()
    self_rows = predictions.loc[predictions["is_self"].astype(bool)]
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
    counterfactual = predictions["candidate_correct"].astype(bool).to_numpy().copy()
    self_mask = predictions["is_self"].astype(bool).to_numpy()
    counterfactual[self_mask] = predictions.loc[self_mask, "baseline_correct"].astype(bool)
    predictions["counterfactual_correct"] = counterfactual
    fold_gains = [
        _gain(frame, "candidate_correct", "baseline_correct")
        for _, frame in predictions.groupby("fold", sort=True)
    ]
    nonself_fold_gains = [
        _gain(
            frame.loc[~frame["is_self"].astype(bool)],
            "candidate_correct",
            "baseline_correct",
        )
        for _, frame in predictions.groupby("fold", sort=True)
    ]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "candidate_top1": root_macro_mean(predictions, "candidate_correct"),
        "top1_gain_pp": 100.0 * _gain(predictions, "candidate_correct", "baseline_correct"),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "self_top1_gain_pp": 100.0 * _gain(self_rows, "candidate_correct", "baseline_correct"),
        "nonself_top1_gain_pp": 100.0 * _gain(nonself, "candidate_correct", "baseline_correct"),
        "nonself_mrr_gain": _gain(nonself, "candidate_rr", "baseline_rr"),
        "counterfactual_top1_gain_pp": 100.0 * _gain(
            predictions, "counterfactual_correct", "baseline_correct"
        ),
        "raw_increment_top1_pp": 100.0 * _gain(
            predictions, "candidate_correct", "no_raw_correct"
        ),
        "raw_increment_mrr": _gain(predictions, "candidate_rr", "no_raw_rr"),
        "positive_fold_count": float(sum(value > 0 for value in fold_gains)),
        "nonself_positive_fold_count": float(sum(value > 0 for value in nonself_fold_gains)),
        "mean_exit_hazard": float(predictions["exit_hazard"].mean()),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    specs = {
        "top1_gain_pp": (predictions, "candidate_correct", "baseline_correct", 100.0),
        "mrr_gain": (predictions, "candidate_rr", "baseline_rr", 1.0),
        "self_top1_gain_pp": (self_rows, "candidate_correct", "baseline_correct", 100.0),
        "nonself_top1_gain_pp": (nonself, "candidate_correct", "baseline_correct", 100.0),
        "nonself_mrr_gain": (nonself, "candidate_rr", "baseline_rr", 1.0),
        "counterfactual_top1_gain_pp": (
            predictions,
            "counterfactual_correct",
            "baseline_correct",
            100.0,
        ),
        "raw_increment_top1_pp": (
            predictions,
            "candidate_correct",
            "no_raw_correct",
            100.0,
        ),
        "raw_increment_mrr": (predictions, "candidate_rr", "no_raw_rr", 1.0),
    }
    intervals: dict[str, Interval] = {}
    for offset, (name, (frame, candidate, baseline, scale)) in enumerate(specs.items()):
        intervals[name] = cluster_bootstrap_difference(
            frame,
            lambda sample, c=candidate, b=baseline, s=scale: s * _gain(sample, c, b),
            "root",
            n_boot,
            seed + offset,
        )
    gates = {
        "overall_top1": metrics["top1_gain_pp"] >= 2.0
        and intervals["top1_gain_pp"].lower > 0,
        "overall_mrr": metrics["mrr_gain"] >= 0.01
        and intervals["mrr_gain"].lower > 0,
        "hit5": metrics["hit5_gain"] >= -0.005,
        "self_preserved": metrics["self_top1_gain_pp"] >= -0.5,
        "nonself": metrics["nonself_top1_gain_pp"] >= 2.0
        and intervals["nonself_top1_gain_pp"].lower > 0
        and metrics["nonself_mrr_gain"] >= 0.01
        and intervals["nonself_mrr_gain"].lower > 0,
        "counterfactual": metrics["counterfactual_top1_gain_pp"] >= 1.0
        and intervals["counterfactual_top1_gain_pp"].lower > 0,
        "fold_direction": metrics["positive_fold_count"] >= 4
        and metrics["nonself_positive_fold_count"] >= 4,
        "raw_increment": (
            metrics["raw_increment_top1_pp"] >= 0.5
            and intervals["raw_increment_top1_pp"].lower > 0
        )
        or (
            metrics["raw_increment_mrr"] >= 0.003
            and intervals["raw_increment_mrr"].lower > 0
        ),
    }
    gates["PRIMARY"] = all(gates.values())
    return metrics, intervals, gates


def run_development(
    config: QHSMConfig,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Interval], dict[str, bool]]:
    root_path = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    frame, vocab, dag = load_multires_development(root_path)
    assignments = assign_balanced_root_folds(frame, config.n_splits)
    outputs: list[pd.DataFrame] = []
    configs: list[dict[str, Any]] = []
    for fold_id in range(config.n_splits):
        outer = frame.loc[assignments == fold_id].reset_index(drop=True)
        outer_training = frame.loc[assignments != fold_id].reset_index(drop=True)
        roles = make_inner_roles(outer_training, 20, 20)
        fit = outer_training.loc[outer_training["root"].isin(roles.fit_roots)].reset_index(drop=True)
        validation = outer_training.loc[
            outer_training["root"].isin(roles.validation_roots)
        ].reset_index(drop=True)
        predictions, selected = _evaluate_fold(fit, validation, outer, vocab, dag, config)
        predictions["fold"] = fold_id
        outputs.append(predictions)
        configs.append(selected)
    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != len(frame) or set(predictions["root"]) & set(TEST_ROOTS):
        raise AssertionError("OOF coverage or locked-root exclusion failed")
    metrics, intervals, gates = summarize_qhsm(predictions, config.bootstrap, config.seed)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [{"metric": name, **asdict(interval)} for name, interval in intervals.items()]
        ).to_csv(destination / "bootstrap_intervals.csv", index=False)
        write_canonical_json(destination / "gates.json", gates)
        write_canonical_json(destination / "model_configs.json", configs)
        write_manifest(
            destination / "run_manifest.json",
            inputs={"development_rows": len(predictions), "development_roots": predictions["root"].nunique()},
            config=asdict(config),
            split_audit={"test_roots_in_predictions": sorted(set(predictions["root"]) & set(TEST_ROOTS))},
        )
    return predictions, metrics, intervals, gates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = QHSMConfig(seed=args.seed, bootstrap=args.bootstrap)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root / "experiments" / "gsad" / "results" / "development" / f"qhsm_seed{config.seed}"
    )
    _, metrics, _, gates = run_development(config, destination, project_root)
    print(json.dumps({"primary_passed": gates["PRIMARY"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
