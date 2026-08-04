from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import freeze_candidate, sha256_file, write_canonical_json, write_manifest
from .attack_dag import AttackDAG
from .conformal import fit_clustered_aps
from .context_tree import SupportAdaptiveContextTree
from .data_protocol import FrozenSplit, TEST_ROOTS
from .metrics import GateResult, Interval, cluster_bootstrap_difference, root_macro_mean
from .opinion_pool import pool_probabilities
from .rank_conformal import fit_rank_union, stable_rank_matrix
from .run_development import (
    _global_clusters,
    _probability_candidates,
    _role_overlap_audit,
    _root_macro_mrr,
    _sample_ids,
    _select_probability_model,
    assign_balanced_root_folds,
    development_frame,
    json_key,
    make_inner_roles,
)


@dataclass(frozen=True)
class RacerConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    alpha: float = 0.10
    sar_alpha: float = 0.10
    sar_max_contexts: tuple[int, ...] = (1, 2, 3, 4, 5)
    sar_kappas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    opinion_weights: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    tail_root_support: int = 5

    def __post_init__(self) -> None:
        if int(self.bootstrap) < 1 or int(self.n_splits) < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if not 0 < float(self.alpha) < 1 or float(self.sar_alpha) < 0:
            raise ValueError("invalid conformal or SAR alpha")
        if not self.sar_max_contexts or min(self.sar_max_contexts) < 1:
            raise ValueError("SAR context grid must contain positive integers")
        if not self.sar_kappas or min(self.sar_kappas) < 0:
            raise ValueError("SAR kappa grid must be nonnegative")
        if (
            not self.opinion_weights
            or min(self.opinion_weights) < 0
            or max(self.opinion_weights) > 1
        ):
            raise ValueError("opinion weights must lie in [0, 1]")
        if int(self.tail_root_support) < 1:
            raise ValueError("tail root support must be positive")


@dataclass(frozen=True)
class RacerFoldResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]
    model_config: dict[str, Any]


@dataclass(frozen=True)
class RacerSummary:
    metrics: dict[str, float]
    intervals: dict[str, Interval]
    gates: dict[str, GateResult]


@dataclass(frozen=True)
class RacerDevelopmentResult:
    predictions: pd.DataFrame
    summary: RacerSummary
    negative_control: RacerSummary
    fold_audits: tuple[dict[str, Any], ...]
    model_configs: tuple[dict[str, Any], ...]
    output_dir: Path | None


FROZEN_INPUT_FILES = (
    "data_v2/core/sim_train_parent_min3.csv",
    "data_v2/core/sim_val_parent_min3.csv",
    "data_v2/core/sim_test_parent_min3.csv",
    "data_v2/core/rl_label_vocab.csv",
    "data/enterprise-attack-18.1.json",
)

FROZEN_SOURCE_FILES = (
    "experiments/gsad/artifacts.py",
    "experiments/gsad/attack_dag.py",
    "experiments/gsad/conformal.py",
    "experiments/gsad/context_tree.py",
    "experiments/gsad/data_protocol.py",
    "experiments/gsad/metrics.py",
    "experiments/gsad/opinion_pool.py",
    "experiments/gsad/probability_models.py",
    "experiments/gsad/rank_conformal.py",
    "experiments/gsad/run_development.py",
    "experiments/gsad/run_racer_development.py",
    "experiments/gsad/run_racer_locked.py",
)


def frozen_file_hashes(project_root: Path | None = None) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    paths = FROZEN_INPUT_FILES + FROZEN_SOURCE_FILES
    return {relative: sha256_file(root / relative) for relative in paths}


def _root_top1(
    probabilities: np.ndarray,
    targets: Sequence[str],
    roots: Sequence[str],
    vocab: Sequence[str],
) -> float:
    ranks = stable_rank_matrix(probabilities)
    label_to_index = {str(label): index for index, label in enumerate(vocab)}
    correct = [
        target in label_to_index and ranks[row, label_to_index[str(target)]] == 1
        for row, target in enumerate(targets)
    ]
    return root_macro_mean(
        pd.DataFrame({"root": [str(root) for root in roots], "correct": correct}),
        "correct",
    )


def _select_sar_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    config: RacerConfig,
) -> tuple[SupportAdaptiveContextTree, dict[str, Any], np.ndarray]:
    scored = []
    for max_context in config.sar_max_contexts:
        for kappa in config.sar_kappas:
            model = SupportAdaptiveContextTree(
                vocab,
                max_context=int(max_context),
                alpha=config.sar_alpha,
                kappa=float(kappa),
            ).fit(fit["prefix_ids"], fit["target"], fit["root"])
            probabilities, _ = model.predict_proba_with_meta(validation["prefix_ids"])
            mrr = _root_macro_mrr(
                probabilities, validation["target"], validation["root"], vocab
            )
            top1 = _root_top1(
                probabilities, validation["target"], validation["root"], vocab
            )
            scored.append(
                (
                    -mrr,
                    -top1,
                    int(max_context),
                    float(kappa),
                    model,
                    probabilities,
                )
            )
    _, _, max_context, kappa, model, probabilities = min(
        scored, key=lambda item: item[:4]
    )
    return (
        model,
        {
            "kind": "sar_ctw",
            "alpha": config.sar_alpha,
            "max_context": max_context,
            "kappa": kappa,
        },
        probabilities,
    )


def _family_name(model_config: dict[str, Any]) -> str:
    if model_config["kind"] == "tactic_aware":
        return "tactic_aware"
    order = int(model_config["order"])
    return {1: "unigram", 2: "bigram", 3: "trigram"}[order]


def _select_family_experts(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    dag: AttackDAG,
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    best: dict[str, tuple[tuple[float, str], object, dict[str, Any]]] = {}
    for model_config, model in _probability_candidates(vocab, dag):
        family = _family_name(model_config)
        model.fit(fit["prefix_ids"], fit["target"], fit["root"])
        probabilities, _ = model.predict_proba_with_meta(validation["prefix_ids"])
        score = _root_macro_mrr(
            probabilities, validation["target"], validation["root"], vocab
        )
        key = (-score, json_key(model_config))
        if family not in best or key < best[family][0]:
            best[family] = (key, model, model_config)
    required = ("unigram", "bigram", "trigram", "tactic_aware")
    if set(best) != set(required):
        raise AssertionError("rank-union expert family is incomplete")
    return (
        {family: best[family][1] for family in required},
        {family: best[family][2] for family in required},
    )


def _predict_experts(
    experts: dict[str, object], prefixes: Sequence[Sequence[str]]
) -> tuple[np.ndarray, ...]:
    return tuple(
        experts[family].predict_proba_with_meta(prefixes)[0]
        for family in ("unigram", "bigram", "trigram", "tactic_aware")
    )


def _select_opinion_pool(
    sar_probabilities: np.ndarray,
    other_probabilities: dict[str, np.ndarray],
    validation: pd.DataFrame,
    vocab: Sequence[str],
    weights: Sequence[float],
) -> dict[str, Any]:
    scored = []
    for expert_name in sorted(other_probabilities):
        for kind in ("linear", "log"):
            for weight in weights:
                probabilities = pool_probabilities(
                    sar_probabilities,
                    other_probabilities[expert_name],
                    weight=float(weight),
                    kind=kind,
                )
                mrr = _root_macro_mrr(
                    probabilities,
                    validation["target"],
                    validation["root"],
                    vocab,
                )
                top1 = _root_top1(
                    probabilities,
                    validation["target"],
                    validation["root"],
                    vocab,
                )
                scored.append(
                    (
                        -mrr,
                        -top1,
                        expert_name,
                        kind,
                        float(weight),
                    )
                )
    _, _, expert_name, kind, weight = min(scored)
    return {"expert": expert_name, "kind": kind, "sar_weight": weight}


def _ranks_and_top_labels(
    probabilities: np.ndarray, vocab: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    ranks = stable_rank_matrix(probabilities)
    top_indices = np.argmin(ranks, axis=1)
    return ranks, np.asarray(vocab, dtype=object)[top_indices]


def evaluate_racer_outer_fold(
    inner_fit: pd.DataFrame,
    validation: pd.DataFrame,
    calibration: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: AttackDAG,
    config: RacerConfig,
    fold_id: int,
) -> RacerFoldResult:
    overlaps = _role_overlap_audit(inner_fit, validation, calibration, outer)
    if overlaps:
        raise ValueError(f"root overlap in RACER outer fold: {overlaps}")
    labels = tuple(str(label) for label in vocab)

    baseline, baseline_config, validation_baseline, _ = _select_probability_model(
        inner_fit, validation, labels, dag
    )
    sar, sar_config, validation_sar = _select_sar_model(
        inner_fit, validation, labels, config
    )
    experts, expert_configs = _select_family_experts(
        inner_fit, validation, labels, dag
    )

    calibration_baseline, _ = baseline.predict_proba_with_meta(
        calibration["prefix_ids"]
    )
    outer_baseline, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    outer_sar, _ = sar.predict_proba_with_meta(outer["prefix_ids"])
    calibration_experts = _predict_experts(experts, calibration["prefix_ids"])
    outer_experts = _predict_experts(experts, outer["prefix_ids"])
    expert_order = ("unigram", "bigram", "trigram", "tactic_aware")
    validation_experts = _predict_experts(experts, validation["prefix_ids"])
    validation_sources = {"baseline": validation_baseline}
    validation_sources.update(dict(zip(expert_order, validation_experts, strict=True)))
    opinion_config = _select_opinion_pool(
        validation_sar,
        validation_sources,
        validation,
        labels,
        config.opinion_weights,
    )
    outer_sources = {"baseline": outer_baseline}
    outer_sources.update(dict(zip(expert_order, outer_experts, strict=True)))
    outer_racer = pool_probabilities(
        outer_sar,
        outer_sources[opinion_config["expert"]],
        weight=opinion_config["sar_weight"],
        kind=opinion_config["kind"],
    )

    rank_union = fit_rank_union(
        calibration_experts,
        calibration["target"],
        labels,
        alpha=config.alpha,
    )
    racer_sets = rank_union.predict_sets(outer_experts)
    global_predictor = fit_clustered_aps(
        calibration_baseline,
        calibration["target"],
        _global_clusters(labels, len(validation)),
        alpha=config.alpha,
        sample_ids=_sample_ids(calibration, f"racer-fold{fold_id}:global-cal"),
        min_calibration_support=1,
        seed=config.seed + fold_id,
    )
    baseline_sets = global_predictor.predict_sets(
        outer_baseline, _sample_ids(outer, f"racer-fold{fold_id}:global-outer")
    )

    baseline_ranks, baseline_top1 = _ranks_and_top_labels(outer_baseline, labels)
    sar_ranks, sar_top1 = _ranks_and_top_labels(outer_sar, labels)
    racer_ranks, racer_top1 = _ranks_and_top_labels(outer_racer, labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    baseline_true_ranks = []
    sar_true_ranks = []
    racer_true_ranks = []
    for row_index, target in enumerate(outer["target"].astype(str)):
        target_index = label_to_index.get(target)
        baseline_true_ranks.append(
            int(baseline_ranks[row_index, target_index])
            if target_index is not None
            else len(labels) + 1
        )
        sar_true_ranks.append(
            int(sar_ranks[row_index, target_index])
            if target_index is not None
            else len(labels) + 1
        )
        racer_true_ranks.append(
            int(racer_ranks[row_index, target_index])
            if target_index is not None
            else len(labels) + 1
        )

    target_root_support = (
        inner_fit[["target", "root"]]
        .astype(str)
        .drop_duplicates()
        .groupby("target")["root"]
        .nunique()
        .to_dict()
    )
    clean_targets = outer["target"].astype(str).to_numpy()
    is_self = [
        bool(prefix) and str(prefix[-1]) == target
        for prefix, target in zip(outer["prefix_ids"], clean_targets, strict=True)
    ]
    predictions = pd.DataFrame(
        {
            "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
            "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
            "root": outer["root"].astype(str).to_numpy(),
            "target": clean_targets,
            "fold": int(fold_id),
            "baseline_top1": baseline_top1.astype(str),
            "sar_top1": sar_top1.astype(str),
            "racer_top1": racer_top1.astype(str),
            "baseline_rank": baseline_true_ranks,
            "sar_rank": sar_true_ranks,
            "racer_rank": racer_true_ranks,
            "baseline_hit5": np.asarray(baseline_true_ranks) <= 5,
            "sar_hit5": np.asarray(sar_true_ranks) <= 5,
            "racer_hit5": np.asarray(racer_true_ranks) <= 5,
            "baseline_set": baseline_sets,
            "racer_set": racer_sets,
            "is_self": is_self,
            "tail_label": [
                int(target_root_support.get(target, 0)) <= config.tail_root_support
                for target in clean_targets
            ],
            "vocab_size": len(labels),
        }
    )
    audit = {
        "fold": int(fold_id),
        "role_overlaps": overlaps,
        "role_roots": {
            "fit": int(inner_fit["root"].nunique()),
            "validation": int(validation["root"].nunique()),
            "calibration": int(calibration["root"].nunique()),
            "outer": int(outer["root"].nunique()),
        },
        "role_rows": {
            "fit": len(inner_fit),
            "validation": len(validation),
            "calibration": len(calibration),
            "outer": len(outer),
        },
        "rank_union": rank_union.audit(),
        "global_aps_forced_nonempty": global_predictor.last_forced_nonempty_count,
        "tail_labels": int(predictions["tail_label"].sum()),
    }
    return RacerFoldResult(
        predictions=predictions,
        audit=audit,
        model_config={
            "baseline": baseline_config,
            "sar_ctw": sar_config,
            "opinion_pool": opinion_config,
            "rank_union_experts": expert_configs,
        },
    )


def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "root",
        "target",
        "baseline_top1",
        "sar_top1",
        "racer_top1",
        "baseline_rank",
        "sar_rank",
        "racer_rank",
        "baseline_hit5",
        "sar_hit5",
        "racer_hit5",
        "baseline_set",
        "racer_set",
        "is_self",
        "tail_label",
        "vocab_size",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"RACER predictions lack {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["_baseline_correct"] = (
        work["baseline_top1"].astype(str) == work["target"].astype(str)
    )
    work["_sar_correct"] = (
        work["sar_top1"].astype(str) == work["target"].astype(str)
    )
    work["_racer_correct"] = (
        work["racer_top1"].astype(str) == work["target"].astype(str)
    )
    work["_baseline_rr"] = 1.0 / work["baseline_rank"].astype(float)
    work["_sar_rr"] = 1.0 / work["sar_rank"].astype(float)
    work["_racer_rr"] = 1.0 / work["racer_rank"].astype(float)
    work["_baseline_set_hit"] = [
        str(target) in prediction_set
        for target, prediction_set in zip(
            work["target"], work["baseline_set"], strict=True
        )
    ]
    work["_racer_set_hit"] = [
        str(target) in prediction_set
        for target, prediction_set in zip(work["target"], work["racer_set"], strict=True)
    ]
    work["_baseline_set_size"] = work["baseline_set"].map(len)
    work["_racer_set_size"] = work["racer_set"].map(len)
    work["_baseline_full"] = (
        work["_baseline_set_size"] >= work["vocab_size"].astype(int)
    )
    work["_racer_full"] = (
        work["_racer_set_size"] >= work["vocab_size"].astype(int)
    )
    return work


def _root_mean(frame: pd.DataFrame, column: str) -> float:
    return root_macro_mean(frame, column)


def _slice_gain(
    work: pd.DataFrame, mask: pd.Series, candidate: str, baseline: str
) -> float:
    subset = work.loc[mask]
    if len(subset) == 0:
        return 0.0
    return _root_mean(subset, candidate) - _root_mean(subset, baseline)


def _summary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    work = _prepared(frame)
    baseline_size = _root_mean(work, "_baseline_set_size")
    racer_size = _root_mean(work, "_racer_set_size")
    tail = work["tail_label"].astype(bool)
    nonself = ~work["is_self"].astype(bool)
    return {
        "baseline_top1": _root_mean(work, "_baseline_correct"),
        "sar_top1": _root_mean(work, "_sar_correct"),
        "racer_top1": _root_mean(work, "_racer_correct"),
        "top1_gain_pp": 100.0
        * (_root_mean(work, "_racer_correct") - _root_mean(work, "_baseline_correct")),
        "baseline_mrr": _root_mean(work, "_baseline_rr"),
        "sar_mrr": _root_mean(work, "_sar_rr"),
        "racer_mrr": _root_mean(work, "_racer_rr"),
        "mrr_gain": _root_mean(work, "_racer_rr") - _root_mean(work, "_baseline_rr"),
        "hit5_gain": _root_mean(work, "racer_hit5")
        - _root_mean(work, "baseline_hit5"),
        "nonself_top1_gain": _slice_gain(
            work, nonself, "_racer_correct", "_baseline_correct"
        ),
        "nonself_mrr_gain": _slice_gain(work, nonself, "_racer_rr", "_baseline_rr"),
        "baseline_row_coverage": float(work["_baseline_set_hit"].mean()),
        "racer_row_coverage": float(work["_racer_set_hit"].mean()),
        "baseline_root_coverage": _root_mean(work, "_baseline_set_hit"),
        "racer_root_coverage": _root_mean(work, "_racer_set_hit"),
        "row_coverage_difference": float(
            work["_racer_set_hit"].mean() - work["_baseline_set_hit"].mean()
        ),
        "root_coverage_difference": _root_mean(work, "_racer_set_hit")
        - _root_mean(work, "_baseline_set_hit"),
        "baseline_mean_set_size": baseline_size,
        "racer_mean_set_size": racer_size,
        "set_reduction_relative": (
            (baseline_size - racer_size) / baseline_size if baseline_size > 0 else 0.0
        ),
        "baseline_full_set_rate": float(work["_baseline_full"].mean()),
        "racer_full_set_rate": float(work["_racer_full"].mean()),
        "tail_coverage_difference": _slice_gain(
            work, tail, "_racer_set_hit", "_baseline_set_hit"
        ),
        "self_rate": float(work["is_self"].astype(bool).mean()),
        "rows": float(len(work)),
        "roots": float(work["root"].nunique()),
    }


def _racer_gates(
    metrics: dict[str, float], intervals: dict[str, Interval]
) -> dict[str, GateResult]:
    top1 = intervals["top1_gain_pp"]
    mrr = intervals["mrr_gain"]
    reduction = intervals["set_reduction_relative"]
    gates = {
        "R1": GateResult(
            metrics["top1_gain_pp"] >= 1.0 and top1.lower > 0,
            metrics["top1_gain_pp"],
            ">= 1pp and CI lower > 0",
            top1.lower,
            "RACER opinion-pool Top-1 improvement",
        ),
        "R2": GateResult(
            metrics["mrr_gain"] >= 0.01 and mrr.lower > 0,
            metrics["mrr_gain"],
            ">= 0.01 and CI lower > 0",
            mrr.lower,
            "RACER opinion-pool MRR improvement",
        ),
        "R3": GateResult(
            metrics["hit5_gain"] >= -0.005,
            metrics["hit5_gain"],
            ">= -0.005",
            None,
            "Hit@5 non-inferiority",
        ),
        "R4": GateResult(
            metrics["nonself_top1_gain"] >= 0
            and metrics["nonself_mrr_gain"] >= 0,
            min(metrics["nonself_top1_gain"], metrics["nonself_mrr_gain"]),
            "non-self Top-1 and MRR gains >= 0",
            None,
            "parent-collapse artifact control",
        ),
        "S1": GateResult(
            0.88 <= metrics["racer_row_coverage"] <= 0.92
            and 0.88 <= metrics["racer_root_coverage"] <= 0.92
            and abs(metrics["row_coverage_difference"]) <= 0.005
            and abs(metrics["root_coverage_difference"]) <= 0.005,
            metrics["racer_root_coverage"],
            "coverage 0.88-0.92 and matched within 0.005",
            None,
            "rank-union coverage",
        ),
        "S2": GateResult(
            metrics["set_reduction_relative"] >= 0.05 and reduction.lower > 0,
            metrics["set_reduction_relative"],
            ">= 5% and CI lower > 0",
            reduction.lower,
            "rank-union set efficiency",
        ),
        "S3": GateResult(
            metrics["racer_full_set_rate"] <= metrics["baseline_full_set_rate"]
            and metrics["tail_coverage_difference"] >= -0.02,
            metrics["tail_coverage_difference"],
            "full-set no worse and tail coverage loss <= 0.02",
            None,
            "tail/full-set safeguard",
        ),
    }
    primary = all(gate.passed for gate in gates.values())
    gates["PRIMARY"] = GateResult(
        primary,
        None,
        "R1-R4 and S1-S3",
        None,
        "all RACER development conditions" if primary else "one or more RACER conditions failed",
    )
    return gates


def summarize_racer_predictions(
    frame: pd.DataFrame, n_boot: int, seed: int
) -> RacerSummary:
    work = _prepared(frame)

    def top1_gain(sample: pd.DataFrame) -> float:
        return 100.0 * (
            _root_mean(sample, "_racer_correct")
            - _root_mean(sample, "_baseline_correct")
        )

    def mrr_gain(sample: pd.DataFrame) -> float:
        return _root_mean(sample, "_racer_rr") - _root_mean(sample, "_baseline_rr")

    def set_reduction(sample: pd.DataFrame) -> float:
        baseline = _root_mean(sample, "_baseline_set_size")
        candidate = _root_mean(sample, "_racer_set_size")
        return (baseline - candidate) / baseline if baseline > 0 else 0.0

    intervals = {
        "top1_gain_pp": cluster_bootstrap_difference(
            work, top1_gain, "root", n_boot, seed
        ),
        "mrr_gain": cluster_bootstrap_difference(
            work, mrr_gain, "root", n_boot, seed + 1
        ),
        "set_reduction_relative": cluster_bootstrap_difference(
            work, set_reduction, "root", n_boot, seed + 2
        ),
    }
    metrics = _summary_metrics(work)
    return RacerSummary(metrics=metrics, intervals=intervals, gates=_racer_gates(metrics, intervals))


def permute_racer_targets(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if "target" not in frame or len(frame) < 2:
        raise ValueError("negative control requires at least two targets")
    output = frame.copy()
    values = output["target"].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    output["target"] = values[rng.permutation(len(values))]
    return output


def run_racer_development(
    config: RacerConfig,
    split: FrozenSplit,
    vocab: Sequence[str],
    dag: AttackDAG,
    output_dir: Path | None = None,
) -> RacerDevelopmentResult:
    frame = development_frame(split).reset_index(drop=True)
    folds = assign_balanced_root_folds(frame, config.n_splits)
    fold_results = []
    for fold_id in range(config.n_splits):
        outer = frame.loc[folds == fold_id].copy()
        outer_training = frame.loc[folds != fold_id].copy()
        roles = make_inner_roles(
            outer_training, validation_root_count=20, calibration_root_count=20
        )
        fold_results.append(
            evaluate_racer_outer_fold(
                inner_fit=outer_training.loc[
                    outer_training["root"].isin(roles.fit_roots)
                ],
                validation=outer_training.loc[
                    outer_training["root"].isin(roles.validation_roots)
                ],
                calibration=outer_training.loc[
                    outer_training["root"].isin(roles.calibration_roots)
                ],
                outer=outer,
                vocab=vocab,
                dag=dag,
                config=config,
                fold_id=fold_id,
            )
        )
    predictions = pd.concat(
        [fold.predictions for fold in fold_results], ignore_index=True
    )
    if len(predictions) != len(frame):
        raise AssertionError("RACER OOF did not produce one row per input")
    if set(predictions["root"].astype(str)) != set(frame["root"].astype(str)):
        raise AssertionError("RACER OOF root coverage is incomplete")
    if set(predictions["root"].astype(str)) & set(TEST_ROOTS):
        raise AssertionError("locked test root appeared in RACER development")
    summary = summarize_racer_predictions(
        predictions, n_boot=config.bootstrap, seed=config.seed
    )
    negative = summarize_racer_predictions(
        permute_racer_targets(predictions, config.seed + 9000),
        n_boot=config.bootstrap,
        seed=config.seed + 10000,
    )
    result = RacerDevelopmentResult(
        predictions=predictions,
        summary=summary,
        negative_control=negative,
        fold_audits=tuple(fold.audit for fold in fold_results),
        model_configs=tuple(fold.model_config for fold in fold_results),
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    if output_dir is not None:
        _write_racer_artifacts(result, config, split, dag)
    return result


def _write_racer_artifacts(
    result: RacerDevelopmentResult,
    config: RacerConfig,
    split: FrozenSplit,
    dag: AttackDAG,
) -> None:
    if result.output_dir is None:
        raise ValueError("RACER result has no output directory")
    output_dir = result.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"RACER output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = result.predictions.copy()
    for column in ("baseline_set", "racer_set"):
        serializable[column] = serializable[column].map(
            lambda value: " || ".join(sorted(str(item) for item in value))
        )
    serializable.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame([result.summary.metrics]).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"metric": metric, **asdict(interval)}
            for metric, interval in result.summary.intervals.items()
        ]
    ).to_csv(output_dir / "bootstrap_intervals.csv", index=False, encoding="utf-8")
    write_canonical_json(
        output_dir / "gates.json",
        {name: asdict(gate) for name, gate in result.summary.gates.items()},
    )
    write_canonical_json(
        output_dir / "negative_control_gates.json",
        {name: asdict(gate) for name, gate in result.negative_control.gates.items()},
    )
    write_canonical_json(output_dir / "fold_audit.json", result.fold_audits)
    write_canonical_json(output_dir / "model_configs.json", result.model_configs)
    write_canonical_json(
        output_dir / "data_audit.json",
        {
            "frozen_split": split.audit,
            "attack_dag": dag.mapping_audit,
            "development_rows": len(result.predictions),
            "development_roots": int(result.predictions["root"].nunique()),
            "locked_test_roots_seen": sorted(
                set(result.predictions["root"].astype(str)) & set(TEST_ROOTS)
            ),
            "self_rate": result.summary.metrics["self_rate"],
        },
    )
    manifest = write_manifest(
        output_dir / "run_manifest.json",
        inputs={
            "files": frozen_file_hashes(),
            "split_audit_digest_source": split.audit,
            "attack_mapping_audit": dag.mapping_audit,
        },
        config=asdict(config),
        split_audit={"folds": result.fold_audits},
    )
    primary = result.summary.gates["PRIMARY"].passed
    negative = result.negative_control.gates["PRIMARY"].passed
    if primary and not negative:
        freeze_candidate(
            config={
                "candidate": "racer",
                "development_config": asdict(config),
                "manifest_digest": manifest["manifest_digest"],
            },
            development_gates=result.summary.gates,
            path=output_dir / "freeze_token.json",
        )
    lines = [
        "# RACER 开发迭代摘要",
        "",
        f"- OOF：{len(result.predictions)} 行，{result.predictions['root'].nunique()} 个 roots。",
        f"- PRIMARY：{'通过' if primary else '失败'}。",
        f"- 置换负对照 PRIMARY：{'异常通过' if negative else '按预期失败'}。",
        "- 锁定 SIM test 未访问。",
        "",
    ]
    (output_dir / "iteration_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _load_default_experiment() -> tuple[FrozenSplit, tuple[str, ...], AttackDAG]:
    project_root = Path(__file__).resolve().parents[2]
    core = project_root / "data_v2" / "core"
    from .data_protocol import build_frozen_split

    split = build_frozen_split(
        [
            pd.read_csv(core / "sim_train_parent_min3.csv"),
            pd.read_csv(core / "sim_val_parent_min3.csv"),
            pd.read_csv(core / "sim_test_parent_min3.csv"),
        ]
    )
    vocab = tuple(
        pd.read_csv(core / "rl_label_vocab.csv")
        .sort_values("label_id")["technique_id_parent"]
        .astype(str)
    )
    dag = AttackDAG.from_stix(
        project_root / "data" / "enterprise-attack-18.1.json", vocab
    )
    return split, vocab, dag


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run root-disjoint RACER development OOF.")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = RacerConfig(
        seed=args.seed, bootstrap=args.bootstrap, n_splits=args.n_splits
    )
    split, vocab, dag = _load_default_experiment()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / f"racer_op_v2_seed{config.seed}"
    )
    result = run_racer_development(
        config, split=split, vocab=vocab, dag=dag, output_dir=output_dir
    )
    print(
        json.dumps(
            {
                "candidate": "racer",
                "rows": len(result.predictions),
                "roots": int(result.predictions["root"].nunique()),
                "primary_passed": result.summary.gates["PRIMARY"].passed,
                "negative_control_passed": result.negative_control.gates[
                    "PRIMARY"
                ].passed,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
