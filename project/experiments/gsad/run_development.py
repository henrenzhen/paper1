from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .attack_dag import AttackDAG, StructuredSet, compress_leaf_set
from .artifacts import freeze_candidate, write_canonical_json, write_manifest
from .conformal import LabelClusters, fit_clustered_aps, fit_graph_clusters
from .data_protocol import FrozenSplit, TEST_ROOTS
from .metrics import (
    GateResult,
    Interval,
    cluster_bootstrap_difference,
    evaluate_gates,
    evaluate_predictions,
    root_macro_mean,
)
from .probability_models import InterpolatedNGram, TacticAwareModel
from .shift_policy import (
    ExactThreshold,
    build_inference_features,
    calibrate_exact_threshold,
    choose_action,
    fit_feature_reference,
    fit_root_balanced_logistic,
)


ALLOWED_CANDIDATES = frozenset({"gsad_core", "gsad_shift", "weighted_gsad"})


def threshold_audit(threshold: ExactThreshold) -> dict[str, Any]:
    """Serialize a calibrated exact-action threshold without non-standard JSON."""
    payload = asdict(threshold)
    if not np.isfinite(float(threshold.threshold)):
        if threshold.enabled:
            raise ValueError("an enabled exact threshold must be finite")
        payload["threshold"] = None
    return payload


@dataclass(frozen=True)
class DevelopmentConfig:
    candidate: str = "gsad_core"
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    alpha: float = 0.10
    min_cluster_support: int = 20
    min_calibration_support: int = 5
    max_nodes: int = 20
    compression_lambda: float = 1.0
    leaf_budget_quantile: float = 0.85
    exact_target_risk: float = 0.50
    confidence_z: float = 1.645

    def __post_init__(self) -> None:
        if self.candidate not in ALLOWED_CANDIDATES:
            raise ValueError(f"candidate must be one of {sorted(ALLOWED_CANDIDATES)}")
        if self.bootstrap < 1 or self.n_splits < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.min_cluster_support < 1 or self.min_calibration_support < 1:
            raise ValueError("cluster supports must be positive")
        if self.max_nodes < 1 or self.compression_lambda < 0:
            raise ValueError("invalid DAG compression configuration")
        if not 0 < self.leaf_budget_quantile <= 1:
            raise ValueError("leaf budget quantile must lie in (0, 1]")
        if not 0 <= self.exact_target_risk <= 1 or self.confidence_z < 0:
            raise ValueError("invalid exact-risk calibration configuration")


@dataclass(frozen=True)
class InnerRoles:
    fit_roots: frozenset[str]
    validation_roots: frozenset[str]
    calibration_roots: frozenset[str]


@dataclass(frozen=True)
class FoldResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]
    model_config: dict[str, Any]
    cluster_digest: str
    cluster_audit: pd.DataFrame


@dataclass(frozen=True)
class DevelopmentSummary:
    metrics: dict[str, float]
    intervals: dict[str, Interval]
    gates: dict[str, GateResult]
    row_metrics: dict[str, float]
    root_metrics: dict[str, float]


@dataclass(frozen=True)
class DevelopmentResult:
    predictions: pd.DataFrame
    summary: DevelopmentSummary
    negative_control: DevelopmentSummary
    fold_audits: tuple[dict[str, Any], ...]
    model_configs: tuple[dict[str, Any], ...]
    cluster_digests: tuple[str, ...]
    output_dir: Path | None


def development_frame(split: FrozenSplit) -> pd.DataFrame:
    frame = pd.concat(
        [split.fit, split.validation, split.calibration], ignore_index=True
    )
    roots = set(frame["root"].astype(str))
    if roots & set(TEST_ROOTS):
        raise ValueError("locked test root entered development frame")
    if len(roots) != 133:
        raise ValueError(f"development requires 133 roots, found {len(roots)}")
    return frame


def assign_balanced_root_folds(frame: pd.DataFrame, n_splits: int) -> np.ndarray:
    if "root" not in frame or len(frame) == 0:
        raise ValueError("frame must contain nonempty root column")
    roots = frame["root"].astype(str)
    row_counts = roots.value_counts().sort_index()
    if int(n_splits) < 2 or int(n_splits) > len(row_counts):
        raise ValueError("invalid number of root folds")
    fold_rows = [0] * int(n_splits)
    fold_root_counts = [0] * int(n_splits)
    root_to_fold: dict[str, int] = {}
    ordered_roots = sorted(row_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    for root, count in ordered_roots:
        fold = min(
            range(int(n_splits)),
            key=lambda index: (fold_rows[index], fold_root_counts[index], index),
        )
        root_to_fold[str(root)] = fold
        fold_rows[fold] += int(count)
        fold_root_counts[fold] += 1
    return roots.map(root_to_fold).to_numpy(dtype=int)


def make_inner_roles(
    outer_training: pd.DataFrame,
    validation_root_count: int,
    calibration_root_count: int,
) -> InnerRoles:
    if "root" not in outer_training or len(outer_training) == 0:
        raise ValueError("outer training frame must contain roots")
    counts = outer_training["root"].astype(str).value_counts().sort_index()
    total_roots = len(counts)
    fit_count = total_roots - int(validation_root_count) - int(calibration_root_count)
    target_counts = {
        "fit": fit_count,
        "validation": int(validation_root_count),
        "calibration": int(calibration_root_count),
    }
    if min(target_counts.values()) < 1:
        raise ValueError("each inner role must contain at least one root")
    role_roots: dict[str, list[str]] = {role: [] for role in target_counts}
    role_rows = {role: 0 for role in target_counts}
    for root, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0])):
        available = [
            role
            for role, target in target_counts.items()
            if len(role_roots[role]) < target
        ]
        role = min(
            available,
            key=lambda name: (
                role_rows[name] / target_counts[name],
                len(role_roots[name]) / target_counts[name],
                name,
            ),
        )
        role_roots[role].append(str(root))
        role_rows[role] += int(count)
    result = InnerRoles(
        fit_roots=frozenset(role_roots["fit"]),
        validation_roots=frozenset(role_roots["validation"]),
        calibration_roots=frozenset(role_roots["calibration"]),
    )
    sets = (result.fit_roots, result.validation_roots, result.calibration_roots)
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise AssertionError("inner role root overlap")
    return result


def _topk(probabilities: np.ndarray, vocab: Sequence[str], k: int = 5) -> list[tuple[str, ...]]:
    labels = np.asarray(vocab, dtype=object)
    output = []
    for row in probabilities:
        order = np.lexsort((np.arange(len(row)), -row))[: min(k, len(row))]
        output.append(tuple(str(label) for label in labels[order]))
    return output


def _root_macro_mrr(
    probabilities: np.ndarray,
    targets: Sequence[str],
    roots: Sequence[str],
    vocab: Sequence[str],
) -> float:
    label_to_index = {str(label): index for index, label in enumerate(vocab)}
    rows = []
    for probability, target, root in zip(probabilities, targets, roots, strict=True):
        target = str(target)
        if target not in label_to_index:
            reciprocal_rank = 0.0
        else:
            order = np.lexsort((np.arange(len(probability)), -probability))
            reciprocal_rank = 1.0 / (
                int(np.flatnonzero(order == label_to_index[target])[0]) + 1
            )
        rows.append((str(root), reciprocal_rank))
    score_frame = pd.DataFrame(rows, columns=["root", "mrr"])
    return float(score_frame.groupby("root", sort=True)["mrr"].mean().mean())


def _target_nll(
    probabilities: np.ndarray, targets: Sequence[str], vocab: Sequence[str]
) -> float:
    label_to_index = {str(label): index for index, label in enumerate(vocab)}
    values = []
    for row, target in zip(probabilities, targets, strict=True):
        index = label_to_index.get(str(target))
        values.append(-np.log(max(float(row[index]), 1e-15)) if index is not None else -np.log(1e-15))
    return float(np.mean(values))


def _probability_candidates(
    vocab: Sequence[str], dag: AttackDAG
) -> list[tuple[dict[str, Any], object]]:
    candidates: list[tuple[dict[str, Any], object]] = []
    for alpha in (0.1, 0.5):
        candidates.append(
            (
                {"kind": "ngram", "order": 1, "alpha": alpha, "weights": (1.0,)},
                InterpolatedNGram(vocab, order=1, alpha=alpha, interpolation=(1.0,)),
            )
        )
        for weights in ((0.25, 0.75), (0.5, 0.5)):
            candidates.append(
                (
                    {"kind": "ngram", "order": 2, "alpha": alpha, "weights": weights},
                    InterpolatedNGram(vocab, order=2, alpha=alpha, interpolation=weights),
                )
            )
        for weights in ((0.2, 0.3, 0.5), (0.4, 0.3, 0.3)):
            candidates.append(
                (
                    {"kind": "ngram", "order": 3, "alpha": alpha, "weights": weights},
                    InterpolatedNGram(vocab, order=3, alpha=alpha, interpolation=weights),
                )
            )
        for technique_weight in (0.5, 0.75):
            candidates.append(
                (
                    {
                        "kind": "tactic_aware",
                        "alpha": alpha,
                        "technique_weight": technique_weight,
                        "tactic_weight": 1.0 - technique_weight,
                    },
                    TacticAwareModel(
                        vocab,
                        technique_to_tactics={
                            label: dag.tactics_for(label) for label in vocab
                        },
                        alpha=alpha,
                        technique_weight=technique_weight,
                        tactic_weight=1.0 - technique_weight,
                    ),
                )
            )
    return candidates


def _select_probability_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    dag: AttackDAG,
) -> tuple[object, dict[str, Any], np.ndarray, pd.DataFrame]:
    scored = []
    for config, model in _probability_candidates(vocab, dag):
        model.fit(fit["prefix_ids"], fit["target"], fit["root"])
        probabilities, metadata = model.predict_proba_with_meta(validation["prefix_ids"])
        mrr = _root_macro_mrr(
            probabilities, validation["target"], validation["root"], vocab
        )
        nll = _target_nll(probabilities, validation["target"], vocab)
        scored.append((-mrr, nll, json_key(config), config, model, probabilities, metadata))
    _, _, _, config, model, probabilities, metadata = min(scored, key=lambda item: item[:3])
    return model, config, probabilities, metadata


def json_key(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sample_ids(frame: pd.DataFrame, namespace: str) -> tuple[str, ...]:
    return tuple(
        f"{namespace}|{sequence_id}|{int(prefix_len)}|{index}"
        for index, (sequence_id, prefix_len) in enumerate(
            zip(frame["sequence_id"], frame["prefix_len"], strict=True)
        )
    )


def _global_clusters(vocab: Sequence[str], validation_n: int) -> LabelClusters:
    labels = tuple(str(label) for label in vocab)
    return LabelClusters(
        vocab=labels,
        label_to_cluster={label: "global" for label in labels},
        members={"global": frozenset(labels)},
        validation_support={"global": int(validation_n)},
        min_support=1,
    )


def _compress_many(
    sets: Sequence[frozenset[str]],
    dag: AttackDAG,
    lam: float,
    max_nodes: int,
) -> list[StructuredSet]:
    cache: dict[frozenset[str], StructuredSet] = {}
    output = []
    for prediction_set in sets:
        if prediction_set not in cache:
            cache[prediction_set] = compress_leaf_set(
                prediction_set, dag, lam=lam, max_nodes=max_nodes
            )
        output.append(cache[prediction_set])
    return output


def _role_overlap_audit(*frames: pd.DataFrame) -> dict[str, list[str]]:
    names = ("fit", "validation", "calibration", "outer")
    root_sets = [set(frame["root"].astype(str)) for frame in frames]
    overlaps = {}
    for left_index, left in enumerate(root_sets):
        for right_index in range(left_index + 1, len(root_sets)):
            overlap = sorted(left & root_sets[right_index])
            if overlap:
                overlaps[f"{names[left_index]}:{names[right_index]}"] = overlap
    return overlaps


def evaluate_outer_fold(
    inner_fit: pd.DataFrame,
    validation: pd.DataFrame,
    calibration: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: AttackDAG,
    config: DevelopmentConfig,
    fold_id: int,
) -> FoldResult:
    overlaps = _role_overlap_audit(inner_fit, validation, calibration, outer)
    if overlaps:
        raise ValueError(f"root overlap in outer fold: {overlaps}")
    vocab = tuple(str(label) for label in vocab)
    model, model_config, validation_probs, validation_meta = _select_probability_model(
        inner_fit, validation, vocab, dag
    )
    fit_probs, fit_meta = model.predict_proba_with_meta(inner_fit["prefix_ids"])
    calibration_probs, calibration_meta = model.predict_proba_with_meta(
        calibration["prefix_ids"]
    )
    outer_probs, outer_meta = model.predict_proba_with_meta(outer["prefix_ids"])
    fit_counts = inner_fit["target"].astype(str).value_counts().to_dict()
    clusters = fit_graph_clusters(
        validation_probs,
        validation["target"],
        fit_counts,
        dag,
        vocab,
        min_support=config.min_cluster_support,
    )
    predictor = fit_clustered_aps(
        calibration_probs,
        calibration["target"],
        clusters,
        alpha=config.alpha,
        sample_ids=_sample_ids(calibration, f"fold{fold_id}:cal"),
        min_calibration_support=config.min_calibration_support,
        seed=config.seed + fold_id,
    )
    calibration_gamma = predictor.predict_sets(
        calibration_probs, _sample_ids(calibration, f"fold{fold_id}:cal-predict")
    )
    outer_gamma = predictor.predict_sets(
        outer_probs, _sample_ids(outer, f"fold{fold_id}:outer")
    )
    global_predictor = fit_clustered_aps(
        calibration_probs,
        calibration["target"],
        _global_clusters(vocab, len(validation)),
        alpha=config.alpha,
        sample_ids=_sample_ids(calibration, f"fold{fold_id}:global-cal"),
        min_calibration_support=1,
        seed=config.seed + fold_id,
    )
    outer_global_gamma = global_predictor.predict_sets(
        outer_probs, _sample_ids(outer, f"fold{fold_id}:global-outer")
    )

    calibration_structured = _compress_many(
        calibration_gamma, dag, config.compression_lambda, config.max_nodes
    )
    outer_structured = _compress_many(
        outer_gamma, dag, config.compression_lambda, config.max_nodes
    )
    calibration_leaf_sizes = np.asarray(
        [item.leaf_equivalent_size for item in calibration_structured], dtype=float
    )
    max_leaf_size = max(
        1,
        int(
            np.quantile(
                calibration_leaf_sizes,
                config.leaf_budget_quantile,
                method="higher",
            )
        ),
    )

    if config.candidate == "gsad_core":
        calibration_safety = calibration_probs.max(axis=1)
        outer_safety = outer_probs.max(axis=1)
        safety_audit = {"kind": "top1_probability"}
    else:
        reference = fit_feature_reference(fit_probs, fit_meta, inner_fit["prefix_ids"])
        validation_features = build_inference_features(
            validation_probs, validation_meta, validation["prefix_ids"], reference
        )
        validation_top1 = np.asarray(vocab, dtype=object)[
            np.argmax(validation_probs, axis=1)
        ]
        score_model = fit_root_balanced_logistic(
            validation_features.values,
            validation_top1.astype(str) == validation["target"].astype(str).to_numpy(),
            validation["root"],
            l2=1.0,
        )
        calibration_features = build_inference_features(
            calibration_probs, calibration_meta, calibration["prefix_ids"], reference
        )
        outer_features = build_inference_features(
            outer_probs, outer_meta, outer["prefix_ids"], reference
        )
        calibration_safety = score_model.predict_proba(calibration_features.values)
        outer_safety = score_model.predict_proba(outer_features.values)
        safety_audit = {
            "kind": "root_balanced_logistic",
            "coefficients": score_model.coefficients.tolist(),
            "intercept": score_model.intercept,
            "converged": score_model.converged,
            "iterations": score_model.iterations,
        }

    calibration_singleton = np.asarray([len(item) == 1 for item in calibration_gamma])
    if calibration_singleton.any():
        calibration_exact_labels = np.asarray(
            [next(iter(item)) if len(item) == 1 else "" for item in calibration_gamma]
        )
        threshold = calibrate_exact_threshold(
            calibration_safety[calibration_singleton],
            calibration_exact_labels[calibration_singleton]
            == calibration.loc[calibration_singleton, "target"].astype(str).to_numpy(),
            calibration.loc[calibration_singleton, "root"],
            target_risk=config.exact_target_risk,
            confidence_z=config.confidence_z,
        )
    else:
        threshold = calibrate_exact_threshold(
            np.asarray([0.0]),
            np.asarray([False]),
            np.asarray(["no_singleton"]),
            target_risk=0.0,
            confidence_z=0.0,
        )

    actions = [
        choose_action(
            gamma=gamma,
            structured=structured,
            safety_score=float(score),
            threshold=threshold.threshold,
            max_leaf_size=max_leaf_size,
            support_ok=True,
        )
        for gamma, structured, score in zip(
            outer_gamma, outer_structured, outer_safety, strict=True
        )
    ]
    top5 = _topk(outer_probs, vocab, k=5)
    top1 = [labels[0] for labels in top5]
    exact_labels = [
        next(iter(gamma)) if action.kind == "exact" else ""
        for gamma, action in zip(outer_gamma, actions, strict=True)
    ]
    fit_labels = set(inner_fit["target"].astype(str))
    predictions = pd.DataFrame(
        {
            "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
            "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
            "root": outer["root"].astype(str).to_numpy(),
            "target": outer["target"].astype(str).to_numpy(),
            "fold": int(fold_id),
            "top1_pred": top1,
            "top1_probability": outer_probs.max(axis=1),
            "top5": top5,
            "gamma": outer_gamma,
            "global_gamma": outer_global_gamma,
            "structured_nodes": [item.nodes for item in outer_structured],
            "structured_descendants": [item.descendants for item in outer_structured],
            "action_kind": [action.kind for action in actions],
            "exact_label": exact_labels,
            "descendants": [action.descendants for action in actions],
            "leaf_equivalent_size": [len(action.descendants) for action in actions],
            "display_node_count": [len(action.nodes) for action in actions],
            "raw_structured_leaf_size": [
                item.leaf_equivalent_size for item in outer_structured
            ],
            "safety_score": outer_safety,
            "fit_seen": [str(target) in fit_labels for target in outer["target"]],
            "vocab_size": len(vocab),
        }
    )
    exact_count = int((predictions["action_kind"] == "exact").sum())
    confidence_order = np.lexsort(
        (np.arange(len(predictions)), -predictions["top1_probability"].to_numpy())
    )
    confidence_accept = np.zeros(len(predictions), dtype=bool)
    confidence_accept[confidence_order[:exact_count]] = True
    predictions["confidence_accept_matched"] = confidence_accept
    predictions["base_correct"] = (
        predictions["top1_pred"].astype(str) == predictions["target"].astype(str)
    )
    predictions["global_leaf_equivalent_size"] = [
        len(item) for item in outer_global_gamma
    ]
    predictions["global_leaf_hit"] = [
        str(target) in prediction_set
        for target, prediction_set in zip(
            predictions["target"], outer_global_gamma, strict=True
        )
    ]
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
        "max_leaf_size": max_leaf_size,
        "exact_threshold": threshold_audit(threshold),
        "safety": safety_audit,
        "forced_nonempty": predictor.last_forced_nonempty_count,
    }
    return FoldResult(
        predictions=predictions,
        audit=audit,
        model_config=model_config,
        cluster_digest=clusters.digest(),
        cluster_audit=predictor.audit.reset_index(),
    )


def _selective_root_accuracy(
    frame: pd.DataFrame, accept_column: str, correct_column: str
) -> float:
    accepted = frame.loc[frame[accept_column].astype(bool)]
    if len(accepted) == 0:
        return float("nan")
    return root_macro_mean(accepted, correct_column)


def confidence_acceptance_at_accuracy(
    frame: pd.DataFrame, target_accuracy: float
) -> np.ndarray:
    if not np.isfinite(target_accuracy):
        return np.zeros(len(frame), dtype=bool)
    required = {"root", "top1_probability", "base_correct"}
    if not required.issubset(frame.columns):
        raise ValueError(f"confidence comparator is missing {sorted(required - set(frame.columns))}")
    scores = frame["top1_probability"].to_numpy(dtype=float)
    correct = frame["base_correct"].to_numpy(dtype=bool)
    roots = frame["root"].astype(str).to_numpy()
    order = np.lexsort((np.arange(len(frame)), -scores))
    root_totals: dict[str, int] = {}
    root_correct: dict[str, int] = {}
    best_count = 0
    cursor = 0
    while cursor < len(order):
        score = scores[order[cursor]]
        boundary = cursor
        while boundary < len(order) and scores[order[boundary]] == score:
            index = int(order[boundary])
            root = roots[index]
            root_totals[root] = root_totals.get(root, 0) + 1
            root_correct[root] = root_correct.get(root, 0) + int(correct[index])
            boundary += 1
        macro_accuracy = float(
            np.mean(
                [root_correct[root] / root_totals[root] for root in sorted(root_totals)]
            )
        )
        if macro_accuracy >= target_accuracy:
            best_count = boundary
        cursor = boundary
    accepted = np.zeros(len(frame), dtype=bool)
    accepted[order[:best_count]] = True
    return accepted


def _baseline_coverage_at_accuracy(frame: pd.DataFrame, target_accuracy: float) -> float:
    return float(confidence_acceptance_at_accuracy(frame, target_accuracy).mean())


def _summary_statistics(frame: pd.DataFrame, candidate: str) -> dict[str, float]:
    work = frame.copy()
    work["_gamma_hit"] = [
        str(target) in gamma
        for target, gamma in zip(work["target"], work["gamma"], strict=True)
    ]
    work["_structured_hit"] = [
        str(target) in descendants
        for target, descendants in zip(
            work["target"], work["structured_descendants"], strict=True
        )
    ]
    work["_action_hit"] = [
        str(target) in descendants
        for target, descendants in zip(work["target"], work["descendants"], strict=True)
    ]
    work["_exact"] = work["action_kind"].astype(str) == "exact"
    work["_exact_correct"] = (
        work["exact_label"].astype(str) == work["target"].astype(str)
    )
    work["_global_hit"] = [
        str(target) in gamma
        for target, gamma in zip(work["target"], work["global_gamma"], strict=True)
    ]
    bundle = evaluate_predictions(work)
    exact_accuracy = _selective_root_accuracy(work, "_exact", "_exact_correct")
    confidence_accuracy = _selective_root_accuracy(
        work, "confidence_accept_matched", "base_correct"
    )
    if not np.isfinite(exact_accuracy):
        exact_accuracy = 0.0
    if not np.isfinite(confidence_accuracy):
        confidence_accuracy = 0.0
    exact_coverage = float(work["_exact"].mean())
    confidence_at_risk = confidence_acceptance_at_accuracy(work, exact_accuracy)
    baseline_at_risk = float(confidence_at_risk.mean())
    exact_relative_gain = (
        (exact_coverage - baseline_at_risk) / baseline_at_risk
        if baseline_at_risk > 0
        else (1.0 if exact_coverage > 0 else 0.0)
    )
    mean_gamma_nodes = float(np.mean([len(item) for item in work["gamma"]]))
    dag_win = (
        float(work["display_node_count"].mean()) < mean_gamma_nodes
        and float(work["_structured_hit"].mean())
        >= float(work["_gamma_hit"].mean())
    )
    shift_win = candidate != "gsad_core" and exact_accuracy > confidence_accuracy
    ablation_wins = int(dag_win) + int(shift_win)
    return {
        "coverage_gain_pp_matched_size": 100.0
        * (root_macro_mean(work, "_action_hit") - root_macro_mean(work, "_global_hit")),
        "exact_output_gain_relative": float(exact_relative_gain),
        "exact_coverage": exact_coverage,
        "exact_accuracy_gain_pp": 100.0 * (exact_accuracy - confidence_accuracy),
        "abstain_rate": float((work["action_kind"] == "abstain").mean()),
        "mean_leaf_size": float(work["leaf_equivalent_size"].mean()),
        "baseline_mean_leaf_size": float(
            work["global_leaf_equivalent_size"].mean()
        ),
        "full_set_rate": float(
            (work["leaf_equivalent_size"] >= work["vocab_size"]).mean()
        ),
        "baseline_full_set_rate": float(
            (work["global_leaf_equivalent_size"] >= work["vocab_size"]).mean()
        ),
        "row_coverage": float(work["_gamma_hit"].mean()),
        "root_macro_coverage": root_macro_mean(work, "_gamma_hit"),
        "ablation_wins": float(ablation_wins),
        "exact_accuracy": exact_accuracy,
        "confidence_matched_accuracy": confidence_accuracy,
        "baseline_exact_coverage_at_matched_accuracy": baseline_at_risk,
        "raw_structured_coverage": float(work["_structured_hit"].mean()),
        "global_coverage": float(work["_global_hit"].mean()),
        "mean_display_nodes": float(work["display_node_count"].mean()),
        **{f"row_{key}": float(value) for key, value in bundle.row.items()},
        **{f"root_{key}": float(value) for key, value in bundle.root_macro.items()},
    }


def _safe_interval(
    frame: pd.DataFrame,
    metric_fn,
    n_boot: int,
    seed: int,
) -> Interval:
    try:
        return cluster_bootstrap_difference(
            frame, metric_fn, group_col="root", n_boot=n_boot, seed=seed
        )
    except ValueError:
        point = float(metric_fn(frame))
        if not np.isfinite(point):
            point = 0.0
        return Interval(point=point, lower=-1.0, upper=1.0, valid_replicates=0)


def summarize_development_predictions(
    frame: pd.DataFrame,
    candidate: str,
    n_boot: int,
    seed: int,
) -> DevelopmentSummary:
    if candidate not in ALLOWED_CANDIDATES:
        raise ValueError("unknown candidate")
    work = frame.copy()
    work["_gamma_hit"] = [
        str(target) in gamma
        for target, gamma in zip(work["target"], work["gamma"], strict=True)
    ]
    work["_action_hit"] = [
        str(target) in descendants
        for target, descendants in zip(work["target"], work["descendants"], strict=True)
    ]
    work["_global_hit"] = [
        str(target) in gamma
        for target, gamma in zip(work["target"], work["global_gamma"], strict=True)
    ]
    work["_exact"] = work["action_kind"].astype(str) == "exact"
    work["_exact_correct"] = (
        work["exact_label"].astype(str) == work["target"].astype(str)
    )
    metrics = _summary_statistics(work, candidate)
    target_exact_accuracy = _selective_root_accuracy(work, "_exact", "_exact_correct")
    work["_confidence_accept_risk"] = confidence_acceptance_at_accuracy(
        work, target_exact_accuracy
    )

    def coverage_difference(sample: pd.DataFrame) -> float:
        return 100.0 * (
            root_macro_mean(sample, "_action_hit")
            - root_macro_mean(sample, "_global_hit")
        )

    def exact_accuracy_difference(sample: pd.DataFrame) -> float:
        exact_accuracy = _selective_root_accuracy(sample, "_exact", "_exact_correct")
        baseline_accuracy = _selective_root_accuracy(
            sample, "confidence_accept_matched", "base_correct"
        )
        if not np.isfinite(exact_accuracy) or not np.isfinite(baseline_accuracy):
            return float("nan")
        return 100.0 * (exact_accuracy - baseline_accuracy)

    def exact_relative_gain(sample: pd.DataFrame) -> float:
        candidate_coverage = float(sample["_exact"].mean())
        baseline_coverage = float(sample["_confidence_accept_risk"].mean())
        if baseline_coverage == 0:
            return 1.0 if candidate_coverage > 0 else 0.0
        return (candidate_coverage - baseline_coverage) / baseline_coverage

    intervals = {
        "coverage_gain_pp_matched_size": _safe_interval(
            work, coverage_difference, n_boot, seed
        ),
        "exact_accuracy_gain_pp": _safe_interval(
            work, exact_accuracy_difference, n_boot, seed + 1
        ),
        "exact_output_gain_relative": _safe_interval(
            work, exact_relative_gain, n_boot, seed + 2
        ),
    }
    gates = evaluate_gates(metrics, intervals, ablations={})
    bundle = evaluate_predictions(work)
    return DevelopmentSummary(
        metrics=metrics,
        intervals=intervals,
        gates=gates,
        row_metrics=bundle.row,
        root_metrics=bundle.root_macro,
    )


def permute_targets_for_negative_control(
    frame: pd.DataFrame, seed: int
) -> pd.DataFrame:
    if "target" not in frame or len(frame) < 2:
        raise ValueError("negative control requires at least two targets")
    output = frame.copy()
    original = output["target"].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    permuted = original[rng.permutation(len(original))]
    if np.array_equal(permuted, original):
        permuted = np.roll(original, 1)
    output["target"] = permuted
    if "top1_pred" in output:
        output["base_correct"] = (
            output["top1_pred"].astype(str) == output["target"].astype(str)
        )
    return output


def run_development(
    config: DevelopmentConfig,
    split: FrozenSplit,
    vocab: Sequence[str],
    dag: AttackDAG,
    output_dir: Path | None = None,
) -> DevelopmentResult:
    frame = development_frame(split)
    fold_assignments = assign_balanced_root_folds(frame, config.n_splits)
    fold_results: list[FoldResult] = []
    for fold_id in range(config.n_splits):
        outer = frame.loc[fold_assignments == fold_id].reset_index(drop=True)
        outer_training = frame.loc[fold_assignments != fold_id].reset_index(drop=True)
        roles = make_inner_roles(
            outer_training,
            validation_root_count=20,
            calibration_root_count=20,
        )
        inner_fit = outer_training.loc[
            outer_training["root"].isin(roles.fit_roots)
        ].reset_index(drop=True)
        validation = outer_training.loc[
            outer_training["root"].isin(roles.validation_roots)
        ].reset_index(drop=True)
        calibration = outer_training.loc[
            outer_training["root"].isin(roles.calibration_roots)
        ].reset_index(drop=True)
        fold_results.append(
            evaluate_outer_fold(
                inner_fit,
                validation,
                calibration,
                outer,
                vocab=vocab,
                dag=dag,
                config=config,
                fold_id=fold_id,
            )
        )
    predictions = pd.concat(
        [result.predictions for result in fold_results], ignore_index=True
    )
    if len(predictions) != len(frame):
        raise AssertionError("development OOF did not produce one row per input")
    if set(predictions["root"].astype(str)) != set(frame["root"].astype(str)):
        raise AssertionError("development OOF root coverage is incomplete")
    if set(predictions["root"].astype(str)) & set(TEST_ROOTS):
        raise AssertionError("locked test root appeared in development predictions")
    summary = summarize_development_predictions(
        predictions,
        candidate=config.candidate,
        n_boot=config.bootstrap,
        seed=config.seed,
    )
    negative_frame = permute_targets_for_negative_control(
        predictions, seed=config.seed + 9000
    )
    negative_control = summarize_development_predictions(
        negative_frame,
        candidate=config.candidate,
        n_boot=config.bootstrap,
        seed=config.seed + 10000,
    )
    result = DevelopmentResult(
        predictions=predictions,
        summary=summary,
        negative_control=negative_control,
        fold_audits=tuple(result.audit for result in fold_results),
        model_configs=tuple(result.model_config for result in fold_results),
        cluster_digests=tuple(result.cluster_digest for result in fold_results),
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    if output_dir is not None:
        _write_development_artifacts(result, config, split, dag)
    return result


def _serialize_collection(value: object) -> str:
    if isinstance(value, (set, frozenset, tuple, list)):
        return " || ".join(sorted(str(item) for item in value))
    return str(value)


def _write_development_artifacts(
    result: DevelopmentResult,
    config: DevelopmentConfig,
    split: FrozenSplit,
    dag: AttackDAG,
) -> None:
    if result.output_dir is None:
        raise ValueError("development result has no output directory")
    output_dir = result.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"development output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = result.predictions.copy()
    for column in (
        "top5",
        "gamma",
        "global_gamma",
        "structured_nodes",
        "structured_descendants",
        "descendants",
    ):
        serializable[column] = serializable[column].map(_serialize_collection)
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
            "cluster_digests": result.cluster_digests,
        },
    )
    manifest = write_manifest(
        output_dir / "run_manifest.json",
        inputs={
            "split_audit_digest_source": split.audit,
            "attack_mapping_audit": dag.mapping_audit,
        },
        config=asdict(config),
        split_audit={"folds": result.fold_audits},
    )
    primary_passed = result.summary.gates["PRIMARY"].passed
    negative_passed = result.negative_control.gates["PRIMARY"].passed
    if primary_passed and not negative_passed:
        freeze_candidate(
            config={
                "development_config": asdict(config),
                "manifest_digest": manifest["manifest_digest"],
                "cluster_digests": result.cluster_digests,
            },
            development_gates=result.summary.gates,
            path=output_dir / "freeze_token.json",
        )
    summary_lines = [
        f"# {config.candidate} 迭代摘要",
        "",
        f"- 开发 OOF：{len(result.predictions)} 行，{result.predictions['root'].nunique()} roots。",
        f"- PRIMARY：{'通过' if primary_passed else '失败'}。",
        f"- 标签置换负控制 PRIMARY：{'异常通过' if negative_passed else '按预期失败'}。",
        "- A–G："
        + ", ".join(
            f"{name}={'PASS' if result.summary.gates[name].passed else 'FAIL'}"
            for name in "ABCDEFG"
        ),
        "- 未访问 locked SIM test；本摘要不是最终论文结论。",
        "",
    ]
    (output_dir / "iteration_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-controlled GSAD development OOF experiments."
    )
    parser.add_argument(
        "--candidate", required=True, choices=sorted(ALLOWED_CANDIDATES)
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _load_default_experiment() -> tuple[FrozenSplit, tuple[str, ...], AttackDAG]:
    project_root = Path(__file__).resolve().parents[2]
    core = project_root / "data_v2" / "core"
    frames = [
        pd.read_csv(core / "sim_train_parent_min3.csv"),
        pd.read_csv(core / "sim_val_parent_min3.csv"),
        pd.read_csv(core / "sim_test_parent_min3.csv"),
    ]
    from .data_protocol import build_frozen_split

    split = build_frozen_split(frames)
    vocab = tuple(
        pd.read_csv(core / "rl_label_vocab.csv")
        .sort_values("label_id")["technique_id_parent"]
        .astype(str)
    )
    dag = AttackDAG.from_stix(
        project_root / "data" / "enterprise-attack-18.1.json", vocab
    )
    return split, vocab, dag


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DevelopmentConfig(
        candidate=args.candidate,
        seed=args.seed,
        bootstrap=args.bootstrap,
        n_splits=args.n_splits,
    )
    split, vocab, dag = _load_default_experiment()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / f"{config.candidate}_seed{config.seed}"
    )
    result = run_development(
        config, split=split, vocab=vocab, dag=dag, output_dir=output_dir
    )
    print(
        json.dumps(
            {
                "candidate": config.candidate,
                "rows": len(result.predictions),
                "roots": int(result.predictions["root"].nunique()),
                "primary_passed": result.summary.gates["PRIMARY"].passed,
                "negative_control_passed": result.negative_control.gates[
                    "PRIMARY"
                ].passed,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
