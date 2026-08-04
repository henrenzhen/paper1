from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .attack_dag import StructuredSet


FEATURE_NAMES = (
    "entropy",
    "margin",
    "transition_surprise",
    "backoff_signal",
    "fit_distance",
)

_FUTURE_FIELD_TOKENS = (
    "next_technique",
    "true_label",
    "matched_technique_name",
    "matched_description",
    "matched_command_summary",
)


def _validate_inputs(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    prefixes: Sequence[Sequence[str]],
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional multi-class matrix")
    if len(matrix) != len(metadata) or len(matrix) != len(prefixes):
        raise ValueError("probabilities, metadata, and prefixes differ in length")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    lowered = [str(column).strip().lower() for column in metadata.columns]
    leaked = [
        column
        for column in lowered
        if any(token in column for token in _FUTURE_FIELD_TOKENS)
    ]
    if leaked:
        raise ValueError(f"future context metadata is forbidden: {leaked}")
    if "used_order" not in metadata:
        raise ValueError("metadata must contain inference-time used_order")
    return matrix


def _raw_features(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    prefixes: Sequence[Sequence[str]],
) -> np.ndarray:
    matrix = _validate_inputs(probabilities, metadata, prefixes)
    clipped = np.clip(matrix, 1e-15, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / np.log(matrix.shape[1])
    sorted_probabilities = np.sort(matrix, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    transition_surprise = -np.log(np.maximum(matrix.max(axis=1), 1e-15))
    used_order = metadata["used_order"].to_numpy(dtype=float)
    if np.any(~np.isfinite(used_order)) or np.any(used_order < 0):
        raise ValueError("used_order must be finite and nonnegative")
    backoff_signal = 1.0 / (1.0 + used_order)
    return np.column_stack((entropy, margin, transition_surprise, backoff_signal))


@dataclass(frozen=True)
class FeatureReference:
    median: np.ndarray
    mad: np.ndarray


@dataclass(frozen=True)
class InferenceFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    provenance: tuple[str, ...]


def fit_feature_reference(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    prefixes: Sequence[Sequence[str]],
) -> FeatureReference:
    raw = _raw_features(probabilities, metadata, prefixes)
    median = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - median), axis=0)
    mad[mad < 1e-12] = 1.0
    return FeatureReference(median=median, mad=mad)


def build_inference_features(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    prefixes: Sequence[Sequence[str]],
    fit_reference: FeatureReference,
) -> InferenceFeatures:
    raw = _raw_features(probabilities, metadata, prefixes)
    if fit_reference.median.shape != (4,) or fit_reference.mad.shape != (4,):
        raise ValueError("feature reference must describe four raw features")
    standardized = (raw - fit_reference.median) / fit_reference.mad
    fit_distance = np.sqrt(np.mean(np.square(standardized), axis=1))
    values = np.column_stack((raw, fit_distance))
    return InferenceFeatures(
        values=values,
        names=FEATURE_NAMES,
        provenance=(
            "model_probability",
            "model_probability",
            "model_probability",
            "used_order",
            "fit_probability_reference",
        ),
    )


def _root_balanced_weights(roots: Sequence[str]) -> np.ndarray:
    root_array = np.asarray(roots, dtype=str)
    if root_array.ndim != 1 or len(root_array) == 0:
        raise ValueError("roots must be a nonempty vector")
    unique, counts = np.unique(root_array, return_counts=True)
    count_by_root = dict(zip(unique, counts, strict=True))
    weights = np.asarray(
        [1.0 / (len(unique) * count_by_root[root]) for root in root_array],
        dtype=float,
    )
    return weights / weights.sum()


@dataclass(frozen=True)
class LogisticScore:
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    l2: float
    converged: bool
    iterations: int

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coefficients):
            raise ValueError("feature matrix has the wrong shape")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        logits = self.intercept + standardized @ self.coefficients
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_root_balanced_logistic(
    features: np.ndarray,
    correct: Sequence[bool],
    roots: Sequence[str],
    l2: float,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> LogisticScore:
    matrix = np.asarray(features, dtype=float)
    outcome = np.asarray(correct, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(outcome) or len(matrix) != len(roots):
        raise ValueError("features, outcomes, and roots have incompatible shapes")
    if np.any(~np.isfinite(matrix)) or np.any((outcome != 0) & (outcome != 1)):
        raise ValueError("features must be finite and outcomes binary")
    if l2 < 0:
        raise ValueError("l2 must be nonnegative")
    weights = _root_balanced_weights(roots)
    feature_mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average(np.square(matrix - feature_mean), axis=0, weights=weights)
    feature_scale = np.sqrt(variance)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(matrix)), standardized))
    coefficients = np.zeros(design.shape[1], dtype=float)
    converged = False
    iterations = 0
    penalty = np.diag(np.concatenate(([0.0], np.full(matrix.shape[1], float(l2)))))

    for iteration in range(1, int(max_iterations) + 1):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (weights * (probability - outcome)) + penalty @ coefficients
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        hessian += np.eye(hessian.shape[0]) * 1e-12
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        iterations = iteration
        if float(np.max(np.abs(step))) < float(tolerance):
            converged = True
            break
    return LogisticScore(
        coefficients=coefficients[1:].copy(),
        intercept=float(coefficients[0]),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        l2=float(l2),
        converged=converged,
        iterations=iterations,
    )


@dataclass(frozen=True)
class ExactThreshold:
    threshold: float
    enabled: bool
    coverage: float
    empirical_risk: float
    upper_risk: float
    accepted_roots: int


def calibrate_exact_threshold(
    scores: Sequence[float],
    correct: Sequence[bool],
    roots: Sequence[str],
    target_risk: float,
    confidence_z: float = 1.645,
) -> ExactThreshold:
    score_array = np.asarray(scores, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    root_array = np.asarray(roots, dtype=str)
    if (
        score_array.ndim != 1
        or len(score_array) == 0
        or len(score_array) != len(correct_array)
        or len(score_array) != len(root_array)
    ):
        raise ValueError("scores, correctness, and roots must be equal nonempty vectors")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not 0 <= target_risk <= 1 or confidence_z < 0:
        raise ValueError("invalid target risk or confidence multiplier")

    feasible: list[ExactThreshold] = []
    for threshold in sorted(set(float(value) for value in score_array), reverse=True):
        accepted = score_array >= threshold
        root_risks = []
        for root in sorted(set(root_array[accepted])):
            root_mask = accepted & (root_array == root)
            root_risks.append(float((~correct_array[root_mask]).mean()))
        if not root_risks:
            continue
        risk_values = np.asarray(root_risks, dtype=float)
        empirical_risk = float(risk_values.mean())
        if len(risk_values) > 1:
            standard_error = float(risk_values.std(ddof=1) / np.sqrt(len(risk_values)))
        else:
            standard_error = 1.0
        upper = min(1.0, empirical_risk + float(confidence_z) * standard_error)
        if upper <= target_risk:
            feasible.append(
                ExactThreshold(
                    threshold=threshold,
                    enabled=True,
                    coverage=float(accepted.mean()),
                    empirical_risk=empirical_risk,
                    upper_risk=upper,
                    accepted_roots=len(risk_values),
                )
            )
    if not feasible:
        return ExactThreshold(
            threshold=float("inf"),
            enabled=False,
            coverage=0.0,
            empirical_risk=1.0,
            upper_risk=1.0,
            accepted_roots=0,
        )
    return max(feasible, key=lambda item: (item.coverage, -item.threshold))


@dataclass(frozen=True)
class Action:
    kind: str
    nodes: frozenset[str]
    descendants: frozenset[str]
    reason: str


def choose_action(
    gamma: frozenset[str],
    structured: StructuredSet,
    safety_score: float,
    threshold: float,
    max_leaf_size: int,
    support_ok: bool,
) -> Action:
    if not support_ok:
        return Action("abstain", frozenset(), frozenset(), "calibration_support")
    if (
        len(gamma) == 1
        and np.isfinite(threshold)
        and float(safety_score) >= float(threshold)
    ):
        label = next(iter(gamma))
        return Action("exact", frozenset({label}), frozenset({label}), "singleton_safe")
    if structured.leaf_equivalent_size <= int(max_leaf_size):
        return Action("dag", structured.nodes, structured.descendants, "structured_supported")
    return Action("abstain", frozenset(), frozenset(), "leaf_budget")
