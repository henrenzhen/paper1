from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validated_probabilities(
    probabilities: object,
    targets: Sequence[str],
    vocabulary: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    vocab = [str(label) for label in vocabulary]
    clean_targets = [str(target) for target in targets]
    if probs.ndim != 2 or probs.shape != (len(clean_targets), len(vocab)):
        raise ValueError("probability shape must match targets and vocabulary")
    if not len(clean_targets):
        raise ValueError("probability rows must be nonempty")
    if len(set(vocab)) != len(vocab):
        raise ValueError("vocabulary labels must be unique")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if (probs < 0.0).any():
        raise ValueError("probabilities must be nonnegative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    index = {label: position for position, label in enumerate(vocab)}
    unknown = sorted(set(clean_targets).difference(index))
    if unknown:
        raise ValueError(f"unknown target labels: {unknown}")
    target_index = np.asarray([index[target] for target in clean_targets], dtype=int)
    return probs, target_index


def prediction_diagnostics(
    probabilities: object,
    targets: Sequence[str],
    vocabulary: Sequence[str],
) -> pd.DataFrame:
    probs, target_index = _validated_probabilities(
        probabilities,
        targets,
        vocabulary,
    )
    row_index = np.arange(len(probs))
    true_probability = probs[row_index, target_index]
    one_hot = np.zeros_like(probs)
    one_hot[row_index, target_index] = 1.0
    ordering = np.argsort(-probs, axis=1, kind="stable")
    inverse_ordering = np.argsort(ordering, axis=1, kind="stable")
    target_rank = inverse_ordering[row_index, target_index] + 1
    confidence = probs.max(axis=1)
    correct = (target_rank == 1).astype(float)
    return pd.DataFrame(
        {
            "nll": -np.log(np.clip(true_probability, 1e-15, 1.0)),
            "brier": np.square(probs - one_hot).sum(axis=1),
            "rr": 1.0 / target_rank,
            "hit1": (target_rank <= 1).astype(float),
            "hit3": (target_rank <= 3).astype(float),
            "hit5": (target_rank <= 5).astype(float),
            "hit10": (target_rank <= 10).astype(float),
            "confidence": confidence,
            "correct": correct,
        }
    )


def root_macro_diagnostics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    required = {
        "root",
        f"{prefix}_nll",
        f"{prefix}_brier",
        f"{prefix}_hit1",
        f"{prefix}_rr",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing diagnostic columns: {missing}")
    if frame.empty:
        raise ValueError("diagnostic frame must be nonempty")
    grouped = frame.groupby("root", sort=False)
    return {
        "nll": float(grouped[f"{prefix}_nll"].mean().mean()),
        "brier": float(grouped[f"{prefix}_brier"].mean().mean()),
        "top1": float(grouped[f"{prefix}_hit1"].mean().mean()),
        "mrr": float(grouped[f"{prefix}_rr"].mean().mean()),
        **{
            f"hit{k}": float(grouped[f"{prefix}_hit{k}"].mean().mean())
            for k in (3, 5, 10)
            if f"{prefix}_hit{k}" in frame.columns
        },
    }


def expected_calibration_error(
    frame: pd.DataFrame,
    prefix: str,
    n_bins: int = 10,
) -> float:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence_col = f"{prefix}_confidence"
    correct_col = f"{prefix}_correct"
    missing = sorted({confidence_col, correct_col}.difference(frame.columns))
    if missing:
        raise ValueError(f"missing calibration columns: {missing}")
    if frame.empty:
        raise ValueError("calibration frame must be nonempty")
    confidence = frame[confidence_col].to_numpy(dtype=float)
    correct = frame[correct_col].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or not np.isfinite(correct).all():
        raise ValueError("calibration values must be finite")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, n_bins - 1)
    total = float(len(frame))
    error = 0.0
    for bin_index in range(n_bins):
        selected = bins == bin_index
        if selected.any():
            error += selected.sum() / total * abs(
                float(correct[selected].mean() - confidence[selected].mean())
            )
    return float(error)
