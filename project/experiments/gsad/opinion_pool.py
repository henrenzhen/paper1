from __future__ import annotations

import numpy as np


def _probability_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a nonempty multiclass probability matrix")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{name} rows must sum to one")
    return matrix


def pool_probabilities(
    left: np.ndarray,
    right: np.ndarray,
    weight: float,
    kind: str,
) -> np.ndarray:
    """Pool two experts with a validation-frozen simplex weight."""
    left_matrix = _probability_matrix(left, "left expert")
    right_matrix = _probability_matrix(right, "right expert")
    if left_matrix.shape != right_matrix.shape:
        raise ValueError("opinion experts must have identical shapes")
    if not np.isfinite(weight) or not 0 <= float(weight) <= 1:
        raise ValueError("opinion weight must lie in [0, 1]")
    if kind not in {"linear", "log"}:
        raise ValueError("opinion kind must be linear or log")
    weight = float(weight)
    if kind == "linear":
        pooled = weight * left_matrix + (1.0 - weight) * right_matrix
    else:
        logits = weight * np.log(np.maximum(left_matrix, 1e-15)) + (
            1.0 - weight
        ) * np.log(np.maximum(right_matrix, 1e-15))
        logits -= logits.max(axis=1, keepdims=True)
        pooled = np.exp(logits)
    pooled /= pooled.sum(axis=1, keepdims=True)
    return pooled

