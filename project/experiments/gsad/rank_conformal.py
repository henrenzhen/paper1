from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np


def _validate_probability_matrix(
    probabilities: np.ndarray, *, n_labels: int | None = None
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("probabilities must be a nonempty two-dimensional matrix")
    if n_labels is not None and matrix.shape[1] != int(n_labels):
        raise ValueError("probability matrix does not match the vocabulary")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    return matrix


def stable_rank_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Return one-based label ranks with vocabulary index as the tie breaker."""
    matrix = _validate_probability_matrix(probabilities)
    ranks = np.empty(matrix.shape, dtype=int)
    label_indices = np.arange(matrix.shape[1])
    one_based = np.arange(1, matrix.shape[1] + 1)
    for row_index, row in enumerate(matrix):
        order = np.lexsort((label_indices, -row))
        ranks[row_index, order] = one_based
    return ranks


def minimum_expert_ranks(
    expert_probabilities: Sequence[np.ndarray],
) -> np.ndarray:
    if not expert_probabilities:
        raise ValueError("at least one expert is required")
    rank_matrices = [stable_rank_matrix(matrix) for matrix in expert_probabilities]
    shape = rank_matrices[0].shape
    if any(matrix.shape != shape for matrix in rank_matrices[1:]):
        raise ValueError("expert probability matrices must have identical shapes")
    return np.minimum.reduce(rank_matrices)


def finite_sample_rank_quantile(scores: Sequence[float], alpha: float) -> int:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("calibration scores must be a nonempty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 1):
        raise ValueError("rank scores must be finite and at least one")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    index = ceil((len(values) + 1) * (1.0 - float(alpha))) - 1
    index = min(max(index, 0), len(values) - 1)
    return int(np.sort(values)[index])


@dataclass(frozen=True)
class RankUnionPredictor:
    vocab: tuple[str, ...]
    threshold: int
    alpha: float
    expert_count: int
    calibration_size: int

    def predict_sets(
        self, expert_probabilities: Sequence[np.ndarray]
    ) -> list[frozenset[str]]:
        if len(expert_probabilities) != self.expert_count:
            raise ValueError("prediction expert count differs from calibration")
        minimum_ranks = minimum_expert_ranks(expert_probabilities)
        if minimum_ranks.shape[1] != len(self.vocab):
            raise ValueError("prediction matrices do not match the vocabulary")
        return [
            frozenset(
                self.vocab[index]
                for index in np.flatnonzero(row <= self.threshold)
            )
            for row in minimum_ranks
        ]

    def audit(self) -> dict[str, int | float]:
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "expert_count": self.expert_count,
            "calibration_size": self.calibration_size,
        }


def fit_rank_union(
    expert_probabilities: Sequence[np.ndarray],
    targets: Sequence[str],
    vocab: Sequence[str],
    alpha: float = 0.10,
) -> RankUnionPredictor:
    labels = tuple(str(label) for label in vocab)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("vocabulary must be nonempty and unique")
    minimum_ranks = minimum_expert_ranks(expert_probabilities)
    if minimum_ranks.shape[1] != len(labels):
        raise ValueError("calibration matrices do not match the vocabulary")
    clean_targets = tuple(str(target) for target in targets)
    if len(clean_targets) != minimum_ranks.shape[0]:
        raise ValueError("targets and calibration probabilities differ in length")
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown = sorted(set(clean_targets) - set(label_to_index))
    if unknown:
        raise ValueError(f"calibration targets outside vocabulary: {unknown}")
    scores = [
        minimum_ranks[row_index, label_to_index[target]]
        for row_index, target in enumerate(clean_targets)
    ]
    threshold = finite_sample_rank_quantile(scores, alpha=alpha)
    return RankUnionPredictor(
        vocab=labels,
        threshold=threshold,
        alpha=float(alpha),
        expert_count=len(expert_probabilities),
        calibration_size=len(clean_targets),
    )

