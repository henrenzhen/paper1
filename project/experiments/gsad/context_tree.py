from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .probability_models import (
    UnigramModel,
    _smoothed_probability,
    _validate_parallel_lengths,
)


class SupportAdaptiveContextTree:
    """Root-balanced context tree with support-adaptive recursive shrinkage."""

    def __init__(
        self,
        vocab: Sequence[str],
        max_context: int = 3,
        alpha: float = 0.1,
        kappa: float = 2.0,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if int(max_context) < 1:
            raise ValueError("max_context must be at least one")
        if not np.isfinite(alpha) or float(alpha) < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(kappa) or float(kappa) < 0:
            raise ValueError("kappa must be finite and nonnegative")
        self.max_context = int(max_context)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.unigram = UnigramModel(self.vocab, alpha=self.alpha)
        self.tables_: list[dict[tuple[str, ...], np.ndarray]] = [
            {} for _ in range(self.max_context + 1)
        ]
        self.context_root_support_: list[dict[tuple[str, ...], int]] = [
            {} for _ in range(self.max_context + 1)
        ]
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "SupportAdaptiveContextTree":
        n_rows = _validate_parallel_lengths(
            prefixes=prefixes, targets=targets, groups=groups
        )
        if n_rows == 0:
            raise ValueError("training data must not be empty")
        clean_prefixes = [tuple(str(item) for item in prefix) for prefix in prefixes]
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        unknown = sorted(set(clean_targets) - set(self.label_to_index))
        if unknown:
            raise ValueError(f"training labels outside vocabulary: {unknown}")

        self.unigram.fit(clean_targets, clean_groups)
        self.tables_ = [{} for _ in range(self.max_context + 1)]
        self.context_root_support_ = [
            {} for _ in range(self.max_context + 1)
        ]

        for context_size in range(1, self.max_context + 1):
            grouped_counts: dict[
                str, dict[tuple[str, ...], np.ndarray]
            ] = defaultdict(dict)
            for prefix, target, group in zip(
                clean_prefixes, clean_targets, clean_groups, strict=True
            ):
                if len(prefix) < context_size:
                    continue
                context = prefix[-context_size:]
                vector = grouped_counts[group].setdefault(
                    context, np.zeros(len(self.vocab), dtype=float)
                )
                vector[self.label_to_index[target]] += 1.0

            contexts = sorted(
                {
                    context
                    for root_table in grouped_counts.values()
                    for context in root_table
                }
            )
            for context in contexts:
                root_conditionals = []
                for root_table in grouped_counts.values():
                    vector = root_table.get(context)
                    if vector is not None:
                        root_conditionals.append(vector / vector.sum())
                self.tables_[context_size][context] = _smoothed_probability(
                    np.mean(root_conditionals, axis=0), self.alpha
                )
                self.context_root_support_[context_size][context] = len(
                    root_conditionals
                )

        self.fitted_ = True
        return self

    def predict_proba_with_meta(
        self, prefixes: Iterable[Sequence[str]]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_ or self.unigram.probabilities_ is None:
            raise RuntimeError("model is not fitted")
        rows: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for raw_prefix in prefixes:
            prefix = tuple(str(item) for item in raw_prefix)
            probability = self.unigram.probabilities_.copy()
            used_context = 0
            root_support = 0
            shrinkage_weight = 0.0
            for context_size in range(1, min(len(prefix), self.max_context) + 1):
                context = prefix[-context_size:]
                conditional = self.tables_[context_size].get(context)
                if conditional is None:
                    continue
                support = self.context_root_support_[context_size][context]
                weight = (
                    1.0
                    if self.kappa == 0
                    else float(support) / (float(support) + self.kappa)
                )
                probability = weight * conditional + (1.0 - weight) * probability
                probability /= probability.sum()
                used_context = context_size
                root_support = support
                shrinkage_weight = weight
            rows.append(probability)
            metadata.append(
                {
                    "used_order": used_context,
                    "used_context": used_context,
                    "context_seen": used_context > 0,
                    "context_root_support": root_support,
                    "shrinkage_weight": shrinkage_weight,
                }
            )
        matrix = np.stack(rows) if rows else np.empty((0, len(self.vocab)))
        return matrix, pd.DataFrame(metadata)

