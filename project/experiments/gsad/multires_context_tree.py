from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .context_tree import SupportAdaptiveContextTree
from .probability_models import (
    UnigramModel,
    _smoothed_probability,
    _validate_parallel_lengths,
)


class MultiResolutionContextTree:
    """Hierarchical backoff from parent-technique to raw-technique contexts.

    The parent view supplies the stable distribution.  A raw sub-technique
    suffix updates that distribution only when the exact suffix was observed
    in training.  Every update is a Dirichlet-style interpolation whose
    evidence is the training-row count and whose audit metadata records the
    number of independent roots supporting the context.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        max_parent_context: int = 2,
        max_raw_context: int = 2,
        alpha: float = 0.1,
        backoff_strength: float = 5.0,
        raw_backoff_strength: float = 5.0,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if int(max_parent_context) < 0 or int(max_raw_context) < 0:
            raise ValueError("context orders must be nonnegative")
        for name, value in (
            ("alpha", alpha),
            ("backoff_strength", backoff_strength),
            ("raw_backoff_strength", raw_backoff_strength),
        ):
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.max_parent_context = int(max_parent_context)
        self.max_raw_context = int(max_raw_context)
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.raw_backoff_strength = float(raw_backoff_strength)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.unigram = UnigramModel(self.vocab, alpha=self.alpha)
        self.parent_tables_: list[dict[tuple[str, ...], np.ndarray]] = []
        self.raw_tables_: list[dict[tuple[str, ...], np.ndarray]] = []
        self.parent_root_support_: list[dict[tuple[str, ...], int]] = []
        self.raw_root_support_: list[dict[tuple[str, ...], int]] = []
        self.fitted_ = False

    def fit(
        self,
        parent_prefixes: Sequence[Sequence[str]],
        raw_prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "MultiResolutionContextTree":
        n_rows = _validate_parallel_lengths(
            parent_prefixes=parent_prefixes,
            raw_prefixes=raw_prefixes,
            targets=targets,
            groups=groups,
        )
        if n_rows == 0:
            raise ValueError("training data must not be empty")
        clean_parent = [tuple(str(item) for item in prefix) for prefix in parent_prefixes]
        clean_raw = [tuple(str(item) for item in prefix) for prefix in raw_prefixes]
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        unknown = sorted(set(clean_targets) - set(self.label_to_index))
        if unknown:
            raise ValueError(f"training labels outside vocabulary: {unknown}")

        self.unigram.fit(clean_targets, clean_groups)
        self.parent_tables_, self.parent_root_support_ = self._build_tables(
            clean_parent, clean_targets, clean_groups, self.max_parent_context
        )
        self.raw_tables_, self.raw_root_support_ = self._build_tables(
            clean_raw, clean_targets, clean_groups, self.max_raw_context
        )
        self.fitted_ = True
        return self

    def _build_tables(
        self,
        prefixes: Sequence[tuple[str, ...]],
        targets: Sequence[str],
        groups: Sequence[str],
        max_context: int,
    ) -> tuple[
        list[dict[tuple[str, ...], np.ndarray]],
        list[dict[tuple[str, ...], int]],
    ]:
        tables: list[dict[tuple[str, ...], np.ndarray]] = [
            {} for _ in range(max_context + 1)
        ]
        supports: list[dict[tuple[str, ...], int]] = [
            {} for _ in range(max_context + 1)
        ]
        for order in range(1, max_context + 1):
            roots_by_context: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for prefix, target, group in zip(prefixes, targets, groups, strict=True):
                if len(prefix) < order:
                    continue
                context = prefix[-order:]
                vector = tables[order].setdefault(
                    context, np.zeros(len(self.vocab), dtype=float)
                )
                vector[self.label_to_index[target]] += 1.0
                roots_by_context[context].add(group)
            supports[order] = {
                context: len(root_set) for context, root_set in roots_by_context.items()
            }
        return tables, supports

    @staticmethod
    def _update(
        prior: np.ndarray, counts: np.ndarray, backoff_strength: float
    ) -> np.ndarray:
        total = float(counts.sum())
        if total <= 0:
            return prior
        probability = (counts + float(backoff_strength) * prior) / (
            total + float(backoff_strength)
        )
        probability /= probability.sum()
        return probability

    def predict_proba_with_meta(
        self,
        parent_prefixes: Iterable[Sequence[str]],
        raw_prefixes: Iterable[Sequence[str]],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_ or self.unigram.probabilities_ is None:
            raise RuntimeError("model is not fitted")
        clean_parent = [tuple(str(item) for item in prefix) for prefix in parent_prefixes]
        clean_raw = [tuple(str(item) for item in prefix) for prefix in raw_prefixes]
        _validate_parallel_lengths(parent_prefixes=clean_parent, raw_prefixes=clean_raw)

        rows: list[np.ndarray] = []
        metadata: list[dict[str, int]] = []
        for parent_prefix, raw_prefix in zip(clean_parent, clean_raw, strict=True):
            probability = self.unigram.probabilities_.copy()
            parent_order = 0
            parent_support = 0
            for order in range(1, min(len(parent_prefix), self.max_parent_context) + 1):
                context = parent_prefix[-order:]
                counts = self.parent_tables_[order].get(context)
                if counts is None:
                    continue
                probability = self._update(probability, counts, self.backoff_strength)
                parent_order = order
                parent_support = self.parent_root_support_[order][context]

            raw_order = 0
            raw_support = 0
            for order in range(1, min(len(raw_prefix), self.max_raw_context) + 1):
                context = raw_prefix[-order:]
                counts = self.raw_tables_[order].get(context)
                if counts is None:
                    continue
                probability = self._update(
                    probability, counts, self.raw_backoff_strength
                )
                raw_order = order
                raw_support = self.raw_root_support_[order][context]

            rows.append(probability)
            metadata.append(
                {
                    "parent_used_order": parent_order,
                    "parent_root_support": parent_support,
                    "raw_used_order": raw_order,
                    "raw_root_support": raw_support,
                }
            )
        matrix = np.stack(rows) if rows else np.empty((0, len(self.vocab)))
        return matrix, pd.DataFrame(metadata)


class QuotientMultiResolutionContextTree:
    """Root-balanced raw-context residual over a parent quotient tree.

    Raw ATT&CK sub-techniques are mapped to parent labels for the prediction
    target.  The parent context tree estimates the quotient-space posterior.
    A raw suffix may update that posterior only when it differs from the
    aligned parent suffix and was observed in independent training roots.
    Consequently a fully collapsed raw view reduces exactly to the parent
    model instead of counting the same evidence twice.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        max_parent_context: int = 2,
        max_raw_context: int = 1,
        alpha: float = 0.1,
        parent_kappa: float = 2.0,
        raw_kappa: float = 2.0,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if int(max_parent_context) < 1 or int(max_raw_context) < 0:
            raise ValueError("parent context must be positive and raw context nonnegative")
        for name, value in (
            ("alpha", alpha),
            ("parent_kappa", parent_kappa),
            ("raw_kappa", raw_kappa),
        ):
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.max_parent_context = int(max_parent_context)
        self.max_raw_context = int(max_raw_context)
        self.alpha = float(alpha)
        self.parent_kappa = float(parent_kappa)
        self.raw_kappa = float(raw_kappa)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.parent_tree = SupportAdaptiveContextTree(
            self.vocab,
            max_context=self.max_parent_context,
            alpha=self.alpha,
            kappa=self.parent_kappa,
        )
        self.raw_tables_: list[dict[tuple[str, ...], np.ndarray]] = [
            {} for _ in range(self.max_raw_context + 1)
        ]
        self.raw_root_support_: list[dict[tuple[str, ...], int]] = [
            {} for _ in range(self.max_raw_context + 1)
        ]
        self.fitted_ = False

    def fit(
        self,
        parent_prefixes: Sequence[Sequence[str]],
        raw_prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "QuotientMultiResolutionContextTree":
        n_rows = _validate_parallel_lengths(
            parent_prefixes=parent_prefixes,
            raw_prefixes=raw_prefixes,
            targets=targets,
            groups=groups,
        )
        if n_rows == 0:
            raise ValueError("training data must not be empty")
        clean_parent = [tuple(str(item) for item in prefix) for prefix in parent_prefixes]
        clean_raw = [tuple(str(item) for item in prefix) for prefix in raw_prefixes]
        if any(len(parent) != len(raw) for parent, raw in zip(clean_parent, clean_raw, strict=True)):
            raise ValueError("raw and parent prefixes must align position by position")
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        unknown = sorted(set(clean_targets) - set(self.label_to_index))
        if unknown:
            raise ValueError(f"training labels outside vocabulary: {unknown}")

        self.parent_tree.fit(clean_parent, clean_targets, clean_groups)
        self.raw_tables_ = [{} for _ in range(self.max_raw_context + 1)]
        self.raw_root_support_ = [
            {} for _ in range(self.max_raw_context + 1)
        ]
        for order in range(1, self.max_raw_context + 1):
            grouped: dict[str, dict[tuple[str, ...], np.ndarray]] = defaultdict(dict)
            for parent, raw, target, group in zip(
                clean_parent, clean_raw, clean_targets, clean_groups, strict=True
            ):
                if len(raw) < order:
                    continue
                raw_context = raw[-order:]
                if raw_context == parent[-order:]:
                    continue
                vector = grouped[group].setdefault(
                    raw_context, np.zeros(len(self.vocab), dtype=float)
                )
                vector[self.label_to_index[target]] += 1.0
            contexts = sorted(
                {
                    context
                    for root_table in grouped.values()
                    for context in root_table
                }
            )
            for context in contexts:
                root_conditionals = []
                for root_table in grouped.values():
                    counts = root_table.get(context)
                    if counts is not None:
                        root_conditionals.append(counts / counts.sum())
                self.raw_tables_[order][context] = _smoothed_probability(
                    np.mean(root_conditionals, axis=0), self.alpha
                )
                self.raw_root_support_[order][context] = len(root_conditionals)
        self.fitted_ = True
        return self

    def predict_proba_with_meta(
        self,
        parent_prefixes: Iterable[Sequence[str]],
        raw_prefixes: Iterable[Sequence[str]],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        clean_parent = [tuple(str(item) for item in prefix) for prefix in parent_prefixes]
        clean_raw = [tuple(str(item) for item in prefix) for prefix in raw_prefixes]
        _validate_parallel_lengths(parent_prefixes=clean_parent, raw_prefixes=clean_raw)
        if any(len(parent) != len(raw) for parent, raw in zip(clean_parent, clean_raw, strict=True)):
            raise ValueError("raw and parent prefixes must align position by position")
        parent_probabilities, parent_meta = self.parent_tree.predict_proba_with_meta(
            clean_parent
        )
        rows: list[np.ndarray] = []
        metadata: list[dict[str, int | float]] = []
        for row_index, (parent, raw) in enumerate(
            zip(clean_parent, clean_raw, strict=True)
        ):
            probability = parent_probabilities[row_index].copy()
            raw_order = 0
            raw_support = 0
            raw_weight = 0.0
            for order in range(1, min(len(raw), self.max_raw_context) + 1):
                raw_context = raw[-order:]
                if raw_context == parent[-order:]:
                    continue
                conditional = self.raw_tables_[order].get(raw_context)
                if conditional is None:
                    continue
                support = self.raw_root_support_[order][raw_context]
                weight = (
                    1.0
                    if self.raw_kappa == 0
                    else float(support) / (float(support) + self.raw_kappa)
                )
                probability = weight * conditional + (1.0 - weight) * probability
                probability /= probability.sum()
                raw_order = order
                raw_support = support
                raw_weight = weight
            rows.append(probability)
            metadata.append(
                {
                    "parent_used_order": int(parent_meta.iloc[row_index]["used_order"]),
                    "parent_root_support": int(
                        parent_meta.iloc[row_index]["context_root_support"]
                    ),
                    "raw_used_order": raw_order,
                    "raw_root_support": raw_support,
                    "raw_shrinkage_weight": raw_weight,
                }
            )
        matrix = np.stack(rows) if rows else np.empty((0, len(self.vocab)))
        return matrix, pd.DataFrame(metadata)
