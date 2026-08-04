from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _validate_parallel_lengths(**values: Sequence[object]) -> int:
    lengths = {name: len(value) for name, value in values.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"parallel inputs have different lengths: {lengths}")
    return next(iter(lengths.values()), 0)


def _smoothed_probability(vector: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    values = values + float(alpha) / len(values)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability mass must be positive and finite")
    return values / total


def _root_balanced_vector(
    labels: Sequence[str],
    groups: Sequence[str],
    label_to_index: Mapping[str, int],
    alpha: float,
) -> np.ndarray:
    _validate_parallel_lengths(labels=labels, groups=groups)
    if not labels:
        raise ValueError("training data must not be empty")
    per_group: dict[str, np.ndarray] = {}
    for label, raw_group in zip(labels, groups, strict=True):
        if label not in label_to_index:
            raise ValueError(f"training label outside vocabulary: {label}")
        group = str(raw_group)
        vector = per_group.setdefault(group, np.zeros(len(label_to_index), dtype=float))
        vector[label_to_index[label]] += 1.0
    normalized = []
    for vector in per_group.values():
        normalized.append(vector / vector.sum())
    return _smoothed_probability(np.mean(normalized, axis=0), alpha)


def _domain_root_balanced_vector(
    labels: Sequence[str],
    groups: Sequence[str],
    domains: Sequence[str],
    label_to_index: Mapping[str, int],
    alpha: float,
    domain_power: float,
    domain_kappa: float,
    leave_one_domain_prior: bool,
) -> np.ndarray:
    _validate_parallel_lengths(labels=labels, groups=groups, domains=domains)
    if not labels:
        raise ValueError("training data must not be empty")
    per_root: dict[tuple[str, str], np.ndarray] = {}
    for label, raw_group, raw_domain in zip(labels, groups, domains, strict=True):
        if label not in label_to_index:
            raise ValueError(f"training label outside vocabulary: {label}")
        key = (str(raw_domain), str(raw_group))
        vector = per_root.setdefault(key, np.zeros(len(label_to_index), dtype=float))
        vector[label_to_index[label]] += 1.0
    per_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for (domain, _), vector in per_root.items():
        per_domain[domain].append(vector / vector.sum())
    all_roots = np.stack(
        [vector for vectors in per_domain.values() for vector in vectors]
    )
    global_mean = np.mean(all_roots, axis=0)
    empirical_means = [np.mean(vectors, axis=0) for vectors in per_domain.values()]
    domain_means = []
    for index, vectors in enumerate(per_domain.values()):
        empirical = np.mean(vectors, axis=0)
        support = len(vectors)
        prior = (
            np.mean(
                [
                    mean
                    for other_index, mean in enumerate(empirical_means)
                    if other_index != index
                ],
                axis=0,
            )
            if leave_one_domain_prior and len(empirical_means) > 1
            else global_mean
        )
        pooled = (
            empirical
            if float(domain_kappa) == 0
            else (
                support * empirical + float(domain_kappa) * prior
            )
            / (support + float(domain_kappa))
        )
        domain_means.append(pooled)
    domain_weights = np.asarray(
        [len(vectors) ** float(domain_power) for vectors in per_domain.values()],
        dtype=float,
    )
    return _smoothed_probability(
        np.average(np.stack(domain_means), axis=0, weights=domain_weights), alpha
    )


@dataclass
class UnigramModel:
    vocab: tuple[str, ...]
    alpha: float = 0.5

    def __init__(self, vocab: Sequence[str], alpha: float = 0.5):
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = float(alpha)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.probabilities_: np.ndarray | None = None

    def fit(self, targets: Sequence[str], groups: Sequence[str]) -> "UnigramModel":
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        self.probabilities_ = _root_balanced_vector(
            clean_targets, clean_groups, self.label_to_index, self.alpha
        )
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        if self.probabilities_ is None:
            raise RuntimeError("model is not fitted")
        if int(n_samples) < 0:
            raise ValueError("n_samples must be nonnegative")
        return np.tile(self.probabilities_, (int(n_samples), 1))


class InterpolatedNGram:
    """Root-balanced, smoothed n-gram with deterministic backoff."""

    def __init__(
        self,
        vocab: Sequence[str],
        order: int = 3,
        alpha: float = 0.5,
        interpolation: Sequence[float] | None = None,
        domain_power: float = 0.0,
        domain_kappa: float = 0.0,
        leave_one_domain_prior: bool = False,
    ):
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if int(order) < 1:
            raise ValueError("order must be at least one")
        self.order = int(order)
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = float(alpha)
        if interpolation is None:
            interpolation = np.ones(self.order, dtype=float)
        weights = np.asarray(interpolation, dtype=float)
        if weights.shape != (self.order,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("interpolation must contain one nonnegative weight per order")
        self.interpolation = weights / weights.sum()
        if not np.isfinite(domain_power) or not 0 <= float(domain_power) <= 1:
            raise ValueError("domain_power must be finite and in [0, 1]")
        self.domain_power = float(domain_power)
        if not np.isfinite(domain_kappa) or float(domain_kappa) < 0:
            raise ValueError("domain_kappa must be finite and nonnegative")
        self.domain_kappa = float(domain_kappa)
        if not isinstance(leave_one_domain_prior, bool):
            raise ValueError("leave_one_domain_prior must be boolean")
        self.leave_one_domain_prior = leave_one_domain_prior
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.unigram = UnigramModel(self.vocab, alpha=self.alpha)
        self.tables_: list[dict[tuple[str, ...], np.ndarray]] = [
            {} for _ in range(self.order)
        ]
        self.context_root_support_: list[dict[tuple[str, ...], int]] = [
            {} for _ in range(self.order)
        ]
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
        domains: Sequence[str] | None = None,
    ) -> "InterpolatedNGram":
        _validate_parallel_lengths(prefixes=prefixes, targets=targets, groups=groups)
        if domains is not None:
            _validate_parallel_lengths(prefixes=prefixes, domains=domains)
        clean_prefixes = [tuple(str(item) for item in prefix) for prefix in prefixes]
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        clean_domains = (
            [str(domain) for domain in domains]
            if domains is not None
            else ["__single_domain__"] * len(clean_prefixes)
        )
        if domains is None:
            self.unigram.fit(clean_targets, clean_groups)
        else:
            self.unigram.probabilities_ = _domain_root_balanced_vector(
                clean_targets,
                clean_groups,
                clean_domains,
                self.label_to_index,
                self.alpha,
                self.domain_power,
                self.domain_kappa,
                self.leave_one_domain_prior,
            )

        for context_size in range(1, self.order):
            grouped_counts: dict[
                tuple[str, str], dict[tuple[str, ...], np.ndarray]
            ] = defaultdict(dict)
            for prefix, target, group, domain in zip(
                clean_prefixes,
                clean_targets,
                clean_groups,
                clean_domains,
                strict=True,
            ):
                if target not in self.label_to_index:
                    raise ValueError(f"training label outside vocabulary: {target}")
                if len(prefix) < context_size:
                    continue
                context = prefix[-context_size:]
                group_contexts = grouped_counts[(domain, group)]
                vector = group_contexts.setdefault(
                    context, np.zeros(len(self.vocab), dtype=float)
                )
                vector[self.label_to_index[target]] += 1.0

            contexts = sorted(
                {context for group_table in grouped_counts.values() for context in group_table}
            )
            table: dict[tuple[str, ...], np.ndarray] = {}
            support: dict[tuple[str, ...], int] = {}
            for context in contexts:
                root_vectors: list[np.ndarray] = []
                domain_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
                for (domain, _), group_table in grouped_counts.items():
                    if context not in group_table:
                        continue
                    vector = group_table[context]
                    normalized = vector / vector.sum()
                    root_vectors.append(normalized)
                    domain_vectors[domain].append(normalized)
                if domains is not None:
                    global_mean = np.mean(root_vectors, axis=0)
                    pooled_domain_means = []
                    empirical_domain_means = [
                        np.mean(vectors, axis=0)
                        for vectors in domain_vectors.values()
                    ]
                    for domain_index, vectors in enumerate(domain_vectors.values()):
                        empirical = empirical_domain_means[domain_index]
                        domain_support = len(vectors)
                        prior = (
                            np.mean(
                                [
                                    mean
                                    for other_index, mean in enumerate(
                                        empirical_domain_means
                                    )
                                    if other_index != domain_index
                                ],
                                axis=0,
                            )
                            if self.leave_one_domain_prior
                            and len(empirical_domain_means) > 1
                            else global_mean
                        )
                        pooled_domain_means.append(
                            empirical
                            if self.domain_kappa == 0
                            else (
                                domain_support * empirical
                                + self.domain_kappa * prior
                            )
                            / (domain_support + self.domain_kappa)
                        )
                    estimate = np.average(
                        np.stack(pooled_domain_means),
                        axis=0,
                        weights=np.asarray(
                            [
                                len(vectors) ** self.domain_power
                                for vectors in domain_vectors.values()
                            ],
                            dtype=float,
                        ),
                    )
                else:
                    estimate = np.mean(root_vectors, axis=0)
                table[context] = _smoothed_probability(
                    estimate, self.alpha
                )
                support[context] = len(root_vectors)
            self.tables_[context_size] = table
            self.context_root_support_[context_size] = support
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
            components: list[np.ndarray] = [self.unigram.probabilities_]
            weights: list[float] = [float(self.interpolation[0])]
            used_order = 0
            root_support = 0
            for context_size in range(1, self.order):
                if len(prefix) < context_size:
                    continue
                context = prefix[-context_size:]
                distribution = self.tables_[context_size].get(context)
                if distribution is None:
                    continue
                components.append(distribution)
                weights.append(float(self.interpolation[context_size]))
                if self.interpolation[context_size] > 0:
                    used_order = context_size
                    root_support = self.context_root_support_[context_size][context]
            weight_array = np.asarray(weights, dtype=float)
            if weight_array.sum() <= 0:
                components = [self.unigram.probabilities_]
                weight_array = np.ones(1, dtype=float)
                used_order = 0
                root_support = 0
            weight_array /= weight_array.sum()
            probability = np.average(np.stack(components), axis=0, weights=weight_array)
            probability /= probability.sum()
            rows.append(probability)
            metadata.append(
                {
                    "used_order": used_order,
                    "context_seen": used_order > 0,
                    "context_root_support": root_support,
                }
            )
        return np.stack(rows) if rows else np.empty((0, len(self.vocab))), pd.DataFrame(
            metadata
        )


class TacticAwareModel:
    """Blend exact technique transitions with root-balanced tactic transfer."""

    def __init__(
        self,
        vocab: Sequence[str],
        technique_to_tactics: Mapping[str, Iterable[str]],
        alpha: float = 0.5,
        technique_weight: float = 0.7,
        tactic_weight: float = 0.3,
    ):
        self.vocab = tuple(str(label) for label in vocab)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.technique_to_tactics = {
            str(label): frozenset(str(tactic) for tactic in tactics)
            for label, tactics in technique_to_tactics.items()
        }
        weights = np.asarray([technique_weight, tactic_weight], dtype=float)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("technique and tactic weights must be nonnegative with positive sum")
        self.technique_weight, self.tactic_weight = (weights / weights.sum()).tolist()
        self.alpha = float(alpha)
        self.technique_model = InterpolatedNGram(
            self.vocab,
            order=2,
            alpha=alpha,
            interpolation=(0.25, 0.75),
        )
        self.tactic_tables_: dict[str, np.ndarray] = {}
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "TacticAwareModel":
        _validate_parallel_lengths(prefixes=prefixes, targets=targets, groups=groups)
        clean_prefixes = [tuple(str(item) for item in prefix) for prefix in prefixes]
        clean_targets = [str(target) for target in targets]
        clean_groups = [str(group) for group in groups]
        self.technique_model.fit(clean_prefixes, clean_targets, clean_groups)

        grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for prefix, target, group in zip(
            clean_prefixes, clean_targets, clean_groups, strict=True
        ):
            if not prefix:
                continue
            if target not in self.label_to_index:
                raise ValueError(f"training label outside vocabulary: {target}")
            for tactic in self.technique_to_tactics.get(prefix[-1], frozenset()):
                vector = grouped[group].setdefault(
                    tactic, np.zeros(len(self.vocab), dtype=float)
                )
                vector[self.label_to_index[target]] += 1.0

        tactics = sorted({tactic for root_table in grouped.values() for tactic in root_table})
        self.tactic_tables_ = {}
        for tactic in tactics:
            root_vectors = []
            for root_table in grouped.values():
                if tactic not in root_table:
                    continue
                vector = root_table[tactic]
                root_vectors.append(vector / vector.sum())
            self.tactic_tables_[tactic] = _smoothed_probability(
                np.mean(root_vectors, axis=0), self.alpha
            )
        self.fitted_ = True
        return self

    def predict_proba_with_meta(
        self, prefixes: Iterable[Sequence[str]]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        prefix_list = [tuple(str(item) for item in prefix) for prefix in prefixes]
        technique_probs, technique_meta = self.technique_model.predict_proba_with_meta(
            prefix_list
        )
        rows: list[np.ndarray] = []
        tactic_seen: list[bool] = []
        for row_index, prefix in enumerate(prefix_list):
            tactic_distributions = []
            if prefix:
                for tactic in sorted(
                    self.technique_to_tactics.get(prefix[-1], frozenset())
                ):
                    if tactic in self.tactic_tables_:
                        tactic_distributions.append(self.tactic_tables_[tactic])
            if tactic_distributions:
                tactic_probability = np.mean(tactic_distributions, axis=0)
                probability = (
                    self.technique_weight * technique_probs[row_index]
                    + self.tactic_weight * tactic_probability
                )
                tactic_seen.append(True)
            else:
                probability = technique_probs[row_index]
                tactic_seen.append(False)
            rows.append(probability / probability.sum())
        metadata = technique_meta.copy()
        metadata["tactic_context_seen"] = tactic_seen
        return np.stack(rows) if rows else np.empty((0, len(self.vocab))), metadata
