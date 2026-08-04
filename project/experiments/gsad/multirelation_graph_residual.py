from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "may",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "uses",
        "using",
        "with",
    }
)


def _attack_id(obj: dict[str, object]) -> str | None:
    for reference in obj.get("external_references", []):
        if not isinstance(reference, dict):
            continue
        candidate = str(reference.get("external_id", ""))
        if reference.get("source_name") == "mitre-attack" and candidate.startswith("T"):
            return candidate
    return None


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN.findall(str(text).lower())
        if token not in _STOP and len(token) > 1
    )


def _tfidf_cosine(documents: Sequence[str]) -> np.ndarray:
    tokenized = [_tokens(document) for document in documents]
    terms = sorted({token for document in tokenized for token in document})
    if not terms:
        return np.zeros((len(documents), len(documents)), dtype=float)
    term_index = {term: index for index, term in enumerate(terms)}
    document_frequency = Counter(
        token for document in tokenized for token in set(document)
    )
    matrix = np.zeros((len(documents), len(terms)), dtype=float)
    n_documents = len(documents)
    for row, document in enumerate(tokenized):
        counts = Counter(document)
        if not counts:
            continue
        scale = float(sum(counts.values()))
        for token, count in counts.items():
            inverse_frequency = np.log(
                (1.0 + n_documents) / (1.0 + document_frequency[token])
            ) + 1.0
            matrix[row, term_index[token]] = (float(count) / scale) * inverse_frequency
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.divide(
        matrix,
        norms[:, None],
        out=np.zeros_like(matrix),
        where=norms[:, None] > 0,
    )
    cosine = normalized @ normalized.T
    np.fill_diagonal(cosine, 0.0)
    return cosine


def _symmetric_top_k(matrix: np.ndarray, neighbors: int) -> np.ndarray:
    if int(neighbors) < 1:
        raise ValueError("semantic_neighbors must be positive")
    selected = np.zeros_like(matrix)
    for row in range(len(matrix)):
        positive = np.flatnonzero(matrix[row] > 0)
        ordered = sorted(positive, key=lambda column: (-matrix[row, column], column))
        for column in ordered[: int(neighbors)]:
            selected[row, column] = matrix[row, column]
    selected = np.maximum(selected, selected.T)
    np.fill_diagonal(selected, 0.0)
    return selected


def build_attack_relation_matrices(
    stix_path: Path,
    vocab: Sequence[str],
    semantic_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build static ATT&CK description and tactic relations without outcome data."""

    vocabulary = tuple(str(label) for label in vocab)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary must be nonempty and unique")
    with Path(stix_path).open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    objects: dict[str, dict[str, object]] = {}
    for obj in bundle.get("objects", []):
        if not isinstance(obj, dict) or obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue
        attack_id = _attack_id(obj)
        if attack_id in vocabulary:
            objects[str(attack_id)] = obj

    documents: list[str] = []
    tactic_sets: list[frozenset[str]] = []
    for label in vocabulary:
        obj = objects.get(label, {})
        documents.append(f"{obj.get('name', '')} {obj.get('description', '')}")
        tactics = {
            str(phase.get("phase_name", ""))
            for phase in obj.get("kill_chain_phases", [])
            if isinstance(phase, dict)
            and phase.get("kill_chain_name") == "mitre-attack"
            and phase.get("phase_name")
        }
        tactic_sets.append(frozenset(tactics))

    semantic = _symmetric_top_k(
        _tfidf_cosine(documents), int(semantic_neighbors)
    )
    tactic = np.zeros((len(vocabulary), len(vocabulary)), dtype=float)
    for left in range(len(vocabulary)):
        for right in range(left + 1, len(vocabulary)):
            union = tactic_sets[left] | tactic_sets[right]
            if not union:
                continue
            score = len(tactic_sets[left] & tactic_sets[right]) / float(len(union))
            tactic[left, right] = score
            tactic[right, left] = score
    audit = {
        "vocab_size": len(vocabulary),
        "mapped_descriptions": sum(bool(_tokens(document)) for document in documents),
        "mapped_tactics": sum(bool(tactics) for tactics in tactic_sets),
        "semantic_edges": int(np.count_nonzero(np.triu(semantic, k=1))),
        "tactic_edges": int(np.count_nonzero(np.triu(tactic, k=1))),
    }
    return semantic, tactic, audit


def combine_relations(
    semantic: np.ndarray,
    tactic: np.ndarray,
    semantic_weight: float,
    tactic_weight: float,
) -> np.ndarray:
    if semantic.shape != tactic.shape or semantic.ndim != 2:
        raise ValueError("relation matrices must be aligned square matrices")
    if semantic.shape[0] != semantic.shape[1]:
        raise ValueError("relation matrices must be square")
    if min(float(semantic_weight), float(tactic_weight)) < 0:
        raise ValueError("relation weights must be nonnegative")
    combined = float(semantic_weight) * semantic + float(tactic_weight) * tactic
    np.fill_diagonal(combined, 0.0)
    return combined


class MultiRelationGraphResidual:
    """Support-gated transfer of neighbor transition distributions.

    Training outcomes only estimate root-balanced transition rows.  The ATT&CK
    relation graph is static, has no self loops, and is used most strongly when
    the current source technique has little independent-root support.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        relation_matrix: np.ndarray,
        support_kappa: float = 5.0,
        residual_weight: float = 0.25,
        local_kappa: float = 1.0,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        relation = np.asarray(relation_matrix, dtype=float)
        if relation.shape != (len(self.vocab), len(self.vocab)):
            raise ValueError("relation matrix does not match vocabulary")
        if not np.isfinite(relation).all() or (relation < 0).any():
            raise ValueError("relation matrix must be finite and nonnegative")
        if np.any(np.diag(relation) != 0):
            raise ValueError("relation graph must not contain self loops")
        for name, value in (
            ("support_kappa", support_kappa),
            ("residual_weight", residual_weight),
            ("local_kappa", local_kappa),
        ):
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(residual_weight) > 1:
            raise ValueError("residual_weight must not exceed one")
        self.relation_matrix = relation.copy()
        self.support_kappa = float(support_kappa)
        self.residual_weight = float(residual_weight)
        self.local_kappa = float(local_kappa)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.local_transition_ = np.empty((0, 0))
        self.graph_transition_ = np.empty((0, 0))
        self.source_support_ = np.empty(0)
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "MultiRelationGraphResidual":
        if not (len(prefixes) == len(targets) == len(groups)) or not prefixes:
            raise ValueError("training arrays must be nonempty and aligned")
        root_targets: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
        root_all_targets: dict[str, Counter[int]] = defaultdict(Counter)
        for prefix, target, group in zip(prefixes, targets, groups, strict=True):
            target_index = self.label_to_index.get(str(target))
            if target_index is None:
                raise ValueError(f"target outside vocabulary: {target}")
            clean_group = str(group)
            root_all_targets[clean_group][target_index] += 1
            if not prefix:
                continue
            source_index = self.label_to_index.get(str(prefix[-1]))
            if source_index is not None:
                root_targets[(source_index, clean_group)][target_index] += 1

        root_unigrams = []
        for counts in root_all_targets.values():
            row = np.zeros(len(self.vocab), dtype=float)
            for target_index, count in counts.items():
                row[target_index] = count
            root_unigrams.append(row / row.sum())
        unigram = np.mean(root_unigrams, axis=0)

        local = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        support = np.zeros(len(self.vocab), dtype=float)
        grouped_rows: dict[int, list[np.ndarray]] = defaultdict(list)
        for (source_index, _), counts in root_targets.items():
            row = np.zeros(len(self.vocab), dtype=float)
            for target_index, count in counts.items():
                row[target_index] = count
            grouped_rows[source_index].append(row / row.sum())
        for source_index in range(len(self.vocab)):
            rows = grouped_rows.get(source_index, [])
            support[source_index] = len(rows)
            empirical = np.mean(rows, axis=0) if rows else unigram
            if self.local_kappa == 0:
                reliability = 1.0 if rows else 0.0
            else:
                reliability = support[source_index] / (
                    support[source_index] + self.local_kappa
                )
            local[source_index] = reliability * empirical + (1.0 - reliability) * unigram

        neighbor_reliability = (
            support / (support + self.local_kappa)
            if self.local_kappa > 0
            else (support > 0).astype(float)
        )
        weighted_graph = self.relation_matrix * neighbor_reliability[None, :]
        row_sums = weighted_graph.sum(axis=1)
        graph = np.tile(unigram, (len(self.vocab), 1))
        available = row_sums > 0
        if available.any():
            graph[available] = (
                weighted_graph[available] @ local
            ) / row_sums[available, None]
        self.local_transition_ = local
        self.graph_transition_ = graph
        self.source_support_ = support
        self.fitted_ = True
        return self

    def predict_proba_with_meta(
        self,
        base_probabilities: np.ndarray,
        prefixes: Sequence[Sequence[str]],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        base = np.asarray(base_probabilities, dtype=float)
        if base.shape != (len(prefixes), len(self.vocab)):
            raise ValueError("base probabilities do not align with prefixes/vocabulary")
        if not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base probabilities must be finite and nonnegative")
        if not np.allclose(base.sum(axis=1), np.ones(len(base))):
            raise ValueError("base probabilities must be normalized")
        output = base.copy()
        metadata: list[dict[str, float | int | str]] = []
        for row_index, prefix in enumerate(prefixes):
            source = str(prefix[-1]) if prefix else ""
            source_index = self.label_to_index.get(source)
            if source_index is None:
                support = 0.0
                weight = 0.0
            else:
                support = self.source_support_[source_index]
                uncertainty = (
                    0.0
                    if self.support_kappa == 0
                    else self.support_kappa / (support + self.support_kappa)
                )
                weight = self.residual_weight * uncertainty
                output[row_index] = (
                    (1.0 - weight) * base[row_index]
                    + weight * self.graph_transition_[source_index]
                )
                output[row_index] /= output[row_index].sum()
            metadata.append(
                {
                    "source_technique": source,
                    "source_root_support": int(support),
                    "graph_weight": float(weight),
                }
            )
        return output, pd.DataFrame(metadata)


def _row_stochastic(matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    clean = np.asarray(matrix, dtype=float).copy()
    row_sums = clean.sum(axis=1)
    available = row_sums > 0
    clean[available] /= row_sums[available, None]
    clean[~available] = fallback
    return clean


class SelfCensoredMultiRelationGraphResidual:
    """Root-balanced, transition-first graph residual with self censorship."""

    def __init__(
        self,
        vocab: Sequence[str],
        semantic_matrix: np.ndarray,
        tactic_matrix: np.ndarray,
        history_length: int = 3,
        history_decay: float = 0.5,
        prior_smoothing: float = 0.05,
        residual_clip: float = 2.0,
        transition_kappa: float = 0.0,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        expected = (len(self.vocab), len(self.vocab))
        semantic = np.asarray(semantic_matrix, dtype=float)
        tactic = np.asarray(tactic_matrix, dtype=float)
        for name, matrix in (("semantic", semantic), ("tactic", tactic)):
            if matrix.shape != expected:
                raise ValueError(f"{name} matrix does not match vocabulary")
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(f"{name} matrix must be finite and nonnegative")
            if np.any(np.diag(matrix) != 0):
                raise ValueError(f"{name} matrix must not contain self loops")
        if int(history_length) < 1:
            raise ValueError("history_length must be positive")
        if not 0 < float(history_decay) <= 1:
            raise ValueError("history_decay must be in (0, 1]")
        if not 0 <= float(prior_smoothing) <= 1:
            raise ValueError("prior_smoothing must be in [0, 1]")
        if not np.isfinite(residual_clip) or float(residual_clip) <= 0:
            raise ValueError("residual_clip must be finite and positive")
        if not np.isfinite(transition_kappa) or float(transition_kappa) < 0:
            raise ValueError("transition_kappa must be finite and nonnegative")
        self.semantic_matrix = semantic.copy()
        self.tactic_matrix = tactic.copy()
        self.history_length = int(history_length)
        self.history_decay = float(history_decay)
        self.prior_smoothing = float(prior_smoothing)
        self.residual_clip = float(residual_clip)
        self.transition_kappa = float(transition_kappa)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.prior_ = np.empty(0)
        self.transition_matrix_ = np.empty((0, 0))
        self.semantic_kernel_ = np.empty((0, 0))
        self.tactic_kernel_ = np.empty((0, 0))
        self.transition_root_support_ = np.empty(0)
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "SelfCensoredMultiRelationGraphResidual":
        if len(prefixes) == 0 or not (len(prefixes) == len(targets) == len(groups)):
            raise ValueError("training arrays must be nonempty and aligned")
        root_target_counts: dict[str, Counter[int]] = defaultdict(Counter)
        root_transition_counts: dict[tuple[int, str], Counter[int]] = defaultdict(
            Counter
        )
        for prefix, target, group in zip(prefixes, targets, groups, strict=True):
            target_index = self.label_to_index.get(str(target))
            if target_index is None:
                raise ValueError(f"target outside vocabulary: {target}")
            clean_group = str(group)
            root_target_counts[clean_group][target_index] += 1
            if not prefix:
                continue
            source_index = self.label_to_index.get(str(prefix[-1]))
            if source_index is None or source_index == target_index:
                continue
            root_transition_counts[(source_index, clean_group)][target_index] += 1

        root_priors: list[np.ndarray] = []
        for counts in root_target_counts.values():
            row = np.zeros(len(self.vocab), dtype=float)
            for target_index, count in counts.items():
                row[target_index] = count
            root_priors.append(row / row.sum())
        prior = np.mean(root_priors, axis=0)
        prior /= prior.sum()

        per_source: dict[int, list[np.ndarray]] = defaultdict(list)
        for (source_index, _), counts in root_transition_counts.items():
            row = np.zeros(len(self.vocab), dtype=float)
            for target_index, count in counts.items():
                row[target_index] = count
            per_source[source_index].append(row / row.sum())

        transition = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        support = np.zeros(len(self.vocab), dtype=float)
        for source_index in range(len(self.vocab)):
            rows = per_source.get(source_index, [])
            support[source_index] = len(rows)
            fallback = prior.copy()
            fallback[source_index] = 0.0
            if fallback.sum() == 0:
                fallback[:] = 1.0
                fallback[source_index] = 0.0
            fallback /= fallback.sum()
            if rows:
                empirical = np.mean(rows, axis=0)
                reliability = (
                    1.0
                    if self.transition_kappa == 0
                    else support[source_index]
                    / (support[source_index] + self.transition_kappa)
                )
                transition[source_index] = (
                    reliability * empirical + (1.0 - reliability) * fallback
                )
            else:
                transition[source_index] = fallback
            transition[source_index, source_index] = 0.0
            transition[source_index] /= transition[source_index].sum()

        self.prior_ = prior
        self.transition_matrix_ = transition
        self.semantic_kernel_ = _row_stochastic(self.semantic_matrix, prior)
        self.tactic_kernel_ = _row_stochastic(self.tactic_matrix, prior)
        self.transition_root_support_ = support
        self.fitted_ = True
        return self

    def _history_seed(self, prefix: Sequence[str]) -> np.ndarray:
        seed = np.zeros(len(self.vocab), dtype=float)
        for offset, label in enumerate(reversed(tuple(prefix)[-self.history_length :])):
            index = self.label_to_index.get(str(label))
            if index is not None:
                seed[index] += self.history_decay**offset
        if seed.sum() == 0:
            return self.prior_.copy()
        return seed / seed.sum()

    def component_probabilities(
        self, prefixes: Sequence[Sequence[str]]
    ) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        seed = np.stack([self._history_seed(prefix) for prefix in prefixes])
        transition = seed @ self.transition_matrix_
        tactic = transition @ self.tactic_kernel_
        semantic = transition @ self.semantic_kernel_
        for matrix in (transition, tactic, semantic):
            matrix /= matrix.sum(axis=1, keepdims=True)
        return {"transition": transition, "tactic": tactic, "semantic": semantic}

    def predict_proba_with_meta(
        self,
        base_probabilities: np.ndarray,
        prefixes: Sequence[Sequence[str]],
        relation_weights: tuple[float, float, float],
        residual_strength: float,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        base = np.asarray(base_probabilities, dtype=float)
        if base.shape != (len(prefixes), len(self.vocab)):
            raise ValueError("base probabilities do not align with prefixes/vocabulary")
        if not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base probabilities must be finite and nonnegative")
        if not np.allclose(base.sum(axis=1), np.ones(len(base))):
            raise ValueError("base probabilities must be normalized")
        weights = np.asarray(relation_weights, dtype=float)
        if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("relation weights must be three nonnegative values summing to one")
        if not 0 <= float(residual_strength) <= 1:
            raise ValueError("residual_strength must be in [0, 1]")

        components = self.component_probabilities(prefixes)
        mixed = (
            weights[0] * components["transition"]
            + weights[1] * components["tactic"]
            + weights[2] * components["semantic"]
        )
        q = (1.0 - self.prior_smoothing) * mixed + self.prior_smoothing * self.prior_
        q /= q.sum(axis=1, keepdims=True)
        epsilon = 1e-12
        residual = np.clip(
            np.log(q + epsilon) - np.log(self.prior_[None, :] + epsilon),
            -self.residual_clip,
            self.residual_clip,
        )
        current_indices: list[int | None] = []
        for row, prefix in enumerate(prefixes):
            current = self.label_to_index.get(str(prefix[-1])) if prefix else None
            current_indices.append(current)
            if current is not None:
                residual[row, current] = 0.0

        logits = np.log(base + epsilon) + float(residual_strength) * residual
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
        output /= output.sum(axis=1, keepdims=True)
        metadata = pd.DataFrame(
            {
                "current_parent_residual": [
                    0.0 if index is not None else float("nan")
                    for index in current_indices
                ],
                "transition_root_support": [
                    int(self.transition_root_support_[index]) if index is not None else 0
                    for index in current_indices
                ],
                "residual_l1": np.abs(residual).sum(axis=1),
            }
        )
        return output, metadata


class TypedGraphEscapeRedistributor:
    """Redistribute only a fitted VOM's existing unseen-successor mass.

    Probabilities assigned by the base model to successors observed after the
    longest fitted suffix are held exactly fixed.  Static ATT&CK relations act
    on source transition rows, never as temporal target edges.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        semantic_matrix: np.ndarray,
        tactic_matrix: np.ndarray,
        max_context: int = 3,
        history_length: int = 3,
        history_decay: float = 0.5,
    ) -> None:
        if int(max_context) < 1:
            raise ValueError("max_context must be positive")
        self.vocab = tuple(str(label) for label in vocab)
        self.max_context = int(max_context)
        self.graph_model = SelfCensoredMultiRelationGraphResidual(
            vocab=self.vocab,
            semantic_matrix=semantic_matrix,
            tactic_matrix=tactic_matrix,
            history_length=history_length,
            history_decay=history_decay,
        )
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.context_successors_: list[dict[tuple[str, ...], np.ndarray]] = []
        self.context_root_support_: list[dict[tuple[str, ...], int]] = []
        self.semantic_source_transition_ = np.empty((0, 0))
        self.tactic_source_transition_ = np.empty((0, 0))
        self.fitted_ = False

    def fit(
        self,
        prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
    ) -> "TypedGraphEscapeRedistributor":
        if len(prefixes) == 0 or not (len(prefixes) == len(targets) == len(groups)):
            raise ValueError("training arrays must be nonempty and aligned")
        self.graph_model.fit(prefixes, targets, groups)
        successor_sets: list[dict[tuple[str, ...], set[int]]] = [
            {} for _ in range(self.max_context + 1)
        ]
        supporting_roots: list[dict[tuple[str, ...], set[str]]] = [
            {} for _ in range(self.max_context + 1)
        ]
        for prefix, target, group in zip(prefixes, targets, groups, strict=True):
            clean_prefix = tuple(str(label) for label in prefix)
            target_index = self.label_to_index.get(str(target))
            if target_index is None:
                raise ValueError(f"target outside vocabulary: {target}")
            for order in range(1, min(len(clean_prefix), self.max_context) + 1):
                context = clean_prefix[-order:]
                successor_sets[order].setdefault(context, set()).add(target_index)
                supporting_roots[order].setdefault(context, set()).add(str(group))
        self.context_successors_ = [
            {
                context: np.asarray(sorted(indices), dtype=int)
                for context, indices in order_table.items()
            }
            for order_table in successor_sets
        ]
        self.context_root_support_ = [
            {context: len(roots) for context, roots in order_table.items()}
            for order_table in supporting_roots
        ]
        transition = self.graph_model.transition_matrix_
        self.semantic_source_transition_ = (
            self.graph_model.semantic_kernel_ @ transition
        )
        self.tactic_source_transition_ = self.graph_model.tactic_kernel_ @ transition
        self.fitted_ = True
        return self

    def _context(self, prefix: Sequence[str]) -> tuple[int, np.ndarray, int]:
        clean = tuple(str(label) for label in prefix)
        for order in range(min(len(clean), self.max_context), 0, -1):
            context = clean[-order:]
            successors = self.context_successors_[order].get(context)
            if successors is not None:
                return (
                    order,
                    successors,
                    self.context_root_support_[order][context],
                )
        return 0, np.arange(len(self.vocab), dtype=int), 0

    def is_seen_successor(self, prefix: Sequence[str], target: str) -> bool:
        _, seen, _ = self._context(prefix)
        target_index = self.label_to_index.get(str(target))
        return target_index is not None and bool(np.any(seen == target_index))

    def component_probabilities(
        self, prefixes: Sequence[Sequence[str]]
    ) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        seeds = np.stack([self.graph_model._history_seed(prefix) for prefix in prefixes])
        transition = seeds @ self.graph_model.transition_matrix_
        semantic = seeds @ self.semantic_source_transition_
        tactic = seeds @ self.tactic_source_transition_
        for matrix in (transition, semantic, tactic):
            matrix /= matrix.sum(axis=1, keepdims=True)
        return {
            "transition": transition,
            "tactic_source": tactic,
            "semantic_source": semantic,
        }

    def predict_proba_with_meta(
        self,
        base_probabilities: np.ndarray,
        prefixes: Sequence[Sequence[str]],
        relation_weights: tuple[float, float, float],
        graph_fraction: float,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        base = np.asarray(base_probabilities, dtype=float)
        if base.shape != (len(prefixes), len(self.vocab)):
            raise ValueError("base probabilities do not align with prefixes/vocabulary")
        if not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base probabilities must be finite and nonnegative")
        if not np.allclose(base.sum(axis=1), np.ones(len(base))):
            raise ValueError("base probabilities must be normalized")
        weights = np.asarray(relation_weights, dtype=float)
        if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("relation weights must be three nonnegative values summing to one")
        if not 0 <= float(graph_fraction) <= 1:
            raise ValueError("graph_fraction must be in [0, 1]")

        components = self.component_probabilities(prefixes)
        graph = (
            weights[0] * components["transition"]
            + weights[1] * components["tactic_source"]
            + weights[2] * components["semantic_source"]
        )
        output = base.copy()
        metadata: list[dict[str, float | int]] = []
        for row, prefix in enumerate(prefixes):
            order, seen_indices, support = self._context(prefix)
            seen = np.zeros(len(self.vocab), dtype=bool)
            seen[seen_indices] = True
            unseen = ~seen
            escape_mass = float(base[row, unseen].sum())
            graph_shift_l1 = 0.0
            if unseen.any() and escape_mass > 0 and graph_fraction > 0:
                base_escape = base[row, unseen] / escape_mass
                graph_total = float(graph[row, unseen].sum())
                graph_escape = (
                    graph[row, unseen] / graph_total
                    if graph_total > 0
                    else base_escape
                )
                redistributed = (
                    (1.0 - float(graph_fraction)) * base_escape
                    + float(graph_fraction) * graph_escape
                )
                replacement = escape_mass * redistributed
                graph_shift_l1 = float(np.abs(replacement - base[row, unseen]).sum())
                output[row, unseen] = replacement
                output[row, seen] = base[row, seen]
                output[row] /= output[row].sum()
            metadata.append(
                {
                    "context_order": order,
                    "context_root_support": support,
                    "seen_successor_count": int(seen.sum()),
                    "escape_mass": escape_mass,
                    "graph_shift_l1": graph_shift_l1,
                }
            )
        return output, pd.DataFrame(metadata)
