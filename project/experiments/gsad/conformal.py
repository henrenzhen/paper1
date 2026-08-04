from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .attack_dag import AttackDAG


def _validate_probability_matrix(probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != n_classes:
        raise ValueError(
            f"probability matrix must have shape (n, {n_classes}), got {matrix.shape}"
        )
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("every probability row must sum to one")
    return matrix


def deterministic_uniform(sample_id: str, label: str, seed: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}|{sample_id}|{label}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return integer / float(1 << 53)


def aps_score(probability: np.ndarray, label_index: int, u: float) -> float:
    vector = np.asarray(probability, dtype=float)
    if vector.ndim != 1 or not 0 <= int(label_index) < len(vector):
        raise ValueError("invalid probability vector or label index")
    if not 0 <= float(u) < 1:
        raise ValueError("u must lie in [0, 1)")
    order = np.lexsort((np.arange(len(vector)), -vector))
    position = int(np.flatnonzero(order == int(label_index))[0])
    mass_before = float(vector[order[:position]].sum())
    return mass_before + float(u) * float(vector[int(label_index)])


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    index = math.ceil((len(values) + 1) * (1.0 - float(alpha))) - 1
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


@dataclass(frozen=True)
class LabelClusters:
    vocab: tuple[str, ...]
    label_to_cluster: Mapping[str, str]
    members: Mapping[str, frozenset[str]]
    validation_support: Mapping[str, int]
    min_support: int

    def cluster_for(self, label: str) -> str:
        try:
            return self.label_to_cluster[str(label)]
        except KeyError as exc:
            raise ValueError(f"label outside clustered vocabulary: {label}") from exc

    def digest(self) -> str:
        payload = {
            "vocab": list(self.vocab),
            "label_to_cluster": dict(sorted(self.label_to_cluster.items())),
            "members": {
                cluster: sorted(labels) for cluster, labels in sorted(self.members.items())
            },
            "validation_support": dict(sorted(self.validation_support.items())),
            "min_support": int(self.min_support),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _label_features(
    probabilities: np.ndarray,
    targets: Sequence[str],
    fit_counts: Mapping[str, int],
    dag: AttackDAG,
    vocab: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    label_to_index = {label: index for index, label in enumerate(vocab)}
    scores_by_label: dict[str, list[float]] = {label: [] for label in vocab}
    for row_index, target in enumerate(targets):
        if target not in label_to_index:
            raise ValueError(f"validation target outside vocabulary: {target}")
        scores_by_label[target].append(
            aps_score(probabilities[row_index], label_to_index[target], u=0.5)
        )
    all_scores = [score for values in scores_by_label.values() for score in values]
    fallback_quantiles = (
        np.quantile(all_scores, [0.25, 0.5, 0.75])
        if all_scores
        else np.asarray([0.5, 0.5, 0.5])
    )
    features = []
    support: dict[str, int] = {}
    for label in vocab:
        tactic_features = [float(tactic in dag.tactics_for(label)) for tactic in dag.tactic_ids]
        label_scores = scores_by_label[label]
        quantiles = (
            np.quantile(label_scores, [0.25, 0.5, 0.75])
            if label_scores
            else fallback_quantiles
        )
        features.append(
            tactic_features
            + [float(np.log1p(max(0, int(fit_counts.get(label, 0)))))]
            + [float(value) for value in quantiles]
        )
        support[label] = len(label_scores)
    matrix = np.asarray(features, dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (matrix - mean) / scale, support


def fit_graph_clusters(
    validation_probs: np.ndarray,
    validation_targets: Sequence[str],
    fit_counts: Mapping[str, int],
    dag: AttackDAG,
    vocab: Sequence[str],
    min_support: int,
) -> LabelClusters:
    vocab_tuple = tuple(str(label) for label in vocab)
    if not vocab_tuple or len(set(vocab_tuple)) != len(vocab_tuple):
        raise ValueError("cluster vocabulary must be nonempty and unique")
    if int(min_support) < 1:
        raise ValueError("min_support must be positive")
    matrix = _validate_probability_matrix(validation_probs, len(vocab_tuple))
    targets = tuple(str(target) for target in validation_targets)
    if len(matrix) != len(targets):
        raise ValueError("validation probabilities and targets have different lengths")
    features, label_support = _label_features(
        matrix, targets, fit_counts, dag, vocab_tuple
    )
    feature_by_label = {label: features[index] for index, label in enumerate(vocab_tuple)}

    clusters: dict[str, set[str]] = {label: {label} for label in vocab_tuple}

    def cluster_support(labels: set[str]) -> int:
        return sum(label_support[label] for label in labels)

    def cluster_tactics(labels: set[str]) -> frozenset[str]:
        tactics: set[str] = set()
        for label in labels:
            tactics.update(dag.tactics_for(label))
        return frozenset(tactics)

    def centroid(labels: set[str]) -> np.ndarray:
        return np.mean([feature_by_label[label] for label in sorted(labels)], axis=0)

    while True:
        merge_options: list[tuple[int, str, float, str, str, str]] = []
        cluster_keys = sorted(clusters)
        for left_key in cluster_keys:
            left = clusters[left_key]
            support = cluster_support(left)
            if support >= int(min_support):
                continue
            left_tactics = cluster_tactics(left)
            if not left_tactics:
                continue
            left_centroid = centroid(left)
            for right_key in cluster_keys:
                if right_key == left_key:
                    continue
                right = clusters[right_key]
                if left_tactics.isdisjoint(cluster_tactics(right)):
                    continue
                distance = float(np.linalg.norm(left_centroid - centroid(right)))
                merge_options.append(
                    (support, min(left), distance, min(right), left_key, right_key)
                )
        if not merge_options:
            break
        _, _, _, _, selected, right_key = min(merge_options)
        merged_labels = clusters.pop(selected) | clusters.pop(right_key)
        merged_key = min(merged_labels)
        while merged_key in clusters:
            merged_key = f"{merged_key}+"
        clusters[merged_key] = merged_labels

    sorted_members = sorted((tuple(sorted(labels)) for labels in clusters.values()))
    members: dict[str, frozenset[str]] = {}
    label_to_cluster: dict[str, str] = {}
    validation_support: dict[str, int] = {}
    for index, labels in enumerate(sorted_members):
        cluster_id = f"cluster_{index:03d}"
        member_set = frozenset(labels)
        members[cluster_id] = member_set
        validation_support[cluster_id] = sum(label_support[label] for label in member_set)
        for label in member_set:
            label_to_cluster[label] = cluster_id
    return LabelClusters(
        vocab=vocab_tuple,
        label_to_cluster=label_to_cluster,
        members=members,
        validation_support=validation_support,
        min_support=int(min_support),
    )


@dataclass
class ClusteredAPS:
    clusters: LabelClusters
    alpha: float
    thresholds: dict[str, float]
    global_threshold: float
    audit: pd.DataFrame
    seed: int
    last_forced_nonempty_count: int = 0

    def predict_sets(
        self, probabilities: np.ndarray, sample_ids: Sequence[str]
    ) -> list[frozenset[str]]:
        matrix = _validate_probability_matrix(probabilities, len(self.clusters.vocab))
        if len(matrix) != len(sample_ids):
            raise ValueError("prediction probabilities and sample IDs have different lengths")
        output: list[frozenset[str]] = []
        forced = 0
        for row_index, sample_id in enumerate(sample_ids):
            probability = matrix[row_index]
            selected: set[str] = set()
            for label_index, label in enumerate(self.clusters.vocab):
                score = aps_score(
                    probability,
                    label_index,
                    deterministic_uniform(str(sample_id), label, self.seed),
                )
                threshold = self.thresholds[self.clusters.cluster_for(label)]
                if score <= threshold:
                    selected.add(label)
            if not selected:
                selected.add(self.clusters.vocab[int(np.argmax(probability))])
                forced += 1
            output.append(frozenset(selected))
        self.last_forced_nonempty_count = forced
        return output


def fit_clustered_aps(
    cal_probs: np.ndarray,
    cal_targets: Sequence[str],
    clusters: LabelClusters,
    alpha: float,
    sample_ids: Sequence[str],
    min_calibration_support: int = 5,
    seed: int = 20260730,
) -> ClusteredAPS:
    matrix = _validate_probability_matrix(cal_probs, len(clusters.vocab))
    targets = tuple(str(target) for target in cal_targets)
    ids = tuple(str(sample_id) for sample_id in sample_ids)
    if len(matrix) != len(targets) or len(matrix) != len(ids):
        raise ValueError("calibration probabilities, targets, and sample IDs differ in length")
    if int(min_calibration_support) < 1:
        raise ValueError("min_calibration_support must be positive")
    label_to_index = {label: index for index, label in enumerate(clusters.vocab)}
    scores_by_cluster: dict[str, list[float]] = {
        cluster_id: [] for cluster_id in clusters.members
    }
    all_scores: list[float] = []
    for row_index, (target, sample_id) in enumerate(zip(targets, ids, strict=True)):
        if target not in label_to_index:
            raise ValueError(f"calibration target outside vocabulary: {target}")
        score = aps_score(
            matrix[row_index],
            label_to_index[target],
            deterministic_uniform(sample_id, target, seed),
        )
        all_scores.append(score)
        scores_by_cluster[clusters.cluster_for(target)].append(score)
    global_threshold = finite_sample_quantile(all_scores, alpha)
    thresholds: dict[str, float] = {}
    audit_rows = []
    for cluster_id in sorted(clusters.members):
        scores = scores_by_cluster[cluster_id]
        fallback = len(scores) < int(min_calibration_support)
        threshold = (
            global_threshold if fallback else finite_sample_quantile(scores, alpha)
        )
        thresholds[cluster_id] = threshold
        audit_rows.append(
            {
                "cluster_id": cluster_id,
                "members": " || ".join(sorted(clusters.members[cluster_id])),
                "validation_support": clusters.validation_support[cluster_id],
                "calibration_support": len(scores),
                "fallback": bool(fallback),
                "threshold": threshold,
            }
        )
    audit = pd.DataFrame(audit_rows).set_index("cluster_id")
    return ClusteredAPS(
        clusters=clusters,
        alpha=float(alpha),
        thresholds=thresholds,
        global_threshold=global_threshold,
        audit=audit,
        seed=int(seed),
    )
