#!/usr/bin/env python3
"""Run the frozen post-result candidate-level RGAF learnability audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
BASE_RESULTS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1"
B0_RANKINGS = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1/b0_rankings.csv"
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/rgaf_candidate_gate_learnability_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/rgaf_candidate_gate_exploratory_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
METHODS = ("B0", "A", "UniformResidual", "RGAF", "RGAF-Shuffle")
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
FEATURE_NAMES = (
    "log_context1_support",
    "log_context2_support",
    "log_pair1_support",
    "log_pair2_support",
    "pair1_source_fraction",
    "pair2_source_fraction",
    "one_minus_normalized_entropy",
    "b0_reciprocal_rank",
    "absolute_transition_evidence",
    "a_rank_percentile",
    "log_prefix_len",
)
L2_GRID = (0.001, 0.01, 0.1)
EPOCHS = 60
LEARNING_RATE = 0.05
SHUFFLE_SHIFT = 37
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = import_module("rgaf_nonsemantic_base", BASE_SCRIPT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_b0(labels: Sequence[str]) -> dict[str, tuple[int, ...]]:
    label_index = {label: index for index, label in enumerate(labels)}
    output: dict[str, tuple[int, ...]] = {}
    for row in read_csv(B0_RANKINGS):
        sample_id = row["sample_id"]
        ranked = tuple(label_index[label] for label in json.loads(row["predicted_next_ttps"]))
        if len(ranked) != 5 or len(set(ranked)) != 5:
            raise AssertionError(f"invalid B0 ranking for {sample_id}")
        output[sample_id] = ranked
    if len(output) != 784:
        raise AssertionError(f"expected 784 B0 rows, found {len(output)}")
    return output


class EvidenceModel:
    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        labels: Sequence[str],
        label_index: dict[str, int],
    ) -> None:
        self.labels = labels
        self.label_index = label_index
        self.output_size = len(labels)
        self.relevance = BASE.RelevanceModel(
            train,
            self.output_size,
            lambda row: {label_index[label] for label in row["targets"]},
        )
        self.sources = tuple(sorted({row["source"] for row in train}))
        self.source_count = max(1, len(self.sources))
        self.context1_sources: dict[str, set[str]] = defaultdict(set)
        self.context2_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.pair1_sources: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.pair2_sources: dict[tuple[tuple[str, str], int], set[str]] = defaultdict(set)
        for row in train:
            history = row["history"]
            source = row["source"]
            h1 = history[-1]
            self.context1_sources[h1].add(source)
            h2 = (history[-2], history[-1]) if len(history) >= 2 else None
            if h2 is not None:
                self.context2_sources[h2].add(source)
            for label in set(row["targets"]):
                candidate = label_index[label]
                self.pair1_sources[(h1, candidate)].add(source)
                if h2 is not None:
                    self.pair2_sources[(h2, candidate)].add(source)

    def bundle(
        self,
        row: dict[str, Any],
        b0_ranking: tuple[int, ...],
        shuffled: bool = False,
    ) -> dict[str, Any]:
        history = row["history"]
        h1 = history[-1]
        h2 = (history[-2], history[-1]) if len(history) >= 2 else None
        context1_count = self.relevance.order1_counts.get(h1, 0)
        context2_count = self.relevance.order2_counts.get(h2, 0) if h2 is not None else 0
        pair1 = self.relevance.order1_targets.get(h1, [0] * self.output_size)
        pair2 = self.relevance.order2_targets.get(h2, [0] * self.output_size) if h2 is not None else [0] * self.output_size
        a_values = self.relevance.score(history)
        prior = self.relevance.unigram
        total = sum(max(value, 0.0) for value in a_values)
        probabilities = [max(value, 0.0) / total for value in a_values] if total else [1.0 / self.output_size] * self.output_size
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0) / math.log(self.output_size)
        a_order = sorted(range(self.output_size), key=lambda index: (-a_values[index], index))
        a_rank = [0] * self.output_size
        for rank, candidate in enumerate(a_order):
            a_rank[candidate] = rank
        prior_order = sorted(range(self.output_size), key=lambda index: (-prior[index], index))
        b0_scores = [0.0] * self.output_size
        for rank, candidate in enumerate(b0_ranking, start=1):
            b0_scores[candidate] = 1.0 / rank
        evidence = [
            max(-4.0, min(4.0, math.log((a_values[index] + 1e-9) / (prior[index] + 1e-9)))) / 4.0
            for index in range(self.output_size)
        ]
        raw_features: list[tuple[float, ...]] = []
        aligned_evidence: list[float] = []
        aligned_a_values: list[float] = []
        for candidate in range(self.output_size):
            donor = (candidate + SHUFFLE_SHIFT) % self.output_size if shuffled else candidate
            donor_evidence = evidence[donor]
            aligned_evidence.append(donor_evidence)
            aligned_a_values.append(a_values[donor])
            raw_features.append(
                (
                    math.log1p(context1_count),
                    math.log1p(context2_count),
                    math.log1p(pair1[donor]),
                    math.log1p(pair2[donor]),
                    len(self.pair1_sources.get((h1, donor), set())) / self.source_count,
                    len(self.pair2_sources.get((h2, donor), set())) / self.source_count if h2 is not None else 0.0,
                    1.0 - entropy,
                    b0_scores[candidate],
                    abs(donor_evidence),
                    1.0 - a_rank[donor] / max(1, self.output_size - 1),
                    math.log1p(row["prefix_len"]),
                )
            )
        aligned_order = sorted(range(self.output_size), key=lambda index: (-aligned_a_values[index], index))
        return {
            "row": row,
            "b0_ranking": b0_ranking,
            "base": b0_scores,
            "evidence": aligned_evidence,
            "raw_features": raw_features,
            "a_values": a_values,
            "a_order": a_order,
            "transition_order": aligned_order,
            "prior_order": prior_order,
            "context1_count": context1_count,
            "context2_count": context2_count,
        }


def campaign_loo_bundles(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    b0: dict[str, tuple[int, ...]],
    shuffled: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["campaign_id"])].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        excluded = grouped[key]
        excluded_ids = {row["sample_id"] for row in excluded}
        train = [row for row in rows if row["sample_id"] not in excluded_ids]
        if not train:
            raise AssertionError(f"empty campaign-LOO training set for {key}")
        model = EvidenceModel(train, labels, label_index)
        output.extend(model.bundle(row, b0[row["sample_id"]], shuffled) for row in excluded)
    output.sort(key=lambda bundle: bundle["row"]["sample_id"])
    if len(output) != len(rows):
        raise AssertionError("campaign-LOO bundle count changed")
    return output


def feature_normalization(bundles: Sequence[dict[str, Any]]) -> tuple[list[float], list[float]]:
    dimension = len(FEATURE_NAMES)
    count = len(bundles) * len(bundles[0]["raw_features"])
    means = [0.0] * dimension
    for bundle in bundles:
        for features in bundle["raw_features"]:
            for index, value in enumerate(features):
                means[index] += value
    means = [value / count for value in means]
    variances = [0.0] * dimension
    for bundle in bundles:
        for features in bundle["raw_features"]:
            for index, value in enumerate(features):
                variances[index] += (value - means[index]) ** 2
    scales = [math.sqrt(value / count) or 1.0 for value in variances]
    return means, scales


def apply_normalization(
    bundles: Sequence[dict[str, Any]], means: Sequence[float], scales: Sequence[float]
) -> None:
    for bundle in bundles:
        bundle["features"] = [
            (1.0, *( (value - means[index]) / scales[index] for index, value in enumerate(features) ))
            for features in bundle["raw_features"]
        ]


def hard_negatives(bundle: dict[str, Any], label_index: dict[str, int]) -> tuple[list[int], list[int]]:
    positives = sorted({label_index[label] for label in bundle["row"]["targets"]})
    positive_set = set(positives)
    candidates = set(bundle["transition_order"][:20])
    candidates.update(bundle["b0_ranking"])
    candidates.update(bundle["prior_order"][:10])
    negatives = sorted(candidate for candidate in candidates if candidate not in positive_set)
    if not negatives:
        raise AssertionError(f"no hard negatives for {bundle['row']['sample_id']}")
    return positives, negatives


def candidate_scores(bundle: dict[str, Any], weights: Sequence[float]) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    gates: list[float] = []
    for candidate, features in enumerate(bundle["features"]):
        gate = sigmoid(sum(weight * value for weight, value in zip(weights, features)))
        gates.append(gate)
        scores.append(bundle["base"][candidate] + gate * bundle["evidence"][candidate])
    return scores, gates


def train_gate(
    bundles: Sequence[dict[str, Any]],
    label_index: dict[str, int],
    l2: float,
) -> list[float]:
    dimension = len(FEATURE_NAMES) + 1
    weights = [math.log(0.1 / 0.9), *([0.0] * len(FEATURE_NAMES))]
    first_moment = [0.0] * dimension
    second_moment = [0.0] * dimension
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for epoch in range(1, EPOCHS + 1):
        gradient = [0.0] * dimension
        pairs = 0
        for bundle in bundles:
            scores, gates = candidate_scores(bundle, weights)
            positives, negatives = hard_negatives(bundle, label_index)
            derivatives = []
            for candidate, features in enumerate(bundle["features"]):
                gate_derivative = gates[candidate] * (1.0 - gates[candidate]) * bundle["evidence"][candidate]
                derivatives.append([gate_derivative * value for value in features])
            for positive in positives:
                for negative in negatives:
                    difference = scores[positive] - scores[negative]
                    factor = -sigmoid(-difference)
                    for index in range(dimension):
                        gradient[index] += factor * (derivatives[positive][index] - derivatives[negative][index])
                    pairs += 1
        if not pairs:
            raise AssertionError("gate training produced zero pairs")
        for index in range(dimension):
            gradient[index] /= pairs
            if index:
                gradient[index] += l2 * weights[index]
            first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * gradient[index]
            second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * gradient[index] ** 2
            corrected_first = first_moment[index] / (1.0 - beta1 ** epoch)
            corrected_second = second_moment[index] / (1.0 - beta2 ** epoch)
            weights[index] -= LEARNING_RATE * corrected_first / (math.sqrt(corrected_second) + epsilon)
    return weights


def ranked_indices(scores: Sequence[float], labels: Sequence[str]) -> list[int]:
    return sorted(range(len(labels)), key=lambda index: (-scores[index], index))


def campaign_macro(rows: Sequence[dict[str, Any]], metric: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["campaign_id"]].append(float(row[metric]))
    return statistics.fmean(statistics.fmean(values) for values in grouped.values())


def evaluate_bundles(
    bundles: Sequence[dict[str, Any]],
    weights: Sequence[float],
    labels: Sequence[str],
    method: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bundle in bundles:
        row = bundle["row"]
        if method in {"RGAF", "RGAF-Shuffle"}:
            scores, gates = candidate_scores(bundle, weights)
            ranked = ranked_indices(scores, labels)
        elif method == "UniformResidual":
            scores = [base + evidence for base, evidence in zip(bundle["base"], bundle["evidence"])]
            gates = [1.0] * len(labels)
            ranked = ranked_indices(scores, labels)
        elif method == "B0":
            ranked = [*bundle["b0_ranking"], *[index for index in range(len(labels)) if index not in set(bundle["b0_ranking"])]]
            scores = bundle["base"]
            gates = [0.0] * len(labels)
        elif method == "A":
            scores = bundle["a_values"]
            gates = [0.0] * len(labels)
            ranked = bundle["a_order"]
        else:
            raise AssertionError(method)
        top5_labels = [labels[index] for index in ranked[:5]]
        metrics = BASE.sample_metrics(top5_labels, row["targets"])
        top20 = ranked[:20]
        output.append(
            {
                "held_out_source": row["source"],
                "method": method,
                "sample_id": row["sample_id"],
                "campaign_id": row["campaign_id"],
                "prefix_len": row["prefix_len"],
                "target_parent_ids": compact_json(row["targets"]),
                "target_size": row["target_size"],
                "top20_ids": compact_json([labels[index] for index in top20]),
                "top20_scores": compact_json([round(scores[index], 12) for index in top20]),
                "mean_gate": statistics.fmean(gates),
                "mean_top5_gate": statistics.fmean(gates[index] for index in ranked[:5]),
                "gate_gt_half_fraction": sum(gate > 0.5 for gate in gates) / len(gates),
                "context1_support": bundle["context1_count"],
                "context2_support": bundle["context2_count"],
                **metrics,
            }
        )
    return output


def select_l2(
    outer_train_sources: Sequence[str],
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    b0: dict[str, tuple[int, ...]],
) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    per_l2: dict[float, list[float]] = defaultdict(list)
    for validation_source in outer_train_sources:
        inner_train = [row for row in rows if row["source"] in outer_train_sources and row["source"] != validation_source]
        validation = [row for row in rows if row["source"] == validation_source]
        train_bundles = campaign_loo_bundles(inner_train, labels, label_index, b0, False)
        means, scales = feature_normalization(train_bundles)
        apply_normalization(train_bundles, means, scales)
        validation_model = EvidenceModel(inner_train, labels, label_index)
        validation_bundles = [validation_model.bundle(row, b0[row["sample_id"]], False) for row in validation]
        apply_normalization(validation_bundles, means, scales)
        for l2 in L2_GRID:
            weights = train_gate(train_bundles, label_index, l2)
            predictions = evaluate_bundles(validation_bundles, weights, labels, "RGAF")
            value = campaign_macro(predictions, "ndcg5")
            per_l2[l2].append(value)
            details.append(
                {
                    "inner_validation_source": validation_source,
                    "l2": l2,
                    "campaign_macro_ndcg5": value,
                    "inner_training_rows": len(inner_train),
                    "validation_rows": len(validation),
                }
            )
    mean_values = {l2: statistics.fmean(values) for l2, values in per_l2.items()}
    best = max(mean_values.values())
    selected = max(l2 for l2, value in mean_values.items() if abs(value - best) <= 1e-12)
    for row in details:
        row["two_source_mean_ndcg5"] = mean_values[row["l2"]]
        row["selected"] = int(row["l2"] == selected)
    return selected, details


def fold_predictions(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    b0: dict[str, tuple[int, ...]],
    l2: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    main_train = campaign_loo_bundles(train, labels, label_index, b0, False)
    main_means, main_scales = feature_normalization(main_train)
    apply_normalization(main_train, main_means, main_scales)
    main_weights = train_gate(main_train, label_index, l2)
    shuffle_train = campaign_loo_bundles(train, labels, label_index, b0, True)
    shuffle_means, shuffle_scales = feature_normalization(shuffle_train)
    apply_normalization(shuffle_train, shuffle_means, shuffle_scales)
    shuffle_weights = train_gate(shuffle_train, label_index, l2)
    outer_model = EvidenceModel(train, labels, label_index)
    main_test = [outer_model.bundle(row, b0[row["sample_id"]], False) for row in test]
    shuffle_test = [outer_model.bundle(row, b0[row["sample_id"]], True) for row in test]
    apply_normalization(main_test, main_means, main_scales)
    apply_normalization(shuffle_test, shuffle_means, shuffle_scales)
    predictions: list[dict[str, Any]] = []
    predictions.extend(evaluate_bundles(main_test, main_weights, labels, "RGAF"))
    predictions.extend(evaluate_bundles(shuffle_test, shuffle_weights, labels, "RGAF-Shuffle"))
    predictions.extend(evaluate_bundles(main_test, main_weights, labels, "UniformResidual"))
    predictions.extend(evaluate_bundles(main_test, main_weights, labels, "B0"))
    predictions.extend(evaluate_bundles(main_test, main_weights, labels, "A"))
    weights = {
        "held_out_source": held_out,
        "selected_l2": l2,
        "rgaf_weights": compact_json([round(value, 12) for value in main_weights]),
        "shuffle_weights": compact_json([round(value, 12) for value in shuffle_weights]),
        "feature_means": compact_json([round(value, 12) for value in main_means]),
        "feature_scales": compact_json([round(value, 12) for value in main_scales]),
    }
    return predictions, weights


def campaign_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
            "campaign_id": campaign,
            "rows": len(values),
            **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, method, campaign), values in sorted(grouped.items())
    ]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in campaigns:
        grouped[(row["held_out_source"], row["method"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
            "campaigns": len(values),
            **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, method), values in sorted(grouped.items())
    ]


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in campaigns}
    ids = {
        source: sorted(row["campaign_id"] for row in campaigns if row["held_out_source"] == source and row["method"] == "B0")
        for source in SOURCES
    }
    comparisons = (("RGAF", "B0"), ("RGAF", "RGAF-Shuffle"), ("RGAF", "UniformResidual"), ("UniformResidual", "B0"))
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(ids[source]) for _ in ids[source]] for source in SOURCES}
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                source_values = [
                    statistics.fmean(
                        float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    for source in SOURCES
                ]
                replicates[(comparison, metric)].append(statistics.fmean(source_values))
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        comparison = f"{left}-{right}"
        for metric in METRICS:
            point = statistics.fmean(
                statistics.fmean(
                    float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                    for campaign in ids[source]
                )
                for source in SOURCES
            )
            values = sorted(replicates[(comparison, metric)])
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "point_estimate": point,
                    "ci95_low": quantile(values, 0.025),
                    "ci95_high": quantile(values, 0.975),
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return output


def report(folds: Sequence[dict[str, Any]], differences: Sequence[dict[str, Any]], selected: dict[str, float]) -> str:
    lookup = {(row["held_out_source"], row["method"]): row for row in folds}
    lines = [
        "# Candidate-level RGAF learnability audit v1",
        "",
        "**Post-result exploratory method development; prospective confirmation is required.**",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [float(lookup[(source, method)]["campaign_macro_ndcg5"]) for source in SOURCES]
        lines.append(f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | **{statistics.fmean(values):.4f}** |")
    lines.extend(["", "Selected L2: " + ", ".join(f"{source}={selected[source]:g}" for source in SOURCES), "", "| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("RGAF-B0", "RGAF-RGAF-Shuffle", "RGAF-UniformResidual", "UniformResidual-B0"):
        row = next(item for item in differences if item["comparison"] == comparison and item["metric"] == "ndcg5")
        lines.append(f"| {comparison} | {float(row['point_estimate']):+.4f} | [{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] |")
    rgaf_real = all(float(lookup[(source, "RGAF")]["campaign_macro_ndcg5"]) > float(lookup[(source, "B0")]["campaign_macro_ndcg5"]) for source in ("ctid", "attack_flow"))
    rgaf_overall = statistics.fmean(float(lookup[(source, "RGAF")]["campaign_macro_ndcg5"]) for source in SOURCES)
    b0_overall = statistics.fmean(float(lookup[(source, "B0")]["campaign_macro_ndcg5"]) for source in SOURCES)
    shuffle_overall = statistics.fmean(float(lookup[(source, "RGAF-Shuffle")]["campaign_macro_ndcg5"]) for source in SOURCES)
    if rgaf_real and rgaf_overall > b0_overall and rgaf_overall > shuffle_overall:
        primary = next(item for item in differences if item["comparison"] == "RGAF-B0" and item["metric"] == "ndcg5")
        status = "strong exploratory signal" if float(primary["ci95_low"]) > 0 else "learnability signal"
    else:
        status = "no learnability evidence"
    lines.extend(["", f"Frozen decision: **{status}**.", "", "Exact B0 Top-5 and A Top-20 reproduction gates passed for all 784 rows.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    b0 = load_b0(labels)
    if set(b0) != {row["sample_id"] for row in rows}:
        raise AssertionError("B0/sample key mismatch")
    if args.validate_only:
        print(compact_json({"rows": len(rows), "b0_rows": len(b0), "labels": len(labels), "status": "PASS"}))
        return
    committed_a = {
        row["sample_id"]: tuple(json.loads(row["top20_ids"]))
        for row in read_csv(BASE_RESULTS / "predictions.csv")
        if row["method"] == "A"
    }
    inner_rows: list[dict[str, Any]] = []
    selected_l2: dict[str, float] = {}
    predictions: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for held_out in SOURCES:
        outer_train_sources = tuple(source for source in SOURCES if source != held_out)
        l2, details = select_l2(outer_train_sources, rows, labels, label_index, b0)
        selected_l2[held_out] = l2
        for row in details:
            row["held_out_source"] = held_out
        inner_rows.extend(details)
        fold, weights = fold_predictions(held_out, rows, labels, label_index, b0, l2)
        predictions.extend(fold)
        weight_rows.append(weights)
    predictions.sort(key=lambda row: (row["held_out_source"], row["method"], row["sample_id"]))
    counts: dict[str, int] = defaultdict(int)
    for row in predictions:
        counts[row["method"]] += 1
    if set(counts) != set(METHODS) or any(counts[method] != 784 for method in METHODS):
        raise AssertionError(f"prediction count gate failed: {dict(counts)}")
    for row in predictions:
        if row["method"] == "B0":
            expected = tuple(labels[index] for index in b0[row["sample_id"]])
            actual = tuple(json.loads(row["top20_ids"]))[:5]
            if actual != expected:
                raise AssertionError(f"B0 reproduction failed: {row['sample_id']}")
        if row["method"] == "A":
            actual = tuple(json.loads(row["top20_ids"]))
            if actual != committed_a[row["sample_id"]]:
                raise AssertionError(f"A reproduction failed: {row['sample_id']}")
    campaigns = campaign_rows(predictions)
    folds = fold_rows(campaigns)
    differences = bootstrap(campaigns)
    markdown = report(folds, differences, selected_l2)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "inner_selection.csv", sorted(inner_rows, key=lambda row: (row["held_out_source"], row["inner_validation_source"], row["l2"])), ("held_out_source", "inner_validation_source", "l2", "campaign_macro_ndcg5", "two_source_mean_ndcg5", "selected", "inner_training_rows", "validation_rows"))
    write_csv(output / "predictions.csv", predictions, ("held_out_source", "method", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "top20_ids", "top20_scores", "mean_gate", "mean_top5_gate", "gate_gt_half_fraction", "context1_support", "context2_support", *METRICS))
    write_csv(output / "campaign_results.csv", campaigns, ("held_out_source", "method", "campaign_id", "rows", *METRICS))
    write_csv(output / "fold_results.csv", folds, ("held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]))
    write_csv(output / "paired_bootstrap_differences.csv", differences, ("comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"))
    write_csv(output / "gate_parameters.csv", weight_rows, ("held_out_source", "selected_l2", "rgaf_weights", "shuffle_weights", "feature_means", "feature_scales"))
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "stdout.log").write_text(markdown, encoding="utf-8")
    managed = ("inner_selection.csv", "predictions.csv", "campaign_results.csv", "fold_results.csv", "paired_bootstrap_differences.csv", "gate_parameters.csv", "report.md", "stdout.log")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post-result exploratory method development",
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "protocol": {"path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "inputs": {
            BASE.SAMPLES_PATH.relative_to(PROJECT_ROOT).as_posix(): sha256(BASE.SAMPLES_PATH),
            BASE.VOCAB_PATH.relative_to(PROJECT_ROOT).as_posix(): sha256(BASE.VOCAB_PATH),
            B0_RANKINGS.relative_to(PROJECT_ROOT).as_posix(): sha256(B0_RANKINGS),
            (BASE_RESULTS / "predictions.csv").relative_to(PROJECT_ROOT).as_posix(): sha256(BASE_RESULTS / "predictions.csv"),
        },
        "parameters": {
            "features": list(FEATURE_NAMES),
            "l2_grid": list(L2_GRID),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "shuffle_shift": SHUFFLE_SHIFT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "selected_l2": selected_l2,
        },
        "leakage_gate": "PASS: feature/scoring path does not read targets or target-conditioned visibility; targets enter only gate fitting and post-ranking metrics",
        "reproduction_gates": {"B0_top5": "PASS 784/784", "A_top20": "PASS 784/784"},
        "outputs_sha256": {name: sha256(output / name) for name in managed},
    }
    (output / "results_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
