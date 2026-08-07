#!/usr/bin/env python3
"""Run the frozen A0/CO/A/K/T future-3 LODO baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_v4/semantic_alignment"
SAMPLES_PATH = DATA_DIR / "future3_samples.csv"
TACTIC_MAP_PATH = DATA_DIR / "technique_tactic_multihot.csv"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/nonsemantic_future3_baselines_v1.md"
ADDENDUM = (
    PROJECT_ROOT
    / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.1_addendum.md"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
METHODS = ("A0", "CO", "A", "K", "T")
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
LAMBDA_GRID = tuple(index / 10 for index in range(11))
ALPHA = 0.1
LAYER_WEIGHTS = (0.5, 0.3, 0.2)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_samples() -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in read_csv(SAMPLES_PATH):
        if int(row["is_development"]):
            continue
        parsed.append(
            {
                **row,
                "prefix_len": int(row["prefix_len"]),
                "history": tuple(json.loads(row["observed_parent_ids"])),
                "targets": tuple(json.loads(row["target_parent_ids"])),
                "target_size": int(row["target_size"]),
                "last_description_chars": int(row["last_description_chars"]),
            }
        )
    if len(parsed) != 784:
        raise AssertionError(f"expected 784 main rows, found {len(parsed)}")
    if Counter(row["source"] for row in parsed) != Counter(
        {"ctid": 263, "attack_flow": 412, "stockpile": 109}
    ):
        raise AssertionError("main source counts changed")
    return parsed


def parse_vocabulary() -> tuple[list[str], dict[str, int]]:
    labels = [row["technique_id_parent"] for row in read_csv(VOCAB_PATH)]
    if len(labels) != 184 or len(set(labels)) != 184:
        raise AssertionError("vocabulary is not 184 unique labels")
    return labels, {label: index for index, label in enumerate(labels)}


def parse_tactics() -> tuple[list[tuple[int, ...]], dict[str, tuple[int, ...]]]:
    candidate: list[tuple[int, ...]] = []
    by_label: dict[str, tuple[int, ...]] = {}
    for row in read_csv(TACTIC_MAP_PATH):
        indices = tuple(int(value) for value in json.loads(row["tactic_indices"]))
        if not indices:
            raise AssertionError(f"missing tactic map for {row['parent_technique_id']}")
        candidate.append(indices)
        by_label[row["parent_technique_id"]] = indices
    if len(candidate) != 184:
        raise AssertionError("candidate tactic mapping is not 184 rows")
    return candidate, by_label


class RelevanceModel:
    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        output_size: int,
        target_getter: Any,
    ) -> None:
        self.output_size = output_size
        self.n = len(train)
        self.target_counts = [0] * output_size
        self.order1_counts: Counter[str] = Counter()
        self.order2_counts: Counter[tuple[str, str]] = Counter()
        self.order1_targets: dict[str, list[int]] = {}
        self.order2_targets: dict[tuple[str, str], list[int]] = {}
        for row in train:
            targets = tuple(sorted(set(target_getter(row))))
            history = row["history"]
            for target in targets:
                self.target_counts[target] += 1
            h1 = history[-1]
            self.order1_counts[h1] += 1
            bucket1 = self.order1_targets.setdefault(h1, [0] * output_size)
            for target in targets:
                bucket1[target] += 1
            if len(history) >= 2:
                h2 = (history[-2], history[-1])
                self.order2_counts[h2] += 1
                bucket2 = self.order2_targets.setdefault(h2, [0] * output_size)
                for target in targets:
                    bucket2[target] += 1
        self.unigram = [
            (count + 0.5) / (self.n + 1.0) for count in self.target_counts
        ]

    def score(self, history: Sequence[str]) -> list[float]:
        layers: list[tuple[float, list[float]]] = []
        if len(history) >= 2:
            context2 = (history[-2], history[-1])
            count2 = self.order2_counts.get(context2, 0)
            if count2:
                targets2 = self.order2_targets[context2]
                layers.append(
                    (
                        LAYER_WEIGHTS[0],
                        [
                            (targets2[index] + ALPHA * self.unigram[index])
                            / (count2 + ALPHA)
                            for index in range(self.output_size)
                        ],
                    )
                )
        context1 = history[-1]
        count1 = self.order1_counts.get(context1, 0)
        if count1:
            targets1 = self.order1_targets[context1]
            layers.append(
                (
                    LAYER_WEIGHTS[1],
                    [
                        (targets1[index] + ALPHA * self.unigram[index])
                        / (count1 + ALPHA)
                        for index in range(self.output_size)
                    ],
                )
            )
        layers.append((LAYER_WEIGHTS[2], self.unigram))
        total_weight = sum(weight for weight, _ in layers)
        return [
            sum(weight * values[index] for weight, values in layers) / total_weight
            for index in range(self.output_size)
        ]


class CooccurrenceModel:
    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        label_index: dict[str, int],
    ) -> None:
        self.n = len(train)
        self.label_index = label_index
        self.history_counts: Counter[str] = Counter()
        self.target_counts = [0] * len(label_index)
        self.joint: dict[str, list[int]] = {}
        for row in train:
            histories = sorted(set(row["history"]))
            targets = sorted({label_index[label] for label in row["targets"]})
            for label in targets:
                self.target_counts[label] += 1
            for history in histories:
                self.history_counts[history] += 1
                values = self.joint.setdefault(history, [0] * len(label_index))
                for label in targets:
                    values[label] += 1

    def score(self, history: Sequence[str], a0: Sequence[float]) -> tuple[list[float], bool]:
        values = [0.0] * len(self.label_index)
        for item in sorted(set(history)):
            history_count = self.history_counts.get(item, 0)
            if not history_count:
                continue
            joints = self.joint[item]
            for label, joint in enumerate(joints):
                target_count = self.target_counts[label]
                if joint and target_count:
                    pmi = math.log((joint * self.n) / (history_count * target_count))
                    if pmi > 0:
                        values[label] += pmi
        fallback = not any(value > 0 for value in values)
        return (list(a0), True) if fallback else (values, False)


def clipped_logit(value: float) -> float:
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def standardize(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    sd = math.sqrt(variance)
    if sd < 1e-6:
        return [0.0] * len(values)
    return [(value - mean) / sd for value in values]


def ranking(scores: Sequence[float], labels: Sequence[str]) -> tuple[list[str], list[float]]:
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    return [labels[index] for index in order], [scores[index] for index in order]


def sample_metrics(top5: Sequence[str], targets: Sequence[str]) -> dict[str, float]:
    truth = set(targets)
    hits = [int(label in truth) for label in top5]
    found = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(5, len(truth))))
    return {
        "ndcg5": dcg / ideal,
        "hit5": float(found > 0),
        "precision5": found / 5,
        "recall5": found / len(truth),
    }


def campaign_macro(records: Sequence[dict[str, Any]], metric: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["campaign_id"]].append(float(row[metric]))
    return statistics.fmean(statistics.fmean(values) for values in grouped.values())


def tactic_target_indices(row: dict[str, Any], tactic_by_label: dict[str, tuple[int, ...]]) -> set[int]:
    result: set[int] = set()
    for label in row["targets"]:
        result.update(tactic_by_label[label])
    return result


def tactic_candidate_scores(
    tactic_probabilities: Sequence[float],
    candidate_tactics: Sequence[tuple[int, ...]],
) -> list[float]:
    return [statistics.fmean(tactic_probabilities[index] for index in indices) for indices in candidate_tactics]


def fused_scores(a_values: Sequence[float], t_values: Sequence[float], weight: float) -> list[float]:
    a_z = standardize([clipped_logit(value) for value in a_values])
    t_z = standardize([clipped_logit(value) for value in t_values])
    return [(1 - weight) * left + weight * right for left, right in zip(a_z, t_z)]


def choose_lambda(
    outer_train_sources: Sequence[str],
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
) -> tuple[float, list[dict[str, Any]]]:
    per_lambda: dict[float, list[float]] = {value: [] for value in LAMBDA_GRID}
    details: list[dict[str, Any]] = []
    for validation_source in outer_train_sources:
        inner_train = [
            row
            for row in rows
            if row["source"] in outer_train_sources and row["source"] != validation_source
        ]
        validation = [row for row in rows if row["source"] == validation_source]
        a_model = RelevanceModel(
            inner_train, len(labels), lambda row: {label_index[label] for label in row["targets"]}
        )
        t_model = RelevanceModel(
            inner_train,
            14,
            lambda row: tactic_target_indices(row, tactic_by_label),
        )
        predictions: dict[float, list[dict[str, Any]]] = {value: [] for value in LAMBDA_GRID}
        for row in validation:
            a_values = a_model.score(row["history"])
            tactic_values = tactic_candidate_scores(
                t_model.score(row["history"]), candidate_tactics
            )
            for value in LAMBDA_GRID:
                scores = fused_scores(a_values, tactic_values, value)
                ranked, _ = ranking(scores, labels)
                predictions[value].append(
                    {**row, **sample_metrics(ranked[:5], row["targets"])}
                )
        for value in LAMBDA_GRID:
            score = campaign_macro(predictions[value], "ndcg5")
            per_lambda[value].append(score)
            details.append(
                {
                    "inner_validation_source": validation_source,
                    "lambda": value,
                    "campaign_macro_ndcg5": score,
                    "training_rows": len(inner_train),
                    "validation_rows": len(validation),
                }
            )
    mean_scores = {value: statistics.fmean(scores) for value, scores in per_lambda.items()}
    best_score = max(mean_scores.values())
    chosen = min(value for value, score in mean_scores.items() if abs(score - best_score) <= 1e-12)
    for row in details:
        row["source_equal_mean_ndcg5"] = mean_scores[row["lambda"]]
        row["selected"] = int(row["lambda"] == chosen)
    return chosen, details


def visibility(
    test: dict[str, Any],
    seen_pairs: set[tuple[str, str]],
    seen_targets: set[str],
) -> tuple[str, str]:
    last = test["history"][-1]
    pair_flags = [(last, target) in seen_pairs for target in test["targets"]]
    if all(pair_flags):
        transition = "all_seen"
    elif any(pair_flags):
        transition = "mixed"
    else:
        transition = "all_unseen"
    target_flags = [target in seen_targets for target in test["targets"]]
    if all(target_flags):
        target_visibility = "all_seen"
    elif any(target_flags):
        target_visibility = "mixed"
    else:
        target_visibility = "all_unseen"
    return transition, target_visibility


def evaluate_fold(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    outer_train_sources = tuple(source for source in SOURCES if source != held_out)
    train = [row for row in rows if row["source"] in outer_train_sources]
    test = [row for row in rows if row["source"] == held_out]
    chosen_lambda, inner = choose_lambda(
        outer_train_sources,
        rows,
        labels,
        label_index,
        candidate_tactics,
        tactic_by_label,
    )
    for row in inner:
        row["held_out_source"] = held_out
        row["outer_training_sources"] = compact_json(outer_train_sources)

    a0 = [0.0] * len(labels)
    for row in train:
        for target in set(row["targets"]):
            a0[label_index[target]] += 1
    co_model = CooccurrenceModel(train, label_index)
    a_model = RelevanceModel(
        train, len(labels), lambda row: {label_index[label] for label in row["targets"]}
    )
    t_model = RelevanceModel(
        train,
        14,
        lambda row: tactic_target_indices(row, tactic_by_label),
    )
    seen_pairs = {
        (row["history"][-1], target) for row in train for target in row["targets"]
    }
    seen_targets = {target for row in train for target in row["targets"]}

    output: list[dict[str, Any]] = []
    for row in test:
        a_values = a_model.score(row["history"])
        co_values, co_fallback = co_model.score(row["history"], a0)
        last_tactics = tactic_by_label.get(row["history"][-1], tuple())
        compatible = [
            bool(last_tactics)
            and any(candidate >= last for candidate in indices for last in last_tactics)
            for indices in candidate_tactics
        ]
        k_values = [int(flag) + value for flag, value in zip(compatible, a_values)]
        tactic_values = tactic_candidate_scores(
            t_model.score(row["history"]), candidate_tactics
        )
        t_values = fused_scores(a_values, tactic_values, chosen_lambda)
        method_scores = {
            "A0": a0,
            "CO": co_values,
            "A": a_values,
            "K": k_values,
            "T": t_values,
        }
        transition_group, target_group = visibility(row, seen_pairs, seen_targets)
        for method in METHODS:
            ranked, ordered_scores = ranking(method_scores[method], labels)
            metrics = sample_metrics(ranked[:5], row["targets"])
            output.append(
                {
                    "held_out_source": held_out,
                    "method": method,
                    "sample_id": row["sample_id"],
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "target_parent_ids": compact_json(row["targets"]),
                    "target_size": row["target_size"],
                    "transition_visibility": transition_group,
                    "target_label_visibility": target_group,
                    "text_length_group": row["text_length_group"],
                    "chosen_tactic_lambda": chosen_lambda if method == "T" else "",
                    "co_fallback_to_a0": int(co_fallback) if method == "CO" else "",
                    "top20_ids": compact_json(ranked[:20]),
                    "top20_scores": compact_json([round(value, 12) for value in ordered_scores[:20]]),
                    **metrics,
                }
            )
    return output, inner, chosen_lambda


def campaign_results(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (source, method, campaign), rows in sorted(grouped.items()):
        output.append(
            {
                "held_out_source": source,
                "method": method,
                "campaign_id": campaign,
                "rows": len(rows),
                **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
            }
        )
    return output


def fold_results(
    predictions: Sequence[dict[str, Any]],
    campaigns: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        for method in METHODS:
            rows = [
                row for row in predictions if row["held_out_source"] == source and row["method"] == method
            ]
            camps = [
                row for row in campaigns if row["held_out_source"] == source and row["method"] == method
            ]
            output.append(
                {
                    "held_out_source": source,
                    "method": method,
                    "rows": len(rows),
                    "campaigns": len(camps),
                    **{
                        f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in camps)
                        for metric in METRICS
                    },
                    **{
                        f"row_micro_{metric}": statistics.fmean(float(row[metric]) for row in rows)
                        for metric in METRICS
                    },
                }
            )
    return output


def aggregate_results(folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in folds if row["method"] == method]
        output.append(
            {
                "method": method,
                "source_folds": len(rows),
                **{
                    f"source_equal_campaign_macro_{metric}": statistics.fmean(
                        float(row[f"campaign_macro_{metric}"]) for row in rows
                    )
                    for metric in METRICS
                },
                **{
                    f"all_row_micro_{metric}": sum(
                        float(row[f"row_micro_{metric}"]) * int(row["rows"]) for row in rows
                    )
                    / sum(int(row["rows"]) for row in rows)
                    for metric in METRICS
                },
            }
        )
    return output


def stratified_results(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "transition_visibility": lambda row: row["transition_visibility"],
        "target_label_visibility": lambda row: row["target_label_visibility"],
        "text_length": lambda row: row["text_length_group"],
        "target_size": lambda row: str(row["target_size"]),
    }
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        for method in METHODS:
            method_rows = [
                row for row in predictions if row["held_out_source"] == source and row["method"] == method
            ]
            for stratum_name, getter in definitions.items():
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in method_rows:
                    groups[getter(row)].append(row)
                for group, rows in sorted(groups.items()):
                    campaign_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in rows:
                        campaign_values[row["campaign_id"]].append(row)
                    output.append(
                        {
                            "held_out_source": source,
                            "method": method,
                            "stratum": stratum_name,
                            "group": group,
                            "rows": len(rows),
                            "campaigns": len(campaign_values),
                            "inferentially_eligible": int(
                                len(rows) >= 20 and len(campaign_values) >= 5
                            ),
                            **{
                                f"campaign_macro_{metric}": statistics.fmean(
                                    statistics.fmean(float(row[metric]) for row in values)
                                    for values in campaign_values.values()
                                )
                                for metric in METRICS
                            },
                        }
                    )
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap(
    campaigns: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    campaign_ids: dict[str, list[str]] = {}
    for source in SOURCES:
        campaign_ids[source] = sorted(
            {row["campaign_id"] for row in campaigns if row["held_out_source"] == source}
        )
    for row in campaigns:
        values[row["held_out_source"]][row["method"]][row["campaign_id"]] = {
            metric: float(row[metric]) for metric in METRICS
        }

    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[dict[str, list[str]]] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draws.append(
            {
                source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]]
                for source in SOURCES
            }
        )

    replicate_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for draw in draws:
        for method in METHODS:
            for metric in METRICS:
                source_values: list[float] = []
                for source in SOURCES:
                    value = statistics.fmean(
                        values[source][method][campaign][metric] for campaign in draw[source]
                    )
                    replicate_values[(source, method, metric)].append(value)
                    source_values.append(value)
                replicate_values[("source_equal_overall", method, metric)].append(
                    statistics.fmean(source_values)
                )

    point_source: dict[tuple[str, str, str], float] = {}
    for source in SOURCES:
        for method in METHODS:
            for metric in METRICS:
                point_source[(source, method, metric)] = statistics.fmean(
                    values[source][method][campaign][metric] for campaign in campaign_ids[source]
                )
    for row in aggregates:
        for metric in METRICS:
            point_source[("source_equal_overall", row["method"], metric)] = float(
                row[f"source_equal_campaign_macro_{metric}"]
            )

    method_rows: list[dict[str, Any]] = []
    for (scope, method, metric), replicates in sorted(replicate_values.items()):
        method_rows.append(
            {
                "scope": scope,
                "method": method,
                "metric": metric,
                "point_estimate": point_source[(scope, method, metric)],
                "ci95_low": percentile(replicates, 0.025),
                "ci95_high": percentile(replicates, 0.975),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            }
        )

    comparisons = (
        ("A", "A0"),
        ("CO", "A0"),
        ("K", "A"),
        ("T", "A"),
        ("T", "K"),
        ("T", "A0"),
    )
    difference_rows: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
            for metric in METRICS:
                replicates = [
                    left_value - right_value
                    for left_value, right_value in zip(
                        replicate_values[(scope, left, metric)],
                        replicate_values[(scope, right, metric)],
                    )
                ]
                difference_rows.append(
                    {
                        "scope": scope,
                        "comparison": f"{left}-{right}",
                        "metric": metric,
                        "point_estimate": (
                            point_source[(scope, left, metric)]
                            - point_source[(scope, right, metric)]
                        ),
                        "ci95_low": percentile(replicates, 0.025),
                        "ci95_high": percentile(replicates, 0.975),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return method_rows, difference_rows


def report_markdown(
    folds: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
    differences: Sequence[dict[str, Any]],
    lambdas: dict[str, float],
    predictions: Sequence[dict[str, Any]],
) -> str:
    fold_lookup = {(row["held_out_source"], row["method"]): row for row in folds}
    aggregate_lookup = {row["method"]: row for row in aggregates}
    difference_lookup = {
        (row["scope"], row["comparison"], row["metric"]): row for row in differences
    }
    lines = [
        "# Non-semantic future-3 LODO results",
        "",
        "No LLM, text encoder, external API, or outer-test hyperparameter selection was used.",
        "",
        "## Campaign-macro NDCG@5",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {fold_lookup[('ctid', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{fold_lookup[('attack_flow', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{fold_lookup[('stockpile', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{aggregate_lookup[method]['source_equal_campaign_macro_ndcg5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Source-equal campaign-macro metrics",
            "",
            "| Method | NDCG@5 | Hit@5 | Precision@5 | Recall@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = aggregate_lookup[method]
        lines.append(
            f"| {method} | {row['source_equal_campaign_macro_ndcg5']:.4f} | "
            f"{row['source_equal_campaign_macro_hit5']:.4f} | "
            f"{row['source_equal_campaign_macro_precision5']:.4f} | "
            f"{row['source_equal_campaign_macro_recall5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Soft-tactic inner selection",
            "",
            "| Held-out source | Chosen lambda |",
            "|---|---:|",
        ]
    )
    for source in SOURCES:
        lines.append(f"| {source} | {lambdas[source]:.1f} |")
    lines.extend(
        [
            "",
            "## Primary paired differences",
            "",
            "| Scope | T - A NDCG@5 | 95% campaign-bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for scope in (*SOURCES, "source_equal_overall"):
        row = difference_lookup[(scope, "T-A", "ndcg5")]
        lines.append(
            f"| {scope} | {row['point_estimate']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    co_fallback = sum(
        int(row["co_fallback_to_a0"])
        for row in predictions
        if row["method"] == "CO"
    )
    transition_counts = Counter(
        row["transition_visibility"] for row in predictions if row["method"] == "A"
    )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- CO fell back to A0 for {co_fallback}/784 test rows.",
            f"- Transition-visibility rows: {dict(sorted(transition_counts.items()))}.",
            "- `stratified_results.csv` contains transition, target-label, text-length, "
            "and target-cardinality strata; cells under 5 campaigns or 20 rows are marked descriptive-only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = [
        "predictions.csv",
        "inner_lambda_selection.csv",
        "campaign_results.csv",
        "fold_results.csv",
        "aggregate_results.csv",
        "stratified_results.csv",
        "bootstrap_method_metrics.csv",
        "bootstrap_differences.csv",
        "results_manifest.json",
        "report.md",
        "stdout.log",
    ]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing results: {existing}")

    rows = parse_samples()
    labels, label_index = parse_vocabulary()
    candidate_tactics, tactic_by_label = parse_tactics()
    predictions: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    chosen_lambdas: dict[str, float] = {}
    for held_out in SOURCES:
        fold_predictions, fold_inner, chosen = evaluate_fold(
            held_out,
            rows,
            labels,
            label_index,
            candidate_tactics,
            tactic_by_label,
        )
        predictions.extend(fold_predictions)
        inner_rows.extend(fold_inner)
        chosen_lambdas[held_out] = chosen

    predictions.sort(key=lambda row: (row["held_out_source"], row["method"], row["sample_id"]))
    inner_rows.sort(
        key=lambda row: (
            row["held_out_source"],
            row["inner_validation_source"],
            float(row["lambda"]),
        )
    )
    campaigns = campaign_results(predictions)
    folds = fold_results(predictions, campaigns)
    aggregates = aggregate_results(folds)
    stratified = stratified_results(predictions)
    bootstrap_methods, bootstrap_differences = bootstrap(campaigns, aggregates)

    prediction_columns = [
        "held_out_source",
        "method",
        "sample_id",
        "campaign_id",
        "prefix_len",
        "target_parent_ids",
        "target_size",
        "transition_visibility",
        "target_label_visibility",
        "text_length_group",
        "chosen_tactic_lambda",
        "co_fallback_to_a0",
        "top20_ids",
        "top20_scores",
        *METRICS,
    ]
    inner_columns = [
        "held_out_source",
        "outer_training_sources",
        "inner_validation_source",
        "lambda",
        "campaign_macro_ndcg5",
        "source_equal_mean_ndcg5",
        "selected",
        "training_rows",
        "validation_rows",
    ]
    campaign_columns = ["held_out_source", "method", "campaign_id", "rows", *METRICS]
    fold_columns = [
        "held_out_source",
        "method",
        "rows",
        "campaigns",
        *[f"campaign_macro_{metric}" for metric in METRICS],
        *[f"row_micro_{metric}" for metric in METRICS],
    ]
    aggregate_columns = [
        "method",
        "source_folds",
        *[f"source_equal_campaign_macro_{metric}" for metric in METRICS],
        *[f"all_row_micro_{metric}" for metric in METRICS],
    ]
    stratum_columns = [
        "held_out_source",
        "method",
        "stratum",
        "group",
        "rows",
        "campaigns",
        "inferentially_eligible",
        *[f"campaign_macro_{metric}" for metric in METRICS],
    ]
    bootstrap_columns = [
        "scope",
        "method",
        "metric",
        "point_estimate",
        "ci95_low",
        "ci95_high",
        "replicates",
        "seed",
    ]
    difference_columns = [
        "scope",
        "comparison",
        "metric",
        "point_estimate",
        "ci95_low",
        "ci95_high",
        "replicates",
        "seed",
    ]
    write_csv(output / "predictions.csv", predictions, prediction_columns)
    write_csv(output / "inner_lambda_selection.csv", inner_rows, inner_columns)
    write_csv(output / "campaign_results.csv", campaigns, campaign_columns)
    write_csv(output / "fold_results.csv", folds, fold_columns)
    write_csv(output / "aggregate_results.csv", aggregates, aggregate_columns)
    write_csv(output / "stratified_results.csv", stratified, stratum_columns)
    write_csv(output / "bootstrap_method_metrics.csv", bootstrap_methods, bootstrap_columns)
    write_csv(output / "bootstrap_differences.csv", bootstrap_differences, difference_columns)
    report = report_markdown(
        folds,
        aggregates,
        bootstrap_differences,
        chosen_lambdas,
        predictions,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    stdout_text = report + f"\nWrote results to {output}\n"
    (output / "stdout.log").write_text(stdout_text, encoding="utf-8")

    output_hashes = {
        name: sha256(output / name)
        for name in managed
        if name != "results_manifest.json" and (output / name).exists()
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "method_card": {
            "path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(METHOD_CARD),
        },
        "v8_1_addendum": {
            "path": ADDENDUM.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(ADDENDUM),
        },
        "inputs": {
            "samples_sha256": sha256(SAMPLES_PATH),
            "tactic_mapping_sha256": sha256(TACTIC_MAP_PATH),
            "vocabulary_sha256": sha256(VOCAB_PATH),
            "main_rows": len(rows),
            "source_counts": dict(Counter(row["source"] for row in rows)),
        },
        "parameters": {
            "methods": list(METHODS),
            "alpha_s": ALPHA,
            "layer_weights": list(LAYER_WEIGHTS),
            "lambda_grid": list(LAMBDA_GRID),
            "chosen_lambda_by_held_out_source": chosen_lambdas,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "outputs_sha256": output_hashes,
    }
    write_json(output / "results_manifest.json", manifest)
    print(stdout_text, end="")


if __name__ == "__main__":
    main()
