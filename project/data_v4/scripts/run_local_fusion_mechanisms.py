#!/usr/bin/env python3
"""Run frozen F1--F4 local fusion mechanisms and mandatory controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import local_fusion_common as C


PROTOCOL = C.PROJECT_ROOT / "data_v4/protocols/local_fusion_mechanism_search_v1.md"
ADDENDUM = C.PROJECT_ROOT / "data_v4/protocols/local_fusion_implementation_addendum_v1.md"
ORACLE = C.PROJECT_ROOT / "data_v4/local_fusion_search/oracles_v1/oracle_summary.csv"
DEFAULT_OUTPUT = C.PROJECT_ROOT / "data_v4/local_fusion_search/mechanisms_v1"
L2_GRID = (0.001, 0.01, 0.1, 1.0)
F2_GRID = tuple(
    (experts, support, margin, weight2)
    for experts in (3, 2)
    for support in (2, 1, 0)
    for margin in (1.0, 0.5, 0.25, 0.0)
    for weight2 in (0.0, 0.5, 1.0)
)
GAMMA_GRID = (1.64, 1.0, 0.5, 0.0)
ACTION_ORDER = ("identity", "reorder_A", "reorder_T", "reorder_K", "replace_A", "replace_T", "replace_K")


def campaign_macro(predictions: Sequence[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        grouped[row["campaign_id"]].append(float(row["ndcg5"]))
    return statistics.fmean(statistics.fmean(values) for values in grouped.values())


def prepare_folds(
    rows: Sequence[dict[str, Any]], labels: Sequence[str], label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]], tactic_by_label: dict[str, tuple[int, ...]],
    b0: dict[str, tuple[str, ...]], lambdas: dict[str, float],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for held_out in C.SOURCES:
        outer_train = [row for row in rows if row["source"] != held_out]
        outer_test = [row for row in rows if row["source"] == held_out]
        tactic_lambda = lambdas[held_out]
        train_bundles = C.campaign_loo_bundles(outer_train, labels, label_index, candidate_tactics, tactic_by_label, b0, tactic_lambda)
        test_bundles = C.build_eval_bundles(outer_train, outer_test, labels, label_index, candidate_tactics, tactic_by_label, b0, tactic_lambda)
        inner = []
        for validation_source in sorted({row["source"] for row in outer_train}):
            inner_train_rows = [row for row in outer_train if row["source"] != validation_source]
            validation_rows = [row for row in outer_train if row["source"] == validation_source]
            inner.append({
                "validation_source": validation_source,
                "train": C.campaign_loo_bundles(inner_train_rows, labels, label_index, candidate_tactics, tactic_by_label, b0, tactic_lambda),
                "validation": C.build_eval_bundles(inner_train_rows, validation_rows, labels, label_index, candidate_tactics, tactic_by_label, b0, tactic_lambda),
            })
        output[held_out] = {"train": train_bundles, "test": test_bundles, "inner": inner}
    return output


def sample_value(bundle: dict[str, Any], name: str, variant: str, index: int) -> float:
    if variant == "equal_capacity":
        return C.pseudo_value(bundle["row"]["sample_id"], "__sample__", index)
    return float(bundle["evidence"][name])


def f1_features(bundle: dict[str, Any], label: str, variant: str, labels: Sequence[str]) -> list[float]:
    b0_rank = bundle["b0"].index(label) + 1
    one_hot = [float(b0_rank == rank) for rank in range(1, 6)]
    if variant == "equal_capacity":
        counts = [C.pseudo_value(bundle["row"]["sample_id"], label, 20 + index) for index in range(6)]
    else:
        stats = bundle["evidence"]["stats"][label]
        counts = [math.log1p(bundle["evidence"]["n1"]), math.log1p(bundle["evidence"]["n2"]), math.log1p(stats["k1"]), math.log1p(stats["k2"]), stats["s1"], stats["s2"]]
    e1 = C.candidate_stat(bundle, label, "e1", variant, 30)
    e2 = C.candidate_stat(bundle, label, "e2", variant, 31)
    rank_maps = {expert: C.rank_map(bundle, expert, labels, variant) for expert in C.EXPERTS}
    reciprocal = [1.0 / rank_maps[expert][label] for expert in C.EXPERTS]
    votes = float(sum(rank_maps[expert][label] <= 5 for expert in C.EXPERTS))
    entropy = sample_value(bundle, "entropy", variant, 32)
    prefix = math.log1p(bundle["row"]["prefix_len"])
    rr = 1.0 / b0_rank
    return [*one_hot, *counts, e1, e2, *reciprocal, votes, entropy, prefix, rr * e1, rr * e2, rr * votes]


def train_f1(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], l2: float) -> tuple[list[float], list[float], list[float]]:
    raw = [f1_features(bundle, label, variant, labels) for bundle in bundles for label in bundle["b0"]]
    means, scales = C.normalize_fit(raw)
    examples = []
    for bundle in bundles:
        vectors = {label: [1.0, *C.normalize(f1_features(bundle, label, variant, labels), means, scales)] for label in bundle["b0"]}
        truth = set(bundle["row"]["targets"])
        positives = [label for label in bundle["b0"] if label in truth]
        negatives = [label for label in bundle["b0"] if label not in truth]
        examples.extend((vectors[pos], vectors[neg]) for pos in positives for neg in negatives)
    dimension = len(raw[0]) + 1
    weights = [0.0] * dimension
    m = [0.0] * dimension
    v = [0.0] * dimension
    for epoch in range(1, 81):
        gradient = [0.0] * dimension
        for positive, negative in examples:
            difference = sum(weights[index] * (positive[index] - negative[index]) for index in range(dimension))
            factor = -C.sigmoid(-difference)
            for index in range(dimension):
                gradient[index] += factor * (positive[index] - negative[index])
        denominator = max(1, len(examples))
        for index in range(dimension):
            gradient[index] /= denominator
            if index > 5:
                gradient[index] += l2 * weights[index]
            m[index] = 0.9 * m[index] + 0.1 * gradient[index]
            v[index] = 0.999 * v[index] + 0.001 * gradient[index] ** 2
            mh = m[index] / (1 - 0.9**epoch)
            vh = v[index] / (1 - 0.999**epoch)
            weights[index] -= 0.03 * mh / (math.sqrt(vh) + 1e-8)
    return weights, means, scales


def predict_f1(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], model: tuple[list[float], list[float], list[float]], method: str) -> list[dict[str, Any]]:
    weights, means, scales = model
    output = []
    for bundle in bundles:
        original = {label: index for index, label in enumerate(bundle["b0"])}
        values = {}
        for label in bundle["b0"]:
            vector = [1.0, *C.normalize(f1_features(bundle, label, variant, labels), means, scales)]
            values[label] = sum(weight * value for weight, value in zip(weights, vector))
        ranked = sorted(bundle["b0"], key=lambda label: (-values[label], original[label]))
        output.append(C.prediction_row(method, bundle, ranked))
    return output


def select_f1(fold: dict[str, Any], variant: str, labels: Sequence[str]) -> tuple[float, list[dict[str, Any]]]:
    details = []
    means: dict[float, list[float]] = defaultdict(list)
    for split in fold["inner"]:
        train = C.variant_bundles(split["train"], variant)
        validation = C.variant_bundles(split["validation"], variant)
        for l2 in L2_GRID:
            model = train_f1(train, variant, labels, l2)
            value = campaign_macro(predict_f1(validation, variant, labels, model, "F1"))
            means[l2].append(value)
            details.append({"validation_source": split["validation_source"], "l2": l2, "campaign_macro_ndcg5": value})
    averaged = {value: statistics.fmean(scores) for value, scores in means.items()}
    best = max(averaged.values())
    selected = max(value for value, score in averaged.items() if abs(score - best) <= 1e-12)
    for row in details:
        row["source_equal_mean"] = averaged[row["l2"]]
        row["selected"] = int(row["l2"] == selected)
    return selected, details


def f2_rank(bundle: dict[str, Any], variant: str, labels: Sequence[str], config: tuple[int, int, float, float]) -> tuple[list[str], bool]:
    min_experts, min_support, margin, weight2 = config
    maps = {expert: C.rank_map(bundle, expert, labels, variant) for expert in C.EXPERTS}
    existing = set(bundle["b0"])
    candidates = []
    for label in labels:
        if label in existing:
            continue
        expert_count = sum(maps[expert][label] <= 10 for expert in C.EXPERTS)
        if expert_count < min_experts:
            continue
        support = max(C.candidate_stat(bundle, label, "s1", variant, 40), C.candidate_stat(bundle, label, "s2", variant, 41))
        if support < min_support:
            continue
        rr = sum(1.0 / maps[expert][label] for expert in C.EXPERTS if maps[expert][label] <= 10)
        value = rr + C.candidate_stat(bundle, label, "e1", variant, 42) + weight2 * C.candidate_stat(bundle, label, "e2", variant, 43)
        candidates.append((value, expert_count, support, label))
    rank5 = bundle["b0"][4]
    rank5_value = sum(1.0 / maps[expert][rank5] for expert in C.EXPERTS if maps[expert][rank5] <= 10) + C.candidate_stat(bundle, rank5, "e1", variant, 42) + weight2 * C.candidate_stat(bundle, rank5, "e2", variant, 43)
    if not candidates:
        return list(bundle["b0"]), False
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    best = candidates[0]
    if best[0] <= rank5_value + margin:
        return list(bundle["b0"]), False
    return [*bundle["b0"][:4], best[3]], True


def predict_f2(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], config: tuple[int, int, float, float], method: str) -> list[dict[str, Any]]:
    output = []
    for bundle in bundles:
        ranked, replaced = f2_rank(bundle, variant, labels, config)
        output.append(C.prediction_row(method, bundle, ranked, {"action_applied": int(replaced)}))
    return output


def select_grid(fold: dict[str, Any], variant: str, labels: Sequence[str], grid: Sequence[Any], predictor: Any, method: str) -> tuple[Any, list[dict[str, Any]]]:
    details = []
    values: dict[Any, list[float]] = defaultdict(list)
    for split in fold["inner"]:
        train = C.variant_bundles(split["train"], variant)
        validation = C.variant_bundles(split["validation"], variant)
        for config in grid:
            predictions = predictor(train, validation, variant, labels, config, method)
            score = campaign_macro(predictions)
            values[config].append(score)
            details.append({"validation_source": split["validation_source"], "config": C.compact_json(config), "campaign_macro_ndcg5": score})
    averaged = {config: statistics.fmean(scores) for config, scores in values.items()}
    best = max(averaged.values())
    selected = next(config for config in grid if abs(averaged[config] - best) <= 1e-12)
    for row in details:
        config = json.loads(row["config"])
        key = tuple(config) if isinstance(config, list) else config
        row["source_equal_mean"] = averaged[key]
        row["selected"] = int(key == selected)
    return selected, details


def f2_predictor(_train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], config: Any, method: str) -> list[dict[str, Any]]:
    return predict_f2(validation, variant, labels, tuple(config), method)


def rbo(left: Sequence[str], right: Sequence[str], p: float = 0.9) -> float:
    return (1 - p) * sum(len(set(left[:depth]) & set(right[:depth])) / depth * p ** (depth - 1) for depth in range(1, 6))


def f3_features(bundle: dict[str, Any], variant: str, labels: Sequence[str]) -> list[float]:
    rankings = {expert: tuple(C.rank_map(bundle, expert, labels, variant).keys()) for expert in ()}  # type anchor
    rankings = {expert: tuple(sorted(labels, key=C.rank_map(bundle, expert, labels, variant).get)) for expert in C.EXPERTS}
    b0 = bundle["b0"]
    jaccards = [len(set(b0) & set(rankings[expert][:5])) / len(set(b0) | set(rankings[expert][:5])) for expert in C.EXPERTS]
    rbos = [rbo(b0, rankings[expert][:5]) for expert in C.EXPERTS]
    mutual = []
    for left, right in (("A", "T"), ("A", "K"), ("T", "K")):
        mutual.append(len(set(rankings[left][:5]) & set(rankings[right][:5])) / len(set(rankings[left][:5]) | set(rankings[right][:5])))
    if variant == "equal_capacity":
        n1 = C.pseudo_value(bundle["row"]["sample_id"], "__sample__", 50)
        n2 = C.pseudo_value(bundle["row"]["sample_id"], "__sample__", 51)
        entropy = C.pseudo_value(bundle["row"]["sample_id"], "__sample__", 52)
    else:
        n1 = math.log1p(bundle["evidence"]["n1"])
        n2 = math.log1p(bundle["evidence"]["n2"])
        entropy = bundle["evidence"]["entropy"]
    support1 = statistics.fmean(float(C.candidate_stat(bundle, label, "k1", variant, 53) > 0) for label in b0)
    support2 = statistics.fmean(float(C.candidate_stat(bundle, label, "k2", variant, 54) > 0) for label in b0)
    both = statistics.fmean(float(C.candidate_stat(bundle, label, "s1", variant, 55) >= 2) for label in b0)
    e1 = [C.candidate_stat(bundle, label, "e1", variant, 56) for label in b0]
    e2 = [C.candidate_stat(bundle, label, "e2", variant, 57) for label in b0]
    union = set(b0)
    for expert in C.EXPERTS:
        union.update(rankings[expert][:5])
    outside_votes = Counter(label for expert in C.EXPERTS for label in rankings[expert][:5] if label not in set(b0))
    return [n1, n2, entropy, math.log1p(bundle["row"]["prefix_len"]), float(len(union)), *jaccards, *rbos, *mutual, support1, support2, both, statistics.fmean(e1), max(e1), statistics.fmean(e2), max(e2), float(max(outside_votes.values(), default=0))]


def best_expert_label(bundle: dict[str, Any], variant: str, labels: Sequence[str]) -> int:
    order = ("B0", "T", "K", "A")
    rankings = {"B0": bundle["b0"]}
    for expert in C.EXPERTS:
        rank = C.rank_map(bundle, expert, labels, variant)
        rankings[expert] = tuple(sorted(labels, key=rank.get))[:5]
    values = {name: C.sample_metric(rankings[name], bundle["row"])["ndcg5"] for name in order}
    best = max(values.values())
    return order.index(next(name for name in order if abs(values[name] - best) <= 1e-12))


def train_f3(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], l2: float) -> tuple[list[list[float]], list[float], list[float]]:
    raw = [f3_features(bundle, variant, labels) for bundle in bundles]
    means, scales = C.normalize_fit(raw)
    vectors = [[1.0, *C.normalize(vector, means, scales)] for vector in raw]
    targets = [best_expert_label(bundle, variant, labels) for bundle in bundles]
    dimension = len(vectors[0])
    weights = [[0.0] * dimension for _ in range(4)]
    m = [[0.0] * dimension for _ in range(4)]
    v = [[0.0] * dimension for _ in range(4)]
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, bundle in enumerate(bundles):
        source_indices[bundle["row"]["source"]].append(index)
    for epoch in range(1, 101):
        gradient = [[0.0] * dimension for _ in range(4)]
        for indices in source_indices.values():
            source_weight = 1.0 / (len(source_indices) * len(indices))
            for index in indices:
                probabilities = C.softmax([sum(weight * value for weight, value in zip(row, vectors[index])) for row in weights])
                for expert in range(4):
                    factor = (probabilities[expert] - float(targets[index] == expert)) * source_weight
                    for feature in range(dimension):
                        gradient[expert][feature] += factor * vectors[index][feature]
        for expert in range(4):
            for feature in range(dimension):
                if feature:
                    gradient[expert][feature] += l2 * weights[expert][feature]
                m[expert][feature] = 0.9 * m[expert][feature] + 0.1 * gradient[expert][feature]
                v[expert][feature] = 0.999 * v[expert][feature] + 0.001 * gradient[expert][feature] ** 2
                mh = m[expert][feature] / (1 - 0.9**epoch)
                vh = v[expert][feature] / (1 - 0.999**epoch)
                weights[expert][feature] -= 0.03 * mh / (math.sqrt(vh) + 1e-8)
    return weights, means, scales


def predict_f3(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], model: tuple[list[list[float]], list[float], list[float]], method: str) -> list[dict[str, Any]]:
    weights, means, scales = model
    order = ("B0", "T", "K", "A")
    output = []
    for bundle in bundles:
        vector = [1.0, *C.normalize(f3_features(bundle, variant, labels), means, scales)]
        logits = [sum(weight * value for weight, value in zip(row, vector)) for row in weights]
        choice = max(range(4), key=lambda index: (logits[index], -index))
        name = order[choice]
        if name == "B0":
            ranking = bundle["b0"]
        else:
            ranks = C.rank_map(bundle, name, labels, variant)
            ranking = tuple(sorted(labels, key=ranks.get))[:5]
        output.append(C.prediction_row(method, bundle, ranking, {"selected_expert": name, "action_applied": int(name != "B0")}))
    return output


def select_f3(fold: dict[str, Any], variant: str, labels: Sequence[str]) -> tuple[float, list[dict[str, Any]]]:
    details = []
    values: dict[float, list[float]] = defaultdict(list)
    for split in fold["inner"]:
        train = C.variant_bundles(split["train"], variant)
        validation = C.variant_bundles(split["validation"], variant)
        for l2 in L2_GRID:
            model = train_f3(train, variant, labels, l2)
            score = campaign_macro(predict_f3(validation, variant, labels, model, "F3"))
            values[l2].append(score)
            details.append({"validation_source": split["validation_source"], "l2": l2, "campaign_macro_ndcg5": score})
    averaged = {value: statistics.fmean(scores) for value, scores in values.items()}
    best = max(averaged.values())
    selected = max(value for value, score in averaged.items() if abs(score - best) <= 1e-12)
    for row in details:
        row["source_equal_mean"] = averaged[row["l2"]]
        row["selected"] = int(row["l2"] == selected)
    return selected, details


def action_rankings(bundle: dict[str, Any], variant: str, labels: Sequence[str]) -> dict[str, tuple[str, ...]]:
    output = {"identity": bundle["b0"]}
    existing = set(bundle["b0"])
    for expert in C.EXPERTS:
        ranks = C.rank_map(bundle, expert, labels, variant)
        full = tuple(sorted(labels, key=ranks.get))
        original = {label: index for index, label in enumerate(bundle["b0"])}
        output[f"reorder_{expert}"] = tuple(sorted(bundle["b0"], key=lambda label: (ranks[label], original[label])))
        outside = next((label for label in full if label not in existing), None)
        output[f"replace_{expert}"] = bundle["b0"] if outside is None else (*bundle["b0"][:4], outside)
    return output


def action_cell(bundle: dict[str, Any], variant: str, labels: Sequence[str], action: str, ranking: Sequence[str]) -> tuple[Any, ...]:
    if action == "identity":
        return (action, 0, 0, "na", "na")
    expert = action.split("_", 1)[1]
    ranks = C.rank_map(bundle, expert, labels, variant)
    expert_top = set(sorted(labels, key=ranks.get)[:5])
    jac = len(set(bundle["b0"]) & expert_top) / len(set(bundle["b0"]) | expert_top)
    jac_bin = "lt025" if jac < 0.25 else ("le050" if jac <= 0.5 else "gt050")
    n2 = int(sample_value(bundle, "n2", variant, 60) > 0)
    two_source = int(any(C.candidate_stat(bundle, label, "s1", variant, 61) >= 2 for label in bundle["b0"]))
    outside_sign = "na"
    if action.startswith("replace_") and ranking[4] not in set(bundle["b0"]):
        outside_sign = "pos" if C.candidate_stat(bundle, ranking[4], "e1", variant, 62) > 0 else "nonpos"
    return (action, n2, two_source, jac_bin, outside_sign)


def fit_f4_policy(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], gamma: float) -> dict[tuple[Any, ...], float]:
    per_campaign: dict[tuple[tuple[Any, ...], str, str], list[float]] = defaultdict(list)
    for bundle in bundles:
        actions = action_rankings(bundle, variant, labels)
        base = C.sample_metric(bundle["b0"], bundle["row"])["ndcg5"]
        for action in ACTION_ORDER[1:]:
            cell = action_cell(bundle, variant, labels, action, actions[action])
            delta = C.sample_metric(actions[action], bundle["row"])["ndcg5"] - base
            per_campaign[(cell, bundle["row"]["source"], bundle["row"]["campaign_id"])].append(delta)
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (cell, _source, _campaign), values in per_campaign.items():
        grouped[cell].append(statistics.fmean(values))
    policy = {}
    for cell, values in grouped.items():
        if len(values) < 3:
            continue
        sd = statistics.stdev(values)
        lower = statistics.fmean(values) - gamma * sd / math.sqrt(len(values))
        if lower > 0:
            policy[cell] = lower
    return policy


def predict_f4(bundles: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], policy: dict[tuple[Any, ...], float], method: str) -> list[dict[str, Any]]:
    output = []
    for bundle in bundles:
        actions = action_rankings(bundle, variant, labels)
        candidates = []
        for order, action in enumerate(ACTION_ORDER[1:], start=1):
            cell = action_cell(bundle, variant, labels, action, actions[action])
            lower = policy.get(cell, 0.0)
            if lower > 0:
                candidates.append((lower, -order, action))
        action = max(candidates)[2] if candidates else "identity"
        output.append(C.prediction_row(method, bundle, actions[action], {"selected_action": action, "action_applied": int(action != "identity")}))
    return output


def f4_predictor(train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]], variant: str, labels: Sequence[str], gamma: float, method: str) -> list[dict[str, Any]]:
    return predict_f4(validation, variant, labels, fit_f4_policy(train, variant, labels, gamma), method)


def method_name(base: str, variant: str) -> str:
    suffix = {"main": "", "permuted": "-Perm", "no_prior": "-NoPrior", "equal_capacity": "-Equal"}[variant]
    return base + suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows, labels, label_index, candidate_tactics, tactic_by_label, b0, lambdas = C.load_inputs()
    folds = prepare_folds(rows, labels, label_index, candidate_tactics, tactic_by_label, b0, lambdas)
    predictions: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for held_out in C.SOURCES:
        fold = folds[held_out]
        for bundle in fold["test"]:
            predictions.append(C.prediction_row("B0", bundle, bundle["b0"]))
        for variant in C.VARIANTS:
            train = C.variant_bundles(fold["train"], variant)
            test = C.variant_bundles(fold["test"], variant)

            l2, details = select_f1(fold, variant, labels)
            model1 = train_f1(train, variant, labels, l2)
            predictions.extend(predict_f1(test, variant, labels, model1, method_name("F1", variant)))
            inner_rows.extend({"held_out_source": held_out, "mechanism": "F1", "variant": variant, **row} for row in details)
            parameter_rows.append({"held_out_source": held_out, "mechanism": "F1", "variant": variant, "selected": l2})

            config2, details = select_grid(fold, variant, labels, F2_GRID, f2_predictor, "F2")
            predictions.extend(predict_f2(test, variant, labels, config2, method_name("F2", variant)))
            inner_rows.extend({"held_out_source": held_out, "mechanism": "F2", "variant": variant, **row} for row in details)
            parameter_rows.append({"held_out_source": held_out, "mechanism": "F2", "variant": variant, "selected": C.compact_json(config2)})

            l2, details = select_f3(fold, variant, labels)
            model3 = train_f3(train, variant, labels, l2)
            predictions.extend(predict_f3(test, variant, labels, model3, method_name("F3", variant)))
            inner_rows.extend({"held_out_source": held_out, "mechanism": "F3", "variant": variant, **row} for row in details)
            parameter_rows.append({"held_out_source": held_out, "mechanism": "F3", "variant": variant, "selected": l2})

            gamma, details = select_grid(fold, variant, labels, GAMMA_GRID, f4_predictor, "F4")
            policy = fit_f4_policy(train, variant, labels, gamma)
            predictions.extend(predict_f4(test, variant, labels, policy, method_name("F4", variant)))
            inner_rows.extend({"held_out_source": held_out, "mechanism": "F4", "variant": variant, **row} for row in details)
            parameter_rows.append({"held_out_source": held_out, "mechanism": "F4", "variant": variant, "selected": gamma, "policy_cells": len(policy)})

    expected_methods = {"B0"} | {method_name(base, variant) for base in ("F1", "F2", "F3", "F4") for variant in C.VARIANTS}
    counts = Counter(row["method"] for row in predictions)
    if set(counts) != expected_methods or any(counts[method] != 784 for method in expected_methods):
        raise AssertionError(f"prediction denominator gate failed: {counts}")
    campaigns = C.campaign_rows(predictions)
    fold_results = C.fold_rows(campaigns)
    comparisons = []
    for base in ("F1", "F2", "F3", "F4"):
        comparisons.extend((base, other) for other in ("B0", f"{base}-Perm", f"{base}-NoPrior", f"{base}-Equal"))
    differences = C.bootstrap(campaigns, comparisons)
    influence = C.ctid_influence(campaigns, ("F1", "F2", "F3", "F4"))

    fold_lookup = {(row["held_out_source"], row["method"]): row for row in fold_results}
    diff_lookup = {(row["comparison"], row["metric"]): row for row in differences}
    decisions = []
    for base in ("F1", "F2", "F3", "F4"):
        main_b0 = diff_lookup[(f"{base}-B0", "ndcg5")]
        main_perm = diff_lookup[(f"{base}-{base}-Perm", "ndcg5")]
        real_positive = all(float(fold_lookup[source, base]["campaign_macro_ndcg5"]) > float(fold_lookup[source, "B0"]["campaign_macro_ndcg5"]) for source in ("ctid", "attack_flow"))
        stable = not any(row["method"] == base and int(row["sign_reversed"]) for row in influence)
        extra_dead = False
        if base == "F2":
            rate = statistics.fmean(float(row.get("action_applied", 0)) for row in predictions if row["method"] == base)
            extra_dead = rate < 0.01 or rate > 0.90
        elif base == "F3":
            selected = [row for row in predictions if row["method"] == base]
            extra_dead = statistics.fmean(float(row.get("action_applied", 0)) for row in selected) < 0.01 or any(not any(row["held_out_source"] == source and float(row.get("action_applied", 0)) for row in selected) for source in ("ctid", "attack_flow"))
        elif base == "F4":
            selected_campaigns = {source: {row["campaign_id"] for row in predictions if row["method"] == base and row["held_out_source"] == source and float(row.get("action_applied", 0))} for source in ("ctid", "attack_flow")}
            extra_dead = any(len(selected_campaigns[source]) < 3 for source in selected_campaigns)
        passed = float(main_b0["ci95_low"]) > 0 and real_positive and float(main_perm["ci95_low"]) > 0 and stable and not extra_dead
        decisions.append({"mechanism": base, "delta_vs_b0": main_b0["point_estimate"], "ci95_low": main_b0["ci95_low"], "ci95_high": main_b0["ci95_high"], "real_sources_positive": int(real_positive), "permutation_significant": int(float(main_perm["ci95_low"]) > 0), "ctid_loo_stable": int(stable), "extra_dead_condition": int(extra_dead), "discovery_positive": int(passed)})

    output.mkdir(parents=True, exist_ok=False)
    prediction_columns = ("held_out_source", "method", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "top5_ids", "ndcg5", "hit5", "precision5", "recall5", "action_applied", "selected_expert", "selected_action")
    C.write_csv(output / "predictions.csv", predictions, prediction_columns)
    C.write_csv(output / "campaign_results.csv", campaigns, ("held_out_source", "method", "campaign_id", "rows", "ndcg5", "hit5", "precision5", "recall5"))
    C.write_csv(output / "fold_results.csv", fold_results, ("held_out_source", "method", "campaigns", "campaign_macro_ndcg5", "campaign_macro_hit5", "campaign_macro_precision5", "campaign_macro_recall5"))
    C.write_csv(output / "bootstrap_differences.csv", differences, ("comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"))
    C.write_csv(output / "ctid_influence.csv", influence, ("method", "removed_ctid_campaign", "full_delta", "leave_one_out_delta", "sign_reversed"))
    C.write_csv(output / "inner_selection.csv", inner_rows, sorted({key for row in inner_rows for key in row}))
    C.write_csv(output / "parameters.csv", parameter_rows, ("held_out_source", "mechanism", "variant", "selected", "policy_cells"))
    C.write_csv(output / "decisions.csv", decisions, ("mechanism", "delta_vs_b0", "ci95_low", "ci95_high", "real_sources_positive", "permutation_significant", "ctid_loo_stable", "extra_dead_condition", "discovery_positive"))
    lines = ["# Zero-cost local fusion mechanisms", "", "Four implemented mechanisms; F5 was dropped at the oracle gate. No network/API call.", "", "| Mechanism | CTID | Attack Flow | Stockpile | Overall | Delta B0 (95% CI) | Decision |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for item in decisions:
        base = item["mechanism"]
        values = [float(fold_lookup[source, base]["campaign_macro_ndcg5"]) for source in C.SOURCES]
        overall = statistics.fmean(values)
        lines.append(f"| {base} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {overall:.4f} | {float(item['delta_vs_b0']):+.4f} [{float(item['ci95_low']):+.4f},{float(item['ci95_high']):+.4f}] | {'PASS' if item['discovery_positive'] else 'FAIL'} |")
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text(report, encoding="utf-8")
    managed = ("predictions.csv", "campaign_results.csv", "fold_results.csv", "bootstrap_differences.csv", "ctid_influence.csv", "inner_selection.csv", "parameters.csv", "decisions.csv", "report.md", "stdout.log")
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "analysis_status": "post-result discovery search", "network_calls": 0, "api_cost": 0, "mechanisms_explored": 5, "mechanisms_implemented": 4, "oracle_dropped": ["F5"], "protocol": {"path": PROTOCOL.relative_to(C.PROJECT_ROOT).as_posix(), "sha256": C.sha256(PROTOCOL)}, "addendum": {"path": ADDENDUM.relative_to(C.PROJECT_ROOT).as_posix(), "sha256": C.sha256(ADDENDUM)}, "script": {"path": Path(__file__).relative_to(C.PROJECT_ROOT).as_posix(), "sha256": C.sha256(Path(__file__))}, "common_script_sha256": C.sha256(Path(C.__file__)), "inputs": {path.relative_to(C.PROJECT_ROOT).as_posix(): C.sha256(path) for path in (C.BASE.SAMPLES_PATH, C.B0_PATH, C.BASE_RESULTS, C.BASE.VOCAB_PATH, ORACLE)}, "outputs": {name: C.sha256(output / name) for name in managed}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
