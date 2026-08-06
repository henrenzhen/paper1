#!/usr/bin/env python3
"""Exploratory, unblinded feasibility screen on the frozen 30-row pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_INPUT = PROJECT_ROOT / "data_v4/external_reasoning/pilot/pilot_sample_30.csv"
RUN_DIR = (
    PROJECT_ROOT
    / "data_v4/external_reasoning/pilot/runs/20260806T081038Z_da72df77"
)
PILOT_RESULTS = RUN_DIR / "pilot_raw_results.csv"
OUTPUT_DIR = RUN_DIR / "exploratory_feasibility_screen_v1"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
SOURCE_PATHS = {
    "ctid": PROJECT_ROOT / "data_v2/repro_external/closed_set/ctid_in184.csv",
    "attack_flow": PROJECT_ROOT
    / "data_v2/repro_external/closed_set/attack_flow_cumulative_in184.csv",
    "stockpile": PROJECT_ROOT
    / "data_v2/repro_external/closed_set/stockpile_cumulative_in184.csv",
}

ALPHA = 0.1
EPS = 1e-12
LAMBDA_GRID = [i / 20 for i in range(21)]
RANK_WEIGHTS = [math.exp(-i) for i in range(5)]
RANK_WEIGHT_SUM = sum(RANK_WEIGHTS)
RANK_WEIGHTS = [value / RANK_WEIGHT_SUM for value in RANK_WEIGHTS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["source"], row["campaign_id"], int(row["prefix_len"])


def parse_sequence(row: dict[str, str]) -> list[str]:
    value = json.loads(row["prefix"])
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid prefix for {key(row)}")
    return [str(item) for item in value]


def parse_top5(row: dict[str, str], vocab: set[str]) -> list[str]:
    try:
        value = json.loads(row.get("predicted_next_ttps") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    items = [str(item) for item in value]
    if len(items) != 5 or len(set(items)) != 5 or any(x not in vocab for x in items):
        return []
    return items


def load_source(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = dict(raw)
        row["prefix_len"] = int(raw["prefix_len"])
        row["sequence"] = parse_sequence(raw)
        rows.append(row)
    return rows


class Backbone:
    def __init__(self, rows: list[dict[str, Any]], vocab: list[str]) -> None:
        self.vocab = vocab
        self.order2: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.order1: dict[str, Counter[str]] = defaultdict(Counter)
        self.unigram: Counter[str] = Counter()
        for row in rows:
            sequence = row["sequence"]
            target = row["true_label"]
            self.unigram[target] += 1
            self.order1[sequence[-1]][target] += 1
            if len(sequence) >= 2:
                self.order2[(sequence[-2], sequence[-1])][target] += 1
        self.unigram_probability = self._distribution(self.unigram)

    def _distribution(self, counts: Counter[str]) -> list[float]:
        denominator = sum(counts.values()) + ALPHA
        uniform = ALPHA / len(self.vocab)
        return [(counts.get(label, 0) + uniform) / denominator for label in self.vocab]

    def probability(self, sequence: list[str]) -> list[float]:
        layers: list[tuple[float, list[float]]] = []
        if len(sequence) >= 2:
            context2 = (sequence[-2], sequence[-1])
            if context2 in self.order2:
                layers.append((0.5, self._distribution(self.order2[context2])))
        if sequence[-1] in self.order1:
            layers.append((0.3, self._distribution(self.order1[sequence[-1]])))
        layers.append((0.2, self.unigram_probability))
        weight_sum = sum(weight for weight, _ in layers)
        return [
            sum(weight * distribution[index] for weight, distribution in layers)
            / weight_sum
            for index in range(len(self.vocab))
        ]


def prior_probability(rows: list[dict[str, Any]], vocab: list[str]) -> list[float]:
    counts = Counter(row["true_label"] for row in rows)
    denominator = len(rows) + ALPHA
    uniform = ALPHA / len(vocab)
    return [(counts.get(label, 0) + uniform) / denominator for label in vocab]


def rank_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def rank_labels(scores: list[float], vocab: list[str]) -> list[str]:
    return [vocab[index] for index in rank_indices(scores)]


def rank_of(ranking: list[str], target: str) -> int:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return 0


def rank_prior(top5: list[str], vocab: list[str]) -> list[float]:
    base = 0.1 / len(vocab)
    probability = [base for _ in vocab]
    index = {label: position for position, label in enumerate(vocab)}
    for rank, label in enumerate(top5):
        probability[index[label]] += 0.9 * RANK_WEIGHTS[rank]
    return probability


def fuse(
    backbone_probability: list[float],
    top5: list[str],
    lambda_value: float,
    vocab: list[str],
) -> list[float]:
    if not top5:
        return backbone_probability
    semantic_probability = rank_prior(top5, vocab)
    logits = [
        (1.0 - lambda_value) * math.log(a + EPS)
        + lambda_value * math.log(b + EPS)
        for a, b in zip(backbone_probability, semantic_probability)
    ]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def campaign_macro_top1(rows: list[dict[str, Any]], rank_field: str) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["campaign_id"]].append(1 if row[rank_field] == 1 else 0)
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def select_lambda(
    outer_train_sources: list[str],
    source_rows: dict[str, list[dict[str, Any]]],
    pilot_by_source: dict[str, list[dict[str, Any]]],
    vocab: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    mean_scores: dict[float, float] = {}
    for lambda_value in LAMBDA_GRID:
        fold_scores: list[float] = []
        for validation_source in outer_train_sources:
            training_source = next(
                source for source in outer_train_sources if source != validation_source
            )
            backbone = Backbone(source_rows[training_source], vocab)
            evaluated: list[dict[str, Any]] = []
            for row in pilot_by_source[validation_source]:
                probability = backbone.probability(row["sequence"])
                ranking = rank_labels(
                    fuse(probability, row["llm_top5"], lambda_value, vocab), vocab
                )
                evaluated.append(
                    {
                        "campaign_id": row["campaign_id"],
                        "rank": rank_of(ranking, row["true_label"]),
                    }
                )
            score = campaign_macro_top1(evaluated, "rank")
            fold_scores.append(score)
            audit_rows.append(
                {
                    "outer_train_sources": "+".join(outer_train_sources),
                    "lambda": lambda_value,
                    "inner_train_source": training_source,
                    "inner_validation_source": validation_source,
                    "validation_n": len(evaluated),
                    "campaign_macro_top1": score,
                }
            )
        mean_scores[lambda_value] = sum(fold_scores) / len(fold_scores)
    best = max(mean_scores.values())
    selected = min(
        value for value, score in mean_scores.items() if abs(score - best) <= 1e-15
    )
    for row in audit_rows:
        row["selected"] = row["lambda"] == selected
        row["mean_inner_top1_for_lambda"] = mean_scores[row["lambda"]]
    return selected, audit_rows


def metric_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rank_field = f"{method}_rank"
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["campaign_id"]].append(int(row[rank_field]))

    def summarize(ranks: list[int]) -> dict[str, float | int]:
        return {
            "n": len(ranks),
            "top1_count": sum(rank == 1 for rank in ranks),
            "top1": sum(rank == 1 for rank in ranks) / len(ranks),
            "hit5_count": sum(1 <= rank <= 5 for rank in ranks),
            "hit5": sum(1 <= rank <= 5 for rank in ranks) / len(ranks),
            "mrr": sum(1.0 / rank if rank > 0 else 0.0 for rank in ranks)
            / len(ranks),
        }

    row_micro = summarize([int(row[rank_field]) for row in rows])
    campaign_metrics = [summarize(ranks) for ranks in grouped.values()]
    return {
        "row_micro": row_micro,
        "campaign_macro": {
            "campaigns": len(grouped),
            "top1": sum(item["top1"] for item in campaign_metrics) / len(grouped),
            "hit5": sum(item["hit5"] for item in campaign_metrics) / len(grouped),
            "mrr": sum(item["mrr"] for item in campaign_metrics) / len(grouped),
        },
        "mrr_scope": "top5_only" if method == "B0" else "full_184",
    }


def main() -> None:
    vocab_rows = read_csv(VOCAB_PATH)
    vocab = [row["technique_id_parent"] for row in vocab_rows]
    if len(vocab) != 184 or len(set(vocab)) != 184:
        raise ValueError("Expected exactly 184 unique labels")
    vocab_set = set(vocab)

    source_rows = {name: load_source(path) for name, path in SOURCE_PATHS.items()}
    pilot_input = read_csv(PILOT_INPUT)
    generated = read_csv(PILOT_RESULTS)
    generated_by_key = {key(row): row for row in generated}
    if len(generated_by_key) != 30 or len(pilot_input) != 30:
        raise ValueError("Expected 30 unique pilot generations and 30 pilot inputs")

    pilot_rows: list[dict[str, Any]] = []
    for raw in pilot_input:
        generated_row = generated_by_key[key(raw)]
        row: dict[str, Any] = dict(raw)
        row["prefix_len"] = int(raw["prefix_len"])
        row["sequence"] = parse_sequence(raw)
        row["generation_status"] = generated_row["generation_status"]
        row["llm_top5"] = parse_top5(generated_row, vocab_set)
        row["valid_top5"] = bool(row["llm_top5"])
        pilot_rows.append(row)

    pilot_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pilot_rows:
        pilot_by_source[row["source"]].append(row)
    if {source: len(rows) for source, rows in pilot_by_source.items()} != {
        "ctid": 10,
        "attack_flow": 10,
        "stockpile": 10,
    }:
        raise ValueError("Pilot source counts are not 10/10/10")
    for source, rows in pilot_by_source.items():
        if len({row["campaign_id"] for row in rows}) != 10:
            raise ValueError(f"Pilot source {source} does not have 10 distinct campaigns")

    all_prediction_rows: list[dict[str, Any]] = []
    all_inner_rows: list[dict[str, Any]] = []
    lambda_by_outer_source: dict[str, float] = {}
    source_names = list(SOURCE_PATHS)

    for outer_source in source_names:
        outer_train_sources = [name for name in source_names if name != outer_source]
        outer_train_rows = sum(
            (source_rows[name] for name in outer_train_sources), start=[]
        )
        backbone = Backbone(outer_train_rows, vocab)
        prior = prior_probability(outer_train_rows, vocab)
        selected_lambda, inner_rows = select_lambda(
            outer_train_sources, source_rows, pilot_by_source, vocab
        )
        lambda_by_outer_source[outer_source] = selected_lambda
        for inner_row in inner_rows:
            inner_row["outer_held_source"] = outer_source
        all_inner_rows.extend(inner_rows)

        for row in pilot_by_source[outer_source]:
            a_probability = backbone.probability(row["sequence"])
            a_ranking = rank_labels(a_probability, vocab)
            a0_ranking = rank_labels(prior, vocab)
            b0_ranking = row["llm_top5"]
            b2_probability = fuse(
                a_probability, b0_ranking, selected_lambda, vocab
            )
            b2_ranking = rank_labels(b2_probability, vocab)
            target = row["true_label"]
            all_prediction_rows.append(
                {
                    "source": outer_source,
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "true_label": target,
                    "generation_status": row["generation_status"],
                    "valid_top5": row["valid_top5"],
                    "selected_lambda_B2": selected_lambda,
                    "A0_rank": rank_of(a0_ranking, target),
                    "A0_top20": json.dumps(a0_ranking[:20], ensure_ascii=False),
                    "A_rank": rank_of(a_ranking, target),
                    "A_top20": json.dumps(a_ranking[:20], ensure_ascii=False),
                    "B0_rank": rank_of(b0_ranking, target),
                    "B0_top5": json.dumps(b0_ranking, ensure_ascii=False),
                    "B2_rank": rank_of(b2_ranking, target),
                    "B2_top20": json.dumps(b2_ranking[:20], ensure_ascii=False),
                }
            )

    results: dict[str, Any] = {
        "warning": (
            "EXPLORATORY UNBLINDED SCREEN. Do not provide this directory to the "
            "groundedness reviewer. These 30 rows are development/screening data."
        ),
        "methods": ["A0", "A", "B0", "B2"],
        "lambda_by_outer_source": lambda_by_outer_source,
        "by_source": {},
        "pooled_30": {},
    }
    for source in source_names:
        source_predictions = [
            row for row in all_prediction_rows if row["source"] == source
        ]
        results["by_source"][source] = {
            method: metric_summary(source_predictions, method)
            for method in results["methods"]
        }
    results["pooled_30"] = {
        method: metric_summary(all_prediction_rows, method)
        for method in results["methods"]
    }
    results["invalid_top5_rows"] = [
        {
            "source": row["source"],
            "campaign_id": row["campaign_id"],
            "prefix_len": row["prefix_len"],
            "generation_status": row["generation_status"],
        }
        for row in all_prediction_rows
        if not row["valid_top5"]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "predictions.csv", all_prediction_rows)
    write_csv(OUTPUT_DIR / "inner_lambda_selection.csv", all_inner_rows)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exploratory_unblinded": True,
        "protocol": "project/data_v4/protocols/LLM_semantic_pilot_feasibility_screen_v1.md",
        "vocab_size": len(vocab),
        "alpha_s_total_mass": ALPHA,
        "lambda_grid": LAMBDA_GRID,
        "rank_weights": RANK_WEIGHTS,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in [PILOT_INPUT, PILOT_RESULTS, VOCAB_PATH, *SOURCE_PATHS.values()]
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "row_count": len(all_prediction_rows),
        "source_counts": {
            source: sum(row["source"] == source for row in all_prediction_rows)
            for source in source_names
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stdout = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    (OUTPUT_DIR / "stdout.log").write_text(stdout, encoding="utf-8")
    print(stdout, end="")


if __name__ == "__main__":
    main()
