#!/usr/bin/env python3
"""Run the frozen Markov-only beam future-3 LODO baseline."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
BASELINE_PREDICTIONS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/markov_beam_future3_baseline_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/markov_beam_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
HORIZON = 3
BEAM_WIDTH = 50
BRANCH_CAP = 20
SMOOTHING = 0.1
FLOOR = 1e-12
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def load_baseline_module() -> Any:
    spec = importlib.util.spec_from_file_location("nonsemantic_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_baseline_module()


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MarkovBeam:
    def __init__(self, train: Sequence[dict[str, Any]], labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.label_index = {label: index for index, label in enumerate(labels)}
        self.edges: dict[str, Counter[str]] = defaultdict(Counter)
        self.target_counts: Counter[str] = Counter()
        for row in train:
            immediate = row["targets"][0]
            self.edges[row["history"][-1]][immediate] += 1
            self.target_counts.update(set(row["targets"]))
        self.fallback = sorted(
            labels,
            key=lambda label: (-self.target_counts[label], self.label_index[label]),
        )[:BRANCH_CAP]

    def candidates(self, last: str) -> list[tuple[str, float]]:
        counts = self.edges.get(last)
        if counts:
            labels = sorted(
                counts,
                key=lambda label: (-counts[label], self.label_index[label]),
            )[:BRANCH_CAP]
            weights = [counts[label] + SMOOTHING for label in labels]
        else:
            labels = self.fallback
            weights = [self.target_counts[label] + SMOOTHING for label in labels]
        denominator = sum(weights)
        return [
            (label, max(weight / denominator, FLOOR))
            for label, weight in zip(labels, weights)
        ]

    def score(self, history: Sequence[str]) -> tuple[list[float], int]:
        beams: list[tuple[tuple[str, ...], float]] = [(tuple(), 0.0)]
        fallback_expansions = 0
        for _ in range(HORIZON):
            expanded: dict[tuple[str, ...], float] = {}
            for path, log_probability in beams:
                last = path[-1] if path else history[-1]
                if last not in self.edges:
                    fallback_expansions += 1
                for label, probability in self.candidates(last):
                    new_path = path + (label,)
                    new_score = log_probability + math.log(probability)
                    if new_path not in expanded or new_score > expanded[new_path]:
                        expanded[new_path] = new_score
            beams = sorted(
                expanded.items(),
                key=lambda item: (
                    -item[1],
                    tuple(self.label_index[label] for label in item[0]),
                ),
            )[:BEAM_WIDTH]
        maximum = max(value for _, value in beams)
        weights = [math.exp(value - maximum) for _, value in beams]
        denominator = sum(weights)
        marginals = [0.0] * len(self.labels)
        for (path, _), weight in zip(beams, weights):
            probability = weight / denominator
            for label in set(path):
                marginals[self.label_index[label]] += probability
        return marginals, fallback_expansions


def references() -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {method: {} for method in ("A", "A0", "K")}
    for row in read_csv(BASELINE_PREDICTIONS):
        if row["method"] in result:
            result[row["method"]][row["sample_id"]] = row
    if any(len(values) != 784 for values in result.values()):
        raise AssertionError("frozen A/A0/K references must each contain 784 rows")
    return result


def evaluate(rows: Sequence[dict[str, Any]], labels: Sequence[str], frozen: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for held_out in SOURCES:
        train = [row for row in rows if row["source"] != held_out]
        test = [row for row in rows if row["source"] == held_out]
        model = MarkovBeam(train, labels)
        for row in test:
            scores, fallbacks = model.score(row["history"])
            ranked, ordered_scores = BASE.ranking(scores, labels)
            reference = frozen["A"][row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": "MB",
                    "sample_id": row["sample_id"],
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "target_parent_ids": compact_json(row["targets"]),
                    "target_size": row["target_size"],
                    "transition_visibility": reference["transition_visibility"],
                    "target_label_visibility": reference["target_label_visibility"],
                    "text_length_group": reference["text_length_group"],
                    "fallback_beam_expansions": fallbacks,
                    "top20_ids": compact_json(ranked[:20]),
                    "top20_scores": compact_json([round(float(value), 12) for value in ordered_scores[:20]]),
                    **BASE.sample_metrics(ranked[:5], row["targets"]),
                }
            )
    return sorted(output, key=lambda row: (row["held_out_source"], row["sample_id"]))


def campaign_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["held_out_source"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": "MB",
            "campaign_id": campaign,
            "rows": len(values),
            **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, campaign), values in sorted(grouped.items())
    ]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        values = [row for row in campaigns if row["held_out_source"] == source]
        output.append(
            {
                "held_out_source": source,
                "method": "MB",
                "campaigns": len(values),
                **{
                    f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values)
                    for metric in METRICS
                },
            }
        )
    return output


def reference_campaigns(reference: dict[str, dict[str, str]], method: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in reference.values():
        grouped[(row["held_out_source"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
            "campaign_id": campaign,
            **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, campaign), values in sorted(grouped.items())
    ]


def paired_bootstrap(
    campaigns: Sequence[dict[str, Any]], frozen: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    all_rows = list(campaigns)
    for method in ("A", "A0", "K"):
        all_rows.extend(reference_campaigns(frozen[method], method))
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in all_rows}
    campaign_ids = {
        source: sorted(row["campaign_id"] for row in campaigns if row["held_out_source"] == source)
        for source in SOURCES
    }
    rng = random.Random(BOOTSTRAP_SEED)
    replicate: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
        for right in ("A", "A0", "K"):
            for metric in METRICS:
                source_deltas: list[float] = []
                for source in SOURCES:
                    value = statistics.fmean(
                        float(lookup[(source, "MB", campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    source_deltas.append(value)
                    replicate[(source, f"MB-{right}", metric)].append(value)
                replicate[("source_equal_overall", f"MB-{right}", metric)].append(statistics.fmean(source_deltas))
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for right in ("A", "A0", "K"):
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(
                        statistics.fmean(
                            float(lookup[(source, "MB", campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                            for campaign in campaign_ids[source]
                        )
                        for source in SOURCES
                    )
                else:
                    point = statistics.fmean(
                        float(lookup[(scope, "MB", campaign)][metric]) - float(lookup[(scope, right, campaign)][metric])
                        for campaign in campaign_ids[scope]
                    )
                values = replicate[(scope, f"MB-{right}", metric)]
                output.append(
                    {
                        "scope": scope,
                        "comparison": f"MB-{right}",
                        "metric": metric,
                        "point_estimate": point,
                        "ci95_low": BASE.percentile(values, 0.025),
                        "ci95_high": BASE.percentile(values, 0.975),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return output


def report_markdown(folds: Sequence[dict[str, Any]], differences: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> str:
    fold = {row["held_out_source"]: row for row in folds}
    delta = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    overall = {metric: statistics.fmean(float(row[f"campaign_macro_{metric}"]) for row in folds) for metric in METRICS}
    lines = [
        "# Markov-only beam future-3 LODO results",
        "",
        "No neural model, text, tactic feature, LLM, or external API was used.",
        "",
        "## Campaign-macro metrics",
        "",
        "| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        row = fold[source]
        lines.append(f"| {source} | {row['campaign_macro_ndcg5']:.4f} | {row['campaign_macro_hit5']:.4f} | {row['campaign_macro_precision5']:.4f} | {row['campaign_macro_recall5']:.4f} |")
    lines.append(f"| **Source-equal overall** | **{overall['ndcg5']:.4f}** | **{overall['hit5']:.4f}** | **{overall['precision5']:.4f}** | **{overall['recall5']:.4f}** |")
    lines.extend(["", "## Source-equal paired NDCG@5 differences", "", "| Comparison | Delta | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("MB-A", "MB-A0", "MB-K"):
        row = delta[("source_equal_overall", comparison, "ndcg5")]
        lines.append(f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    total_fallback = sum(int(row["fallback_beam_expansions"]) for row in predictions)
    lines.extend(["", f"Total no-outedge fallback expansions across 784 test beams: {total_fallback}.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ["predictions.csv", "campaign_results.csv", "fold_results.csv", "paired_bootstrap_differences.csv", "report.md", "stdout.log", "results_manifest.json"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing MB results: {existing}")
    rows = BASE.parse_samples()
    labels, _ = BASE.parse_vocabulary()
    frozen = references()
    predictions = evaluate(rows, labels, frozen)
    campaigns = campaign_rows(predictions)
    folds = fold_rows(campaigns)
    differences = paired_bootstrap(campaigns, frozen)
    write_csv(output / "predictions.csv", predictions, ["held_out_source", "method", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "fallback_beam_expansions", "top20_ids", "top20_scores", *METRICS])
    write_csv(output / "campaign_results.csv", campaigns, ["held_out_source", "method", "campaign_id", "rows", *METRICS])
    write_csv(output / "fold_results.csv", folds, ["held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]])
    write_csv(output / "paired_bootstrap_differences.csv", differences, ["scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"])
    report = report_markdown(folds, differences, predictions)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "inputs": {"samples_sha256": sha256(BASE.SAMPLES_PATH), "vocabulary_sha256": sha256(BASE.VOCAB_PATH), "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS), "main_rows": len(rows)},
        "parameters": {"horizon": HORIZON, "beam_width": BEAM_WIDTH, "branch_cap": BRANCH_CAP, "smoothing": SMOOTHING, "floor": FLOOR, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED},
        "outputs_sha256": {name: sha256(output / name) for name in managed if name != "results_manifest.json" and (output / name).exists()},
    }
    write_json(output / "results_manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
