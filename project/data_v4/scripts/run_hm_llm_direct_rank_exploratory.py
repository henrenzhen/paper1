#!/usr/bin/env python3
"""Run exploratory nested-LODO fusion of frozen HM and direct LLM Top-5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_hybrid_llm_summary_future3_lodo.py"
SUMMARY_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_llm_summary_future3_lodo_v1"
B0_RANKINGS = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1/b0_rankings.csv"
NONSEMANTIC = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
NEURAL = PROJECT_ROOT / "data_v4/results/id_neural_future3_lodo_v1/mean_sample_metrics.csv"
MB = PROJECT_ROOT / "data_v4/results/markov_beam_future3_lodo_v1/predictions.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/hm_llm_direct_rank_exploratory_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/hm_llm_direct_rank_exploratory_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)
LAMBDAS = tuple(index / 10 for index in range(11))
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
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


SUMMARY = import_module("llm_summary_for_direct_fusion", SUMMARY_SCRIPT)
BASE = SUMMARY.BASE
HMR = SUMMARY.HMR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_b0(labels: Sequence[str]) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    label_index = {label: index for index, label in enumerate(labels)}
    scores: dict[str, list[float]] = {}
    rankings: dict[str, list[str]] = {}
    for row in read_csv(B0_RANKINGS):
        sample_id = row["sample_id"]
        ranked = json.loads(row["predicted_next_ttps"])
        if len(ranked) != 5 or len(set(ranked)) != 5 or any(item not in label_index for item in ranked):
            raise AssertionError(f"invalid frozen B0 row: {sample_id}")
        values = [0.0] * len(labels)
        for rank, label in enumerate(ranked, start=1):
            values[label_index[label]] = 1.0 / rank
        scores[sample_id] = values
        rankings[sample_id] = ranked
    if len(scores) != 784:
        raise AssertionError(f"expected 784 B0 score rows, found {len(scores)}")
    return scores, rankings


def fuse(hm_values: Sequence[float], b0_values: Sequence[float], weight: float) -> list[float]:
    hm_z = BASE.standardize([BASE.clipped_logit(float(value)) for value in hm_values])
    b0_z = BASE.standardize([float(value) for value in b0_values])
    return [(1 - weight) * left + weight * right for left, right in zip(hm_z, b0_z)]


def score_rows(
    rows: Sequence[dict[str, Any]],
    hm: np.ndarray,
    b0: dict[str, list[float]],
    weight: float,
    labels: Sequence[str],
) -> float:
    records: list[dict[str, Any]] = []
    for row, hm_row in zip(rows, hm):
        ranked, _ = BASE.ranking(fuse(hm_row, b0[row["sample_id"]], weight), labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_lambda(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    b0: dict[str, list[float]],
    inner_matrix: np.ndarray,
    inner_lookup: dict[tuple[str, str, int, str], int],
) -> tuple[float, list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    for validation_source in training_sources:
        validation = [row for row in rows if row["source"] == validation_source]
        for seed in SEEDS:
            hm = HMR.hm_rows(
                inner_matrix,
                [inner_lookup[(held_out, validation_source, seed, row["sample_id"])] for row in validation],
            )
            for weight in LAMBDAS:
                details.append(
                    {
                        "held_out_source": held_out,
                        "inner_validation_source": validation_source,
                        "seed": seed,
                        "lambda": weight,
                        "campaign_macro_ndcg5": score_rows(validation, hm, b0, weight, labels),
                        "validation_rows": len(validation),
                    }
                )
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in details:
        grouped[row["lambda"]].append(row["campaign_macro_ndcg5"])
    means = {weight: statistics.fmean(values) for weight, values in grouped.items()}
    best = max(means.values())
    selected = min(weight for weight, value in means.items() if abs(value - best) <= 1e-12)
    for row in details:
        row["five_seed_two_source_mean_ndcg5"] = means[row["lambda"]]
        row["selected"] = int(row["lambda"] == selected)
    return selected, details


def outer_predictions(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    b0: dict[str, list[float]],
    b0_rankings: dict[str, list[str]],
    weight: float,
    outer_matrix: np.ndarray,
    outer_lookup: dict[tuple[str, int, str], int],
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    test = [row for row in rows if row["source"] == held_out]
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        hm = HMR.hm_rows(
            outer_matrix,
            [outer_lookup[(held_out, seed, row["sample_id"])] for row in test],
        )
        for row, hm_row in zip(test, hm):
            hm_ranked, _ = BASE.ranking(fuse(hm_row, b0[row["sample_id"]], 0.0), labels)
            if hm_ranked[:20] != json.loads(frozen_hm[(held_out, seed, row["sample_id"])] ["top20_ids"]):
                raise AssertionError(f"lambda=0 HM reproduction failed: {held_out}/{seed}/{row['sample_id']}")
            b0_ranked, _ = BASE.ranking(fuse(hm_row, b0[row["sample_id"]], 1.0), labels)
            if b0_ranked[:5] != b0_rankings[row["sample_id"]]:
                raise AssertionError(f"lambda=1 B0 reproduction failed: {row['sample_id']}")
            ranked, scores = BASE.ranking(fuse(hm_row, b0[row["sample_id"]], weight), labels)
            reference = frozen_a[row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": "HM+B0",
                    "seed": seed,
                    "selected_lambda": weight,
                    "sample_id": row["sample_id"],
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "target_parent_ids": compact_json(row["targets"]),
                    "target_size": row["target_size"],
                    "transition_visibility": reference["transition_visibility"],
                    "target_label_visibility": reference["target_label_visibility"],
                    "text_length_group": reference["text_length_group"],
                    "top20_ids": compact_json(ranked[:20]),
                    "top20_scores": compact_json([round(float(value), 12) for value in scores[:20]]),
                    **BASE.sample_metrics(ranked[:5], row["targets"]),
                }
            )
    return output


def mean_seed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    for sample_id, values in sorted(grouped.items()):
        if len(values) != 5:
            raise AssertionError(f"not five fused seeds: {sample_id}")
        first = values[0]
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "method": "HM+B0",
                "sample_id": sample_id,
                "campaign_id": first["campaign_id"],
                "prefix_len": first["prefix_len"],
                "target_size": first["target_size"],
                "transition_visibility": first["transition_visibility"],
                "target_label_visibility": first["target_label_visibility"],
                "text_length_group": first["text_length_group"],
                **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
            }
        )
    return output


def normalize_reference_rows(
    fused: Sequence[dict[str, Any]],
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    output = list(fused)
    metadata = {row["sample_id"]: row for row in fused}
    for source in SOURCES:
        sample_ids = sorted({sample_id for held, _, sample_id in frozen_hm if held == source})
        for sample_id in sample_ids:
            first = frozen_hm[(source, SEEDS[0], sample_id)]
            output.append(
                {
                    **{key: metadata[sample_id][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                    "method": "HM",
                    **{metric: statistics.fmean(float(frozen_hm[(source, seed, sample_id)][metric]) for seed in SEEDS) for metric in METRICS},
                }
            )
    b0 = {row["sample_id"]: row for row in read_csv(SUMMARY_RESULTS / "b0_sample_metrics.csv")}
    for sample_id, row in b0.items():
        output.append({**{key: metadata[sample_id][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")}, "method": "B0", **{metric: float(row[metric]) for metric in METRICS}})
    for row in read_csv(NONSEMANTIC):
        if row["method"] in ("A0", "A", "K"):
            output.append({**{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")}, "method": row["method"], **{metric: float(row[metric]) for metric in METRICS}})
    for row in read_csv(NEURAL):
        if row["method"] == "LSTM":
            output.append({**{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")}, "method": "LSTM", **{metric: float(row[metric]) for metric in METRICS}})
    for row in read_csv(MB):
        output.append({**{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")}, "method": "MB", **{metric: float(row[metric]) for metric in METRICS}})
    expected_methods = {"HM+B0", "HM", "B0", "A0", "A", "K", "LSTM", "MB"}
    counts = defaultdict(int)
    for row in output:
        counts[row["method"]] += 1
    if set(counts) != expected_methods or any(counts[method] != 784 for method in expected_methods):
        raise AssertionError(f"reference sample count gate failed: {dict(counts)}")
    return output


def campaign_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    return [{"held_out_source": source, "method": method, "campaign_id": campaign, "rows": len(values), **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS}} for (source, method, campaign), values in sorted(grouped.items())]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in campaigns})
    return [{"held_out_source": source, "method": method, "campaigns": len(values), **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS}} for method in methods for source in SOURCES for values in [[row for row in campaigns if row["method"] == method and row["held_out_source"] == source]]]


def bootstrap(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in campaigns}
    campaign_ids = {source: sorted(row["campaign_id"] for row in campaigns if row["held_out_source"] == source and row["method"] == "B0") for source in SOURCES}
    comparisons = (("HM+B0", "B0"), ("HM+B0", "HM"), ("B0", "A0"), ("B0", "A"), ("B0", "K"), ("B0", "LSTM"), ("B0", "MB"))
    rng = random.Random(BOOTSTRAP_SEED)
    reps: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                source_values = []
                for source in SOURCES:
                    delta = statistics.fmean(float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in draws[source])
                    source_values.append(delta); reps[(source, comparison, metric)].append(delta)
                reps[("source_equal_overall", comparison, metric)].append(statistics.fmean(source_values))
    output = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(statistics.fmean(float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in campaign_ids[source]) for source in SOURCES)
                else:
                    point = statistics.fmean(float(lookup[(scope, left, campaign)][metric]) - float(lookup[(scope, right, campaign)][metric]) for campaign in campaign_ids[scope])
                values = reps[(scope, comparison, metric)]
                output.append({"scope": scope, "comparison": comparison, "metric": metric, "point_estimate": point, "ci95_low": BASE.percentile(values, 0.025), "ci95_high": BASE.percentile(values, 0.975), "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    return output


def stratified(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {"transition_visibility": lambda row: row["transition_visibility"], "target_label_visibility": lambda row: row["target_label_visibility"], "text_length": lambda row: row["text_length_group"], "target_size": lambda row: str(row["target_size"])}
    output = []
    for source in SOURCES:
        for method in ("HM+B0", "B0", "HM"):
            method_rows = [row for row in rows if row["held_out_source"] == source and row["method"] == method]
            for name, getter in definitions.items():
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in method_rows: groups[getter(row)].append(row)
                for group, values in sorted(groups.items()):
                    by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in values: by_campaign[row["campaign_id"]].append(row)
                    output.append({"held_out_source": source, "method": method, "stratum": name, "group": group, "rows": len(values), "campaigns": len(by_campaign), "inferentially_eligible": int(len(values)>=20 and len(by_campaign)>=5), **{f"campaign_macro_{metric}": statistics.fmean(statistics.fmean(float(row[metric]) for row in items) for items in by_campaign.values()) for metric in METRICS}})
    return output


def report(folds: Sequence[dict[str, Any]], selected: dict[str, float], differences: Sequence[dict[str, Any]]) -> str:
    lookup = {(row["method"], row["held_out_source"]): row for row in folds}
    diff = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    lines = ["# Exploratory HM + direct LLM rank fusion", "", "**Post-result exploratory analysis; prospective confirmation is required.**", "", "| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |", "|---|---:|---:|---:|---:|"]
    for method in ("A0", "A", "K", "LSTM", "MB", "HM", "B0", "HM+B0"):
        values = [float(lookup[(method, source)]["campaign_macro_ndcg5"]) for source in SOURCES]
        lines.append(f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | **{statistics.fmean(values):.4f}** |")
    lines.extend(["", "Selected lambda: " + ", ".join(f"{source}={selected[source]:.1f}" for source in SOURCES), "", "| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("HM+B0-B0", "HM+B0-HM", "B0-A0", "B0-K", "B0-LSTM"):
        row = diff[("source_equal_overall", comparison, "ndcg5")]
        lines.append(f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    lines.extend(["", "Lambda 0 reproduced HM and lambda 1 reproduced the exact B0 Top-5 for all outer seed-sample rows.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ("inner_selection.csv", "selected_lambdas.csv", "predictions_by_seed.csv", "mean_sample_metrics.csv", "campaign_results.csv", "fold_results.csv", "paired_bootstrap_differences.csv", "stratified_results.csv", "report.md", "results_manifest.json")
    if any((output/name).exists() for name in managed): raise FileExistsError("refusing to overwrite exploratory direct-rank results")
    rows, labels, _, _ = SUMMARY.load_data()
    inner_matrix, inner_lookup, outer_matrix, outer_lookup = HMR.load_cache()
    frozen_hm, _, frozen_a = SUMMARY.frozen_references()
    b0, b0_rankings = load_b0(labels)
    selected: dict[str, float] = {}
    inner = []
    predictions = []
    for held_out in SOURCES:
        weight, details = select_lambda(held_out, rows, labels, b0, inner_matrix, inner_lookup)
        selected[held_out] = weight; inner.extend(details)
        predictions.extend(outer_predictions(held_out, rows, labels, b0, b0_rankings, weight, outer_matrix, outer_lookup, frozen_hm, frozen_a))
    predictions.sort(key=lambda row: (row["held_out_source"], row["seed"], row["sample_id"]))
    means = mean_seed_rows(predictions)
    samples = normalize_reference_rows(means, frozen_hm)
    campaigns = campaign_rows(samples)
    folds = fold_rows(campaigns)
    differences = bootstrap(campaigns)
    strata = stratified(samples)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"inner_selection.csv", inner, ("held_out_source", "inner_validation_source", "seed", "lambda", "campaign_macro_ndcg5", "five_seed_two_source_mean_ndcg5", "selected", "validation_rows"))
    write_csv(output/"selected_lambdas.csv", [{"held_out_source": source, "selected_lambda": selected[source]} for source in SOURCES], ("held_out_source", "selected_lambda"))
    write_csv(output/"predictions_by_seed.csv", predictions, ("held_out_source", "method", "seed", "selected_lambda", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "top20_ids", "top20_scores", *METRICS))
    sample_columns = ("held_out_source", "method", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", *METRICS)
    write_csv(output/"mean_sample_metrics.csv", means, sample_columns)
    write_csv(output/"campaign_results.csv", campaigns, ("held_out_source", "method", "campaign_id", "rows", *METRICS))
    write_csv(output/"fold_results.csv", folds, ("held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]))
    write_csv(output/"paired_bootstrap_differences.csv", differences, ("scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"))
    write_csv(output/"stratified_results.csv", strata, ("held_out_source", "method", "stratum", "group", "rows", "campaigns", "inferentially_eligible", *[f"campaign_macro_{metric}" for metric in METRICS]))
    markdown = report(folds, selected, differences); (output/"report.md").write_text(markdown, encoding="utf-8")
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "analysis_status": "post-result exploratory; prospective confirmation required", "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))}, "method_card_sha256": sha256(METHOD_CARD), "inputs": {"b0_rankings_sha256": sha256(B0_RANKINGS), "hm_cache_manifest_sha256": sha256(SUMMARY.HM_CACHE/"cache_manifest.json"), "summary_results_manifest_sha256": sha256(SUMMARY_RESULTS/"results_manifest.json")}, "parameters": {"b0_score": "1/rank for returned ranks 1..5; zero otherwise", "lambda_grid": list(LAMBDAS), "selected": selected, "tie_break": "smallest lambda", "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED}, "lambda0_hm_reproduction": "PASS all 3920 outer seed-sample rows", "lambda1_b0_reproduction": "PASS all 3920 outer seed-sample rows", "outputs_sha256": {name: sha256(output/name) for name in managed if name != "results_manifest.json" and (output/name).exists()}}
    (output/"results_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
