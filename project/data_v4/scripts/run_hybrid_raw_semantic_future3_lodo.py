#!/usr/bin/env python3
"""Run frozen HM + raw-description semantic fusion (HM+R)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_raw_semantic_future3_lodo.py"
HM_CACHE = PROJECT_ROOT / "data_v4/model_scores/hm_future3_v1"
HM_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_markov_lstm_future3_lodo_v1"
RAW_RESULTS = PROJECT_ROOT / "data_v4/results/raw_semantic_future3_lodo_v1"
BASELINE_PREDICTIONS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/hybrid_raw_semantic_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/hybrid_raw_semantic_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (20, 40, 60, 80, 100)
LAMBDAS = tuple(index / 10 for index in range(11))
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807
RUN_LOG: list[str] = []


def emit(message: str) -> None:
    RUN_LOG.append(message)
    print(message, flush=True)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RAW = import_module("raw_semantic_for_hmr", RAW_SCRIPT)
BASE = RAW.BASE


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


def load_cache() -> tuple[np.ndarray, dict[tuple[str, str, int, str], int], np.ndarray, dict[tuple[str, int, str], int]]:
    inner = np.load(HM_CACHE / "inner_scores.npy", mmap_mode="r")
    outer = np.load(HM_CACHE / "outer_scores.npy", mmap_mode="r")
    if inner.shape != (7840, 184) or outer.shape != (3920, 184):
        raise AssertionError(f"HM cache shape changed: {inner.shape}/{outer.shape}")
    inner_lookup = {
        (row["held_out_source"], row["inner_validation_source"], int(row["seed"]), row["sample_id"]): int(row["score_row"])
        for row in read_csv(HM_CACHE / "inner_index.csv")
    }
    outer_lookup = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): int(row["score_row"])
        for row in read_csv(HM_CACHE / "outer_index.csv")
    }
    if len(inner_lookup) != 7840 or len(outer_lookup) != 3920:
        raise AssertionError("HM cache index is not unique and complete")
    return inner, inner_lookup, outer, outer_lookup


def hm_rows(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(matrix[list(indices)], dtype=np.float32)


def fuse(hm_values: Sequence[float], semantic_logits: Sequence[float], weight: float) -> list[float]:
    hm_z = BASE.standardize([BASE.clipped_logit(float(value)) for value in hm_values])
    semantic_z = BASE.standardize([float(value) for value in semantic_logits])
    return [(1 - weight) * left + weight * right for left, right in zip(hm_z, semantic_z)]


def score_validation(rows: Sequence[dict[str, Any]], hm: np.ndarray, semantic: np.ndarray, weight: float, labels: Sequence[str]) -> float:
    records: list[dict[str, Any]] = []
    for row, hm_row, semantic_row in zip(rows, hm, semantic):
        ranked, _ = BASE.ranking(fuse(hm_row, semantic_row, weight), labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_hyperparameters(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    inner_matrix: np.ndarray,
    inner_lookup: dict[tuple[str, str, int, str], int],
) -> tuple[int, float, list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    for validation_source in training_sources:
        train = [row for row in rows if row["source"] in training_sources and row["source"] != validation_source]
        validation = [row for row in rows if row["source"] == validation_source]
        for seed in SEEDS:
            started = time.perf_counter()
            semantic = RAW.train_checkpoints(train, validation, label_index, seed, EPOCHS)
            hm = hm_rows(inner_matrix, [inner_lookup[(held_out, validation_source, seed, row["sample_id"])] for row in validation])
            emit(f"inner held_out={held_out} validation={validation_source} seed={seed} elapsed={time.perf_counter()-started:.1f}s")
            for epoch in EPOCHS:
                for weight in LAMBDAS:
                    details.append(
                        {
                            "held_out_source": held_out,
                            "inner_validation_source": validation_source,
                            "seed": seed,
                            "epoch": epoch,
                            "lambda": weight,
                            "campaign_macro_ndcg5": score_validation(validation, hm, semantic[epoch], weight, labels),
                            "training_rows": len(train),
                            "validation_rows": len(validation),
                        }
                    )
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in details:
        grouped[(row["epoch"], row["lambda"])].append(row["campaign_macro_ndcg5"])
    means = {key: statistics.fmean(values) for key, values in grouped.items()}
    best = max(means.values())
    candidates = [key for key, value in means.items() if abs(value - best) <= 1e-12]
    selected_epoch, selected_lambda = min(candidates, key=lambda key: (key[1], key[0]))
    for row in details:
        row["five_seed_two_source_mean_ndcg5"] = means[(row["epoch"], row["lambda"])]
        row["selected"] = int(row["epoch"] == selected_epoch and row["lambda"] == selected_lambda)
    return selected_epoch, selected_lambda, details


def frozen_references() -> tuple[dict[tuple[str, int, str], dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    hm = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): row
        for row in read_csv(HM_RESULTS / "predictions_by_seed.csv")
    }
    raw = {row["sample_id"]: row for row in read_csv(RAW_RESULTS / "mean_sample_metrics.csv")}
    a = {row["sample_id"]: row for row in read_csv(BASELINE_PREDICTIONS) if row["method"] == "A"}
    return hm, raw, a


def evaluate_outer(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    epoch: int,
    weight: float,
    outer_matrix: np.ndarray,
    outer_lookup: dict[tuple[str, int, str], int],
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        semantic = RAW.train_checkpoints(train, test, label_index, seed, (epoch,))[epoch]
        hm = hm_rows(outer_matrix, [outer_lookup[(held_out, seed, row["sample_id"])] for row in test])
        for row, hm_row in zip(test, hm):
            ranked, _ = BASE.ranking(fuse(hm_row, hm_row, 0.0), labels)
            if ranked[:20] != json.loads(frozen_hm[(held_out, seed, row["sample_id"])]["top20_ids"]):
                raise AssertionError(f"lambda=0 HM reproduction failed: {held_out}/{seed}/{row['sample_id']}")
        for row, hm_row, semantic_row in zip(test, hm, semantic):
            ranked, scores = BASE.ranking(fuse(hm_row, semantic_row, weight), labels)
            reference = frozen_a[row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": "HM+R",
                    "seed": seed,
                    "selected_epoch": epoch,
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
        emit(f"outer held_out={held_out} seed={seed} epoch={epoch} lambda={weight:.1f} elapsed={time.perf_counter()-started:.1f}s")
    return output


def mean_samples(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    for sample_id, values in sorted(grouped.items()):
        first = values[0]
        if len(values) != len(SEEDS):
            raise AssertionError(f"expected five seeds: {sample_id}")
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "method": "HM+R",
                "sample_id": sample_id,
                "campaign_id": first["campaign_id"],
                "target_size": first["target_size"],
                "transition_visibility": first["transition_visibility"],
                "target_label_visibility": first["target_label_visibility"],
                "text_length_group": first["text_length_group"],
                **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
            }
        )
    return output


def campaigns(rows: Sequence[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["campaign_id"])].append(row)
    return [
        {"held_out_source": source, "method": method, "campaign_id": campaign, "rows": len(values), **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS}}
        for (source, campaign), values in sorted(grouped.items())
    ]


def fold_results(campaign_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        values = [row for row in campaign_rows if row["held_out_source"] == source]
        output.append({"held_out_source": source, "method": "HM+R", "campaigns": len(values), **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS}})
    return output


def paired_bootstrap(hmr_campaigns: Sequence[dict[str, Any]], frozen_hm: dict[tuple[str, int, str], dict[str, str]], frozen_r: dict[str, dict[str, str]], frozen_a: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    references: dict[str, list[dict[str, Any]]] = {"HM": [], "R": [], "A": []}
    for source in SOURCES:
        sample_ids = sorted({sample_id for held, _, sample_id in frozen_hm if held == source})
        for sample_id in sample_ids:
            first = frozen_hm[(source, SEEDS[0], sample_id)]
            references["HM"].append({"held_out_source": source, "campaign_id": first["campaign_id"], **{metric: statistics.fmean(float(frozen_hm[(source, seed, sample_id)][metric]) for seed in SEEDS) for metric in METRICS}})
    for method, frozen in (("R", frozen_r), ("A", frozen_a)):
        for row in frozen.values():
            references[method].append({"held_out_source": row["held_out_source"], "campaign_id": row["campaign_id"], **{metric: float(row[metric]) for metric in METRICS}})
    all_campaigns = list(hmr_campaigns)
    for method, values in references.items():
        all_campaigns.extend(campaigns(values, method))
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in all_campaigns}
    campaign_ids = {source: sorted(row["campaign_id"] for row in hmr_campaigns if row["held_out_source"] == source) for source in SOURCES}
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
        for right in ("HM", "R", "A"):
            for metric in METRICS:
                source_values: list[float] = []
                for source in SOURCES:
                    delta = statistics.fmean(float(lookup[(source, "HM+R", campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in draws[source])
                    source_values.append(delta)
                    replicates[(source, right, metric)].append(delta)
                replicates[("source_equal_overall", right, metric)].append(statistics.fmean(source_values))
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for right in ("HM", "R", "A"):
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(statistics.fmean(float(lookup[(source, "HM+R", campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in campaign_ids[source]) for source in SOURCES)
                else:
                    point = statistics.fmean(float(lookup[(scope, "HM+R", campaign)][metric]) - float(lookup[(scope, right, campaign)][metric]) for campaign in campaign_ids[scope])
                values = replicates[(scope, right, metric)]
                output.append({"scope": scope, "comparison": f"HM+R-{right}", "metric": metric, "point_estimate": point, "ci95_low": BASE.percentile(values, 0.025), "ci95_high": BASE.percentile(values, 0.975), "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    return output


def report_markdown(folds: Sequence[dict[str, Any]], selections: dict[str, tuple[int, float]], differences: Sequence[dict[str, Any]]) -> str:
    fold = {row["held_out_source"]: row for row in folds}
    delta = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    overall = {metric: statistics.fmean(float(row[f"campaign_macro_{metric}"]) for row in folds) for metric in METRICS}
    lines = ["# HM + raw-description future-3 LODO results", "", "No LLM or external API was used.", "", "## Campaign-macro metrics (five-seed mean)", "", "| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | Epoch | Lambda |", "|---|---:|---:|---:|---:|---:|---:|"]
    for source in SOURCES:
        row = fold[source]
        epoch, weight = selections[source]
        lines.append(f"| {source} | {row['campaign_macro_ndcg5']:.4f} | {row['campaign_macro_hit5']:.4f} | {row['campaign_macro_precision5']:.4f} | {row['campaign_macro_recall5']:.4f} | {epoch} | {weight:.1f} |")
    lines.append(f"| **Source-equal overall** | **{overall['ndcg5']:.4f}** | **{overall['hit5']:.4f}** | **{overall['precision5']:.4f}** | **{overall['recall5']:.4f}** | — | — |")
    lines.extend(["", "## Source-equal paired NDCG@5 differences", "", "| Comparison | Delta | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("HM+R-HM", "HM+R-R", "HM+R-A"):
        row = delta[("source_equal_overall", comparison, "ndcg5")]
        lines.append(f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    lines.extend(["", "All lambda=0 outer rankings reproduced the frozen HM Top-20 rows.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ["inner_selection.csv", "selected_hyperparameters.csv", "predictions_by_seed.csv", "mean_sample_metrics.csv", "campaign_results_five_seed_mean.csv", "fold_results_five_seed_mean.csv", "paired_bootstrap_differences.csv", "report.md", "stdout.log", "results_manifest.json"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite HM+R results: {existing}")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, _ = RAW.load_data()
    inner_matrix, inner_lookup, outer_matrix, outer_lookup = load_cache()
    frozen_hm, frozen_r, frozen_a = frozen_references()
    selections: dict[str, tuple[int, float]] = {}
    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for held_out in SOURCES:
        epoch, weight, details = select_hyperparameters(held_out, rows, labels, label_index, inner_matrix, inner_lookup)
        selections[held_out] = (epoch, weight)
        inner.extend(details)
        emit(f"selected held_out={held_out} epoch={epoch} lambda={weight:.1f}")
        predictions.extend(evaluate_outer(held_out, rows, labels, label_index, epoch, weight, outer_matrix, outer_lookup, frozen_hm, frozen_a))
    elapsed = time.perf_counter() - started
    inner.sort(key=lambda row: (row["held_out_source"], row["inner_validation_source"], row["seed"], row["epoch"], row["lambda"]))
    predictions.sort(key=lambda row: (row["held_out_source"], row["seed"], row["sample_id"]))
    mean_rows = mean_samples(predictions)
    campaign_rows = campaigns(mean_rows, "HM+R")
    folds = fold_results(campaign_rows)
    differences = paired_bootstrap(campaign_rows, frozen_hm, frozen_r, frozen_a)
    selection_rows = [{"held_out_source": source, "selected_epoch": values[0], "selected_lambda": values[1]} for source, values in selections.items()]
    write_csv(output / "inner_selection.csv", inner, ["held_out_source", "inner_validation_source", "seed", "epoch", "lambda", "campaign_macro_ndcg5", "five_seed_two_source_mean_ndcg5", "selected", "training_rows", "validation_rows"])
    write_csv(output / "selected_hyperparameters.csv", selection_rows, ["held_out_source", "selected_epoch", "selected_lambda"])
    write_csv(output / "predictions_by_seed.csv", predictions, ["held_out_source", "method", "seed", "selected_epoch", "selected_lambda", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "top20_ids", "top20_scores", *METRICS])
    write_csv(output / "mean_sample_metrics.csv", mean_rows, ["held_out_source", "method", "sample_id", "campaign_id", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", *METRICS])
    write_csv(output / "campaign_results_five_seed_mean.csv", campaign_rows, ["held_out_source", "method", "campaign_id", "rows", *METRICS])
    write_csv(output / "fold_results_five_seed_mean.csv", folds, ["held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]])
    write_csv(output / "paired_bootstrap_differences.csv", differences, ["scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"])
    report = report_markdown(folds, selections, differences)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text("\n".join(RUN_LOG) + "\n" + report + f"\nelapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "inputs": {"hm_cache_manifest_sha256": sha256(HM_CACHE / "cache_manifest.json"), "embedding_manifest_sha256": sha256(RAW.EMBEDDING_MANIFEST), "hm_results_manifest_sha256": sha256(HM_RESULTS / "results_manifest.json"), "raw_results_manifest_sha256": sha256(RAW_RESULTS / "results_manifest.json"), "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS), "rows": len(rows)},
        "parameters": {"architecture": "1024-256-GELU-dropout0.3-184", "loss": "BCEWithLogitsLoss", "optimizer": "AdamW", "learning_rate": RAW.LEARNING_RATE, "weight_decay": RAW.WEIGHT_DECAY, "batch_size": RAW.BATCH_SIZE, "seeds": list(SEEDS), "epoch_grid": list(EPOCHS), "lambda_grid": list(LAMBDAS), "selected": {source: {"epoch": values[0], "lambda": values[1]} for source, values in selections.items()}, "num_threads": args.num_threads, "deterministic_algorithms": True},
        "lambda0_hm_reproduction_gate": "PASS all 3920 outer seed-sample rows",
        "elapsed_seconds": elapsed,
        "outputs_sha256": {name: sha256(output / name) for name in managed if name != "results_manifest.json" and (output / name).exists()},
    }
    write_json(output / "results_manifest.json", manifest)
    print(report)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"wrote results to {output}")


if __name__ == "__main__":
    main()
