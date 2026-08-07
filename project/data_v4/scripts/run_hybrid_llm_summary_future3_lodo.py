#!/usr/bin/env python3
"""Run frozen HM+S and within-training-stratum permuted HM+P future-3 LODO."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_raw_semantic_future3_lodo.py"
HMR_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_hybrid_raw_semantic_future3_lodo.py"
EMBEDDING_DIR = PROJECT_ROOT / "data_v4/semantic_embeddings/llm_summary_bge_m3_5617a9f"
EMBEDDINGS = EMBEDDING_DIR / "embeddings.npy"
EMBEDDING_INDEX = EMBEDDING_DIR / "embedding_index.csv"
EMBEDDING_MANIFEST = EMBEDDING_DIR / "embedding_manifest.json"
SUMMARY_DIR = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1"
B0_RANKINGS = SUMMARY_DIR / "b0_rankings.csv"
SUMMARY_MANIFEST = SUMMARY_DIR / "summary_manifest.json"
HM_CACHE = PROJECT_ROOT / "data_v4/model_scores/hm_future3_v1"
HM_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_markov_lstm_future3_lodo_v1"
HMR_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_raw_semantic_future3_lodo_v1"
BASELINE_PREDICTIONS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/llm_summary_semantic_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/hybrid_llm_summary_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
METHODS = ("HM+S", "HM+P")
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


RAW = import_module("raw_semantic_for_llm_summary", RAW_SCRIPT)
HMR = import_module("hybrid_raw_for_llm_summary", HMR_SCRIPT)
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_data() -> tuple[list[dict[str, Any]], list[str], dict[str, int], np.ndarray]:
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    matrix = np.load(EMBEDDINGS, mmap_mode="r")
    index_rows = read_csv(EMBEDDING_INDEX)
    if matrix.shape != (784, 1024) or len(index_rows) != 784:
        raise AssertionError(f"unexpected S embedding shape/index: {matrix.shape}/{len(index_rows)}")
    vectors = {
        row["sample_id"]: np.asarray(matrix[int(row["embedding_row"])], dtype=np.float32)
        for row in index_rows
    }
    if len(vectors) != 784 or len(rows) != 784:
        raise AssertionError("S embeddings or formal rows are not unique 784-row sets")
    for row in rows:
        if row["sample_id"] not in vectors:
            raise AssertionError(f"missing S embedding: {row['sample_id']}")
        row["embedding"] = vectors[row["sample_id"]]
    return rows, labels, label_index, matrix


def tercile_assignments(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    for source in sorted(by_source):
        values = sorted(
            by_source[source], key=lambda row: (int(row["prefix_len"]), row["sample_id"])
        )
        n = len(values)
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rank, row in enumerate(values):
            bucket = min(2, (3 * rank) // n)
            buckets[bucket].append(row)
        for bucket in tuple(sorted(buckets)):
            if len(buckets[bucket]) >= 2:
                continue
            neighbors = [value for value in buckets if value != bucket]
            target = min(neighbors, key=lambda value: (abs(value - bucket), value))
            buckets[target].extend(buckets.pop(bucket))
        for bucket, bucket_rows in buckets.items():
            if len(bucket_rows) < 2:
                raise AssertionError(f"permutation cell remains too small: {source}/{bucket}")
            for row in bucket_rows:
                output[row["sample_id"]] = bucket
    if len(output) != len(rows):
        raise AssertionError("tercile assignment is incomplete")
    return output


def permute_training_rows(
    rows: Sequence[dict[str, Any]],
    seed: int,
    stage: str,
    held_out: str,
    validation_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bucket_by_sample = tercile_assignments(rows)
    row_by_sample = {row["sample_id"]: row for row in rows}
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], bucket_by_sample[row["sample_id"]])].append(row["sample_id"])
    rng = random.Random(9000 + seed)
    donor_by_recipient: dict[str, str] = {}
    mapping: list[dict[str, Any]] = []
    for (source, bucket), sample_ids in sorted(groups.items()):
        shuffled = sorted(sample_ids)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise AssertionError(f"permutation group too small: {source}/{bucket}")
        for index, recipient in enumerate(shuffled):
            donor = shuffled[(index + 1) % len(shuffled)]
            if donor == recipient:
                raise AssertionError("permutation fixed point")
            donor_by_recipient[recipient] = donor
            mapping.append(
                {
                    "stage": stage,
                    "held_out_source": held_out,
                    "inner_validation_source": validation_source,
                    "seed": seed,
                    "permutation_seed": 9000 + seed,
                    "source": source,
                    "prefix_tercile": bucket,
                    "recipient_sample_id": recipient,
                    "donor_sample_id": donor,
                    "recipient_prefix_len": row_by_sample[recipient]["prefix_len"],
                    "donor_prefix_len": row_by_sample[donor]["prefix_len"],
                }
            )
    permuted: list[dict[str, Any]] = []
    for row in rows:
        donor = row_by_sample[donor_by_recipient[row["sample_id"]]]
        permuted.append({**row, "embedding": donor["embedding"]})
    if len(donor_by_recipient) != len(rows) or any(
        recipient == donor for recipient, donor in donor_by_recipient.items()
    ):
        raise AssertionError("permutation completeness/fixed-point gate failed")
    return permuted, mapping


def training_rows_for_method(
    method: str,
    rows: Sequence[dict[str, Any]],
    seed: int,
    stage: str,
    held_out: str,
    validation_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if method == "HM+S":
        return list(rows), []
    if method == "HM+P":
        return permute_training_rows(
            rows, seed, stage, held_out, validation_source
        )
    raise ValueError(method)


def score_validation(
    rows: Sequence[dict[str, Any]],
    hm: np.ndarray,
    semantic: np.ndarray,
    weight: float,
    labels: Sequence[str],
) -> float:
    records: list[dict[str, Any]] = []
    for row, hm_row, semantic_row in zip(rows, hm, semantic):
        ranked, _ = BASE.ranking(HMR.fuse(hm_row, semantic_row, weight), labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_hyperparameters(
    method: str,
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    inner_matrix: np.ndarray,
    inner_lookup: dict[tuple[str, str, int, str], int],
) -> tuple[int, float, list[dict[str, Any]], list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for validation_source in training_sources:
        train = [
            row
            for row in rows
            if row["source"] in training_sources and row["source"] != validation_source
        ]
        validation = [row for row in rows if row["source"] == validation_source]
        for seed in SEEDS:
            fitted_train, current_mapping = training_rows_for_method(
                method, train, seed, "inner", held_out, validation_source
            )
            mappings.extend(current_mapping)
            started = time.perf_counter()
            semantic = RAW.train_checkpoints(
                fitted_train, validation, label_index, seed, EPOCHS
            )
            hm = HMR.hm_rows(
                inner_matrix,
                [
                    inner_lookup[(held_out, validation_source, seed, row["sample_id"])]
                    for row in validation
                ],
            )
            emit(
                f"inner method={method} held_out={held_out} "
                f"validation={validation_source} seed={seed} "
                f"elapsed={time.perf_counter()-started:.1f}s"
            )
            for epoch in EPOCHS:
                for weight in LAMBDAS:
                    details.append(
                        {
                            "method": method,
                            "held_out_source": held_out,
                            "inner_validation_source": validation_source,
                            "seed": seed,
                            "epoch": epoch,
                            "lambda": weight,
                            "campaign_macro_ndcg5": score_validation(
                                validation, hm, semantic[epoch], weight, labels
                            ),
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
        row["selected"] = int(
            row["epoch"] == selected_epoch and row["lambda"] == selected_lambda
        )
    return selected_epoch, selected_lambda, details, mappings


def frozen_references() -> tuple[
    dict[tuple[str, int, str], dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    hm = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): row
        for row in read_csv(HM_RESULTS / "predictions_by_seed.csv")
    }
    hmr = {
        row["sample_id"]: row for row in read_csv(HMR_RESULTS / "mean_sample_metrics.csv")
    }
    a = {
        row["sample_id"]: row
        for row in read_csv(BASELINE_PREDICTIONS)
        if row["method"] == "A"
    }
    if len(hm) != 3920 or len(hmr) != 784 or len(a) != 784:
        raise AssertionError("frozen comparison inputs changed")
    return hm, hmr, a


def evaluate_outer(
    method: str,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    output: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for seed in SEEDS:
        fitted_train, current_mapping = training_rows_for_method(
            method, train, seed, "outer", held_out, ""
        )
        mappings.extend(current_mapping)
        started = time.perf_counter()
        semantic = RAW.train_checkpoints(
            fitted_train, test, label_index, seed, (epoch,)
        )[epoch]
        hm = HMR.hm_rows(
            outer_matrix,
            [outer_lookup[(held_out, seed, row["sample_id"])] for row in test],
        )
        for row, hm_row in zip(test, hm):
            ranked, _ = BASE.ranking(HMR.fuse(hm_row, hm_row, 0.0), labels)
            if ranked[:20] != json.loads(
                frozen_hm[(held_out, seed, row["sample_id"])] ["top20_ids"]
            ):
                raise AssertionError(
                    f"lambda=0 HM reproduction failed: {held_out}/{seed}/{row['sample_id']}"
                )
        for row, hm_row, semantic_row in zip(test, hm, semantic):
            ranked, scores = BASE.ranking(HMR.fuse(hm_row, semantic_row, weight), labels)
            reference = frozen_a[row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": method,
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
                    "top20_scores": compact_json(
                        [round(float(value), 12) for value in scores[:20]]
                    ),
                    **BASE.sample_metrics(ranked[:5], row["targets"]),
                }
            )
        emit(
            f"outer method={method} held_out={held_out} seed={seed} "
            f"epoch={epoch} lambda={weight:.1f} "
            f"elapsed={time.perf_counter()-started:.1f}s"
        )
    return output, mappings


def mean_samples(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["method"], row["sample_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, sample_id), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise AssertionError(f"expected five seed rows: {method}/{sample_id}")
        first = values[0]
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "method": method,
                "sample_id": sample_id,
                "campaign_id": first["campaign_id"],
                "prefix_len": first["prefix_len"],
                "target_size": first["target_size"],
                "transition_visibility": first["transition_visibility"],
                "target_label_visibility": first["target_label_visibility"],
                "text_length_group": first["text_length_group"],
                **{
                    metric: statistics.fmean(float(row[metric]) for row in values)
                    for metric in METRICS
                },
            }
        )
    return output


def campaign_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
            "campaign_id": campaign,
            "rows": len(values),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in METRICS
            },
        }
        for (source, method, campaign), values in sorted(grouped.items())
    ]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for source in SOURCES:
            values = [
                row
                for row in campaigns
                if row["method"] == method and row["held_out_source"] == source
            ]
            output.append(
                {
                    "held_out_source": source,
                    "method": method,
                    "campaigns": len(values),
                    **{
                        f"campaign_macro_{metric}": statistics.fmean(
                            float(row[metric]) for row in values
                        )
                        for metric in METRICS
                    },
                }
            )
    return output


def reference_sample_rows(
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
    frozen_hmr: dict[str, dict[str, str]],
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        sample_ids = sorted(
            {sample_id for held, _, sample_id in frozen_hm if held == source}
        )
        for sample_id in sample_ids:
            first = frozen_hm[(source, SEEDS[0], sample_id)]
            output.append(
                {
                    "held_out_source": source,
                    "method": "HM",
                    "sample_id": sample_id,
                    "campaign_id": first["campaign_id"],
                    "prefix_len": first["prefix_len"],
                    "target_size": first["target_size"],
                    "transition_visibility": first["transition_visibility"],
                    "target_label_visibility": first["target_label_visibility"],
                    "text_length_group": first["text_length_group"],
                    **{
                        metric: statistics.fmean(
                            float(frozen_hm[(source, seed, sample_id)][metric])
                            for seed in SEEDS
                        )
                        for metric in METRICS
                    },
                }
            )
    for method, frozen in (("HM+R", frozen_hmr), ("A", frozen_a)):
        for row in frozen.values():
            output.append(
                {
                    "held_out_source": row["held_out_source"],
                    "method": method,
                    "sample_id": row.get("sample_id", ""),
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row.get("prefix_len", ""),
                    "target_size": row["target_size"],
                    "transition_visibility": row["transition_visibility"],
                    "target_label_visibility": row["target_label_visibility"],
                    "text_length_group": row["text_length_group"],
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    return output


def b0_sample_rows(frozen_a: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows = read_csv(B0_RANKINGS)
    if len(rows) != 784:
        raise AssertionError("B0 rankings are not 784 rows")
    for row in rows:
        ranked = json.loads(row["predicted_next_ttps"])
        targets = json.loads(row["target_parent_ids"])
        reference = frozen_a[row["sample_id"]]
        output.append(
            {
                "held_out_source": row["held_out_source"],
                "method": "B0",
                "sample_id": row["sample_id"],
                "campaign_id": row["campaign_id"],
                "prefix_len": row["prefix_len"],
                "target_size": row["target_size"],
                "transition_visibility": reference["transition_visibility"],
                "target_label_visibility": reference["target_label_visibility"],
                "text_length_group": reference["text_length_group"],
                **BASE.sample_metrics(ranked, targets),
            }
        )
    return output


def paired_bootstrap(all_campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["held_out_source"], row["method"], row["campaign_id"]): row
        for row in all_campaigns
    }
    campaign_ids = {
        source: sorted(
            row["campaign_id"]
            for row in all_campaigns
            if row["held_out_source"] == source and row["method"] == "HM+S"
        )
        for source in SOURCES
    }
    comparisons = (
        ("HM+S", "HM"),
        ("HM+S", "HM+P"),
        ("HM+S", "HM+R"),
        ("HM+S", "A"),
        ("HM+P", "HM"),
    )
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {
            source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]]
            for source in SOURCES
        }
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                source_values: list[float] = []
                for source in SOURCES:
                    delta = statistics.fmean(
                        float(lookup[(source, left, campaign)][metric])
                        - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    source_values.append(delta)
                    replicates[(source, comparison, metric)].append(delta)
                replicates[("source_equal_overall", comparison, metric)].append(
                    statistics.fmean(source_values)
                )
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(
                        statistics.fmean(
                            float(lookup[(source, left, campaign)][metric])
                            - float(lookup[(source, right, campaign)][metric])
                            for campaign in campaign_ids[source]
                        )
                        for source in SOURCES
                    )
                else:
                    point = statistics.fmean(
                        float(lookup[(scope, left, campaign)][metric])
                        - float(lookup[(scope, right, campaign)][metric])
                        for campaign in campaign_ids[scope]
                    )
                values = replicates[(scope, comparison, metric)]
                output.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "metric": metric,
                        "point_estimate": point,
                        "ci95_low": BASE.percentile(values, 0.025),
                        "ci95_high": BASE.percentile(values, 0.975),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return output


def stratified(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "transition_visibility": lambda row: row["transition_visibility"],
        "target_label_visibility": lambda row: row["target_label_visibility"],
        "text_length": lambda row: row["text_length_group"],
        "target_size": lambda row: str(row["target_size"]),
        "prefix_len": lambda row: str(row["prefix_len"]),
    }
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        for method in ("HM+S", "HM+P", "HM"):
            method_rows = [
                row
                for row in rows
                if row["held_out_source"] == source and row["method"] == method
            ]
            for name, getter in definitions.items():
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in method_rows:
                    groups[getter(row)].append(row)
                for group, values in sorted(groups.items()):
                    campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in values:
                        campaigns[row["campaign_id"]].append(row)
                    output.append(
                        {
                            "held_out_source": source,
                            "method": method,
                            "stratum": name,
                            "group": group,
                            "rows": len(values),
                            "campaigns": len(campaigns),
                            "inferentially_eligible": int(
                                len(values) >= 20 and len(campaigns) >= 5
                            ),
                            **{
                                f"campaign_macro_{metric}": statistics.fmean(
                                    statistics.fmean(float(row[metric]) for row in rows_)
                                    for rows_ in campaigns.values()
                                )
                                for metric in METRICS
                            },
                        }
                    )
    return output


def report_markdown(
    folds: Sequence[dict[str, Any]],
    all_campaigns: Sequence[dict[str, Any]],
    selections: dict[tuple[str, str], tuple[int, float]],
    differences: Sequence[dict[str, Any]],
) -> str:
    campaign_fold: dict[tuple[str, str], dict[str, float]] = {}
    for method in ("HM", "HM+R", "B0"):
        for source in SOURCES:
            values = [
                row
                for row in all_campaigns
                if row["method"] == method and row["held_out_source"] == source
            ]
            campaign_fold[(method, source)] = {
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in METRICS
            }
    for row in folds:
        campaign_fold[(row["method"], row["held_out_source"])] = {
            metric: float(row[f"campaign_macro_{metric}"]) for metric in METRICS
        }
    diff = {
        (row["scope"], row["comparison"], row["metric"]): row
        for row in differences
    }
    lines = [
        "# HM + LLM summary future-3 LODO results",
        "",
        "## Campaign-macro metrics (five-seed mean for probe methods)",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("HM", "HM+R", "HM+S", "HM+P", "B0"):
        values = [campaign_fold[(method, source)]["ndcg5"] for source in SOURCES]
        lines.append(
            f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | "
            f"**{statistics.fmean(values):.4f}** |"
        )
    lines.extend(
        [
            "",
            "## Selected S/P hyperparameters",
            "",
            "| Method | Held out | Epoch | Lambda |",
            "|---|---|---:|---:|",
        ]
    )
    for method in METHODS:
        for source in SOURCES:
            epoch, weight = selections[(method, source)]
            lines.append(f"| {method} | {source} | {epoch} | {weight:.1f} |")
    lines.extend(
        [
            "",
            "## Source-equal paired NDCG@5 differences",
            "",
            "| Comparison | Delta | 95% campaign-bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for comparison in ("HM+S-HM", "HM+S-HM+P", "HM+S-HM+R", "HM+P-HM"):
        row = diff[("source_equal_overall", comparison, "ndcg5")]
        lines.append(
            f"| {comparison} | {row['point_estimate']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "All lambda=0 outer rankings reproduced the frozen HM Top-20 rows.",
            "All P training mappings were derangements within source and prefix-length tercile; validation and test summaries were never permuted.",
            "The 30 development rows were excluded throughout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = (
        "inner_selection.csv",
        "selected_hyperparameters.csv",
        "permutation_mappings.csv",
        "predictions_by_seed.csv",
        "mean_sample_metrics.csv",
        "campaign_results_five_seed_mean.csv",
        "fold_results_five_seed_mean.csv",
        "b0_sample_metrics.csv",
        "paired_bootstrap_differences.csv",
        "stratified_results.csv",
        "report.md",
        "stdout.log",
        "results_manifest.json",
    )
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite HM+S/P results: {existing}")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, _ = load_data()
    inner_matrix, inner_lookup, outer_matrix, outer_lookup = HMR.load_cache()
    frozen_hm, frozen_hmr, frozen_a = frozen_references()

    selections: dict[tuple[str, str], tuple[int, float]] = {}
    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    started = time.perf_counter()
    for method in METHODS:
        for held_out in SOURCES:
            epoch, weight, details, inner_mappings = select_hyperparameters(
                method,
                held_out,
                rows,
                labels,
                label_index,
                inner_matrix,
                inner_lookup,
            )
            selections[(method, held_out)] = (epoch, weight)
            inner.extend(details)
            mappings.extend(inner_mappings)
            emit(
                f"selected method={method} held_out={held_out} "
                f"epoch={epoch} lambda={weight:.1f}"
            )
            outer_predictions, outer_mappings = evaluate_outer(
                method,
                held_out,
                rows,
                labels,
                label_index,
                epoch,
                weight,
                outer_matrix,
                outer_lookup,
                frozen_hm,
                frozen_a,
            )
            predictions.extend(outer_predictions)
            mappings.extend(outer_mappings)
    elapsed = time.perf_counter() - started
    if len(predictions) != 7840:
        raise AssertionError(f"unexpected S/P prediction rows: {len(predictions)}")
    if len(mappings) != 15680 or any(
        row["recipient_sample_id"] == row["donor_sample_id"] for row in mappings
    ):
        raise AssertionError(f"P mapping count/fixed-point gate failed: {len(mappings)}")
    predictions.sort(
        key=lambda row: (row["method"], row["held_out_source"], row["seed"], row["sample_id"])
    )
    inner.sort(
        key=lambda row: (
            row["method"],
            row["held_out_source"],
            row["inner_validation_source"],
            row["seed"],
            row["epoch"],
            row["lambda"],
        )
    )
    mappings.sort(
        key=lambda row: (
            row["stage"],
            row["held_out_source"],
            row["inner_validation_source"],
            row["seed"],
            row["recipient_sample_id"],
        )
    )
    mean_rows = mean_samples(predictions)
    our_campaigns = campaign_rows(mean_rows)
    folds = fold_rows(our_campaigns)
    reference_rows = reference_sample_rows(frozen_hm, frozen_hmr, frozen_a)
    b0_rows = b0_sample_rows(frozen_a)
    all_sample_rows = [*mean_rows, *reference_rows, *b0_rows]
    all_campaigns = [*our_campaigns, *campaign_rows(reference_rows), *campaign_rows(b0_rows)]
    differences = paired_bootstrap(all_campaigns)
    strata = stratified(all_sample_rows)
    selection_rows = [
        {
            "method": method,
            "held_out_source": source,
            "selected_epoch": selections[(method, source)][0],
            "selected_lambda": selections[(method, source)][1],
        }
        for method in METHODS
        for source in SOURCES
    ]

    write_csv(
        output / "inner_selection.csv",
        inner,
        (
            "method",
            "held_out_source",
            "inner_validation_source",
            "seed",
            "epoch",
            "lambda",
            "campaign_macro_ndcg5",
            "five_seed_two_source_mean_ndcg5",
            "selected",
            "training_rows",
            "validation_rows",
        ),
    )
    write_csv(
        output / "selected_hyperparameters.csv",
        selection_rows,
        ("method", "held_out_source", "selected_epoch", "selected_lambda"),
    )
    write_csv(
        output / "permutation_mappings.csv",
        mappings,
        (
            "stage",
            "held_out_source",
            "inner_validation_source",
            "seed",
            "permutation_seed",
            "source",
            "prefix_tercile",
            "recipient_sample_id",
            "donor_sample_id",
            "recipient_prefix_len",
            "donor_prefix_len",
        ),
    )
    prediction_columns = (
        "held_out_source",
        "method",
        "seed",
        "selected_epoch",
        "selected_lambda",
        "sample_id",
        "campaign_id",
        "prefix_len",
        "target_parent_ids",
        "target_size",
        "transition_visibility",
        "target_label_visibility",
        "text_length_group",
        "top20_ids",
        "top20_scores",
        *METRICS,
    )
    write_csv(output / "predictions_by_seed.csv", predictions, prediction_columns)
    mean_columns = (
        "held_out_source",
        "method",
        "sample_id",
        "campaign_id",
        "prefix_len",
        "target_size",
        "transition_visibility",
        "target_label_visibility",
        "text_length_group",
        *METRICS,
    )
    write_csv(output / "mean_sample_metrics.csv", mean_rows, mean_columns)
    write_csv(output / "b0_sample_metrics.csv", b0_rows, mean_columns)
    write_csv(
        output / "campaign_results_five_seed_mean.csv",
        our_campaigns,
        ("held_out_source", "method", "campaign_id", "rows", *METRICS),
    )
    write_csv(
        output / "fold_results_five_seed_mean.csv",
        folds,
        (
            "held_out_source",
            "method",
            "campaigns",
            *[f"campaign_macro_{metric}" for metric in METRICS],
        ),
    )
    write_csv(
        output / "paired_bootstrap_differences.csv",
        differences,
        (
            "scope",
            "comparison",
            "metric",
            "point_estimate",
            "ci95_low",
            "ci95_high",
            "replicates",
            "seed",
        ),
    )
    write_csv(
        output / "stratified_results.csv",
        strata,
        (
            "held_out_source",
            "method",
            "stratum",
            "group",
            "rows",
            "campaigns",
            "inferentially_eligible",
            *[f"campaign_macro_{metric}" for metric in METRICS],
        ),
    )
    report = report_markdown(folds, all_campaigns, selections, differences)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text(
        "\n".join(RUN_LOG) + "\n" + report + f"\nelapsed_seconds={elapsed:.3f}\n",
        encoding="utf-8",
    )
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
        "inputs": {
            "embedding_manifest_sha256": sha256(EMBEDDING_MANIFEST),
            "summary_manifest_sha256": sha256(SUMMARY_MANIFEST),
            "b0_rankings_sha256": sha256(B0_RANKINGS),
            "hm_cache_manifest_sha256": sha256(HM_CACHE / "cache_manifest.json"),
            "hm_results_manifest_sha256": sha256(HM_RESULTS / "results_manifest.json"),
            "hmr_results_manifest_sha256": sha256(HMR_RESULTS / "results_manifest.json"),
            "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS),
            "rows": len(rows),
        },
        "parameters": {
            "architecture": "1024-256-GELU-dropout0.3-184",
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "learning_rate": RAW.LEARNING_RATE,
            "weight_decay": RAW.WEIGHT_DECAY,
            "batch_size": RAW.BATCH_SIZE,
            "seeds": list(SEEDS),
            "epoch_grid": list(EPOCHS),
            "lambda_grid": list(LAMBDAS),
            "selected": {
                method: {
                    source: {
                        "epoch": selections[(method, source)][0],
                        "lambda": selections[(method, source)][1],
                    }
                    for source in SOURCES
                }
                for method in METHODS
            },
            "permutation": {
                "seed": "9000 + train_seed",
                "strata": "training source x deterministic equal-count prefix-length tercile",
                "mapping": "shuffle sorted IDs once per run/group, circular next-donor",
                "fixed_points": 0,
                "mapping_rows": len(mappings),
            },
            "num_threads": args.num_threads,
            "deterministic_algorithms": True,
        },
        "lambda0_hm_reproduction_gate": "PASS all 7840 S/P outer seed-sample rows",
        "development_rows_used": 0,
        "elapsed_seconds": elapsed,
        "outputs_sha256": {
            name: sha256(output / name)
            for name in managed
            if name != "results_manifest.json" and (output / name).exists()
        },
    }
    write_json(output / "results_manifest.json", manifest)
    print(report)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"wrote results to {output}")


if __name__ == "__main__":
    main()
