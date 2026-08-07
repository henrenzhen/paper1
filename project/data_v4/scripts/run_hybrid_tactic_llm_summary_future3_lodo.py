#!/usr/bin/env python3
"""Complete frozen HM+T and HM+ST future-3 LODO experiments."""

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
SUMMARY_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_hybrid_llm_summary_future3_lodo.py"
SUMMARY_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_llm_summary_future3_lodo_v1"
HM_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_markov_lstm_future3_lodo_v1"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/hybrid_tactic_llm_summary_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/hybrid_tactic_llm_summary_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (20, 40, 60, 80, 100)
LAMBDAS = tuple(index / 10 for index in range(11))
SIMPLEX = tuple(
    (h / 10, s / 10, (10 - h - s) / 10)
    for h in range(11)
    for s in range(11 - h)
)
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


SUMMARY = import_module("summary_for_tactic_completion", SUMMARY_SCRIPT)
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


def standardize_rows(values: np.ndarray, probabilities: bool = False) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if probabilities:
        clipped = np.clip(matrix, 1e-6, 1 - 1e-6)
        matrix = np.log(clipped / (1 - clipped))
    means = matrix.mean(axis=1, keepdims=True)
    standard_deviations = matrix.std(axis=1, keepdims=True)
    safe = np.where(standard_deviations < 1e-6, 1.0, standard_deviations)
    output = (matrix - means) / safe
    output[standard_deviations[:, 0] < 1e-6] = 0.0
    return output


def tactic_scores(
    train: Sequence[dict[str, Any]],
    evaluation: Sequence[dict[str, Any]],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
) -> np.ndarray:
    model = BASE.RelevanceModel(
        train,
        14,
        lambda row: BASE.tactic_target_indices(row, tactic_by_label),
    )
    return np.asarray(
        [
            BASE.tactic_candidate_scores(model.score(row["history"]), candidate_tactics)
            for row in evaluation
        ],
        dtype=np.float64,
    )


def campaign_macro_ndcg(
    rows: Sequence[dict[str, Any]], scores: np.ndarray, label_index: dict[str, int]
) -> float:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    by_campaign: dict[str, list[float]] = defaultdict(list)
    for row, ranked in zip(rows, order):
        targets = {label_index[label] for label in row["targets"]}
        dcg = sum(
            int(int(index) in targets) / np.log2(rank + 2)
            for rank, index in enumerate(ranked)
        )
        ideal = sum(1 / np.log2(rank + 2) for rank in range(min(5, len(targets))))
        by_campaign[row["campaign_id"]].append(float(dcg / ideal))
    return statistics.fmean(statistics.fmean(values) for values in by_campaign.values())


def select_fold(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    inner_matrix: np.ndarray,
    inner_lookup: dict[tuple[str, str, int, str], int],
) -> tuple[float, tuple[int, float, float, float], list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    for validation_source in training_sources:
        train = [
            row
            for row in rows
            if row["source"] in training_sources and row["source"] != validation_source
        ]
        validation = [row for row in rows if row["source"] == validation_source]
        tactic_z = standardize_rows(
            tactic_scores(train, validation, candidate_tactics, tactic_by_label),
            probabilities=True,
        )
        for seed in SEEDS:
            started = time.perf_counter()
            semantic = SUMMARY.RAW.train_checkpoints(
                train, validation, label_index, seed, EPOCHS
            )
            hm = HMR.hm_rows(
                inner_matrix,
                [
                    inner_lookup[(held_out, validation_source, seed, row["sample_id"])]
                    for row in validation
                ],
            )
            hm_z = standardize_rows(hm, probabilities=True)
            emit(
                f"inner held_out={held_out} validation={validation_source} "
                f"seed={seed} elapsed={time.perf_counter()-started:.1f}s"
            )
            for tactic_weight in LAMBDAS:
                score = campaign_macro_ndcg(
                    validation,
                    (1 - tactic_weight) * hm_z + tactic_weight * tactic_z,
                    label_index,
                )
                details.append(
                    {
                        "method": "HM+T",
                        "held_out_source": held_out,
                        "inner_validation_source": validation_source,
                        "seed": seed,
                        "epoch": "",
                        "w_h": 1 - tactic_weight,
                        "w_s": 0.0,
                        "w_t": tactic_weight,
                        "campaign_macro_ndcg5": score,
                        "training_rows": len(train),
                        "validation_rows": len(validation),
                    }
                )
            for epoch in EPOCHS:
                semantic_z = standardize_rows(semantic[epoch])
                for w_h, w_s, w_t in SIMPLEX:
                    score = campaign_macro_ndcg(
                        validation,
                        w_h * hm_z + w_s * semantic_z + w_t * tactic_z,
                        label_index,
                    )
                    details.append(
                        {
                            "method": "HM+ST",
                            "held_out_source": held_out,
                            "inner_validation_source": validation_source,
                            "seed": seed,
                            "epoch": epoch,
                            "w_h": w_h,
                            "w_s": w_s,
                            "w_t": w_t,
                            "campaign_macro_ndcg5": score,
                            "training_rows": len(train),
                            "validation_rows": len(validation),
                        }
                    )

    grouped: dict[tuple[str, str, float, float, float], list[float]] = defaultdict(list)
    for row in details:
        key = (row["method"], str(row["epoch"]), row["w_h"], row["w_s"], row["w_t"])
        grouped[key].append(row["campaign_macro_ndcg5"])
    means = {key: statistics.fmean(values) for key, values in grouped.items()}

    hm_t_keys = [key for key in means if key[0] == "HM+T"]
    hm_t_best = max(means[key] for key in hm_t_keys)
    hm_t_selected = min(
        (key for key in hm_t_keys if abs(means[key] - hm_t_best) <= 1e-12),
        key=lambda key: key[4],
    )
    st_keys = [key for key in means if key[0] == "HM+ST"]
    st_best = max(means[key] for key in st_keys)
    st_selected = min(
        (key for key in st_keys if abs(means[key] - st_best) <= 1e-12),
        key=lambda key: (key[3], key[4], -key[2], int(key[1])),
    )
    for row in details:
        key = (row["method"], str(row["epoch"]), row["w_h"], row["w_s"], row["w_t"])
        row["five_seed_two_source_mean_ndcg5"] = means[key]
        row["selected"] = int(key in (hm_t_selected, st_selected))
    return hm_t_selected[4], (
        int(st_selected[1]), st_selected[2], st_selected[3], st_selected[4]
    ), details


def load_frozen_rankings() -> tuple[
    dict[tuple[str, int, str], dict[str, str]],
    dict[tuple[str, int, str], dict[str, str]],
    dict[str, tuple[int, float]],
]:
    hm = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): row
        for row in read_csv(HM_RESULTS / "predictions_by_seed.csv")
    }
    hm_s = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): row
        for row in read_csv(SUMMARY_RESULTS / "predictions_by_seed.csv")
        if row["method"] == "HM+S"
    }
    selection = {
        row["held_out_source"]: (int(row["selected_epoch"]), float(row["selected_lambda"]))
        for row in read_csv(SUMMARY_RESULTS / "selected_hyperparameters.csv")
        if row["method"] == "HM+S"
    }
    if len(hm) != 3920 or len(hm_s) != 3920 or len(selection) != 3:
        raise AssertionError("frozen HM/HM+S rankings changed")
    return hm, hm_s, selection


def outer_fold(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    hm_t_weight: float,
    st_selection: tuple[int, float, float, float],
    outer_matrix: np.ndarray,
    outer_lookup: dict[tuple[str, int, str], int],
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
    frozen_hm_s: dict[tuple[str, int, str], dict[str, str]],
    frozen_s_selection: dict[str, tuple[int, float]],
) -> list[dict[str, Any]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    tactic_z = standardize_rows(
        tactic_scores(train, test, candidate_tactics, tactic_by_label),
        probabilities=True,
    )
    st_epoch, w_h, w_s, w_t = st_selection
    old_s_epoch, old_s_weight = frozen_s_selection[held_out]
    predictions: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        checkpoints = SUMMARY.RAW.train_checkpoints(
            train, test, label_index, seed, tuple(sorted({st_epoch, old_s_epoch}))
        )
        hm = HMR.hm_rows(
            outer_matrix,
            [outer_lookup[(held_out, seed, row["sample_id"])] for row in test],
        )
        hm_z = standardize_rows(hm, probabilities=True)
        semantic_z = standardize_rows(checkpoints[st_epoch])
        old_semantic_z = standardize_rows(checkpoints[old_s_epoch])
        hm_t_scores = (1 - hm_t_weight) * hm_z + hm_t_weight * tactic_z
        st_scores = w_h * hm_z + w_s * semantic_z + w_t * tactic_z
        for index, row in enumerate(test):
            key = (held_out, seed, row["sample_id"])
            hm_ranked, _ = BASE.ranking(hm_z[index], labels)
            if hm_ranked[:20] != json.loads(frozen_hm[key]["top20_ids"]):
                raise AssertionError(f"HM reproduction failed: {key}")
            old_s_scores = (1 - old_s_weight) * hm_z[index] + old_s_weight * old_semantic_z[index]
            old_s_ranked, _ = BASE.ranking(old_s_scores, labels)
            if old_s_ranked[:20] != json.loads(frozen_hm_s[key]["top20_ids"]):
                raise AssertionError(f"HM+S reproduction failed: {key}")
            for method, values in (("HM+T", hm_t_scores[index]), ("HM+ST", st_scores[index])):
                ranked, ordered = BASE.ranking(values, labels)
                predictions.append(
                    {
                        "held_out_source": held_out,
                        "method": method,
                        "seed": seed,
                        "selected_epoch": st_epoch if method == "HM+ST" else "",
                        "w_h": w_h if method == "HM+ST" else 1 - hm_t_weight,
                        "w_s": w_s if method == "HM+ST" else 0.0,
                        "w_t": w_t if method == "HM+ST" else hm_t_weight,
                        "sample_id": row["sample_id"],
                        "campaign_id": row["campaign_id"],
                        "prefix_len": row["prefix_len"],
                        "target_parent_ids": compact_json(row["targets"]),
                        "target_size": row["target_size"],
                        "transition_visibility": frozen_hm[key]["transition_visibility"],
                        "target_label_visibility": frozen_hm[key]["target_label_visibility"],
                        "text_length_group": frozen_hm[key]["text_length_group"],
                        "top20_ids": compact_json(ranked[:20]),
                        "top20_scores": compact_json([round(float(value), 12) for value in ordered[:20]]),
                        **BASE.sample_metrics(ranked[:5], row["targets"]),
                    }
                )
        emit(
            f"outer held_out={held_out} seed={seed} st_epoch={st_epoch} "
            f"elapsed={time.perf_counter()-started:.1f}s"
        )
    return predictions


def mean_seed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["sample_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, sample_id), values in sorted(grouped.items()):
        if len(values) != 5:
            raise AssertionError(f"not five seeds: {method}/{sample_id}")
        first = values[0]
        output.append(
            {
                **{key: first[key] for key in ("held_out_source", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                "method": method,
                "sample_id": sample_id,
                **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
            }
        )
    return output


def reference_rows(
    fused: Sequence[dict[str, Any]],
    frozen_hm: dict[tuple[str, int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    metadata = {row["sample_id"]: row for row in fused}
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        sample_ids = sorted({sample for held, _, sample in frozen_hm if held == source})
        for sample_id in sample_ids:
            first = frozen_hm[(source, SEEDS[0], sample_id)]
            output.append(
                {
                    **{key: metadata[sample_id][key] for key in ("held_out_source", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                    "method": "HM",
                    "sample_id": sample_id,
                    **{metric: statistics.fmean(float(frozen_hm[(source, seed, sample_id)][metric]) for seed in SEEDS) for metric in METRICS},
                }
            )
    for row in read_csv(SUMMARY_RESULTS / "mean_sample_metrics.csv"):
        if row["method"] != "HM+S":
            continue
        output.append(
            {
                **{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                "method": "HM+S",
                "sample_id": row["sample_id"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    if len(output) != 1568:
        raise AssertionError(f"reference row gate failed: {len(output)}")
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
            **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, method, campaign), values in sorted(grouped.items())
    ]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in ("HM", "HM+S", "HM+T", "HM+ST"):
        for source in SOURCES:
            values = [row for row in campaigns if row["method"] == method and row["held_out_source"] == source]
            output.append(
                {
                    "held_out_source": source,
                    "method": method,
                    "campaigns": len(values),
                    **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
                }
            )
    return output


def bootstrap(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in campaigns}
    campaign_ids = {
        source: sorted(row["campaign_id"] for row in campaigns if row["held_out_source"] == source and row["method"] == "HM")
        for source in SOURCES
    }
    comparisons = (("HM+T", "HM"), ("HM+ST", "HM"), ("HM+ST", "HM+S"), ("HM+ST", "HM+T"))
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                source_values = []
                for source in SOURCES:
                    delta = statistics.fmean(
                        float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    source_values.append(delta)
                    replicates[(source, comparison, metric)].append(delta)
                replicates[("source_equal_overall", comparison, metric)].append(statistics.fmean(source_values))
    output = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(
                        statistics.fmean(
                            float(lookup[(source, left, campaign)][metric]) - float(lookup[(source, right, campaign)][metric])
                            for campaign in campaign_ids[source]
                        )
                        for source in SOURCES
                    )
                else:
                    point = statistics.fmean(
                        float(lookup[(scope, left, campaign)][metric]) - float(lookup[(scope, right, campaign)][metric])
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
    }
    output = []
    for source in SOURCES:
        for method in ("HM", "HM+S", "HM+T", "HM+ST"):
            method_rows = [row for row in rows if row["held_out_source"] == source and row["method"] == method]
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
                            "inferentially_eligible": int(len(values) >= 20 and len(campaigns) >= 5),
                            **{f"campaign_macro_{metric}": statistics.fmean(statistics.fmean(float(row[metric]) for row in items) for items in campaigns.values()) for metric in METRICS},
                        }
                    )
    return output


def report(
    folds: Sequence[dict[str, Any]], selections: dict[str, dict[str, Any]], differences: Sequence[dict[str, Any]]
) -> str:
    lookup = {(row["method"], row["held_out_source"]): row for row in folds}
    diff = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    lines = [
        "# HM + tactic + LLM-summary future-3 LODO results",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("HM", "HM+S", "HM+T", "HM+ST"):
        values = [float(lookup[(method, source)]["campaign_macro_ndcg5"]) for source in SOURCES]
        lines.append(f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | **{statistics.fmean(values):.4f}** |")
    lines.extend(["", "## Selected hyperparameters", "", "| Held out | HM+T w_T | HM+ST epoch | w_H | w_S | w_T |", "|---|---:|---:|---:|---:|---:|"])
    for source in SOURCES:
        value = selections[source]
        lines.append(f"| {source} | {value['hm_t_w_t']:.1f} | {value['st_epoch']} | {value['st_w_h']:.1f} | {value['st_w_s']:.1f} | {value['st_w_t']:.1f} |")
    lines.extend(["", "## Source-equal paired NDCG@5 differences", "", "| Comparison | Delta | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("HM+T-HM", "HM+ST-HM", "HM+ST-HM+S", "HM+ST-HM+T"):
        row = diff[("source_equal_overall", comparison, "ndcg5")]
        lines.append(f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    st_gt_s = sum(float(lookup[("HM+ST", source)]["campaign_macro_ndcg5"]) > float(lookup[("HM+S", source)]["campaign_macro_ndcg5"]) for source in SOURCES)
    st_gt_t = sum(float(lookup[("HM+ST", source)]["campaign_macro_ndcg5"]) > float(lookup[("HM+T", source)]["campaign_macro_ndcg5"]) for source in SOURCES)
    overall = {
        method: statistics.fmean(float(lookup[(method, source)]["campaign_macro_ndcg5"]) for source in SOURCES)
        for method in ("HM+S", "HM+T", "HM+ST")
    }
    complementarity = st_gt_s >= 2 and st_gt_t >= 2 and overall["HM+ST"] > overall["HM+S"] and overall["HM+ST"] > overall["HM+T"]
    lines.extend(
        [
            "",
            "## Frozen complementarity rule",
            "",
            f"- HM+ST > HM+S sources: {st_gt_s}/3",
            f"- HM+ST > HM+T sources: {st_gt_t}/3",
            f"- Overall exceeds both: {overall['HM+ST'] > overall['HM+S'] and overall['HM+ST'] > overall['HM+T']}",
            f"- Complementarity claim: {'SUPPORTED' if complementarity else 'NOT SUPPORTED'}",
            "",
            "HM and HM+S Top-20 reproduction gates passed for all 3,920 outer seed/sample rows.",
            "The negative preregistered HM+S primary result remains unchanged.",
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
        "predictions_by_seed.csv",
        "mean_sample_metrics.csv",
        "campaign_results.csv",
        "fold_results.csv",
        "paired_bootstrap_differences.csv",
        "stratified_results.csv",
        "report.md",
        "stdout.log",
        "results_manifest.json",
    )
    if any((output / name).exists() for name in managed):
        raise FileExistsError("refusing to overwrite HM+T/HM+ST results")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, _ = SUMMARY.load_data()
    candidate_tactics, tactic_by_label = BASE.parse_tactics()
    inner_matrix, inner_lookup, outer_matrix, outer_lookup = HMR.load_cache()
    frozen_hm, frozen_hm_s, frozen_s_selection = load_frozen_rankings()
    selections: dict[str, dict[str, Any]] = {}
    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for held_out in SOURCES:
        hm_t_weight, st_selection, details = select_fold(
            held_out,
            rows,
            labels,
            label_index,
            candidate_tactics,
            tactic_by_label,
            inner_matrix,
            inner_lookup,
        )
        st_epoch, w_h, w_s, w_t = st_selection
        selections[held_out] = {
            "hm_t_w_t": hm_t_weight,
            "st_epoch": st_epoch,
            "st_w_h": w_h,
            "st_w_s": w_s,
            "st_w_t": w_t,
        }
        inner.extend(details)
        emit(f"selected held_out={held_out} hm_t_w_t={hm_t_weight:.1f} st=({st_epoch},{w_h:.1f},{w_s:.1f},{w_t:.1f})")
        predictions.extend(
            outer_fold(
                held_out,
                rows,
                labels,
                label_index,
                candidate_tactics,
                tactic_by_label,
                hm_t_weight,
                st_selection,
                outer_matrix,
                outer_lookup,
                frozen_hm,
                frozen_hm_s,
                frozen_s_selection,
            )
        )
    if len(predictions) != 7840:
        raise AssertionError(f"prediction row gate failed: {len(predictions)}")
    predictions.sort(key=lambda row: (row["method"], row["held_out_source"], row["seed"], row["sample_id"]))
    inner.sort(key=lambda row: (row["method"], row["held_out_source"], row["inner_validation_source"], row["seed"], str(row["epoch"]), row["w_s"], row["w_t"]))
    means = mean_seed_rows(predictions)
    all_samples = [*means, *reference_rows(means, frozen_hm)]
    if len(all_samples) != 3136:
        raise AssertionError(f"all-sample gate failed: {len(all_samples)}")
    campaigns = campaign_rows(all_samples)
    folds = fold_rows(campaigns)
    differences = bootstrap(campaigns)
    strata = stratified(all_samples)
    selection_rows = [{"held_out_source": source, **selections[source]} for source in SOURCES]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "inner_selection.csv", inner, ("method", "held_out_source", "inner_validation_source", "seed", "epoch", "w_h", "w_s", "w_t", "campaign_macro_ndcg5", "five_seed_two_source_mean_ndcg5", "selected", "training_rows", "validation_rows"))
    write_csv(output / "selected_hyperparameters.csv", selection_rows, ("held_out_source", "hm_t_w_t", "st_epoch", "st_w_h", "st_w_s", "st_w_t"))
    prediction_columns = ("held_out_source", "method", "seed", "selected_epoch", "w_h", "w_s", "w_t", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "top20_ids", "top20_scores", *METRICS)
    write_csv(output / "predictions_by_seed.csv", predictions, prediction_columns)
    sample_columns = ("held_out_source", "method", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", *METRICS)
    write_csv(output / "mean_sample_metrics.csv", means, sample_columns)
    write_csv(output / "campaign_results.csv", campaigns, ("held_out_source", "method", "campaign_id", "rows", *METRICS))
    write_csv(output / "fold_results.csv", folds, ("held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]))
    write_csv(output / "paired_bootstrap_differences.csv", differences, ("scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"))
    write_csv(output / "stratified_results.csv", strata, ("held_out_source", "method", "stratum", "group", "rows", "campaigns", "inferentially_eligible", *[f"campaign_macro_{metric}" for metric in METRICS]))
    markdown = report(folds, selections, differences)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "stdout.log").write_text("\n".join(RUN_LOG) + "\n" + markdown + "\n", encoding="utf-8")
    output_hashes = {name: sha256(output / name) for name in managed if name != "results_manifest.json" and (output / name).exists()}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "inputs": {
            "summary_results_manifest_sha256": sha256(SUMMARY_RESULTS / "results_manifest.json"),
            "hm_results_manifest_sha256": sha256(HM_RESULTS / "results_manifest.json"),
            "hm_cache_manifest_sha256": sha256(SUMMARY.HM_CACHE / "cache_manifest.json"),
        },
        "parameters": {
            "seeds": list(SEEDS),
            "epochs": list(EPOCHS),
            "lambda_grid": list(LAMBDAS),
            "simplex_points": len(SIMPLEX),
            "selection": selections,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "hm_reproduction": "PASS all 3920 outer seed-sample rows",
        "hm_plus_s_reproduction": "PASS all 3920 outer seed-sample rows",
        "outputs_sha256": output_hashes,
    }
    (output / "results_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
