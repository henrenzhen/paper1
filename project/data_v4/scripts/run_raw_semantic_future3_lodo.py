#!/usr/bin/env python3
"""Run the frozen BGE-M3 raw-description semantic probe (R)."""

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
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
BASELINE_PREDICTIONS = (
    PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
)
EMBEDDING_DIR = PROJECT_ROOT / "data_v4/semantic_embeddings/bge_m3_5617a9f"
EMBEDDINGS = EMBEDDING_DIR / "embeddings.npy"
EMBEDDING_INDEX = EMBEDDING_DIR / "embedding_index.csv"
EMBEDDING_MANIFEST = EMBEDDING_DIR / "embedding_manifest.json"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/raw_semantic_probe_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/raw_semantic_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (20, 40, 60, 80, 100)
LAMBDAS = tuple(index / 10 for index in range(11))
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RUN_LOG: list[str] = []


def emit(message: str) -> None:
    RUN_LOG.append(message)
    print(message, flush=True)


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_data() -> tuple[list[dict[str, Any]], list[str], dict[str, int], np.ndarray]:
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    matrix = np.load(EMBEDDINGS, mmap_mode="r")
    index_rows = read_csv(EMBEDDING_INDEX)
    if matrix.shape != (814, 1024) or len(index_rows) != 814:
        raise AssertionError(f"unexpected embedding shape/index: {matrix.shape}/{len(index_rows)}")
    vector_by_sample = {
        row["sample_id"]: np.asarray(matrix[int(row["embedding_row"])], dtype=np.float32)
        for row in index_rows
    }
    for row in rows:
        if row["sample_id"] not in vector_by_sample:
            raise AssertionError(f"missing embedding: {row['sample_id']}")
        row["embedding"] = vector_by_sample[row["sample_id"]]
    return rows, labels, label_index, matrix


class Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 184),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def tensors(
    rows: Sequence[dict[str, Any]], label_index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.stack([row["embedding"] for row in rows])).float()
    y = torch.zeros((len(rows), len(label_index)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for target in row["targets"]:
            y[row_index, label_index[target]] = 1.0
    return x, y


def train_checkpoints(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    label_index: dict[str, int],
    seed: int,
    checkpoint_epochs: Sequence[int],
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = Probe()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_x, train_y = tensors(train_rows, label_index)
    validation_x, _ = tensors(validation_rows, label_index)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    outputs: dict[int, np.ndarray] = {}
    checkpoints = set(checkpoint_epochs)
    model.train()
    for epoch in range(1, max(checkpoint_epochs) + 1):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                outputs[epoch] = model(validation_x).cpu().numpy().astype(np.float32)
            model.train()
    return outputs


def a_scores(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
) -> list[list[float]]:
    model = BASE.RelevanceModel(
        train_rows,
        len(labels),
        lambda row: {label_index[label] for label in row["targets"]},
    )
    return [model.score(row["history"]) for row in test_rows]


def fuse(a_values: Sequence[float], semantic_logits: Sequence[float], weight: float) -> list[float]:
    a_z = BASE.standardize([BASE.clipped_logit(value) for value in a_values])
    semantic_z = BASE.standardize([float(value) for value in semantic_logits])
    return [
        (1 - weight) * left + weight * right
        for left, right in zip(a_z, semantic_z)
    ]


def score_validation(
    validation_rows: Sequence[dict[str, Any]],
    a_values: Sequence[Sequence[float]],
    semantic_logits: np.ndarray,
    weight: float,
    labels: Sequence[str],
) -> float:
    records: list[dict[str, Any]] = []
    for row, a_row, semantic_row in zip(validation_rows, a_values, semantic_logits):
        scores = fuse(a_row, semantic_row, weight)
        ranked, _ = BASE.ranking(scores, labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_hyperparameters(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
) -> tuple[int, float, list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    for validation_source in training_sources:
        train_rows = [row for row in rows if row["source"] in training_sources and row["source"] != validation_source]
        validation_rows = [row for row in rows if row["source"] == validation_source]
        a_values = a_scores(train_rows, validation_rows, labels, label_index)
        for seed in SEEDS:
            started = time.perf_counter()
            checkpoints = train_checkpoints(
                train_rows, validation_rows, label_index, seed, EPOCHS
            )
            elapsed = time.perf_counter() - started
            emit(
                f"inner held_out={held_out} validation={validation_source} seed={seed} "
                f"epochs=100 elapsed={elapsed:.1f}s"
            )
            for epoch in EPOCHS:
                for weight in LAMBDAS:
                    details.append(
                        {
                            "held_out_source": held_out,
                            "inner_validation_source": validation_source,
                            "seed": seed,
                            "epoch": epoch,
                            "lambda": weight,
                            "campaign_macro_ndcg5": score_validation(
                                validation_rows,
                                a_values,
                                checkpoints[epoch],
                                weight,
                                labels,
                            ),
                            "training_rows": len(train_rows),
                            "validation_rows": len(validation_rows),
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
    return selected_epoch, selected_lambda, details


def baseline_a_lookup() -> dict[str, dict[str, str]]:
    return {
        row["sample_id"]: row
        for row in read_csv(BASELINE_PREDICTIONS)
        if row["method"] == "A"
    }


def evaluate_outer(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    epoch: int,
    weight: float,
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if row["source"] != held_out]
    test_rows = [row for row in rows if row["source"] == held_out]
    a_values = a_scores(train_rows, test_rows, labels, label_index)
    for row, scores in zip(test_rows, a_values):
        ranked, _ = BASE.ranking(scores, labels)
        if ranked[:20] != json.loads(frozen_a[row["sample_id"]]["top20_ids"]):
            raise AssertionError(f"A ranking does not match frozen baseline: {row['sample_id']}")

    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        semantic = train_checkpoints(
            train_rows, test_rows, label_index, seed, (epoch,)
        )[epoch]
        elapsed = time.perf_counter() - started
        emit(
            f"outer held_out={held_out} seed={seed} epoch={epoch} elapsed={elapsed:.1f}s",
        )
        for row, a_row, semantic_row in zip(test_rows, a_values, semantic):
            scores = fuse(a_row, semantic_row, weight)
            ranked, ordered_scores = BASE.ranking(scores, labels)
            output.append(
                {
                    "held_out_source": held_out,
                    "method": "R",
                    "seed": seed,
                    "selected_epoch": epoch,
                    "selected_lambda": weight,
                    "sample_id": row["sample_id"],
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "target_parent_ids": compact_json(row["targets"]),
                    "target_size": row["target_size"],
                    "transition_visibility": frozen_a[row["sample_id"]]["transition_visibility"],
                    "target_label_visibility": frozen_a[row["sample_id"]]["target_label_visibility"],
                    "text_length_group": frozen_a[row["sample_id"]]["text_length_group"],
                    "top20_ids": compact_json(ranked[:20]),
                    "top20_scores": compact_json(
                        [round(float(value), 12) for value in ordered_scores[:20]]
                    ),
                    **BASE.sample_metrics(ranked[:5], row["targets"]),
                }
            )
    return output


def by_seed_fold_results(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        for seed in SEEDS:
            rows = [row for row in predictions if row["held_out_source"] == source and row["seed"] == seed]
            campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                campaigns[row["campaign_id"]].append(row)
            output.append(
                {
                    "held_out_source": source,
                    "seed": seed,
                    "rows": len(rows),
                    "campaigns": len(campaigns),
                    **{
                        f"campaign_macro_{metric}": statistics.fmean(
                            statistics.fmean(float(row[metric]) for row in values)
                            for values in campaigns.values()
                        )
                        for metric in METRICS
                    },
                }
            )
    return output


def mean_sample_metrics(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    for sample_id, rows in sorted(grouped.items()):
        if len(rows) != len(SEEDS):
            raise AssertionError(f"sample does not have five seeds: {sample_id}")
        first = rows[0]
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "sample_id": sample_id,
                "campaign_id": first["campaign_id"],
                "target_size": first["target_size"],
                "transition_visibility": first["transition_visibility"],
                "target_label_visibility": first["target_label_visibility"],
                "text_length_group": first["text_length_group"],
                **{
                    metric: statistics.fmean(float(row[metric]) for row in rows)
                    for metric in METRICS
                },
            }
        )
    return output


def campaign_means(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "campaign_id": campaign,
            "rows": len(values),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in METRICS
            },
        }
        for (source, campaign), values in sorted(grouped.items())
    ]


def fold_mean_results(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        rows = [row for row in campaigns if row["held_out_source"] == source]
        output.append(
            {
                "held_out_source": source,
                "campaigns": len(rows),
                **{
                    f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in rows)
                    for metric in METRICS
                },
            }
        )
    return output


def paired_bootstrap(
    r_campaigns: Sequence[dict[str, Any]], frozen_a: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    a_sample_rows = [
        {
            "held_out_source": row["held_out_source"],
            "campaign_id": row["campaign_id"],
            **{metric: float(row[metric]) for metric in METRICS},
        }
        for row in frozen_a.values()
    ]
    a_campaigns = campaign_means(a_sample_rows)
    r_lookup = {(row["held_out_source"], row["campaign_id"]): row for row in r_campaigns}
    a_lookup = {(row["held_out_source"], row["campaign_id"]): row for row in a_campaigns}
    campaign_ids = {
        source: sorted(campaign for held, campaign in r_lookup if held == source)
        for source in SOURCES
    }
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        source_deltas: dict[str, dict[str, float]] = {}
        for source in SOURCES:
            draw = [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]]
            source_deltas[source] = {
                metric: statistics.fmean(
                    float(r_lookup[(source, campaign)][metric])
                    - float(a_lookup[(source, campaign)][metric])
                    for campaign in draw
                )
                for metric in METRICS
            }
            for metric in METRICS:
                replicates[(source, metric)].append(source_deltas[source][metric])
        for metric in METRICS:
            replicates[("source_equal_overall", metric)].append(
                statistics.fmean(source_deltas[source][metric] for source in SOURCES)
            )
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for metric in METRICS:
            if scope == "source_equal_overall":
                point = statistics.fmean(
                    statistics.fmean(
                        float(r_lookup[(source, campaign)][metric])
                        - float(a_lookup[(source, campaign)][metric])
                        for campaign in campaign_ids[source]
                    )
                    for source in SOURCES
                )
            else:
                point = statistics.fmean(
                    float(r_lookup[(scope, campaign)][metric])
                    - float(a_lookup[(scope, campaign)][metric])
                    for campaign in campaign_ids[scope]
                )
            values = replicates[(scope, metric)]
            output.append(
                {
                    "scope": scope,
                    "comparison": "R-A",
                    "metric": metric,
                    "point_estimate": point,
                    "ci95_low": BASE.percentile(values, 0.025),
                    "ci95_high": BASE.percentile(values, 0.975),
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return output


def stratified(mean_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "transition_visibility": lambda row: row["transition_visibility"],
        "target_label_visibility": lambda row: row["target_label_visibility"],
        "text_length": lambda row: row["text_length_group"],
        "target_size": lambda row: str(row["target_size"]),
    }
    output: list[dict[str, Any]] = []
    for source in SOURCES:
        source_rows = [row for row in mean_rows if row["held_out_source"] == source]
        for name, getter in definitions.items():
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in source_rows:
                groups[getter(row)].append(row)
            for group, rows in sorted(groups.items()):
                campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    campaigns[row["campaign_id"]].append(row)
                output.append(
                    {
                        "held_out_source": source,
                        "stratum": name,
                        "group": group,
                        "rows": len(rows),
                        "campaigns": len(campaigns),
                        "inferentially_eligible": int(len(rows) >= 20 and len(campaigns) >= 5),
                        **{
                            f"campaign_macro_{metric}": statistics.fmean(
                                statistics.fmean(float(row[metric]) for row in values)
                                for values in campaigns.values()
                            )
                            for metric in METRICS
                        },
                    }
                )
    return output


def report_markdown(
    folds: Sequence[dict[str, Any]],
    selections: dict[str, tuple[int, float]],
    deltas: Sequence[dict[str, Any]],
) -> str:
    fold_lookup = {row["held_out_source"]: row for row in folds}
    delta_lookup = {(row["scope"], row["metric"]): row for row in deltas}
    overall = {
        metric: statistics.fmean(
            float(row[f"campaign_macro_{metric}"]) for row in folds
        )
        for metric in METRICS
    }
    lines = [
        "# Raw-description semantic R results",
        "",
        "BGE-M3 is frozen. No LLM or external generation API is used.",
        "",
        "## Campaign-macro metrics (five-seed mean)",
        "",
        "| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | Selected epoch | Selected lambda |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        row = fold_lookup[source]
        epoch, weight = selections[source]
        lines.append(
            f"| {source} | {row['campaign_macro_ndcg5']:.4f} | "
            f"{row['campaign_macro_hit5']:.4f} | {row['campaign_macro_precision5']:.4f} | "
            f"{row['campaign_macro_recall5']:.4f} | {epoch} | {weight:.1f} |"
        )
    lines.append(
        f"| **Source-equal overall** | **{overall['ndcg5']:.4f}** | "
        f"**{overall['hit5']:.4f}** | **{overall['precision5']:.4f}** | "
        f"**{overall['recall5']:.4f}** | — | — |"
    )
    lines.extend(
        [
            "",
            "## Paired R - A NDCG@5",
            "",
            "| Scope | Delta | 95% campaign-bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for scope in (*SOURCES, "source_equal_overall"):
        row = delta_lookup[(scope, "ndcg5")]
        lines.append(
            f"| {scope} | {row['point_estimate']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Raw-description fusion does not satisfy a cross-source support claim when the "
            "source-equal R-A difference is negative or any held-out source degrades materially. "
            "This result does not evaluate the LLM-normalized S branch.",
            "",
            "The 30 development rows were excluded from inner fitting, hyperparameter selection, and outer evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = [
        "inner_selection.csv",
        "selected_hyperparameters.csv",
        "predictions_by_seed.csv",
        "fold_results_by_seed.csv",
        "mean_sample_metrics.csv",
        "campaign_results_five_seed_mean.csv",
        "fold_results_five_seed_mean.csv",
        "paired_bootstrap_differences.csv",
        "stratified_results.csv",
        "report.md",
        "results_manifest.json",
        "stdout.log",
    ]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite semantic results: {existing}")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, matrix = load_data()
    frozen_a = baseline_a_lookup()
    if len(frozen_a) != 784:
        raise AssertionError("frozen A predictions do not contain 784 rows")

    inner_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selections: dict[str, tuple[int, float]] = {}
    started = time.perf_counter()
    for held_out in SOURCES:
        epoch, weight, details = select_hyperparameters(
            held_out, rows, labels, label_index
        )
        selections[held_out] = (epoch, weight)
        inner_rows.extend(details)
        message = f"selected held_out={held_out} epoch={epoch} lambda={weight:.1f}"
        emit(message)
        predictions.extend(
            evaluate_outer(
                held_out,
                rows,
                labels,
                label_index,
                epoch,
                weight,
                frozen_a,
            )
        )
    elapsed = time.perf_counter() - started
    inner_rows.sort(
        key=lambda row: (
            row["held_out_source"],
            row["inner_validation_source"],
            row["seed"],
            row["epoch"],
            row["lambda"],
        )
    )
    predictions.sort(
        key=lambda row: (row["held_out_source"], row["seed"], row["sample_id"])
    )
    by_seed = by_seed_fold_results(predictions)
    mean_rows = mean_sample_metrics(predictions)
    campaign_rows = campaign_means(mean_rows)
    fold_rows = fold_mean_results(campaign_rows)
    deltas = paired_bootstrap(campaign_rows, frozen_a)
    strata = stratified(mean_rows)
    selection_rows = [
        {"held_out_source": source, "selected_epoch": values[0], "selected_lambda": values[1]}
        for source, values in selections.items()
    ]

    write_csv(
        output / "inner_selection.csv",
        inner_rows,
        [
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
        ],
    )
    write_csv(
        output / "selected_hyperparameters.csv",
        selection_rows,
        ["held_out_source", "selected_epoch", "selected_lambda"],
    )
    write_csv(
        output / "predictions_by_seed.csv",
        predictions,
        [
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
        ],
    )
    write_csv(
        output / "fold_results_by_seed.csv",
        by_seed,
        [
            "held_out_source",
            "seed",
            "rows",
            "campaigns",
            *[f"campaign_macro_{metric}" for metric in METRICS],
        ],
    )
    write_csv(
        output / "mean_sample_metrics.csv",
        mean_rows,
        [
            "held_out_source",
            "sample_id",
            "campaign_id",
            "target_size",
            "transition_visibility",
            "target_label_visibility",
            "text_length_group",
            *METRICS,
        ],
    )
    write_csv(
        output / "campaign_results_five_seed_mean.csv",
        campaign_rows,
        ["held_out_source", "campaign_id", "rows", *METRICS],
    )
    write_csv(
        output / "fold_results_five_seed_mean.csv",
        fold_rows,
        ["held_out_source", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]],
    )
    write_csv(
        output / "paired_bootstrap_differences.csv",
        deltas,
        [
            "scope",
            "comparison",
            "metric",
            "point_estimate",
            "ci95_low",
            "ci95_high",
            "replicates",
            "seed",
        ],
    )
    write_csv(
        output / "stratified_results.csv",
        strata,
        [
            "held_out_source",
            "stratum",
            "group",
            "rows",
            "campaigns",
            "inferentially_eligible",
            *[f"campaign_macro_{metric}" for metric in METRICS],
        ],
    )
    report = report_markdown(fold_rows, selections, deltas)
    (output / "report.md").write_text(report, encoding="utf-8")
    stdout_text = (
        "\n".join(RUN_LOG)
        + "\n"
        + report
        + f"\nelapsed_seconds={elapsed:.3f}\n"
        + f"wrote results to {output}\n"
    )
    (output / "stdout.log").write_text(stdout_text, encoding="utf-8")

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
            "embedding_sha256": sha256(EMBEDDINGS),
            "embedding_index_sha256": sha256(EMBEDDING_INDEX),
            "embedding_manifest_sha256": sha256(EMBEDDING_MANIFEST),
            "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS),
            "rows": len(rows),
        },
        "parameters": {
            "architecture": "1024-256-GELU-dropout0.3-184",
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "seeds": list(SEEDS),
            "epoch_grid": list(EPOCHS),
            "lambda_grid": list(LAMBDAS),
            "selected": {
                source: {"epoch": values[0], "lambda": values[1]}
                for source, values in selections.items()
            },
            "num_threads": args.num_threads,
            "deterministic_algorithms": True,
        },
        "elapsed_seconds": elapsed,
        "a_ranking_reproduction_gate": "PASS: all 784 outer rows Top-20 identical",
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
