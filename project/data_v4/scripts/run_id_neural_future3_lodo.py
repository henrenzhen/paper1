#!/usr/bin/env python3
"""Run frozen ID-only LSTM and Transformer future-3 LODO baselines."""

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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
BASELINE_PREDICTIONS = (
    PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
)
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/id_neural_future3_baselines_v1.md"
RUNTIME_LOCK = PROJECT_ROOT / "data_v4/protocols/semantic_runtime_lock.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/id_neural_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
ARCHITECTURES = ("LSTM", "TR")
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (20, 40, 60, 80, 100)
LEARNING_RATES = (3e-4, 1e-3)
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807
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


def load_rows() -> tuple[list[dict[str, Any]], list[str], dict[str, int], dict[str, int]]:
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    history_labels = sorted({label for row in rows for label in row["history"]})
    input_index = {label: index + 1 for index, label in enumerate(history_labels)}
    input_only = sorted(set(history_labels) - set(labels))
    if len(history_labels) != 120 or input_only != ["T0866", "T1029", "T1533"]:
        raise AssertionError(
            f"history vocabulary changed: size={len(history_labels)} input_only={input_only}"
        )
    return rows, labels, label_index, input_index


def tensors(
    rows: Sequence[dict[str, Any]],
    input_index: dict[str, int],
    label_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(row["history"]) for row in rows)
    x = torch.zeros((len(rows), maximum), dtype=torch.long)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    y = torch.zeros((len(rows), len(label_index)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        encoded = [input_index[label] for label in row["history"]]
        x[row_index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        lengths[row_index] = len(encoded)
        for target in row["targets"]:
            y[row_index, label_index[target]] = 1.0
    return x, lengths, y


class LSTMModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size + 1, 128, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
        )
        self.head = nn.Linear(256, 184)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        values = self.embedding(x)
        packed = pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        return self.head(hidden[-1])


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, maximum_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size + 1, 128, padding_idx=0)
        self.position = nn.Embedding(maximum_length, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(128, 184)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, sequence = x.shape
        positions = torch.arange(sequence, device=x.device).unsqueeze(0).expand(batch, -1)
        values = self.embedding(x) + self.position(positions)
        causal = torch.triu(
            torch.ones((sequence, sequence), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        padding = x.eq(0)
        encoded = self.encoder(values, mask=causal, src_key_padding_mask=padding)
        last = encoded[torch.arange(batch, device=x.device), lengths - 1]
        return self.head(last)


def make_model(architecture: str, input_size: int, maximum_length: int) -> nn.Module:
    if architecture == "LSTM":
        return LSTMModel(input_size)
    if architecture == "TR":
        return TransformerModel(input_size, maximum_length)
    raise ValueError(architecture)


def train_checkpoints(
    architecture: str,
    train_rows: Sequence[dict[str, Any]],
    evaluation_sets: dict[str, Sequence[dict[str, Any]]],
    input_index: dict[str, int],
    label_index: dict[str, int],
    seed: int,
    learning_rate: float,
    checkpoint_epochs: Sequence[int],
) -> dict[str, dict[int, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    maximum_length = max(len(row["history"]) for rows in [train_rows, *evaluation_sets.values()] for row in rows)
    model = make_model(architecture, len(input_index), maximum_length)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_x, train_lengths, train_y = tensors(train_rows, input_index, label_index)
    evaluation_tensors = {
        name: tensors(values, input_index, label_index)[:2]
        for name, values in evaluation_sets.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_lengths, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    checkpoints = set(checkpoint_epochs)
    outputs: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in evaluation_sets
    }
    model.train()
    for epoch in range(1, max(checkpoint_epochs) + 1):
        for batch_x, batch_lengths, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_lengths)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                for name, (values_x, values_lengths) in evaluation_tensors.items():
                    batches: list[np.ndarray] = []
                    for start in range(0, len(values_x), BATCH_SIZE):
                        batches.append(
                            model(
                                values_x[start : start + BATCH_SIZE],
                                values_lengths[start : start + BATCH_SIZE],
                            )
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    outputs[name][epoch] = np.concatenate(batches, axis=0)
            model.train()
    return outputs


def campaign_ndcg(
    rows: Sequence[dict[str, Any]], logits: np.ndarray, labels: Sequence[str]
) -> float:
    records: list[dict[str, Any]] = []
    for row, values in zip(rows, logits):
        ranked, _ = BASE.ranking([float(value) for value in values], labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def inner_cache(
    architecture: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    input_index: dict[str, int],
) -> dict[tuple[str, float, int], dict[str, dict[int, np.ndarray]]]:
    cache: dict[tuple[str, float, int], dict[str, dict[int, np.ndarray]]] = {}
    for training_source in SOURCES:
        train_rows = [row for row in rows if row["source"] == training_source]
        evaluations = {
            source: [row for row in rows if row["source"] == source]
            for source in SOURCES
            if source != training_source
        }
        for learning_rate in LEARNING_RATES:
            for seed in SEEDS:
                started = time.perf_counter()
                cache[(training_source, learning_rate, seed)] = train_checkpoints(
                    architecture,
                    train_rows,
                    evaluations,
                    input_index,
                    label_index,
                    seed,
                    learning_rate,
                    EPOCHS,
                )
                emit(
                    f"inner architecture={architecture} train={training_source} "
                    f"lr={learning_rate:g} seed={seed} elapsed={time.perf_counter()-started:.1f}s"
                )
    return cache


def select_hyperparameters(
    architecture: str,
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    cache: dict[tuple[str, float, int], dict[str, dict[int, np.ndarray]]],
) -> tuple[float, int, list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for validation_source in training_sources:
        inner_train_source = next(
            source for source in training_sources if source != validation_source
        )
        validation_rows = [row for row in rows if row["source"] == validation_source]
        for learning_rate in LEARNING_RATES:
            for seed in SEEDS:
                values = cache[(inner_train_source, learning_rate, seed)][validation_source]
                for epoch in EPOCHS:
                    score = campaign_ndcg(validation_rows, values[epoch], labels)
                    grouped[(learning_rate, epoch)].append(score)
                    details.append(
                        {
                            "architecture": architecture,
                            "held_out_source": held_out,
                            "inner_training_source": inner_train_source,
                            "inner_validation_source": validation_source,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "epoch": epoch,
                            "campaign_macro_ndcg5": score,
                            "training_rows": len([row for row in rows if row["source"] == inner_train_source]),
                            "validation_rows": len(validation_rows),
                        }
                    )
    means = {key: statistics.fmean(values) for key, values in grouped.items()}
    best = max(means.values())
    candidates = [key for key, value in means.items() if abs(value - best) <= 1e-12]
    selected_lr, selected_epoch = min(candidates, key=lambda item: (item[0], item[1]))
    for row in details:
        row["five_seed_two_source_mean_ndcg5"] = means[(row["learning_rate"], row["epoch"])]
        row["selected"] = int(
            row["learning_rate"] == selected_lr and row["epoch"] == selected_epoch
        )
    return selected_lr, selected_epoch, details


def frozen_reference() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv(BASELINE_PREDICTIONS)
    return (
        {row["sample_id"]: row for row in rows if row["method"] == "A"},
        {row["sample_id"]: row for row in rows if row["method"] == "A0"},
    )


def evaluate_outer(
    architecture: str,
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    input_index: dict[str, int],
    learning_rate: float,
    epoch: int,
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if row["source"] != held_out]
    test_rows = [row for row in rows if row["source"] == held_out]
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        logits = train_checkpoints(
            architecture,
            train_rows,
            {held_out: test_rows},
            input_index,
            label_index,
            seed,
            learning_rate,
            (epoch,),
        )[held_out][epoch]
        emit(
            f"outer architecture={architecture} held_out={held_out} lr={learning_rate:g} "
            f"epoch={epoch} seed={seed} elapsed={time.perf_counter()-started:.1f}s"
        )
        for row, values in zip(test_rows, logits):
            ranked, scores = BASE.ranking([float(value) for value in values], labels)
            reference = frozen_a[row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": architecture,
                    "seed": seed,
                    "selected_learning_rate": learning_rate,
                    "selected_epoch": epoch,
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


def mean_sample_metrics(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["method"], row["sample_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, sample_id), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise AssertionError(f"expected five seeds for {method}/{sample_id}")
        first = values[0]
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "method": method,
                "sample_id": sample_id,
                "campaign_id": first["campaign_id"],
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


def campaign_rows(mean_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mean_rows:
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
    for source in SOURCES:
        for method in ARCHITECTURES:
            values = [
                row for row in campaigns
                if row["held_out_source"] == source and row["method"] == method
            ]
            output.append(
                {
                    "held_out_source": source,
                    "method": method,
                    "campaigns": len(values),
                    **{
                        f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values)
                        for metric in METRICS
                    },
                }
            )
    return output


def reference_campaigns(reference: dict[str, dict[str, str]], method: str) -> list[dict[str, Any]]:
    samples = [
        {
            "held_out_source": row["held_out_source"],
            "method": method,
            "campaign_id": row["campaign_id"],
            **{metric: float(row[metric]) for metric in METRICS},
        }
        for row in reference.values()
    ]
    return campaign_rows(samples)


def paired_bootstrap(
    campaigns: Sequence[dict[str, Any]],
    frozen_a: dict[str, dict[str, str]],
    frozen_a0: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    all_campaigns = list(campaigns) + reference_campaigns(frozen_a, "A") + reference_campaigns(frozen_a0, "A0")
    lookup = {
        (row["held_out_source"], row["method"], row["campaign_id"]): row
        for row in all_campaigns
    }
    campaign_ids = {
        source: sorted({
            row["campaign_id"] for row in campaigns if row["held_out_source"] == source
        })
        for source in SOURCES
    }
    comparisons = (("LSTM", "A"), ("TR", "A"), ("LSTM", "A0"), ("TR", "A0"), ("TR", "LSTM"))
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {
            source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]]
            for source in SOURCES
        }
        for left, right in comparisons:
            for metric in METRICS:
                source_values: list[float] = []
                for source in SOURCES:
                    delta = statistics.fmean(
                        float(lookup[(source, left, campaign)][metric])
                        - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    source_values.append(delta)
                    replicates[(source, f"{left}-{right}", metric)].append(delta)
                replicates[("source_equal_overall", f"{left}-{right}", metric)].append(
                    statistics.fmean(source_values)
                )
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
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
                values = replicates[(scope, f"{left}-{right}", metric)]
                output.append(
                    {
                        "scope": scope,
                        "comparison": f"{left}-{right}",
                        "metric": metric,
                        "point_estimate": point,
                        "ci95_low": BASE.percentile(values, 0.025),
                        "ci95_high": BASE.percentile(values, 0.975),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return output


def report_markdown(
    folds: Sequence[dict[str, Any]],
    selections: dict[tuple[str, str], tuple[float, int]],
    differences: Sequence[dict[str, Any]],
) -> str:
    lookup = {(row["held_out_source"], row["method"]): row for row in folds}
    delta = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    lines = [
        "# ID-only neural future-3 LODO results",
        "",
        "No text, tactic feature, LLM, or external API was used.",
        "",
        "## Campaign-macro NDCG@5 (five-seed mean)",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ARCHITECTURES:
        values = [float(lookup[(source, method)]["campaign_macro_ndcg5"]) for source in SOURCES]
        lines.append(
            f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {statistics.fmean(values):.4f} |"
        )
    lines.extend([
        "",
        "## Inner-selected hyperparameters",
        "",
        "| Method | Held-out | Learning rate | Epoch |",
        "|---|---|---:|---:|",
    ])
    for method in ARCHITECTURES:
        for source in SOURCES:
            learning_rate, epoch = selections[(method, source)]
            lines.append(f"| {method} | {source} | {learning_rate:g} | {epoch} |")
    lines.extend([
        "",
        "## Source-equal paired NDCG@5 differences",
        "",
        "| Comparison | Delta | 95% campaign-bootstrap CI |",
        "|---|---:|---:|",
    ])
    for comparison in ("LSTM-A", "TR-A", "LSTM-A0", "TR-A0", "TR-LSTM"):
        row = delta[("source_equal_overall", comparison, "ndcg5")]
        lines.append(
            f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "Metrics average five neural seeds at sample level before campaign aggregation. The 30 development rows are excluded.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    args = parser.parse_args()
    selected_architectures = tuple(args.architectures)
    if selected_architectures != ARCHITECTURES:
        raise ValueError("formal run requires both LSTM and TR")
    output = args.output_dir.resolve()
    managed = [
        "inner_selection.csv",
        "selected_hyperparameters.csv",
        "predictions_by_seed.csv",
        "mean_sample_metrics.csv",
        "campaign_results_five_seed_mean.csv",
        "fold_results_five_seed_mean.csv",
        "paired_bootstrap_differences.csv",
        "report.md",
        "stdout.log",
        "results_manifest.json",
    ]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing neural results: {existing}")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, input_index = load_rows()
    frozen_a, frozen_a0 = frozen_reference()
    if len(frozen_a) != 784 or len(frozen_a0) != 784:
        raise AssertionError("frozen A/A0 reference does not contain 784 rows each")

    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selections: dict[tuple[str, str], tuple[float, int]] = {}
    started = time.perf_counter()
    for architecture in ARCHITECTURES:
        cache = inner_cache(architecture, rows, labels, label_index, input_index)
        for held_out in SOURCES:
            learning_rate, epoch, details = select_hyperparameters(
                architecture, held_out, rows, labels, cache
            )
            selections[(architecture, held_out)] = (learning_rate, epoch)
            inner.extend(details)
            emit(
                f"selected architecture={architecture} held_out={held_out} lr={learning_rate:g} epoch={epoch}"
            )
            predictions.extend(
                evaluate_outer(
                    architecture,
                    held_out,
                    rows,
                    labels,
                    label_index,
                    input_index,
                    learning_rate,
                    epoch,
                    frozen_a,
                )
            )
        del cache
    elapsed = time.perf_counter() - started
    inner.sort(key=lambda row: (row["architecture"], row["held_out_source"], row["inner_validation_source"], row["learning_rate"], row["seed"], row["epoch"]))
    predictions.sort(key=lambda row: (row["method"], row["held_out_source"], row["seed"], row["sample_id"]))
    mean_rows = mean_sample_metrics(predictions)
    campaigns = campaign_rows(mean_rows)
    folds = fold_rows(campaigns)
    differences = paired_bootstrap(campaigns, frozen_a, frozen_a0)
    selection_rows = [
        {
            "architecture": architecture,
            "held_out_source": source,
            "selected_learning_rate": selections[(architecture, source)][0],
            "selected_epoch": selections[(architecture, source)][1],
        }
        for architecture in ARCHITECTURES for source in SOURCES
    ]
    write_csv(output / "inner_selection.csv", inner, [
        "architecture", "held_out_source", "inner_training_source", "inner_validation_source",
        "learning_rate", "seed", "epoch", "campaign_macro_ndcg5",
        "five_seed_two_source_mean_ndcg5", "selected", "training_rows", "validation_rows",
    ])
    write_csv(output / "selected_hyperparameters.csv", selection_rows, [
        "architecture", "held_out_source", "selected_learning_rate", "selected_epoch",
    ])
    write_csv(output / "predictions_by_seed.csv", predictions, [
        "held_out_source", "method", "seed", "selected_learning_rate", "selected_epoch",
        "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size",
        "transition_visibility", "target_label_visibility", "text_length_group",
        "top20_ids", "top20_scores", *METRICS,
    ])
    write_csv(output / "mean_sample_metrics.csv", mean_rows, [
        "held_out_source", "method", "sample_id", "campaign_id", "target_size",
        "transition_visibility", "target_label_visibility", "text_length_group", *METRICS,
    ])
    write_csv(output / "campaign_results_five_seed_mean.csv", campaigns, [
        "held_out_source", "method", "campaign_id", "rows", *METRICS,
    ])
    write_csv(output / "fold_results_five_seed_mean.csv", folds, [
        "held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS],
    ])
    write_csv(output / "paired_bootstrap_differences.csv", differences, [
        "scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed",
    ])
    report = report_markdown(folds, selections, differences)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text(
        "\n".join(RUN_LOG) + "\n" + report + f"\nelapsed_seconds={elapsed:.3f}\n",
        encoding="utf-8",
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
        "inputs": {
            "samples_sha256": sha256(BASE.SAMPLES_PATH),
            "vocabulary_sha256": sha256(BASE.VOCAB_PATH),
            "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS),
            "main_rows": len(rows),
            "input_vocabulary_size_excluding_padding": len(input_index),
            "input_only_ids": sorted(set(input_index) - set(labels)),
        },
        "parameters": {
            "architectures": list(ARCHITECTURES),
            "seeds": list(SEEDS),
            "epoch_grid": list(EPOCHS),
            "learning_rate_grid": list(LEARNING_RATES),
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "selected": {
                f"{architecture}:{source}": {"learning_rate": value[0], "epoch": value[1]}
                for (architecture, source), value in selections.items()
            },
            "num_threads": args.num_threads,
            "deterministic_algorithms": True,
        },
        "elapsed_seconds": elapsed,
        "outputs_sha256": {
            name: sha256(output / name)
            for name in managed if name != "results_manifest.json" and (output / name).exists()
        },
    }
    write_json(output / "results_manifest.json", manifest)
    print(report)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"wrote results to {output}")


if __name__ == "__main__":
    main()
