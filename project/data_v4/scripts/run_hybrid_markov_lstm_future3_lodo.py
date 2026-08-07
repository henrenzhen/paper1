#!/usr/bin/env python3
"""Run the frozen SECRYPT-adapted Hybrid Markov--LSTM baseline."""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEURAL_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_id_neural_future3_lodo.py"
MB_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_markov_beam_future3_lodo.py"
NEURAL_RESULTS = PROJECT_ROOT / "data_v4/results/id_neural_future3_lodo_v1"
MB_RESULTS = PROJECT_ROOT / "data_v4/results/markov_beam_future3_lodo_v1"
BASELINE_PREDICTIONS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/hybrid_markov_lstm_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/hybrid_markov_lstm_future3_lodo_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)
BETAS = tuple(index / 10 for index in range(11))
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807
ROW_CHUNK = 32
INFERENCE_BATCH = 256
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


NEURAL = import_module("id_neural", NEURAL_SCRIPT)
MB = import_module("markov_beam", MB_SCRIPT)
BASE = NEURAL.BASE


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


def selected_lstm_hyperparameters() -> dict[str, tuple[float, int]]:
    rows = read_csv(NEURAL_RESULTS / "selected_hyperparameters.csv")
    selected = {
        row["held_out_source"]: (float(row["selected_learning_rate"]), int(row["selected_epoch"]))
        for row in rows if row["architecture"] == "LSTM"
    }
    if set(selected) != set(SOURCES):
        raise AssertionError(f"missing frozen LSTM selections: {selected}")
    return selected


def train_model(
    train_rows: Sequence[dict[str, Any]],
    input_index: dict[str, int],
    label_index: dict[str, int],
    seed: int,
    learning_rate: float,
    epochs: int,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = NEURAL.LSTMModel(len(input_index))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=NEURAL.WEIGHT_DECAY
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_x, train_lengths, train_y = NEURAL.tensors(train_rows, input_index, label_index)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_lengths, train_y),
        batch_size=NEURAL.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_lengths, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_lengths)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def context_log_probabilities(
    model: nn.Module,
    contexts: Sequence[Sequence[str]],
    input_index: dict[str, int],
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), INFERENCE_BATCH):
            batch = contexts[start : start + INFERENCE_BATCH]
            maximum = max(len(context) for context in batch)
            x = torch.zeros((len(batch), maximum), dtype=torch.long)
            lengths = torch.tensor([len(context) for context in batch], dtype=torch.long)
            for row_index, context in enumerate(batch):
                encoded = [input_index.get(label, 0) for label in context]
                x[row_index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            values = torch.log_softmax(model(x, lengths), dim=1).cpu().numpy().astype(np.float32)
            output.extend(values)
    return output


def score_chunk(
    model: nn.Module,
    rows: Sequence[dict[str, Any]],
    markov: Any,
    betas: Sequence[float],
    input_index: dict[str, int],
    label_index: dict[str, int],
) -> dict[float, list[list[float]]]:
    beams: dict[tuple[int, float], list[tuple[tuple[str, ...], float, float]]] = {
        (row_index, beta): [(tuple(), 0.0, 0.0)]
        for row_index in range(len(rows)) for beta in betas
    }
    for _ in range(MB.HORIZON):
        context_keys = sorted(
            {
                (row_index, path)
                for row_index in range(len(rows))
                for beta in betas
                for path, _, _ in beams[(row_index, beta)]
            },
            key=lambda item: (item[0], tuple(label_index.get(label, -1) for label in item[1])),
        )
        contexts = [tuple(rows[row_index]["history"]) + path for row_index, path in context_keys]
        probabilities = context_log_probabilities(model, contexts, input_index)
        by_context = {key: value for key, value in zip(context_keys, probabilities)}
        next_beams: dict[tuple[int, float], list[tuple[tuple[str, ...], float, float]]] = {}
        for row_index in range(len(rows)):
            for beta in betas:
                expanded: dict[tuple[str, ...], tuple[float, float]] = {}
                for path, markov_log, lstm_log in beams[(row_index, beta)]:
                    last = path[-1] if path else rows[row_index]["history"][-1]
                    neural = by_context[(row_index, path)]
                    for label, probability in markov.candidates(last):
                        new_path = path + (label,)
                        new_markov = markov_log + math.log(probability)
                        new_lstm = lstm_log + float(neural[label_index[label]])
                        old = expanded.get(new_path)
                        new_combined = (1 - beta) * new_markov + beta * new_lstm
                        if old is None or new_combined > (1 - beta) * old[0] + beta * old[1]:
                            expanded[new_path] = (new_markov, new_lstm)
                retained = sorted(
                    expanded.items(),
                    key=lambda item: (
                        -((1 - beta) * item[1][0] + beta * item[1][1]),
                        tuple(label_index[label] for label in item[0]),
                    ),
                )[: MB.BEAM_WIDTH]
                next_beams[(row_index, beta)] = [
                    (path, values[0], values[1]) for path, values in retained
                ]
        beams = next_beams
    result: dict[float, list[list[float]]] = {beta: [] for beta in betas}
    for row_index in range(len(rows)):
        for beta in betas:
            values = beams[(row_index, beta)]
            combined = [(1 - beta) * markov_log + beta * lstm_log for _, markov_log, lstm_log in values]
            maximum = max(combined)
            weights = [math.exp(value - maximum) for value in combined]
            denominator = sum(weights)
            marginals = [0.0] * len(label_index)
            for (path, _, _), weight in zip(values, weights):
                for label in set(path):
                    marginals[label_index[label]] += weight / denominator
            result[beta].append(marginals)
    return result


def beam_scores(
    model: nn.Module,
    rows: Sequence[dict[str, Any]],
    markov: Any,
    betas: Sequence[float],
    input_index: dict[str, int],
    label_index: dict[str, int],
) -> dict[float, list[list[float]]]:
    output: dict[float, list[list[float]]] = {beta: [] for beta in betas}
    for start in range(0, len(rows), ROW_CHUNK):
        chunk = score_chunk(
            model,
            rows[start : start + ROW_CHUNK],
            markov,
            betas,
            input_index,
            label_index,
        )
        for beta in betas:
            output[beta].extend(chunk[beta])
    return output


def campaign_ndcg(rows: Sequence[dict[str, Any]], scores: Sequence[Sequence[float]], labels: Sequence[str]) -> float:
    records: list[dict[str, Any]] = []
    for row, values in zip(rows, scores):
        ranked, _ = BASE.ranking(values, labels)
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_beta(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    input_index: dict[str, int],
    learning_rate: float,
    epochs: int,
) -> tuple[float, list[dict[str, Any]]]:
    outer_training = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    grouped: dict[float, list[float]] = defaultdict(list)
    for validation_source in outer_training:
        inner_train_source = next(source for source in outer_training if source != validation_source)
        train = [row for row in rows if row["source"] == inner_train_source]
        validation = [row for row in rows if row["source"] == validation_source]
        markov = MB.MarkovBeam(train, labels)
        for seed in SEEDS:
            started = time.perf_counter()
            model = train_model(train, input_index, label_index, seed, learning_rate, epochs)
            scores = beam_scores(model, validation, markov, BETAS, input_index, label_index)
            emit(
                f"inner held_out={held_out} train={inner_train_source} validation={validation_source} "
                f"seed={seed} lr={learning_rate:g} epoch={epochs} elapsed={time.perf_counter()-started:.1f}s"
            )
            for beta in BETAS:
                value = campaign_ndcg(validation, scores[beta], labels)
                grouped[beta].append(value)
                details.append(
                    {
                        "held_out_source": held_out,
                        "inner_training_source": inner_train_source,
                        "inner_validation_source": validation_source,
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "epoch": epochs,
                        "beta": beta,
                        "campaign_macro_ndcg5": value,
                        "training_rows": len(train),
                        "validation_rows": len(validation),
                    }
                )
            del model, scores
    means = {beta: statistics.fmean(values) for beta, values in grouped.items()}
    best = max(means.values())
    chosen = min(beta for beta, value in means.items() if abs(value - best) <= 1e-12)
    for row in details:
        row["five_seed_two_source_mean_ndcg5"] = means[row["beta"]]
        row["selected"] = int(row["beta"] == chosen)
    return chosen, details


def frozen_predictions() -> tuple[dict[tuple[str, int], dict[str, dict[str, str]]], dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    neural: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(NEURAL_RESULTS / "predictions_by_seed.csv"):
        if row["method"] == "LSTM":
            neural[(row["held_out_source"], int(row["seed"]))][row["sample_id"]] = row
    mb = {row["sample_id"]: row for row in read_csv(MB_RESULTS / "predictions.csv")}
    simple: dict[str, dict[str, dict[str, str]]] = {method: {} for method in ("A", "K")}
    for row in read_csv(BASELINE_PREDICTIONS):
        if row["method"] in simple:
            simple[row["method"]][row["sample_id"]] = row
    return neural, mb, simple


def direct_logits(model: nn.Module, rows: Sequence[dict[str, Any]], input_index: dict[str, int]) -> np.ndarray:
    return np.stack(context_log_probabilities(model, [row["history"] for row in rows], input_index))


def evaluate_outer(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    input_index: dict[str, int],
    learning_rate: float,
    epochs: int,
    beta: float,
    frozen_neural: dict[tuple[str, int], dict[str, dict[str, str]]],
    frozen_mb: dict[str, dict[str, str]],
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    markov = MB.MarkovBeam(train, labels)
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        model = train_model(train, input_index, label_index, seed, learning_rate, epochs)
        direct = direct_logits(model, test, input_index)
        for row, values in zip(test, direct):
            ranked, _ = BASE.ranking([float(value) for value in values], labels)
            expected = frozen_neural[(held_out, seed)][row["sample_id"]]
            if ranked[:20] != json.loads(expected["top20_ids"]):
                raise AssertionError(f"direct LSTM reproduction failed: {held_out}/{seed}/{row['sample_id']}")
        requested = tuple(sorted({0.0, beta}))
        scores = beam_scores(model, test, markov, requested, input_index, label_index)
        for row, values in zip(test, scores[0.0]):
            ranked, _ = BASE.ranking(values, labels)
            if ranked[:20] != json.loads(frozen_mb[row["sample_id"]]["top20_ids"]):
                raise AssertionError(f"beta=0 MB reproduction failed: {row['sample_id']}")
        for row, values in zip(test, scores[beta]):
            ranked, ordered = BASE.ranking(values, labels)
            reference = frozen_a[row["sample_id"]]
            output.append(
                {
                    "held_out_source": held_out,
                    "method": "HM",
                    "seed": seed,
                    "selected_learning_rate": learning_rate,
                    "selected_epoch": epochs,
                    "selected_beta": beta,
                    "sample_id": row["sample_id"],
                    "campaign_id": row["campaign_id"],
                    "prefix_len": row["prefix_len"],
                    "target_parent_ids": compact_json(row["targets"]),
                    "target_size": row["target_size"],
                    "transition_visibility": reference["transition_visibility"],
                    "target_label_visibility": reference["target_label_visibility"],
                    "text_length_group": reference["text_length_group"],
                    "top20_ids": compact_json(ranked[:20]),
                    "top20_scores": compact_json([round(float(value), 12) for value in ordered[:20]]),
                    **BASE.sample_metrics(ranked[:5], row["targets"]),
                }
            )
        emit(
            f"outer held_out={held_out} seed={seed} lr={learning_rate:g} epoch={epochs} "
            f"beta={beta:.1f} elapsed={time.perf_counter()-started:.1f}s"
        )
        del model, scores, direct
    return output


def mean_sample_metrics(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    for sample_id, values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise AssertionError(f"expected five seeds: {sample_id}")
        first = values[0]
        output.append(
            {
                "held_out_source": first["held_out_source"],
                "method": "HM",
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


def aggregate_campaigns(rows: Sequence[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
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
                "method": "HM",
                "campaigns": len(values),
                **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
            }
        )
    return output


def paired_bootstrap(
    hm_campaigns: Sequence[dict[str, Any]],
    frozen_neural: dict[tuple[str, int], dict[str, dict[str, str]]],
    frozen_mb: dict[str, dict[str, str]],
    simple: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    reference_samples: dict[str, list[dict[str, Any]]] = {method: [] for method in ("MB", "LSTM", "A", "K")}
    for row in frozen_mb.values():
        reference_samples["MB"].append({"held_out_source": row["held_out_source"], "campaign_id": row["campaign_id"], **{metric: float(row[metric]) for metric in METRICS}})
    for source in SOURCES:
        sample_ids = sorted(frozen_neural[(source, SEEDS[0])])
        for sample_id in sample_ids:
            first = frozen_neural[(source, SEEDS[0])][sample_id]
            reference_samples["LSTM"].append({
                "held_out_source": source,
                "campaign_id": first["campaign_id"],
                **{metric: statistics.fmean(float(frozen_neural[(source, seed)][sample_id][metric]) for seed in SEEDS) for metric in METRICS},
            })
    for method in ("A", "K"):
        for row in simple[method].values():
            reference_samples[method].append({"held_out_source": row["held_out_source"], "campaign_id": row["campaign_id"], **{metric: float(row[metric]) for metric in METRICS}})
    all_campaigns = list(hm_campaigns)
    for method, values in reference_samples.items():
        all_campaigns.extend(aggregate_campaigns(values, method))
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in all_campaigns}
    campaign_ids = {source: sorted(row["campaign_id"] for row in hm_campaigns if row["held_out_source"] == source) for source in SOURCES}
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
        for right in ("MB", "LSTM", "A", "K"):
            for metric in METRICS:
                source_values: list[float] = []
                for source in SOURCES:
                    value = statistics.fmean(float(lookup[(source, "HM", campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in draws[source])
                    source_values.append(value)
                    replicates[(source, f"HM-{right}", metric)].append(value)
                replicates[("source_equal_overall", f"HM-{right}", metric)].append(statistics.fmean(source_values))
    output: list[dict[str, Any]] = []
    for scope in (*SOURCES, "source_equal_overall"):
        for right in ("MB", "LSTM", "A", "K"):
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(statistics.fmean(float(lookup[(source, "HM", campaign)][metric]) - float(lookup[(source, right, campaign)][metric]) for campaign in campaign_ids[source]) for source in SOURCES)
                else:
                    point = statistics.fmean(float(lookup[(scope, "HM", campaign)][metric]) - float(lookup[(scope, right, campaign)][metric]) for campaign in campaign_ids[scope])
                values = replicates[(scope, f"HM-{right}", metric)]
                output.append({"scope": scope, "comparison": f"HM-{right}", "metric": metric, "point_estimate": point, "ci95_low": BASE.percentile(values, 0.025), "ci95_high": BASE.percentile(values, 0.975), "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    return output


def report_markdown(folds: Sequence[dict[str, Any]], selections: dict[str, tuple[float, int, float]], differences: Sequence[dict[str, Any]]) -> str:
    fold = {row["held_out_source"]: row for row in folds}
    delta = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    overall = {metric: statistics.fmean(float(row[f"campaign_macro_{metric}"]) for row in folds) for metric in METRICS}
    lines = ["# SECRYPT-adapted Hybrid Markov--LSTM future-3 LODO results", "", "No text, tactic feature, LLM, or external API was used.", "", "## Campaign-macro metrics (five-seed mean)", "", "| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | LR | Epoch | Beta |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source in SOURCES:
        row = fold[source]
        learning_rate, epoch, beta = selections[source]
        lines.append(f"| {source} | {row['campaign_macro_ndcg5']:.4f} | {row['campaign_macro_hit5']:.4f} | {row['campaign_macro_precision5']:.4f} | {row['campaign_macro_recall5']:.4f} | {learning_rate:g} | {epoch} | {beta:.1f} |")
    lines.append(f"| **Source-equal overall** | **{overall['ndcg5']:.4f}** | **{overall['hit5']:.4f}** | **{overall['precision5']:.4f}** | **{overall['recall5']:.4f}** | — | — | — |")
    lines.extend(["", "## Source-equal paired NDCG@5 differences", "", "| Comparison | Delta | 95% campaign-bootstrap CI |", "|---|---:|---:|"])
    for comparison in ("HM-MB", "HM-LSTM", "HM-A", "HM-K"):
        row = delta[("source_equal_overall", comparison, "ndcg5")]
        lines.append(f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    lines.extend(["", "All outer direct-LSTM and beta=0 Markov-beam Top-20 reproduction gates passed.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ["inner_beta_selection.csv", "selected_hyperparameters.csv", "predictions_by_seed.csv", "mean_sample_metrics.csv", "campaign_results_five_seed_mean.csv", "fold_results_five_seed_mean.csv", "paired_bootstrap_differences.csv", "report.md", "stdout.log", "results_manifest.json"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing HM results: {existing}")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, input_index = NEURAL.load_rows()
    frozen_lstm, frozen_mb, simple = frozen_predictions()
    selected_lstm = selected_lstm_hyperparameters()
    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selections: dict[str, tuple[float, int, float]] = {}
    started = time.perf_counter()
    for held_out in SOURCES:
        learning_rate, epochs = selected_lstm[held_out]
        beta, details = select_beta(held_out, rows, labels, label_index, input_index, learning_rate, epochs)
        inner.extend(details)
        selections[held_out] = (learning_rate, epochs, beta)
        emit(f"selected held_out={held_out} lr={learning_rate:g} epoch={epochs} beta={beta:.1f}")
        predictions.extend(evaluate_outer(held_out, rows, labels, label_index, input_index, learning_rate, epochs, beta, frozen_lstm, frozen_mb, simple["A"]))
    elapsed = time.perf_counter() - started
    inner.sort(key=lambda row: (row["held_out_source"], row["inner_validation_source"], row["seed"], row["beta"]))
    predictions.sort(key=lambda row: (row["held_out_source"], row["seed"], row["sample_id"]))
    mean_rows = mean_sample_metrics(predictions)
    campaigns = aggregate_campaigns(mean_rows, "HM")
    folds = fold_rows(campaigns)
    differences = paired_bootstrap(campaigns, frozen_lstm, frozen_mb, simple)
    selection_rows = [{"held_out_source": source, "selected_learning_rate": values[0], "selected_epoch": values[1], "selected_beta": values[2]} for source, values in selections.items()]
    write_csv(output / "inner_beta_selection.csv", inner, ["held_out_source", "inner_training_source", "inner_validation_source", "seed", "learning_rate", "epoch", "beta", "campaign_macro_ndcg5", "five_seed_two_source_mean_ndcg5", "selected", "training_rows", "validation_rows"])
    write_csv(output / "selected_hyperparameters.csv", selection_rows, ["held_out_source", "selected_learning_rate", "selected_epoch", "selected_beta"])
    write_csv(output / "predictions_by_seed.csv", predictions, ["held_out_source", "method", "seed", "selected_learning_rate", "selected_epoch", "selected_beta", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "top20_ids", "top20_scores", *METRICS])
    write_csv(output / "mean_sample_metrics.csv", mean_rows, ["held_out_source", "method", "sample_id", "campaign_id", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", *METRICS])
    write_csv(output / "campaign_results_five_seed_mean.csv", campaigns, ["held_out_source", "method", "campaign_id", "rows", *METRICS])
    write_csv(output / "fold_results_five_seed_mean.csv", folds, ["held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]])
    write_csv(output / "paired_bootstrap_differences.csv", differences, ["scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"])
    report = report_markdown(folds, selections, differences)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text("\n".join(RUN_LOG) + "\n" + report + f"\nelapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "inputs": {"neural_manifest_sha256": sha256(NEURAL_RESULTS / "results_manifest.json"), "mb_manifest_sha256": sha256(MB_RESULTS / "results_manifest.json"), "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS), "main_rows": len(rows)},
        "parameters": {"betas": list(BETAS), "seeds": list(SEEDS), "row_chunk": ROW_CHUNK, "inference_batch": INFERENCE_BATCH, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "selected": {source: {"learning_rate": values[0], "epoch": values[1], "beta": values[2]} for source, values in selections.items()}, "num_threads": args.num_threads, "deterministic_algorithms": True, "generated_unseen_input_representation": "embedding index 0 (zero vector), counted as valid packed step"},
        "reproduction_gates": {"direct_lstm_top20": "PASS all 3920 outer seed-sample rows", "beta0_mb_top20": "PASS all 3920 outer seed-sample rows"},
        "elapsed_seconds": elapsed,
        "outputs_sha256": {name: sha256(output / name) for name in managed if name != "results_manifest.json" and (output / name).exists()},
    }
    write_json(output / "results_manifest.json", manifest)
    print(report)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"wrote results to {output}")


if __name__ == "__main__":
    main()
