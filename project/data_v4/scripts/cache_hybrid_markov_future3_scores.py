#!/usr/bin/env python3
"""Cache frozen inner and outer 184-dimensional HM marginal scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HM_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_hybrid_markov_lstm_future3_lodo.py"
HM_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_markov_lstm_future3_lodo_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/model_scores/hm_future3_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
SEEDS = (42, 43, 44, 45, 46)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HM = import_module("hm_baseline_for_cache", HM_SCRIPT)


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected() -> dict[str, tuple[float, int, float]]:
    rows = read_csv(HM_RESULTS / "selected_hyperparameters.csv")
    result = {
        row["held_out_source"]: (
            float(row["selected_learning_rate"]),
            int(row["selected_epoch"]),
            float(row["selected_beta"]),
        )
        for row in rows
    }
    if set(result) != set(SOURCES):
        raise AssertionError(f"HM selections changed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ["inner_scores.npy", "inner_index.csv", "outer_scores.npy", "outer_index.csv", "cache_manifest.json", "stdout.log"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite HM score cache: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    rows, labels, label_index, input_index = HM.NEURAL.load_rows()
    parameters = selected()
    frozen = {
        (row["held_out_source"], int(row["seed"]), row["sample_id"]): row
        for row in read_csv(HM_RESULTS / "predictions_by_seed.csv")
    }
    inner_blocks: list[np.ndarray] = []
    inner_index: list[dict[str, Any]] = []
    outer_blocks: list[np.ndarray] = []
    outer_index: list[dict[str, Any]] = []
    log: list[str] = []
    started_all = time.perf_counter()
    for held_out in SOURCES:
        learning_rate, epochs, beta = parameters[held_out]
        training_sources = tuple(source for source in SOURCES if source != held_out)
        for validation_source in training_sources:
            training_source = next(source for source in training_sources if source != validation_source)
            train = [row for row in rows if row["source"] == training_source]
            validation = [row for row in rows if row["source"] == validation_source]
            markov = HM.MB.MarkovBeam(train, labels)
            for seed in SEEDS:
                started = time.perf_counter()
                model = HM.train_model(train, input_index, label_index, seed, learning_rate, epochs)
                scores = np.asarray(
                    HM.beam_scores(model, validation, markov, (beta,), input_index, label_index)[beta],
                    dtype=np.float32,
                )
                base = sum(len(block) for block in inner_blocks)
                inner_blocks.append(scores)
                for offset, row in enumerate(validation):
                    inner_index.append(
                        {
                            "score_row": base + offset,
                            "held_out_source": held_out,
                            "inner_training_source": training_source,
                            "inner_validation_source": validation_source,
                            "seed": seed,
                            "learning_rate": learning_rate,
                            "epoch": epochs,
                            "beta": beta,
                            "sample_id": row["sample_id"],
                            "campaign_id": row["campaign_id"],
                        }
                    )
                message = (
                    f"inner held_out={held_out} train={training_source} validation={validation_source} "
                    f"seed={seed} rows={len(validation)} elapsed={time.perf_counter()-started:.1f}s"
                )
                print(message, flush=True)
                log.append(message)
                del model, scores

        train = [row for row in rows if row["source"] != held_out]
        test = [row for row in rows if row["source"] == held_out]
        markov = HM.MB.MarkovBeam(train, labels)
        for seed in SEEDS:
            started = time.perf_counter()
            model = HM.train_model(train, input_index, label_index, seed, learning_rate, epochs)
            scores = np.asarray(
                HM.beam_scores(model, test, markov, (beta,), input_index, label_index)[beta],
                dtype=np.float32,
            )
            base = sum(len(block) for block in outer_blocks)
            for offset, (row, values) in enumerate(zip(test, scores)):
                ranked, _ = HM.BASE.ranking([float(value) for value in values], labels)
                if ranked[:20] != json.loads(frozen[(held_out, seed, row["sample_id"])]["top20_ids"]):
                    raise AssertionError(f"HM cache reproduction failed: {held_out}/{seed}/{row['sample_id']}")
                outer_index.append(
                    {
                        "score_row": base + offset,
                        "held_out_source": held_out,
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "epoch": epochs,
                        "beta": beta,
                        "sample_id": row["sample_id"],
                        "campaign_id": row["campaign_id"],
                    }
                )
            outer_blocks.append(scores)
            message = f"outer held_out={held_out} seed={seed} rows={len(test)} elapsed={time.perf_counter()-started:.1f}s"
            print(message, flush=True)
            log.append(message)
            del model, scores
    inner = np.concatenate(inner_blocks, axis=0)
    outer = np.concatenate(outer_blocks, axis=0)
    if inner.shape != (7840, 184) or outer.shape != (3920, 184):
        raise AssertionError(f"unexpected HM cache shapes: inner={inner.shape} outer={outer.shape}")
    np.save(output / "inner_scores.npy", inner, allow_pickle=False)
    np.save(output / "outer_scores.npy", outer, allow_pickle=False)
    write_csv(output / "inner_index.csv", inner_index, ["score_row", "held_out_source", "inner_training_source", "inner_validation_source", "seed", "learning_rate", "epoch", "beta", "sample_id", "campaign_id"])
    write_csv(output / "outer_index.csv", outer_index, ["score_row", "held_out_source", "seed", "learning_rate", "epoch", "beta", "sample_id", "campaign_id"])
    elapsed = time.perf_counter() - started_all
    (output / "stdout.log").write_text("\n".join(log) + f"\nelapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "inputs": {
            "hm_manifest_sha256": sha256(HM_RESULTS / "results_manifest.json"),
            "hm_predictions_sha256": sha256(HM_RESULTS / "predictions_by_seed.csv"),
            "samples_sha256": sha256(HM.BASE.SAMPLES_PATH),
        },
        "parameters": {
            "seeds": list(SEEDS),
            "selected": {source: {"learning_rate": values[0], "epoch": values[1], "beta": values[2]} for source, values in parameters.items()},
            "num_threads": args.num_threads,
            "deterministic_algorithms": True,
        },
        "shapes": {"inner_scores": list(inner.shape), "outer_scores": list(outer.shape)},
        "outer_top20_reproduction_gate": "PASS all 3920 seed-sample rows",
        "elapsed_seconds": elapsed,
        "outputs_sha256": {name: sha256(output / name) for name in managed if name != "cache_manifest.json" and (output / name).exists()},
    }
    write_json(output / "cache_manifest.json", manifest)
    print(f"inner_shape={inner.shape} outer_shape={outer.shape} elapsed_seconds={elapsed:.1f}")
    print(f"wrote HM score cache to {output}")


if __name__ == "__main__":
    main()
