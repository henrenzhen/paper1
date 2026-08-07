#!/usr/bin/env python3
"""Encode frozen future-3 raw semantic histories with a pinned BGE-M3.

The model is frozen and only inference is performed.  Event-boundary
truncation removes complete oldest events until the tokenized input fits; this
script refuses a sample if its last two complete events alone exceed the limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "data_v4/semantic_preflight/future3_dev_prompts_v1/raw_semantic_inputs.jsonl"
)
RUNTIME_LOCK = PROJECT_ROOT / "data_v4/protocols/semantic_runtime_lock.txt"
DEFAULT_CACHE = PROJECT_ROOT / ".hf_cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_embeddings/bge_m3_5617a9f"

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_TOKENS = 8192
EXPECTED_DIMENSION = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def read_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 814:
        raise AssertionError(f"expected 814 semantic inputs, found {len(rows)}")
    return rows


def event_block(event: dict[str, Any]) -> str:
    return (
        f"事件 {event['event_index']}\n"
        f"父技术: {event['parent_technique']}\n"
        f"可能战术: {', '.join(event['possible_tactics'])}\n"
        f"描述: {event['description']}"
    )


def token_count(tokenizer: Any, value: str) -> int:
    return len(tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"])


def fit_complete_events(tokenizer: Any, record: dict[str, Any]) -> tuple[str, int, int]:
    events = record["model_input"]["events"]
    retained = list(events)
    value = "\n\n".join(event_block(event) for event in retained)
    count = token_count(tokenizer, value)
    removed = 0
    while count > MAX_TOKENS and len(retained) > 2:
        retained.pop(0)
        removed += 1
        value = "\n\n".join(event_block(event) for event in retained)
        count = token_count(tokenizer, value)
    if count > MAX_TOKENS:
        raise ValueError(
            f"last two complete events exceed {MAX_TOKENS} tokens: "
            f"{record['audit_key']['sample_id']} ({count})"
        )
    return value, count, removed


def load_pinned_model(cache_dir: Path) -> tuple[Any, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    if int(model.config.hidden_size) != EXPECTED_DIMENSION:
        raise AssertionError(
            f"expected hidden size {EXPECTED_DIMENSION}, got {model.config.hidden_size}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def model_metadata(tokenizer: Any, model: Any, cache_dir: Path) -> dict[str, Any]:
    model_cache = cache_dir / f"models--{MODEL_ID.replace('/', '--')}"
    snapshot = model_cache / "snapshots" / MODEL_REVISION
    files: dict[str, dict[str, Any]] = {}
    if snapshot.exists():
        for path in sorted(value for value in snapshot.rglob("*") if value.is_file()):
            resolved = path.resolve()
            files[path.relative_to(snapshot).as_posix()] = {
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "hidden_size": int(model.config.hidden_size),
        "model_type": str(model.config.model_type),
        "config_max_position_embeddings": int(model.config.max_position_embeddings),
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cache_snapshot_files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    tokenizer, model = load_pinned_model(cache_dir)
    metadata = model_metadata(tokenizer, model, cache_dir)
    if args.download_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    managed = ["embeddings.npy", "embedding_index.csv", "embedding_manifest.json", "stdout.log"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite embeddings: {existing}")
    rows = read_inputs()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    model.to(device)

    fitted: list[tuple[dict[str, Any], str, int, int]] = []
    for row in rows:
        value, count, removed = fit_complete_events(tokenizer, row)
        fitted.append((row, value, count, removed))
    output.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        output / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(fitted), EXPECTED_DIMENSION),
    )
    index_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    messages: list[str] = []
    for start in range(0, len(fitted), args.batch_size):
        batch = fitted[start : start + args.batch_size]
        texts = [item[1] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output_state = model(**encoded).last_hidden_state[:, 0]
            dense = F.normalize(output_state, p=2, dim=1)
        array = dense.cpu().numpy().astype(np.float32, copy=False)
        embeddings[start : start + len(batch)] = array
        embeddings.flush()
        for offset, (record, text, tokens, removed) in enumerate(batch):
            vector = array[offset]
            index_rows.append(
                {
                    "embedding_row": start + offset,
                    "sample_id": record["audit_key"]["sample_id"],
                    "is_development": int(record["audit_key"]["is_development"]),
                    "development_slot": record["audit_key"]["development_slot"],
                    "token_count": tokens,
                    "oldest_events_removed": removed,
                    "retained_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "l2_norm": float(np.linalg.norm(vector)),
                }
            )
        if (start + len(batch)) % 25 == 0 or start + len(batch) == len(fitted):
            elapsed = time.perf_counter() - started
            message = (
                f"encoded {start + len(batch)}/{len(fitted)} "
                f"elapsed={elapsed:.1f}s rows_per_second={(start + len(batch))/elapsed:.3f}"
            )
            messages.append(message)
            print(message, flush=True)

    elapsed = time.perf_counter() - started
    write_csv(
        output / "embedding_index.csv",
        index_rows,
        [
            "embedding_row",
            "sample_id",
            "is_development",
            "development_slot",
            "token_count",
            "oldest_events_removed",
            "retained_text_sha256",
            "l2_norm",
        ],
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "input": {
            "path": INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(INPUT_PATH),
            "rows_encoded": len(fitted),
            "limit": args.limit,
        },
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "numpy": np.__version__,
            "device": str(device),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "num_threads": args.num_threads,
            "batch_size": args.batch_size,
        },
        "model": metadata,
        "pooling": "last_hidden_state[:,0] (CLS), then L2 normalize",
        "max_tokens": MAX_TOKENS,
        "event_boundary_truncation": "drop complete oldest events; retain at least last two",
        "token_counts": {
            "min": min(row["token_count"] for row in index_rows),
            "max": max(row["token_count"] for row in index_rows),
            "events_removed_total": sum(row["oldest_events_removed"] for row in index_rows),
        },
        "elapsed_seconds": elapsed,
        "rows_per_second": len(fitted) / elapsed,
        "embedding_sha256": sha256(output / "embeddings.npy"),
        "embedding_index_sha256": sha256(output / "embedding_index.csv"),
        "l2_norm_max_abs_error": max(abs(row["l2_norm"] - 1.0) for row in index_rows),
    }
    write_json(output / "embedding_manifest.json", manifest)
    stdout = "\n".join(messages) + "\n"
    (output / "stdout.log").write_text(stdout, encoding="utf-8")
    print(json.dumps(manifest["token_counts"], ensure_ascii=False))
    print(f"wrote embeddings to {output}")


if __name__ == "__main__":
    main()
