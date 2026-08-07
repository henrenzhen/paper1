#!/usr/bin/env python3
"""Encode the frozen 784-row LLM summary text using pinned BGE-M3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch.nn import functional as F

from encode_future3_bge_m3 import (
    EXPECTED_DIMENSION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_LOCK,
    load_pinned_model,
    model_metadata,
    sha256,
    token_count,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1"
INPUT_PATH = INPUT_DIR / "summary_inputs.jsonl"
INPUT_MANIFEST = INPUT_DIR / "summary_manifest.json"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/llm_summary_semantic_future3_v1.md"
DEFAULT_CACHE = PROJECT_ROOT / ".hf_cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_embeddings/llm_summary_bge_m3_5617a9f"


def read_inputs() -> list[dict[str, Any]]:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 784:
        raise AssertionError(f"expected 784 frozen summary inputs, found {len(rows)}")
    if len({row["audit_key"]["sample_id"] for row in rows}) != 784:
        raise AssertionError("summary input sample IDs are not unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-threads", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ("embeddings.npy", "embedding_index.csv", "embedding_manifest.json", "stdout.log")
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite S embeddings: {existing}")
    rows = read_inputs()
    tokenizer, model = load_pinned_model(args.cache_dir.resolve())
    metadata = model_metadata(tokenizer, model, args.cache_dir.resolve())
    fitted: list[tuple[dict[str, Any], str, int]] = []
    for row in rows:
        text = row["model_input"]["summary_text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["summary_text_sha256"]:
            raise AssertionError(f"summary text hash failed: {row['audit_key']['sample_id']}")
        tokens = token_count(tokenizer, text)
        if tokens > MAX_TOKENS:
            raise ValueError(
                f"summary exceeds {MAX_TOKENS} tokens without allowed truncation: "
                f"{row['audit_key']['sample_id']} ({tokens})"
            )
        fitted.append((row, text, tokens))

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    model.to(device)
    output.mkdir(parents=True, exist_ok=False)
    embeddings = np.lib.format.open_memmap(
        output / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(fitted), EXPECTED_DIMENSION),
    )
    index_rows: list[dict[str, Any]] = []
    messages: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(fitted), args.batch_size):
        batch = fitted[start : start + args.batch_size]
        encoded = tokenizer(
            [item[1] for item in batch],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            dense = F.normalize(model(**encoded).last_hidden_state[:, 0], p=2, dim=1)
        array = dense.cpu().numpy().astype(np.float32, copy=False)
        embeddings[start : start + len(batch)] = array
        for offset, (record, text, tokens) in enumerate(batch):
            vector = array[offset]
            index_rows.append(
                {
                    "embedding_row": start + offset,
                    "sample_id": record["audit_key"]["sample_id"],
                    "token_count": tokens,
                    "summary_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "l2_norm": float(np.linalg.norm(vector)),
                }
            )
        completed = start + len(batch)
        if completed % 64 == 0 or completed == len(fitted):
            elapsed = time.perf_counter() - started
            message = (
                f"encoded {completed}/{len(fitted)} elapsed={elapsed:.1f}s "
                f"rows_per_second={completed/elapsed:.3f}"
            )
            messages.append(message)
            print(message, flush=True)
    embeddings.flush()
    elapsed = time.perf_counter() - started
    write_csv(
        output / "embedding_index.csv",
        index_rows,
        ("embedding_row", "sample_id", "token_count", "summary_text_sha256", "l2_norm"),
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "method_card_sha256": sha256(METHOD_CARD),
        "input": {
            "path": INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(INPUT_PATH),
            "manifest_sha256": sha256(INPUT_MANIFEST),
            "rows": len(rows),
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
            "num_threads": args.num_threads,
            "batch_size": args.batch_size,
        },
        "model": metadata,
        "pooling": "last_hidden_state[:,0] (CLS), then L2 normalize",
        "max_tokens": MAX_TOKENS,
        "truncation": "none allowed for S summaries",
        "token_counts": {
            "min": min(row["token_count"] for row in index_rows),
            "max": max(row["token_count"] for row in index_rows),
        },
        "elapsed_seconds": elapsed,
        "rows_per_second": len(fitted) / elapsed,
        "embedding_sha256": sha256(output / "embeddings.npy"),
        "embedding_index_sha256": sha256(output / "embedding_index.csv"),
        "l2_norm_max_abs_error": max(abs(row["l2_norm"] - 1.0) for row in index_rows),
    }
    write_json(output / "embedding_manifest.json", manifest)
    (output / "stdout.log").write_text("\n".join(messages) + "\n", encoding="utf-8")
    print(json.dumps(manifest["token_counts"], ensure_ascii=False))
    print(f"wrote S embeddings to {output}")


if __name__ == "__main__":
    main()
