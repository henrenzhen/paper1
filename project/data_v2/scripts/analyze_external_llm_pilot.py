#!/usr/bin/env python3
"""Evaluate the eight automatic pilot gates and prepare a blinded review sheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*"
)
warnings.filterwarnings(
    "ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax.*"
)

import tokenizers
import transformers
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data_v4/external_reasoning/pilot/runs/20260806T080230Z_1842924a"
)
MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"
MAX_LENGTH = 512
BLIND_SEED = "external-pilot-v4.2-blind-20260806"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["source"], row["campaign_id"], int(row["prefix_len"])


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def gate(
    number: int,
    name: str,
    numerator: int | float,
    denominator: int | float,
    observed: float,
    operator: str,
    threshold: float | tuple[float, float],
    blocking: bool,
) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= float(threshold)
    elif operator == "<=":
        passed = observed <= float(threshold)
    elif operator == "range":
        low, high = threshold  # type: ignore[misc]
        passed = low <= observed <= high
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "gate": number,
        "name": name,
        "numerator": numerator,
        "denominator": denominator,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
        "blocking": blocking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--encoding-mode",
        choices=("standard", "chunked"),
        default="standard",
        help="BGE encoding mode frozen for this pilot round",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "pilot_raw_results.csv"
    attempts_path = run_dir / "raw_attempts.jsonl"
    generation_manifest_path = run_dir / "generation_manifest.json"
    for required in (raw_path, attempts_path, generation_manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)

    _, rows = read_csv(raw_path)
    if len(rows) != 30 or len({key(row) for row in rows}) != 30:
        raise ValueError("pilot raw results must contain 30 unique external keys")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        use_fast=True,
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if tokenizer.model_max_length != MAX_LENGTH or special_tokens != 2:
        raise ValueError(
            f"unexpected tokenizer config: {tokenizer.model_max_length}/{special_tokens}"
        )

    row_audits: list[dict[str, Any]] = []
    valid_reasoning_lengths: list[int] = []
    valid_token_lengths: list[int] = []
    tokenizer_truncated = 0
    rows_with_token_loss_after_chunking = 0
    content_limit = MAX_LENGTH - special_tokens
    for row in rows:
        valid_reasoning = as_bool(row["valid_reasoning"])
        reasoning = row["llm_thinking_process"]
        token_length: int | None = None
        would_truncate: bool | None = None
        chunk_count: int | None = None
        chunk_max_with_special: int | None = None
        chunk_token_loss: int | None = None
        if valid_reasoning:
            content_ids = tokenizer.encode(
                reasoning,
                add_special_tokens=False,
                truncation=False,
            )
            token_length = len(content_ids) + special_tokens
            would_truncate = token_length > MAX_LENGTH
            chunks = [
                content_ids[index : index + content_limit]
                for index in range(0, len(content_ids), content_limit)
            ]
            reconstructed = [token for chunk in chunks for token in chunk]
            if reconstructed != content_ids:
                raise AssertionError("non-overlap chunking did not preserve token IDs")
            chunk_count = len(chunks)
            chunk_max_with_special = max(len(chunk) + special_tokens for chunk in chunks)
            chunk_token_loss = len(content_ids) - len(reconstructed)
            if chunk_max_with_special > MAX_LENGTH:
                raise AssertionError("chunk exceeds BGE model_max_length")
            valid_reasoning_lengths.append(len(reasoning))
            valid_token_lengths.append(token_length)
            tokenizer_truncated += int(would_truncate)
            rows_with_token_loss_after_chunking += int(chunk_token_loss != 0)
        row_audits.append(
            {
                "source": row["source"],
                "campaign_id": row["campaign_id"],
                "prefix_len": int(row["prefix_len"]),
                "generation_status": row["generation_status"],
                "attempt": int(row["attempt"]),
                "json_success": row["generation_status"] == "ok",
                "reasoning_content_len": int(row["reasoning_content_len"]),
                "finish_reason": row["finish_reason"],
                "three_sections_present": as_bool(row["three_sections_present"]),
                "valid_top5": as_bool(row["valid_top5"]),
                "sections_nonempty": as_bool(row["sections_nonempty"]),
                "reasoning_chars": len(reasoning) if valid_reasoning else "",
                "bge_tokens_with_special_tokens": (
                    token_length if token_length is not None else ""
                ),
                "bge_would_truncate_at_512": (
                    would_truncate if would_truncate is not None else ""
                ),
                "bge_chunk_count": chunk_count if chunk_count is not None else "",
                "bge_max_chunk_tokens_with_special": (
                    chunk_max_with_special
                    if chunk_max_with_special is not None
                    else ""
                ),
                "bge_chunk_token_loss": (
                    chunk_token_loss if chunk_token_loss is not None else ""
                ),
            }
        )

    total = len(rows)
    ok_count = sum(row["generation_status"] == "ok" for row in rows)
    internal_zero = sum(int(row["reasoning_content_len"]) == 0 for row in rows)
    finish_length = sum(row["finish_reason"] == "length" for row in rows)
    three_sections = sum(as_bool(row["three_sections_present"]) for row in rows)
    valid_top5 = sum(as_bool(row["valid_top5"]) for row in rows)
    nonempty_sections = sum(as_bool(row["sections_nonempty"]) for row in rows)
    valid_reasoning_count = len(valid_reasoning_lengths)
    mean_reasoning_chars = statistics.mean(valid_reasoning_lengths)
    truncation_rate = rate(tokenizer_truncated, valid_reasoning_count)
    if args.encoding_mode == "standard":
        gate7_numerator = tokenizer_truncated
        gate7_rate = truncation_rate
        gate7_name = "BGE tokenizer truncation among valid reasoning"
    else:
        gate7_numerator = rows_with_token_loss_after_chunking
        gate7_rate = rate(rows_with_token_loss_after_chunking, valid_reasoning_count)
        gate7_name = "BGE token loss after mandatory non-overlap chunking"

    gates = [
        gate(1, "JSON parse success after retries", ok_count, total, rate(ok_count, total), ">=", 0.95, True),
        gate(2, "internal reasoning_content absent or length zero", internal_zero, total, rate(internal_zero, total), ">=", 1.0, True),
        gate(3, "finish_reason=length", finish_length, total, rate(finish_length, total), "<=", 0.02, True),
        gate(4, "three required section titles", three_sections, total, rate(three_sections, total), ">=", 0.95, True),
        gate(5, "valid fixed-vocabulary Top-5", valid_top5, total, rate(valid_top5, total), ">=", 0.90, True),
        gate(6, "all three sections nonempty", nonempty_sections, total, rate(nonempty_sections, total), ">=", 0.95, True),
        gate(7, gate7_name, gate7_numerator, valid_reasoning_count, gate7_rate, "<=", 0.05, True),
        gate(8, "mean reasoning characters among valid reasoning", sum(valid_reasoning_lengths), valid_reasoning_count, mean_reasoning_chars, "range", (600.0, 1100.0), False),
    ]
    automatic_blocking_passed = all(
        item["passed"] for item in gates if item["blocking"]
    )

    gate_rows_path = run_dir / "pilot_gate_rows.csv"
    write_csv(gate_rows_path, list(row_audits[0]), row_audits)

    # The review sheet is deliberately shuffled and contains no true_label,
    # source, campaign_id, prefix_len, or unique sample key.
    blinded: list[tuple[str, dict[str, str]]] = []
    blind_key: dict[str, Any] = {}
    for row in rows:
        row_key = key(row)
        rank = stable_hash(BLIND_SEED, *row_key)
        blinded.append((rank, row))
    blinded.sort(key=lambda item: item[0])

    review_rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(blinded, start=1):
        review_id = f"R{index:02d}"
        predictions = json.loads(row["predicted_next_ttps"] or "[]")
        output_for_review = (
            row["llm_thinking_process"]
            if row["llm_thinking_process"].strip()
            else row["raw_output"]
        )
        review_rows.append(
            {
                "review_id": review_id,
                "generation_status": row["generation_status"],
                "prefix": row["prefix"],
                "kg_context": "",  # populated from frozen pilot input below
                "kg_snippet_ids": row["kg_snippet_ids"],
                "reasoning_or_raw_output": output_for_review,
                "predicted_next_ttps": json.dumps(predictions, ensure_ascii=False),
                "reviewer_id": "",
                "tactic_judgment": "",
                "tactic_reason": "",
                "asset_or_permission_unsupported": "",
                "asset_permission_reason": "",
                "kg_traceable": "",
                "kg_traceability_reason": "",
                "candidate_1_unsupported": "",
                "candidate_2_unsupported": "",
                "candidate_3_unsupported": "",
                "candidate_4_unsupported": "",
                "candidate_5_unsupported": "",
                "candidate_reason": "",
            }
        )
        blind_key[review_id] = {
            "source": row["source"],
            "campaign_id": row["campaign_id"],
            "prefix_len": int(row["prefix_len"]),
            "true_label": row["true_label"],
        }

    pilot_input = (
        PROJECT_ROOT / "data_v4/external_reasoning/pilot/pilot_sample_30.csv"
    )
    _, input_rows = read_csv(pilot_input)
    context_by_key = {key(row): row["kg_context"] for row in input_rows}
    for review_row in review_rows:
        hidden = blind_key[review_row["review_id"]]
        hidden_key = (
            hidden["source"],
            hidden["campaign_id"],
            hidden["prefix_len"],
        )
        review_row["kg_context"] = context_by_key[hidden_key]

    review_sheet_path = run_dir / "blind_review_sheet.csv"
    write_csv(review_sheet_path, list(review_rows[0]), review_rows)
    blind_key_path = run_dir / "blind_review_key.json"
    blind_key_path.write_text(
        json.dumps(
            {
                "warning": "Do not provide this file to the blinded reviewer.",
                "blind_seed_sha256": hashlib.sha256(BLIND_SEED.encode()).hexdigest(),
                "mapping": blind_key,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "generated_at": utc_now(),
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "raw_inputs": {
            raw_path.name: sha256(raw_path),
            attempts_path.name: sha256(attempts_path),
            generation_manifest_path.name: sha256(generation_manifest_path),
            pilot_input.relative_to(PROJECT_ROOT).as_posix(): sha256(pilot_input),
        },
        "status_counts": dict(
            sorted(Counter(row["generation_status"] for row in rows).items())
        ),
        "automatic_gates": gates,
        "automatic_blocking_gates_passed": automatic_blocking_passed,
        "reasoning_character_distribution_valid_rows": {
            "n": valid_reasoning_count,
            "mean": mean_reasoning_chars,
            "median": statistics.median(valid_reasoning_lengths),
            "min": min(valid_reasoning_lengths),
            "max": max(valid_reasoning_lengths),
        },
        "bge_token_distribution_valid_rows": {
            "n": len(valid_token_lengths),
            "mean": statistics.mean(valid_token_lengths),
            "median": statistics.median(valid_token_lengths),
            "min": min(valid_token_lengths),
            "max": max(valid_token_lengths),
        },
        "bge": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_max_length": tokenizer.model_max_length,
            "special_tokens_to_add": special_tokens,
            "content_limit_if_chunking": MAX_LENGTH - special_tokens,
            "encoding_mode": args.encoding_mode,
            "raw_would_truncate_rows": tokenizer_truncated,
            "raw_would_truncate_rate": truncation_rate,
            "rows_with_token_loss_after_chunking": rows_with_token_loss_after_chunking,
            "chunking_required": truncation_rate > 0.05,
            "chunking_resolution_passed": (
                args.encoding_mode == "chunked"
                and rows_with_token_loss_after_chunking == 0
            ),
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "groundedness": {
            "status": "pending_human_blind_review",
            "required_rows": 30,
            "review_sheet": review_sheet_path.name,
            "review_sheet_sha256": sha256(review_sheet_path),
            "private_key_file": blind_key_path.name,
            "private_key_file_sha256": sha256(blind_key_path),
        },
        "outputs": {
            gate_rows_path.name: sha256(gate_rows_path),
            review_sheet_path.name: sha256(review_sheet_path),
            blind_key_path.name: sha256(blind_key_path),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    report_path = run_dir / "pilot_gate_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
