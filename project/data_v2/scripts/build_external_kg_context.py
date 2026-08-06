#!/usr/bin/env python3
"""Build deterministic, prefix-only KG context for the three external sources.

The retrieval and truncation constants reproduce the existing SIM pipeline:
recent 3 parent techniques, top 8 snippets, 260 characters per snippet,
1400 characters before the LLM-pipeline truncation, and 700 final characters.

The retrieval function receives only the prefix. It cannot inspect the target,
future steps, campaign transitions, or any evaluation split information.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUTS = (
    PROJECT_ROOT
    / "data_v2/repro_external/loader_snapshots/ctid_current_loader.csv",
    PROJECT_ROOT
    / "data_v2/repro_external/cumulative/attack_flow_cumulative.csv",
    PROJECT_ROOT
    / "data_v2/repro_external/cumulative/stockpile_cumulative.csv",
)
KG_CSV = PROJECT_ROOT / "data/attack_kg_snippets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_v2/repro_external/kg_context"
OUTPUT_CSV = OUTPUT_DIR / "external_prefixes_with_kg.csv"
REPORT_JSON = OUTPUT_DIR / "kg_context_report.json"

RECENT_K = 3
TOP_K = 8
MAX_CHARS_PER_SNIPPET = 260
MAX_CONTEXT_CHARS_BEFORE_LLM_TRUNCATION = 1400
MAX_CONTEXT_CHARS = 700

TYPE_PRIORITY = {
    "relationship::uses": 4,
    "attack-pattern": 3,
    "intrusion-set": 2,
    "campaign": 2,
    "tool": 2,
    "malware": 2,
}

TECHNIQUE_RE = re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parent_technique(value: object) -> str:
    match = TECHNIQUE_RE.search(clean_text(value).upper())
    return match.group(0).split(".", 1)[0] if match else ""


def parse_prefix(value: str) -> list[str]:
    """Parse the canonical JSON-array prefix without using any other row field."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prefix is not valid JSON: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"prefix is not a JSON array: {value!r}")
    result = [parent_technique(item) for item in parsed]
    if not result or any(not item for item in result):
        raise ValueError(f"prefix contains an invalid ATT&CK technique: {value!r}")
    return result


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    suffix = " ..."
    clipped = text[: max_chars - len(suffix)]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip() + suffix


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_kg() -> tuple[
    list[dict[str, str]], dict[str, list[int]], dict[str, list[int]]
]:
    rows = read_csv(KG_CSV)
    required = {
        "snippet_id",
        "snippet_type",
        "attack_id",
        "source_name",
        "target_name",
        "text",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else ()))
        raise ValueError(f"KG CSV missing required columns: {missing}")

    # Separate indexes preserve the old policy: use exact attack_id matches
    # first, and scan text mentions only if fewer than TOP_K candidates exist.
    exact_index: dict[str, list[int]] = defaultdict(list)
    text_index: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        exact = parent_technique(row.get("attack_id", ""))
        if exact:
            exact_index[exact].append(row_index)
        mentioned = {
            match.group(0).upper().split(".", 1)[0]
            for match in TECHNIQUE_RE.finditer(clean_text(row.get("text", "")))
        }
        for technique in sorted(mentioned):
            text_index[technique].append(row_index)
    return rows, dict(exact_index), dict(text_index)


def score_snippet(row: dict[str, str], query_ids: list[str]) -> float:
    score = float(TYPE_PRIORITY.get(clean_text(row.get("snippet_type", "")), 1))
    matched_attack_id = parent_technique(row.get("attack_id", ""))
    if matched_attack_id in query_ids:
        score += 5.0
    if clean_text(row.get("source_name", "")):
        score += 0.5
    if clean_text(row.get("target_name", "")):
        score += 0.5
    text = clean_text(row.get("text", ""))
    if 80 <= len(text) <= 350:
        score += 1.0
    elif len(text) < 40:
        score -= 0.5
    text_upper = text.upper()
    score += 0.25 * sum(technique in text_upper for technique in query_ids)
    return score


def retrieve(
    prefix: list[str],
    kg_rows: list[dict[str, str]],
    exact_index: dict[str, list[int]],
    text_index: dict[str, list[int]],
) -> list[dict[str, str]]:
    """Retrieve using prefix only; target/future fields are not arguments."""
    recent_ids = prefix[-RECENT_K:]
    candidates = {
        row_index
        for technique in recent_ids
        for row_index in exact_index.get(technique, ())
    }
    if len(candidates) < TOP_K:
        candidates.update(
            row_index
            for technique in recent_ids
            for row_index in text_index.get(technique, ())
        )
    scored = [
        (row_index, score_snippet(kg_rows[row_index], recent_ids))
        for row_index in candidates
    ]
    # The old code sorted only on score after iterating a set. The explicit
    # source-row tie break freezes a reproducible ordering without changing
    # the score definition.
    scored.sort(key=lambda item: (-item[1], item[0]))

    selected: list[dict[str, str]] = []
    seen_snippet_ids: set[str] = set()
    for row_index, _ in scored:
        row = kg_rows[row_index]
        snippet_id = clean_text(row.get("snippet_id", ""))
        if not snippet_id or snippet_id in seen_snippet_ids:
            continue
        seen_snippet_ids.add(snippet_id)
        selected.append(row)
        if len(selected) == TOP_K:
            break
    return selected


def build_context(selected: Iterable[dict[str, str]]) -> dict[str, object]:
    snippets: list[tuple[str, str, str]] = []
    total_chars = 0
    for row in selected:
        text = truncate_text(clean_text(row.get("text", "")), MAX_CHARS_PER_SNIPPET)
        if not text:
            continue
        if total_chars + len(text) > MAX_CONTEXT_CHARS_BEFORE_LLM_TRUNCATION:
            break
        snippets.append(
            (
                clean_text(row.get("snippet_id", "")),
                clean_text(row.get("snippet_type", "")),
                text,
            )
        )
        total_chars += len(text)

    before = " ; ".join(text for _, _, text in snippets).strip()
    after = truncate_text(before, MAX_CONTEXT_CHARS)

    # Keep only IDs whose text starts before the final truncation point.
    contributing_ids: list[str] = []
    contributing_types: list[str] = []
    cursor = 0
    visible_boundary = len(after) - 4 if after.endswith(" ...") else len(after)
    for snippet_id, snippet_type, text in snippets:
        if cursor < visible_boundary:
            contributing_ids.append(snippet_id)
            contributing_types.append(snippet_type)
        cursor += len(text) + 3  # separator: " ; "

    return {
        "kg_context": after,
        "kg_snippet_ids": json.dumps(contributing_ids, ensure_ascii=False),
        "kg_snippet_types": json.dumps(contributing_types, ensure_ascii=False),
        "kg_retrieved_snippet_ids": json.dumps(
            [snippet_id for snippet_id, _, _ in snippets], ensure_ascii=False
        ),
        "kg_chars_before": len(before),
        "kg_chars_after": len(after),
        "kg_empty_reason": "" if after else "no_prefix_snippet",
    }


def validate_unique_keys(rows: list[dict[str, object]]) -> None:
    keys = [
        (str(row["source"]), str(row["campaign_id"]), int(row["prefix_len"]))
        for row in rows
    ]
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"non-unique (source, campaign_id, prefix_len): {duplicates[:5]}")


def main() -> None:
    missing = [str(path) for path in (*INPUTS, KG_CSV) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required input(s): {missing}")

    kg_rows, exact_kg_index, text_kg_index = load_kg()
    output_rows: list[dict[str, object]] = []
    input_counts: dict[str, int] = {}

    for path in INPUTS:
        source_rows = read_csv(path)
        input_counts[str(path.relative_to(PROJECT_ROOT))] = len(source_rows)
        for row in source_rows:
            if not {"source", "campaign_id", "prefix_len", "prefix", "true_label"}.issubset(row):
                raise ValueError(f"{path} lacks a required external-data column")
            prefix = parse_prefix(row["prefix"])
            # Leakage barrier: retrieve() receives only the prefix and fixed KG.
            selected = retrieve(prefix, kg_rows, exact_kg_index, text_kg_index)
            context = build_context(selected)
            output_rows.append(
                {
                    **row,
                    "recent_prefix_ids": json.dumps(prefix[-RECENT_K:], ensure_ascii=False),
                    **context,
                }
            )

    validate_unique_keys(output_rows)
    expected_counts = {"ctid": 281, "attack_flow": 431, "stockpile": 122}
    actual_counts = Counter(str(row["source"]) for row in output_rows)
    if dict(actual_counts) != expected_counts:
        raise ValueError(f"unexpected source counts: {dict(actual_counts)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for output_row in output_rows:
        for field in output_row:
            if field not in seen_fields:
                fieldnames.append(field)
                seen_fields.add(field)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    empty_counts = Counter(
        str(row["source"])
        for row in output_rows
        if row["kg_empty_reason"] == "no_prefix_snippet"
    )
    source_summary = {}
    for source in expected_counts:
        source_rows = [row for row in output_rows if row["source"] == source]
        before_lengths = [int(row["kg_chars_before"]) for row in source_rows]
        after_lengths = [int(row["kg_chars_after"]) for row in source_rows]
        source_summary[source] = {
            "rows": len(source_rows),
            "empty_kg_rows": empty_counts[source],
            "empty_kg_ratio": empty_counts[source] / len(source_rows),
            "kg_chars_before_mean": sum(before_lengths) / len(before_lengths),
            "kg_chars_after_mean": sum(after_lengths) / len(after_lengths),
            "kg_chars_after_max": max(after_lengths),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leakage_guard": "retrieve() receives prefix only; true_label and future rows are never arguments",
        "constants": {
            "recent_k": RECENT_K,
            "top_k": TOP_K,
            "max_chars_per_snippet": MAX_CHARS_PER_SNIPPET,
            "max_context_chars_before_llm_truncation": MAX_CONTEXT_CHARS_BEFORE_LLM_TRUNCATION,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "truncation": "hard maximum includes the ellipsis suffix",
            "tie_break": "descending legacy score, then ascending KG source-row index",
        },
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "sha256": sha256_file(path),
                "rows": (
                    len(kg_rows)
                    if path == KG_CSV
                    else input_counts.get(str(path.relative_to(PROJECT_ROOT)))
                ),
            }
            for path in (*INPUTS, KG_CSV)
        },
        "sources": source_summary,
        "output": {
            "path": str(OUTPUT_CSV.relative_to(PROJECT_ROOT)),
            "rows": len(output_rows),
            "sha256": sha256_file(OUTPUT_CSV),
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    with REPORT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
