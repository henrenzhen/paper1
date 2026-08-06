#!/usr/bin/env python3
"""Freeze the pre-registered 30-row external LLM pilot sample."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "data_v2/repro_external/kg_context/external_prefixes_with_kg_in184.csv"
)
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
PROTOCOL_PATH = (
    PROJECT_ROOT / "data_v4/protocols/LLM_semantic_external_validation_v4.2.md"
)
OUTPUT_DIR = PROJECT_ROOT / "data_v4/external_reasoning/pilot"
OUTPUT_CSV = OUTPUT_DIR / "pilot_sample_30.csv"
MANIFEST_PATH = OUTPUT_DIR / "pilot_sample_manifest.json"

SEED = 20260806
SOURCE_ORDER = ("ctid", "attack_flow", "stockpile")
ALLOCATION = {"short": 3, "medium": 4, "long": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nearest_rank(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(proportion * len(ordered)) - 1]


def stratum(prefix_len: int, q1: int, q2: int) -> str:
    if prefix_len <= q1:
        return "short"
    if prefix_len <= q2:
        return "medium"
    return "long"


def unique_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["source"], row["campaign_id"], int(row["prefix_len"])


def choose_source_rows(
    source: str, rows: list[dict[str, str]], q1: int, q2: int
) -> list[dict[str, Any]]:
    by_stratum_campaign: dict[str, dict[str, list[dict[str, str]]]] = {
        name: defaultdict(list) for name in ALLOCATION
    }
    for row in rows:
        name = stratum(int(row["prefix_len"]), q1, q2)
        by_stratum_campaign[name][row["campaign_id"]].append(row)

    slots = [
        (name, slot_number)
        for name, count in ALLOCATION.items()
        for slot_number in range(1, count + 1)
    ]
    # Allocate the most constrained stratum first; ties follow protocol order.
    stratum_order = {name: index for index, name in enumerate(ALLOCATION)}
    slots.sort(
        key=lambda slot: (
            len(by_stratum_campaign[slot[0]]),
            stratum_order[slot[0]],
            slot[1],
        )
    )

    def ordered_campaigns(name: str, slot_number: int) -> list[str]:
        return sorted(
            by_stratum_campaign[name],
            key=lambda campaign: stable_rank(
                SEED, source, name, slot_number, campaign
            ),
        )

    assignment: list[tuple[str, int, str]] = []

    def backtrack(slot_index: int, used_campaigns: set[str]) -> bool:
        if slot_index == len(slots):
            return True
        name, slot_number = slots[slot_index]
        for campaign in ordered_campaigns(name, slot_number):
            if campaign in used_campaigns:
                continue
            assignment.append((name, slot_number, campaign))
            used_campaigns.add(campaign)
            if backtrack(slot_index + 1, used_campaigns):
                return True
            used_campaigns.remove(campaign)
            assignment.pop()
        return False

    if not backtrack(0, set()):
        raise ValueError(f"pilot constraints are infeasible for source={source}")

    selected: list[dict[str, Any]] = []
    for name, slot_number, campaign in assignment:
        candidates = by_stratum_campaign[name][campaign]
        row = min(
            candidates,
            key=lambda candidate: stable_rank(
                SEED,
                source,
                name,
                slot_number,
                *unique_key(candidate),
            ),
        )
        selected.append(
            {
                "pilot_slot": f"{source}:{name}:{slot_number}",
                "length_stratum": name,
                "source_q1": q1,
                "source_q2": q2,
                **row,
            }
        )

    selected.sort(
        key=lambda row: (
            stratum_order[str(row["length_stratum"])],
            int(str(row["pilot_slot"]).rsplit(":", 1)[1]),
        )
    )
    return selected


def main() -> None:
    missing = [
        str(path)
        for path in (INPUT_PATH, VOCAB_PATH, PROTOCOL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing pilot input(s): {missing}")

    input_fields, rows = read_csv(INPUT_PATH)
    if len(rows) != 830:
        raise ValueError(f"expected 830 closed-set rows, found {len(rows)}")
    required = {
        "source",
        "campaign_id",
        "prefix_len",
        "prefix",
        "true_label",
        "kg_context",
        "kg_snippet_ids",
    }
    if not required.issubset(input_fields):
        raise ValueError(f"pilot input lacks columns: {sorted(required-set(input_fields))}")

    selected: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["source"] == source]
        lengths = [int(row["prefix_len"]) for row in source_rows]
        q1 = nearest_rank(lengths, 1 / 3)
        q2 = nearest_rank(lengths, 2 / 3)
        chosen = choose_source_rows(source, source_rows, q1, q2)
        selected.extend(chosen)
        source_manifest[source] = {
            "input_rows": len(source_rows),
            "campaigns": len({row["campaign_id"] for row in source_rows}),
            "q1": q1,
            "q2": q2,
            "allocation": dict(ALLOCATION),
            "selected_rows": len(chosen),
            "selected_campaigns": len({row["campaign_id"] for row in chosen}),
            "selected_keys": [list(unique_key(row)) for row in chosen],
        }

    if len(selected) != 30:
        raise ValueError(f"expected 30 pilot rows, found {len(selected)}")
    keys = [unique_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot unique-key collision")
    for source in SOURCE_ORDER:
        source_rows = [row for row in selected if row["source"] == source]
        observed = Counter(str(row["length_stratum"]) for row in source_rows)
        if dict(observed) != ALLOCATION:
            raise ValueError(f"pilot stratum mismatch for {source}: {dict(observed)}")
        if len({row["campaign_id"] for row in source_rows}) != 10:
            raise ValueError(f"pilot campaign uniqueness failed for {source}")

    priority_fields = [
        "pilot_slot",
        "length_stratum",
        "source_q1",
        "source_q2",
        "source",
        "campaign_id",
        "prefix_len",
        "prefix",
        "true_label",
        "recent_prefix_ids",
        "kg_context",
        "kg_snippet_ids",
        "kg_chars_before",
        "kg_chars_after",
        "kg_empty_reason",
    ]
    output_fields = priority_fields + [
        field for field in input_fields if field not in priority_fields
    ]
    write_csv(OUTPUT_CSV, output_fields, selected)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "algorithm": {
            "quantiles": (
                "within-source nearest-rank q1=x_(ceil(N/3)), "
                "q2=x_(ceil(2N/3))"
            ),
            "strata": "short<=q1; q1<medium<=q2; long>q2",
            "allocation_per_source": dict(ALLOCATION),
            "campaign_constraint": "10 distinct campaigns per source",
            "ranking": (
                "SHA256(seed, source, stratum-slot, campaign/key); deterministic "
                "backtracking returns the first feasible assignment"
            ),
            "label_usage": "true_label is carried to output but never used for selection",
        },
        "inputs": {
            INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(): sha256(INPUT_PATH),
            VOCAB_PATH.relative_to(PROJECT_ROOT).as_posix(): sha256(VOCAB_PATH),
            PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(): sha256(PROTOCOL_PATH),
        },
        "sources": source_manifest,
        "output": {
            "path": OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "rows": len(selected),
            "sha256": sha256(OUTPUT_CSV),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
