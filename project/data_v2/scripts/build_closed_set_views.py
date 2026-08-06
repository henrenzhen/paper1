#!/usr/bin/env python3
"""Create audited 184-class closed-set views without modifying raw inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
EXPECTED_VOCAB_SHA256 = (
    "9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738"
)

SOURCE_INPUTS = {
    "ctid": PROJECT_ROOT
    / "data_v2/repro_external/loader_snapshots/ctid_current_loader.csv",
    "attack_flow": PROJECT_ROOT
    / "data_v2/repro_external/cumulative/attack_flow_cumulative.csv",
    "stockpile": PROJECT_ROOT
    / "data_v2/repro_external/cumulative/stockpile_cumulative.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "data_v2/repro_external/closed_set"
SOURCE_OUTPUTS = {
    "ctid": OUTPUT_DIR / "ctid_in184.csv",
    "attack_flow": OUTPUT_DIR / "attack_flow_cumulative_in184.csv",
    "stockpile": OUTPUT_DIR / "stockpile_cumulative_in184.csv",
}
KG_INPUT = (
    PROJECT_ROOT
    / "data_v2/repro_external/kg_context/external_prefixes_with_kg.csv"
)
KG_OUTPUT = (
    PROJECT_ROOT
    / "data_v2/repro_external/kg_context/external_prefixes_with_kg_in184.csv"
)
REPORT_PATH = OUTPUT_DIR / "closed_set_report.json"

EXPECTED = {
    "ctid": {"raw_rows": 281, "closed_rows": 281, "campaigns": 10},
    "attack_flow": {"raw_rows": 431, "closed_rows": 428, "campaigns": 35},
    "stockpile": {"raw_rows": 122, "closed_rows": 121, "campaigns": 27},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["source"], row["campaign_id"], int(row["prefix_len"])


def audit_source(
    source: str,
    input_path: Path,
    output_path: Path,
    vocabulary: set[str],
) -> tuple[dict[str, Any], list[tuple[str, str, int]]]:
    fieldnames, raw_rows = read_csv(input_path)
    required = {"source", "campaign_id", "prefix_len", "prefix", "true_label"}
    if not required.issubset(fieldnames):
        raise ValueError(f"{input_path} lacks required columns: {sorted(required)}")
    if any(row["source"] != source for row in raw_rows):
        raise ValueError(f"source mismatch in {input_path}")

    kept = [row for row in raw_rows if row["true_label"].strip() in vocabulary]
    dropped = [row for row in raw_rows if row["true_label"].strip() not in vocabulary]
    raw_keys = [unique_key(row) for row in raw_rows]
    kept_keys = [unique_key(row) for row in kept]
    if len(raw_keys) != len(set(raw_keys)) or len(kept_keys) != len(set(kept_keys)):
        raise ValueError(f"non-unique external key in {input_path}")

    before = Counter(row["campaign_id"] for row in raw_rows)
    after = Counter(row["campaign_id"] for row in kept)
    removed_campaigns = sorted(set(before) - set(after))
    expected = EXPECTED[source]
    observed = {
        "raw_rows": len(raw_rows),
        "closed_rows": len(kept),
        "campaigns": len(after),
    }
    if observed != expected:
        raise ValueError(f"unexpected closed-set shape for {source}: {observed}")
    if removed_campaigns:
        raise ValueError(f"closed-set filter removed campaigns in {source}: {removed_campaigns}")

    write_csv(output_path, fieldnames, kept)
    dropped_details = [
        {
            "source": row["source"],
            "campaign_id": row["campaign_id"],
            "prefix_len": int(row["prefix_len"]),
            "true_label": row["true_label"],
            "campaign_rows_before": before[row["campaign_id"]],
            "campaign_rows_after": after[row["campaign_id"]],
        }
        for row in dropped
    ]
    audit = {
        "raw_path": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "raw_sha256": sha256(input_path),
        "raw_rows": len(raw_rows),
        "closed_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "closed_sha256": sha256(output_path),
        "closed_rows": len(kept),
        "rows_dropped": len(dropped),
        "campaigns_before": len(before),
        "campaigns_after": len(after),
        "campaign_macro_J": len(after),
        "campaigns_removed": removed_campaigns,
        "dropped_rows": dropped_details,
    }
    return audit, kept_keys


def main() -> None:
    required_paths = [VOCAB_PATH, KG_INPUT, *SOURCE_INPUTS.values()]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing closed-set input(s): {missing}")
    vocab_hash = sha256(VOCAB_PATH)
    if vocab_hash != EXPECTED_VOCAB_SHA256:
        raise ValueError(f"fixed vocabulary hash mismatch: {vocab_hash}")
    _, vocab_rows = read_csv(VOCAB_PATH)
    vocabulary = {
        row["technique_id_parent"].strip()
        for row in vocab_rows
        if row["technique_id_parent"].strip()
    }
    if len(vocabulary) != 184:
        raise ValueError(f"expected 184 labels, found {len(vocabulary)}")

    source_audits: dict[str, Any] = {}
    all_kept_keys: set[tuple[str, str, int]] = set()
    for source in ("ctid", "attack_flow", "stockpile"):
        audit, kept_keys = audit_source(
            source,
            SOURCE_INPUTS[source],
            SOURCE_OUTPUTS[source],
            vocabulary,
        )
        source_audits[source] = audit
        all_kept_keys.update(kept_keys)

    kg_fields, kg_rows = read_csv(KG_INPUT)
    kg_keys = [unique_key(row) for row in kg_rows]
    if len(kg_keys) != len(set(kg_keys)):
        raise ValueError("KG input has duplicate external keys")
    kg_kept = [row for row in kg_rows if unique_key(row) in all_kept_keys]
    if len(kg_kept) != 830 or len(all_kept_keys) != 830:
        raise ValueError(
            f"closed-set KG/key count mismatch: {len(kg_kept)}/{len(all_kept_keys)}"
        )
    write_csv(KG_OUTPUT, kg_fields, kg_kept)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "primary": (
                "filter rows only when true_label is outside the frozen 184-class "
                "vocabulary; preserve all prefix tokens"
            ),
            "sensitivity": (
                "use all 834 raw rows; assign every out-of-vocabulary target Top-1/"
                "Hit@5 failure and MRR=0 for every system; do not train OOV targets"
            ),
            "campaign_macro": (
                "J is the number of campaigns remaining in each source; no campaign "
                "is removed; within-campaign n_j uses the selected analysis view"
            ),
        },
        "vocabulary": {
            "path": VOCAB_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": vocab_hash,
            "labels": len(vocabulary),
        },
        "sources": source_audits,
        "totals": {
            "raw_rows": sum(item["raw_rows"] for item in source_audits.values()),
            "closed_rows": sum(
                item["closed_rows"] for item in source_audits.values()
            ),
            "rows_dropped": sum(
                item["rows_dropped"] for item in source_audits.values()
            ),
        },
        "kg_closed_set": {
            "raw_path": KG_INPUT.relative_to(PROJECT_ROOT).as_posix(),
            "raw_sha256": sha256(KG_INPUT),
            "raw_rows": len(kg_rows),
            "closed_path": KG_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
            "closed_sha256": sha256(KG_OUTPUT),
            "closed_rows": len(kg_kept),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
