#!/usr/bin/env python3
"""Build frozen S-branch text and separate B0 rankings from raw LLM output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = (
    PROJECT_ROOT
    / "data_v4/external_reasoning/future3/full/runs/20260807T075707Z_310c2c11"
)
RESULTS_PATH = GENERATION_DIR / "full_raw_results.csv"
GENERATION_MANIFEST = GENERATION_DIR / "generation_manifest.json"
SAMPLES_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/llm_summary_semantic_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1"

ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
SUMMARY_FIELDS = (
    "stage_assessment",
    "observed_capabilities",
    "likely_next_intents",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_summary(row: dict[str, str]) -> str:
    return (
        f"[阶段评估]\n{row['stage_assessment'].strip()}\n\n"
        f"[已观察能力]\n{row['observed_capabilities'].strip()}\n\n"
        f"[可能后续意图]\n{row['likely_next_intents'].strip()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = (
        "summary_inputs.jsonl",
        "summary_index.csv",
        "b0_rankings.csv",
        "summary_manifest.json",
        "report.md",
    )
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite summary artifacts: {existing}")

    results = read_csv(RESULTS_PATH)
    samples = {row["sample_id"]: row for row in read_csv(SAMPLES_PATH)}
    vocabulary = {
        row["technique_id_parent"].strip() for row in read_csv(VOCAB_PATH)
    }
    if len(results) != 784 or len(vocabulary) != 184:
        raise AssertionError(
            f"frozen counts changed: results={len(results)} vocab={len(vocabulary)}"
        )
    summary_records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    b0_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for result in results:
        sample_id = result["sample_id"]
        sample = samples.get(sample_id)
        if sample is None or sample["is_development"] != "0" or sample_id in seen:
            raise AssertionError(f"formal summary key gate failed: {sample_id}")
        seen.add(sample_id)
        if (
            result["generation_status"] != "ok"
            or result["valid_summary"] != "True"
            or result["valid_top5"] != "True"
            or result["reasoning_content_len"] != "0"
        ):
            raise AssertionError(f"invalid formal generation row: {sample_id}")
        values = [result[field].strip() for field in SUMMARY_FIELDS]
        if not all(values):
            raise AssertionError(f"empty summary field: {sample_id}")
        summary_text = serialize_summary(result)
        if ATTACK_ID_RE.search(summary_text):
            raise AssertionError(f"summary contains ATT&CK ID: {sample_id}")
        for literal in (sample["source"], sample["campaign_id"]):
            if literal and literal.casefold() in summary_text.casefold():
                raise AssertionError(
                    f"summary contains forbidden metadata literal {literal!r}: {sample_id}"
                )
        predictions = json.loads(result["predicted_next_ttps"])
        if (
            len(predictions) != 5
            or len(set(predictions)) != 5
            or any(label not in vocabulary for label in predictions)
        ):
            raise AssertionError(f"invalid B0 ranking: {sample_id}")
        targets = json.loads(sample["target_parent_ids"])
        source_counts[sample["source"]] += 1
        summary_records.append(
            {
                "audit_key": {"sample_id": sample_id},
                "model_input": {"summary_text": summary_text},
                "summary_text_sha256": text_sha256(summary_text),
            }
        )
        index_rows.append(
            {
                "sample_id": sample_id,
                "source_audit_only": sample["source"],
                "campaign_id_audit_only": sample["campaign_id"],
                "prefix_len": sample["prefix_len"],
                "summary_characters": len(summary_text),
                "summary_text_sha256": text_sha256(summary_text),
            }
        )
        b0_rows.append(
            {
                "sample_id": sample_id,
                "held_out_source": sample["source"],
                "campaign_id": sample["campaign_id"],
                "prefix_len": sample["prefix_len"],
                "target_parent_ids": compact_json(targets),
                "target_size": sample["target_size"],
                "predicted_next_ttps": compact_json(predictions),
            }
        )
    expected = Counter({"ctid": 263, "attack_flow": 412, "stockpile": 109})
    if source_counts != expected:
        raise AssertionError(f"formal summary denominator changed: {source_counts}")

    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary_inputs.jsonl").open("x", encoding="utf-8") as handle:
        for row in summary_records:
            handle.write(compact_json(row) + "\n")
    for name, rows in (("summary_index.csv", index_rows), ("b0_rankings.csv", b0_rows)):
        with (output / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lengths = [row["summary_characters"] for row in index_rows]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "method_card_sha256": sha256(METHOD_CARD),
        "inputs": {
            "full_raw_results_sha256": sha256(RESULTS_PATH),
            "generation_manifest_sha256": sha256(GENERATION_MANIFEST),
            "future3_samples_sha256": sha256(SAMPLES_PATH),
            "vocabulary_sha256": sha256(VOCAB_PATH),
        },
        "counts": {
            "rows": len(summary_records),
            "source_rows_audit_only": dict(sorted(source_counts.items())),
            "invalid_summary_rows": 0,
            "invalid_b0_rows": 0,
            "summary_attack_id_rows": 0,
            "summary_source_or_campaign_literal_rows": 0,
        },
        "serialization": (
            "[阶段评估]\\n{stage_assessment}\\n\\n[已观察能力]\\n"
            "{observed_capabilities}\\n\\n[可能后续意图]\\n{likely_next_intents}"
        ),
        "predicted_next_ttps_in_summary_input": False,
        "summary_characters": {
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "max": max(lengths),
        },
        "outputs": {
            name: sha256(output / name)
            for name in ("summary_inputs.jsonl", "summary_index.csv", "b0_rankings.csv")
        },
    }
    (output / "summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# Future-3 LLM summary input audit",
            "",
            "- Formal S inputs: **784**.",
            "- Invalid summaries: **0**.",
            "- ATT&CK IDs in summary text: **0**.",
            "- Exact source/campaign literals in summary text: **0**.",
            "- `predicted_next_ttps` included in S text: **no**.",
            f"- Summary characters: min {min(lengths)}, mean {sum(lengths)/len(lengths):.1f}, max {max(lengths)}.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

