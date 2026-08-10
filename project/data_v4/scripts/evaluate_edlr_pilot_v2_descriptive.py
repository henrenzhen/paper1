#!/usr/bin/env python3
"""Compute the frozen development-only EDLR pilot diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2_evaluation.md"
RUN = PROJECT_ROOT / "data_v4/external_reasoning/edlr_pilot_v2_1_repair/runs/20260810T031704Z_43864251"
MERGED = RUN / "merged_pilot_results.csv"
GATES = RUN / "merged_mechanical_gate_report.json"
SAMPLES = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/edlr_pilot_v2_development_descriptive"

EXPECTED_HASHES = {
    MERGED: "7e7e658cd2a59ad5a8ee0f1def52aa9f1f5b7357ad8a5ab73d802665643cbf84",
    GATES: "a20b921f2d5ec19d8b6a34b7007f86de04c9c7d62059f72364c23fe64a55b7ef",
    SAMPLES: "af7d9a6b6358939697730c5de206c51f1b63f7c16954d5dce5a0a4006bbca724",
}
SOURCES = ("ctid", "attack_flow", "stockpile")
METHODS = ("B0", "EA_TOP5", "UNION_LLM", "EDLR", "EDLR_SHUFFLE")
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
CONTRASTS = (
    ("EA_TOP5_minus_B0", "EA_TOP5", "B0"),
    ("UNION_LLM_minus_B0", "UNION_LLM", "B0"),
    ("EDLR_minus_B0", "EDLR", "B0"),
    ("EDLR_minus_UNION_LLM", "EDLR", "UNION_LLM"),
    ("EDLR_minus_EDLR_SHUFFLE", "EDLR", "EDLR_SHUFFLE"),
)


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
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sample_metrics(ranking: Sequence[str], targets: Sequence[str]) -> dict[str, float]:
    truth = set(targets)
    hits = [int(label in truth) for label in ranking[:5]]
    found = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(5, len(truth))))
    return {
        "ndcg5": dcg / ideal,
        "hit5": float(found > 0),
        "precision5": found / 5,
        "recall5": found / len(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = (
        "per_sample_metrics.csv",
        "campaign_metrics.csv",
        "source_metrics.csv",
        "source_equal_summary.csv",
        "contrasts.csv",
        "report.md",
        "evaluation_manifest.json",
    )
    if any((output / name).exists() for name in managed):
        raise FileExistsError("refusing to overwrite EDLR pilot evaluation")
    changed = {str(path): {"expected": expected, "actual": sha256(path)} for path, expected in EXPECTED_HASHES.items() if sha256(path) != expected}
    if changed:
        raise AssertionError(f"frozen evaluation input changed: {changed}")
    gate = json.loads(GATES.read_text(encoding="utf-8"))
    if not gate["mechanical_gates"]["all_mechanical_gates_passed"]:
        raise AssertionError("targets may not be opened before all merged mechanical gates pass")

    samples = {
        row["sample_id"]: row
        for row in read_csv(SAMPLES)
        if row["is_development"] == "1"
    }
    arms = read_csv(MERGED)
    if len(samples) != 30 or len(arms) != 120 or any(row["generation_status"] != "ok" for row in arms):
        raise AssertionError("frozen evaluation denominator/status changed")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in arms:
        by_sample[row["sample_id"]].append(row)
    if set(by_sample) != set(samples) or any(len(rows) != 4 for rows in by_sample.values()):
        raise AssertionError("merged method/sample alignment failed")

    per_sample: list[dict[str, Any]] = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        rows = by_sample[sample_id]
        arm_rankings = {row["arm"]: list(json.loads(row["reranked_next_ttps"])) for row in rows}
        if set(arm_rankings) != set(METHODS) - {"B0"}:
            raise AssertionError(f"arm set changed: {sample_id}")
        b0_values = {row["b0_top5_audit_only"] for row in rows}
        if len(b0_values) != 1:
            raise AssertionError(f"B0 differs across arms: {sample_id}")
        rankings = {"B0": list(json.loads(next(iter(b0_values)))), **arm_rankings}
        targets = list(json.loads(sample["target_parent_ids"]))
        for method in METHODS:
            values = sample_metrics(rankings[method], targets)
            per_sample.append(
                {
                    "status": "development_descriptive_only",
                    "sample_id": sample_id,
                    "source": sample["source"],
                    "campaign_id": sample["campaign_id"],
                    "development_slot": sample["development_slot"],
                    "method": method,
                    "targets": compact(targets),
                    "ranking": compact(rankings[method]),
                    **values,
                }
            )
        base = next(row for row in per_sample if row["sample_id"] == sample_id and row["method"] == "B0")
        ea = next(row for row in per_sample if row["sample_id"] == sample_id and row["method"] == "EA_TOP5")
        if set(rankings["B0"]) != set(rankings["EA_TOP5"]):
            raise AssertionError(f"EA changed the B0 candidate set: {sample_id}")
        if any(abs(float(base[metric]) - float(ea[metric])) > 1e-12 for metric in ("hit5", "precision5", "recall5")):
            raise AssertionError(f"EA non-ranking metric changed: {sample_id}")

    campaign_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        campaign_groups[(row["source"], row["campaign_id"], row["method"])].append(row)
    campaigns: list[dict[str, Any]] = []
    for (source, campaign, method), rows in sorted(campaign_groups.items()):
        campaigns.append(
            {
                "status": "development_descriptive_only",
                "source": source,
                "campaign_id": campaign,
                "method": method,
                "rows": len(rows),
                **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
            }
        )
    source_rows: list[dict[str, Any]] = []
    for source in SOURCES:
        for method in METHODS:
            rows = [row for row in campaigns if row["source"] == source and row["method"] == method]
            source_rows.append(
                {
                    "status": "development_descriptive_only",
                    "source": source,
                    "method": method,
                    "campaigns": len(rows),
                    "prefix_rows": sum(int(row["rows"]) for row in rows),
                    **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
                }
            )
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in source_rows if row["method"] == method]
        summary.append(
            {
                "status": "development_descriptive_only",
                "method": method,
                "source_folds": len(rows),
                **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
            }
        )
    source_lookup = {(row["source"], row["method"]): row for row in source_rows}
    overall_lookup = {row["method"]: row for row in summary}
    contrasts: list[dict[str, Any]] = []
    for name, left, right in CONTRASTS:
        for scope in (*SOURCES, "source_equal"):
            lrow = overall_lookup[left] if scope == "source_equal" else source_lookup[(scope, left)]
            rrow = overall_lookup[right] if scope == "source_equal" else source_lookup[(scope, right)]
            contrasts.append(
                {
                    "status": "development_descriptive_only",
                    "contrast": name,
                    "left_method": left,
                    "right_method": right,
                    "scope": scope,
                    **{metric: float(lrow[metric]) - float(rrow[metric]) for metric in METRICS},
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    common_metrics = (*METRICS,)
    write_csv(output / "per_sample_metrics.csv", per_sample, ("status", "sample_id", "source", "campaign_id", "development_slot", "method", "targets", "ranking", *common_metrics))
    write_csv(output / "campaign_metrics.csv", campaigns, ("status", "source", "campaign_id", "method", "rows", *common_metrics))
    write_csv(output / "source_metrics.csv", source_rows, ("status", "source", "method", "campaigns", "prefix_rows", *common_metrics))
    write_csv(output / "source_equal_summary.csv", summary, ("status", "method", "source_folds", *common_metrics))
    write_csv(output / "contrasts.csv", contrasts, ("status", "contrast", "left_method", "right_method", "scope", *common_metrics))

    ndcg_sources = {method: {source: source_lookup[(source, method)]["ndcg5"] for source in SOURCES} for method in METHODS}
    report = [
        "# EDLR pilot v2 development-only descriptive results",
        "",
        "> These 30 rows are prompt-development data. Numbers below are descriptive and cannot establish effectiveness or significance.",
        "",
        "## NDCG@5",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        report.append(
            f"| {method} | {ndcg_sources[method]['ctid']:.4f} | {ndcg_sources[method]['attack_flow']:.4f} | "
            f"{ndcg_sources[method]['stockpile']:.4f} | {overall_lookup[method]['ndcg5']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Frozen NDCG@5 contrasts",
            "",
            "| Contrast | CTID | Attack Flow | Stockpile | Source-equal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    contrast_lookup = {(row["contrast"], row["scope"]): row for row in contrasts}
    for name, _left, _right in CONTRASTS:
        report.append(
            f"| {name} | {contrast_lookup[(name, 'ctid')]['ndcg5']:+.4f} | "
            f"{contrast_lookup[(name, 'attack_flow')]['ndcg5']:+.4f} | "
            f"{contrast_lookup[(name, 'stockpile')]['ndcg5']:+.4f} | "
            f"{contrast_lookup[(name, 'source_equal')]['ndcg5']:+.4f} |"
        )
    report.extend(
        [
            "",
            "Mechanism attribution would require EDLR to exceed both UNION_LLM and EDLR_SHUFFLE; this pilot can only guide whether a separately authorized exploratory full run is worth considering.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_descriptive_only",
        "targets_first_opened_after_merged_mechanical_gate_passed": True,
        "network_calls": 0,
        "api_cost": 0,
        "denominators": {
            "samples": 30,
            "methods": len(METHODS),
            "per_sample_rows": len(per_sample),
            "source_sample_counts": {source: sum(sample["source"] == source for sample in samples.values()) for source in SOURCES},
            "source_campaign_counts": {source: len({sample["campaign_id"] for sample in samples.values() if sample["source"] == source}) for source in SOURCES},
        },
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "inputs": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in (MERGED, GATES, SAMPLES)},
        "outputs": {name: sha256(output / name) for name in managed if name != "evaluation_manifest.json"},
    }
    write_json(output / "evaluation_manifest.json", manifest)
    print("\n".join(report))


if __name__ == "__main__":
    main()
