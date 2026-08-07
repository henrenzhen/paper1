#!/usr/bin/env python3
"""Create a unified, campaign-aware status and stratification audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "data_v4/results"
NONSEMANTIC = RESULTS / "nonsemantic_future3_lodo_v1/predictions.csv"
RAW = RESULTS / "raw_semantic_future3_lodo_v1/mean_sample_metrics.csv"
NEURAL = RESULTS / "id_neural_future3_lodo_v1/mean_sample_metrics.csv"
MB = RESULTS / "markov_beam_future3_lodo_v1/predictions.csv"
HM = RESULTS / "hybrid_markov_lstm_future3_lodo_v1/mean_sample_metrics.csv"
DEFAULT_OUTPUT = RESULTS / "future3_validation_status_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
METHODS = ("A0", "CO", "A", "K", "T", "R", "LSTM", "TR", "MB", "HM")
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
STRATA = (
    ("transition_visibility", "transition_visibility"),
    ("target_label_visibility", "target_label_visibility"),
    ("text_length", "text_length_group"),
    ("target_size", "target_size"),
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


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


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def normalize(rows: Sequence[dict[str, str]], forced_method: str | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        method = forced_method or row["method"]
        output.append(
            {
                "held_out_source": row["held_out_source"],
                "method": method,
                "sample_id": row["sample_id"],
                "campaign_id": row["campaign_id"],
                "target_size": str(row["target_size"]),
                "transition_visibility": row["transition_visibility"],
                "target_label_visibility": row["target_label_visibility"],
                "text_length_group": row["text_length_group"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    return output


def load_unified() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nonsemantic = read_csv(NONSEMANTIC)
    rows.extend(normalize([row for row in nonsemantic if row["method"] in ("A0", "CO", "A", "K", "T")]))
    rows.extend(normalize(read_csv(RAW), "R"))
    rows.extend(normalize(read_csv(NEURAL)))
    rows.extend(normalize(read_csv(MB)))
    rows.extend(normalize(read_csv(HM)))
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if set(by_method) != set(METHODS):
        raise AssertionError(f"unified methods changed: {sorted(by_method)}")
    expected_ids: set[str] | None = None
    for method in METHODS:
        values = by_method[method]
        ids = {row["sample_id"] for row in values}
        if len(values) != 784 or len(ids) != 784:
            raise AssertionError(f"{method} does not contain 784 unique rows")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise AssertionError(f"sample universe differs for {method}")
    return rows


def campaign_values(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["campaign_id"]].append(row)
    return {
        campaign: {
            metric: statistics.fmean(float(row[metric]) for row in values)
            for metric in METRICS
        }
        for campaign, values in grouped.items()
    }


def main_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_points: dict[tuple[str, str, str], float] = {}
    for source in SOURCES:
        for method in METHODS:
            values = [row for row in rows if row["held_out_source"] == source and row["method"] == method]
            campaigns = campaign_values(values)
            record = {"scope": source, "method": method, "rows": len(values), "campaigns": len(campaigns)}
            for metric in METRICS:
                point = statistics.fmean(value[metric] for value in campaigns.values())
                record[f"campaign_macro_{metric}"] = point
                source_points[(source, method, metric)] = point
            output.append(record)
    for method in METHODS:
        output.append(
            {
                "scope": "source_equal_overall",
                "method": method,
                "rows": 784,
                "campaigns": 72,
                **{
                    f"campaign_macro_{metric}": statistics.fmean(source_points[(source, method, metric)] for source in SOURCES)
                    for metric in METRICS
                },
            }
        )
    return output


def stratified_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum, field in STRATA:
        groups = sorted({str(row[field]) for row in rows})
        for group in groups:
            for source in SOURCES:
                for method in METHODS:
                    values = [
                        row for row in rows
                        if row["held_out_source"] == source
                        and row["method"] == method
                        and str(row[field]) == group
                    ]
                    campaigns = campaign_values(values)
                    if not values:
                        continue
                    output.append(
                        {
                            "stratum": stratum,
                            "group": group,
                            "scope": source,
                            "method": method,
                            "rows": len(values),
                            "campaigns": len(campaigns),
                            "inferentially_eligible": int(len(values) >= 20 and len(campaigns) >= 5),
                            **{
                                f"campaign_macro_{metric}": statistics.fmean(value[metric] for value in campaigns.values())
                                for metric in METRICS
                            },
                        }
                    )
    return output


def stratified_bootstrap(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum, field in STRATA:
        groups = sorted({str(row[field]) for row in rows})
        for group in groups:
            subset = [row for row in rows if str(row[field]) == group]
            lookup: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
            campaign_ids: dict[str, list[str]] = {}
            valid = True
            for source in SOURCES:
                for method in METHODS:
                    values = [row for row in subset if row["held_out_source"] == source and row["method"] == method]
                    lookup[(source, method)] = campaign_values(values)
                campaign_ids[source] = sorted(lookup[(source, "A")])
                if not campaign_ids[source]:
                    valid = False
            if not valid:
                continue
            rng = random.Random(BOOTSTRAP_SEED)
            replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for _ in range(BOOTSTRAP_REPLICATES):
                draws = {source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]] for source in SOURCES}
                for method in METHODS:
                    if method == "A":
                        continue
                    for metric in METRICS:
                        source_values: list[float] = []
                        for source in SOURCES:
                            delta = statistics.fmean(
                                lookup[(source, method)][campaign][metric] - lookup[(source, "A")][campaign][metric]
                                for campaign in draws[source]
                            )
                            source_values.append(delta)
                            replicates[(source, method, metric)].append(delta)
                        replicates[("source_equal_overall", method, metric)].append(statistics.fmean(source_values))
            for scope in (*SOURCES, "source_equal_overall"):
                for method in METHODS:
                    if method == "A":
                        continue
                    for metric in METRICS:
                        if scope == "source_equal_overall":
                            point = statistics.fmean(
                                statistics.fmean(
                                    lookup[(source, method)][campaign][metric] - lookup[(source, "A")][campaign][metric]
                                    for campaign in campaign_ids[source]
                                )
                                for source in SOURCES
                            )
                            row_count = sum(len([row for row in subset if row["held_out_source"] == source and row["method"] == method]) for source in SOURCES)
                            campaign_count = sum(len(campaign_ids[source]) for source in SOURCES)
                        else:
                            point = statistics.fmean(
                                lookup[(scope, method)][campaign][metric] - lookup[(scope, "A")][campaign][metric]
                                for campaign in campaign_ids[scope]
                            )
                            row_count = len([row for row in subset if row["held_out_source"] == scope and row["method"] == method])
                            campaign_count = len(campaign_ids[scope])
                        values = replicates[(scope, method, metric)]
                        output.append(
                            {
                                "stratum": stratum,
                                "group": group,
                                "scope": scope,
                                "comparison": f"{method}-A",
                                "metric": metric,
                                "rows": row_count,
                                "campaigns": campaign_count,
                                "inferentially_eligible": int(row_count >= 20 and campaign_count >= 5),
                                "point_estimate": point,
                                "ci95_low": percentile(values, 0.025),
                                "ci95_high": percentile(values, 0.975),
                                "replicates": BOOTSTRAP_REPLICATES,
                                "seed": BOOTSTRAP_SEED,
                            }
                        )
    return output


def report_markdown(main: Sequence[dict[str, Any]], differences: Sequence[dict[str, Any]]) -> str:
    overall = {row["method"]: row for row in main if row["scope"] == "source_equal_overall"}
    fold = {(row["scope"], row["method"]): row for row in main if row["scope"] in SOURCES}
    unseen = {
        row["comparison"]: row
        for row in differences
        if row["stratum"] == "transition_visibility"
        and row["group"] == "all_unseen"
        and row["scope"] == "source_equal_overall"
        and row["metric"] == "ndcg5"
    }
    lines = [
        "# Future-3 cross-source validation status v1",
        "",
        "All values use the same 784-row main set, campaign-macro aggregation, and source-equal outer summary.",
        "",
        "## Campaign-macro NDCG@5",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {fold[('ctid', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{fold[('attack_flow', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{fold[('stockpile', method)]['campaign_macro_ndcg5']:.4f} | "
            f"{overall[method]['campaign_macro_ndcg5']:.4f} |"
        )
    lines.extend([
        "",
        "## All-unseen transition stratum: source-equal NDCG@5 delta vs A",
        "",
        "| Comparison | Delta | 95% campaign-bootstrap CI |",
        "|---|---:|---:|",
    ])
    for method in ("T", "R", "LSTM", "TR", "MB", "HM"):
        row = unseen[f"{method}-A"]
        lines.append(f"| {method}-A | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    lines.extend([
        "",
        "Current evidence rejects stable gains from raw-description probing, ID-only neural capacity, Markov beam restriction, and the SECRYPT-adapted hybrid. It does not evaluate the LLM-normalized S branch.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ["main_metrics.csv", "stratified_metrics.csv", "paired_stratified_bootstrap.csv", "report.md", "status_manifest.json"]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite validation status: {existing}")
    rows = load_unified()
    main_rows = main_metrics(rows)
    strata = stratified_metrics(rows)
    differences = stratified_bootstrap(rows)
    write_csv(output / "main_metrics.csv", main_rows, ["scope", "method", "rows", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]])
    write_csv(output / "stratified_metrics.csv", strata, ["stratum", "group", "scope", "method", "rows", "campaigns", "inferentially_eligible", *[f"campaign_macro_{metric}" for metric in METRICS]])
    write_csv(output / "paired_stratified_bootstrap.csv", differences, ["stratum", "group", "scope", "comparison", "metric", "rows", "campaigns", "inferentially_eligible", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"])
    report = report_markdown(main_rows, differences)
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "inputs_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in (NONSEMANTIC, RAW, NEURAL, MB, HM)},
        "methods": list(METHODS),
        "sample_rows_per_method": 784,
        "campaigns": 72,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "llm_normalized_s_status": "NOT_RUN",
        "outputs_sha256": {name: sha256(output / name) for name in managed if name != "status_manifest.json" and (output / name).exists()},
    }
    write_json(output / "status_manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
