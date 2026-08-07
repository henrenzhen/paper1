#!/usr/bin/env python3
"""Audit whether one CTID campaign drives the direct-LLM result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTIC_RESULTS = PROJECT_ROOT / "data_v4/results/tactic_llm_direct_rank_exploratory_v1/campaign_results.csv"
SUMMARY_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_tactic_llm_summary_future3_lodo_v1/campaign_results.csv"
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/ctid_campaign_influence_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/audits/ctid_campaign_influence_v1"
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
COMPARISONS = (
    ("B0-A0", "B0", "A0"),
    ("B0-K", "B0", "K"),
    ("B0-T", "B0", "T"),
    ("B0-HM", "B0", "HM"),
    ("B0-HM+S", "B0", "HM+S"),
    ("T+B0-B0", "T+B0", "B0"),
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def load_campaigns() -> tuple[list[str], dict[tuple[str, str], dict[str, float | int]]]:
    wanted_tactic = {"A0", "K", "T", "B0", "T+B0"}
    wanted_summary = {"HM", "HM+S"}
    rows = [
        row
        for row in read_csv(TACTIC_RESULTS)
        if row["held_out_source"] == "ctid" and row["method"] in wanted_tactic
    ]
    rows.extend(
        row
        for row in read_csv(SUMMARY_RESULTS)
        if row["held_out_source"] == "ctid" and row["method"] in wanted_summary
    )
    values: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in rows:
        key = (row["method"], row["campaign_id"])
        if key in values:
            raise AssertionError(f"duplicate campaign-method row: {key}")
        values[key] = {
            "rows": int(row["rows"]),
            **{metric: float(row[metric]) for metric in METRICS},
        }
    campaigns = sorted({campaign for method, campaign in values if method == "B0"})
    if len(campaigns) != 10:
        raise AssertionError(f"expected 10 CTID campaigns, found {len(campaigns)}")
    methods = {method for _, method, _ in COMPARISONS} | {method for _, _, method in COMPARISONS}
    for method in methods:
        method_campaigns = sorted(campaign for candidate, campaign in values if candidate == method)
        if method_campaigns != campaigns:
            raise AssertionError(f"campaign mismatch for {method}")
    for campaign in campaigns:
        row_counts = {int(values[(method, campaign)]["rows"]) for method in methods}
        if len(row_counts) != 1:
            raise AssertionError(f"row-count mismatch for {campaign}: {row_counts}")
    return campaigns, values


def calculate(
    campaigns: list[str], values: dict[tuple[str, str], dict[str, float | int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    campaign_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for comparison, left, right in COMPARISONS:
        for metric in METRICS:
            differences = {
                campaign: float(values[(left, campaign)][metric])
                - float(values[(right, campaign)][metric])
                for campaign in campaigns
            }
            full = statistics.fmean(differences.values())
            for campaign in campaigns:
                campaign_rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "campaign_id": campaign,
                        "rows": values[(left, campaign)]["rows"],
                        "left_value": values[(left, campaign)][metric],
                        "right_value": values[(right, campaign)][metric],
                        "difference": differences[campaign],
                    }
                )
                kept = [value for candidate, value in differences.items() if candidate != campaign]
                loco_rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "omitted_campaign": campaign,
                        "campaigns_remaining": len(kept),
                        "full_effect": full,
                        "leave_one_out_effect": statistics.fmean(kept),
                    }
                )
            current = [
                row
                for row in loco_rows
                if row["comparison"] == comparison and row["metric"] == metric
            ]
            minimum = min(current, key=lambda row: float(row["leave_one_out_effect"]))
            maximum = max(current, key=lambda row: float(row["leave_one_out_effect"]))
            positive = sum(value > 0 for value in differences.values())
            zero = sum(value == 0 for value in differences.values())
            negative = sum(value < 0 for value in differences.values())
            if full > 0:
                stability = "single-campaign sign-stable" if float(minimum["leave_one_out_effect"]) > 0 else "single-campaign fragile"
            elif full < 0:
                stability = "full effect non-positive"
            else:
                stability = "full effect zero"
            summary_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "campaigns": len(campaigns),
                    "full_effect": full,
                    "loco_min": minimum["leave_one_out_effect"],
                    "loco_min_omitted": minimum["omitted_campaign"],
                    "loco_max": maximum["leave_one_out_effect"],
                    "loco_max_omitted": maximum["omitted_campaign"],
                    "campaign_positive": positive,
                    "campaign_zero": zero,
                    "campaign_negative": negative,
                    "stability": stability,
                }
            )
    return campaign_rows, loco_rows, summary_rows


def make_report(summary_rows: list[dict[str, Any]], campaign_rows: list[dict[str, Any]]) -> str:
    primary = [row for row in summary_rows if row["metric"] == "ndcg5"]
    lines = [
        "# CTID campaign influence audit v1",
        "",
        "Post-result sensitivity analysis; not a new confirmatory test.",
        "",
        "## Leave-one-campaign-out NDCG@5",
        "",
        "| Comparison | Full delta | LOCO min | Omitted at min | LOCO max | Positive / zero / negative campaigns | Stability |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in primary:
        lines.append(
            f"| {row['comparison']} | {float(row['full_effect']):+.4f} | "
            f"{float(row['loco_min']):+.4f} | {row['loco_min_omitted']} | "
            f"{float(row['loco_max']):+.4f} | "
            f"{row['campaign_positive']} / {row['campaign_zero']} / {row['campaign_negative']} | "
            f"{row['stability']} |"
        )
    lines.extend(
        [
            "",
            "## Per-campaign NDCG@5 differences",
            "",
            "| Campaign | B0-A0 | B0-K | B0-T | B0-HM | B0-HM+S | T+B0-B0 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lookup = {
        (row["campaign_id"], row["comparison"]): float(row["difference"])
        for row in campaign_rows
        if row["metric"] == "ndcg5"
    }
    campaigns = sorted({campaign for campaign, _ in lookup})
    for campaign in campaigns:
        lines.append(
            f"| {campaign} | "
            + " | ".join(f"{lookup[(campaign, comparison)]:+.4f}" for comparison, _, _ in COMPARISONS)
            + " |"
        )
    lines.extend(
        [
            "",
            "A sign-stable label means that no single CTID campaign can reverse the positive point estimate. It does not imply statistical significance or generalization to a new source.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    campaigns, values = load_campaigns()
    campaign_rows, loco_rows, summary_rows = calculate(campaigns, values)
    report = make_report(summary_rows, campaign_rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(
        output / "campaign_differences.csv",
        campaign_rows,
        ("comparison", "metric", "campaign_id", "rows", "left_value", "right_value", "difference"),
    )
    write_csv(
        output / "leave_one_campaign_out.csv",
        loco_rows,
        ("comparison", "metric", "omitted_campaign", "campaigns_remaining", "full_effect", "leave_one_out_effect"),
    )
    write_csv(
        output / "summary.csv",
        summary_rows,
        (
            "comparison",
            "metric",
            "campaigns",
            "full_effect",
            "loco_min",
            "loco_min_omitted",
            "loco_max",
            "loco_max_omitted",
            "campaign_positive",
            "campaign_zero",
            "campaign_negative",
            "stability",
        ),
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "stdout.log").write_text(report, encoding="utf-8")
    output_names = (
        "campaign_differences.csv",
        "leave_one_campaign_out.csv",
        "summary.csv",
        "report.md",
        "stdout.log",
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post-result sensitivity audit",
        "scope": "CTID; 10 campaigns; campaign-macro leave-one-campaign-out",
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "protocol": {
            "path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(PROTOCOL),
        },
        "inputs": {
            TACTIC_RESULTS.relative_to(PROJECT_ROOT).as_posix(): sha256(TACTIC_RESULTS),
            SUMMARY_RESULTS.relative_to(PROJECT_ROOT).as_posix(): sha256(SUMMARY_RESULTS),
        },
        "comparisons": [comparison for comparison, _, _ in COMPARISONS],
        "primary_metric": "ndcg5",
        "campaign_count": len(campaigns),
        "outputs_sha256": {name: sha256(output / name) for name in output_names},
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
