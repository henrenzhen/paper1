#!/usr/bin/env python3
"""Run post-result exploratory T + direct DeepSeek Top-5 rank fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRECT_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_hm_llm_direct_rank_exploratory.py"
NONSEMANTIC_RESULTS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1"
SUMMARY_RESULTS = PROJECT_ROOT / "data_v4/results/hybrid_llm_summary_future3_lodo_v1"
HM_B0_RESULTS = PROJECT_ROOT / "data_v4/results/hm_llm_direct_rank_exploratory_v1"
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/tactic_llm_direct_rank_exploratory_future3_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/results/tactic_llm_direct_rank_exploratory_v1"

SOURCES = ("ctid", "attack_flow", "stockpile")
GRID = tuple(index / 10 for index in range(11))
METRICS = ("ndcg5", "hit5", "precision5", "recall5")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIRECT = import_module("direct_rank_for_tactic_fusion", DIRECT_SCRIPT)
BASE = DIRECT.SUMMARY.BASE


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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def t_components(
    train: Sequence[dict[str, Any]],
    evaluation: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
) -> tuple[list[list[float]], list[list[float]]]:
    a_model = BASE.RelevanceModel(
        train,
        len(labels),
        lambda row: {label_index[label] for label in row["targets"]},
    )
    tactic_model = BASE.RelevanceModel(
        train,
        14,
        lambda row: BASE.tactic_target_indices(row, tactic_by_label),
    )
    a_scores: list[list[float]] = []
    tactic_scores: list[list[float]] = []
    for row in evaluation:
        a_scores.append(a_model.score(row["history"]))
        tactic_scores.append(
            BASE.tactic_candidate_scores(
                tactic_model.score(row["history"]), candidate_tactics
            )
        )
    return a_scores, tactic_scores


def build_t_score(
    a_values: Sequence[float], tactic_values: Sequence[float], tactic_weight: float
) -> list[float]:
    return BASE.fused_scores(a_values, tactic_values, tactic_weight)


def fuse_t_b0(
    t_values: Sequence[float], b0_values: Sequence[float], b0_weight: float
) -> list[float]:
    t_z = BASE.standardize([float(value) for value in t_values])
    b0_z = BASE.standardize([float(value) for value in b0_values])
    return [
        (1 - b0_weight) * left + b0_weight * right
        for left, right in zip(t_z, b0_z)
    ]


def campaign_macro_ndcg(
    rows: Sequence[dict[str, Any]],
    a_scores: Sequence[Sequence[float]],
    tactic_scores: Sequence[Sequence[float]],
    b0_scores: dict[str, list[float]],
    tactic_weight: float,
    b0_weight: float,
    labels: Sequence[str],
) -> float:
    records: list[dict[str, Any]] = []
    for row, a_values, tactic_values in zip(rows, a_scores, tactic_scores):
        t_values = build_t_score(a_values, tactic_values, tactic_weight)
        ranked, _ = BASE.ranking(
            fuse_t_b0(t_values, b0_scores[row["sample_id"]], b0_weight), labels
        )
        records.append({**row, **BASE.sample_metrics(ranked[:5], row["targets"])})
    return BASE.campaign_macro(records, "ndcg5")


def select_weights(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    b0_scores: dict[str, list[float]],
) -> tuple[float, float, list[dict[str, Any]]]:
    training_sources = tuple(source for source in SOURCES if source != held_out)
    details: list[dict[str, Any]] = []
    for validation_source in training_sources:
        train = [
            row
            for row in rows
            if row["source"] in training_sources and row["source"] != validation_source
        ]
        validation = [row for row in rows if row["source"] == validation_source]
        a_scores, tactic_scores = t_components(
            train,
            validation,
            labels,
            label_index,
            candidate_tactics,
            tactic_by_label,
        )
        for tactic_weight in GRID:
            for b0_weight in GRID:
                details.append(
                    {
                        "held_out_source": held_out,
                        "inner_validation_source": validation_source,
                        "tactic_lambda": tactic_weight,
                        "b0_lambda": b0_weight,
                        "campaign_macro_ndcg5": campaign_macro_ndcg(
                            validation,
                            a_scores,
                            tactic_scores,
                            b0_scores,
                            tactic_weight,
                            b0_weight,
                            labels,
                        ),
                        "training_rows": len(train),
                        "validation_rows": len(validation),
                    }
                )
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in details:
        grouped[(row["tactic_lambda"], row["b0_lambda"])].append(
            row["campaign_macro_ndcg5"]
        )
    means = {key: statistics.fmean(values) for key, values in grouped.items()}
    best = max(means.values())
    candidates = [key for key, value in means.items() if abs(value - best) <= 1e-12]
    selected_tactic, selected_b0 = min(candidates, key=lambda key: (key[1], key[0]))
    for row in details:
        key = (row["tactic_lambda"], row["b0_lambda"])
        row["two_source_mean_ndcg5"] = means[key]
        row["selected"] = int(key == (selected_tactic, selected_b0))
    return selected_tactic, selected_b0, details


def frozen_inputs() -> tuple[
    dict[str, dict[str, str]], dict[str, float], dict[str, dict[str, str]]
]:
    t_rows = {
        row["sample_id"]: row
        for row in read_csv(NONSEMANTIC_RESULTS / "predictions.csv")
        if row["method"] == "T"
    }
    manifest = json.loads(
        (NONSEMANTIC_RESULTS / "results_manifest.json").read_text(encoding="utf-8")
    )
    frozen_lambdas = {
        source: float(value)
        for source, value in manifest["parameters"]["chosen_lambda_by_held_out_source"].items()
    }
    a_rows = {
        row["sample_id"]: row
        for row in read_csv(NONSEMANTIC_RESULTS / "predictions.csv")
        if row["method"] == "A"
    }
    if len(t_rows) != 784 or len(a_rows) != 784 or set(frozen_lambdas) != set(SOURCES):
        raise AssertionError("frozen T inputs changed")
    return t_rows, frozen_lambdas, a_rows


def outer_predictions(
    held_out: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    b0_scores: dict[str, list[float]],
    b0_rankings: dict[str, list[str]],
    tactic_weight: float,
    b0_weight: float,
    frozen_t: dict[str, dict[str, str]],
    frozen_tactic_lambdas: dict[str, float],
    frozen_a: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    train = [row for row in rows if row["source"] != held_out]
    test = [row for row in rows if row["source"] == held_out]
    a_scores, tactic_scores = t_components(
        train,
        test,
        labels,
        label_index,
        candidate_tactics,
        tactic_by_label,
    )
    output: list[dict[str, Any]] = []
    for row, a_values, tactic_values in zip(test, a_scores, tactic_scores):
        sample_id = row["sample_id"]
        frozen_t_values = build_t_score(
            a_values, tactic_values, frozen_tactic_lambdas[held_out]
        )
        frozen_ranked, _ = BASE.ranking(frozen_t_values, labels)
        if frozen_ranked[:20] != json.loads(frozen_t[sample_id]["top20_ids"]):
            raise AssertionError(f"frozen T reproduction failed: {sample_id}")
        selected_t_values = build_t_score(a_values, tactic_values, tactic_weight)
        b0_ranked, _ = BASE.ranking(
            fuse_t_b0(selected_t_values, b0_scores[sample_id], 1.0), labels
        )
        if b0_ranked[:5] != b0_rankings[sample_id]:
            raise AssertionError(f"B0 reproduction failed: {sample_id}")
        ranked, scores = BASE.ranking(
            fuse_t_b0(selected_t_values, b0_scores[sample_id], b0_weight), labels
        )
        reference = frozen_a[sample_id]
        output.append(
            {
                "held_out_source": held_out,
                "method": "T+B0",
                "selected_tactic_lambda": tactic_weight,
                "selected_b0_lambda": b0_weight,
                "sample_id": sample_id,
                "campaign_id": row["campaign_id"],
                "prefix_len": row["prefix_len"],
                "target_parent_ids": compact_json(row["targets"]),
                "target_size": row["target_size"],
                "transition_visibility": reference["transition_visibility"],
                "target_label_visibility": reference["target_label_visibility"],
                "text_length_group": reference["text_length_group"],
                "top20_ids": compact_json(ranked[:20]),
                "top20_scores": compact_json(
                    [round(float(value), 12) for value in scores[:20]]
                ),
                **BASE.sample_metrics(ranked[:5], row["targets"]),
            }
        )
    return output


def reference_rows(fused: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {row["sample_id"]: row for row in fused}
    output: list[dict[str, Any]] = []
    for row in read_csv(NONSEMANTIC_RESULTS / "predictions.csv"):
        if row["method"] not in ("A0", "A", "K", "T"):
            continue
        output.append(
            {
                **{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                "method": row["method"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    for row in read_csv(SUMMARY_RESULTS / "b0_sample_metrics.csv"):
        output.append(
            {
                **{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                "method": "B0",
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    for row in read_csv(HM_B0_RESULTS / "mean_sample_metrics.csv"):
        output.append(
            {
                **{key: metadata[row["sample_id"]][key] for key in ("held_out_source", "sample_id", "campaign_id", "prefix_len", "target_size", "transition_visibility", "target_label_visibility", "text_length_group")},
                "method": "HM+B0",
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    counts: dict[str, int] = defaultdict(int)
    for row in output:
        counts[row["method"]] += 1
    if set(counts) != {"A0", "A", "K", "T", "B0", "HM+B0"} or any(
        value != 784 for value in counts.values()
    ):
        raise AssertionError(f"reference count gate failed: {dict(counts)}")
    return output


def campaign_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    return [
        {
            "held_out_source": source,
            "method": method,
            "campaign_id": campaign,
            "rows": len(values),
            **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
        }
        for (source, method, campaign), values in sorted(grouped.items())
    ]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = ("A0", "A", "K", "T", "B0", "HM+B0", "T+B0")
    output: list[dict[str, Any]] = []
    for method in methods:
        for source in SOURCES:
            values = [
                row
                for row in campaigns
                if row["method"] == method and row["held_out_source"] == source
            ]
            output.append(
                {
                    "held_out_source": source,
                    "method": method,
                    "campaigns": len(values),
                    **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in METRICS},
                }
            )
    return output


def bootstrap(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["held_out_source"], row["method"], row["campaign_id"]): row
        for row in campaigns
    }
    campaign_ids = {
        source: sorted(
            row["campaign_id"]
            for row in campaigns
            if row["held_out_source"] == source and row["method"] == "B0"
        )
        for source in SOURCES
    }
    comparisons = (
        ("T+B0", "B0"),
        ("T+B0", "T"),
        ("T+B0", "HM+B0"),
        ("T+B0", "K"),
        ("B0", "T"),
        ("B0", "K"),
        ("B0", "A0"),
    )
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {
            source: [rng.choice(campaign_ids[source]) for _ in campaign_ids[source]]
            for source in SOURCES
        }
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                source_values = []
                for source in SOURCES:
                    delta = statistics.fmean(
                        float(lookup[(source, left, campaign)][metric])
                        - float(lookup[(source, right, campaign)][metric])
                        for campaign in draws[source]
                    )
                    source_values.append(delta)
                    replicates[(source, comparison, metric)].append(delta)
                replicates[("source_equal_overall", comparison, metric)].append(
                    statistics.fmean(source_values)
                )
    output = []
    for scope in (*SOURCES, "source_equal_overall"):
        for left, right in comparisons:
            comparison = f"{left}-{right}"
            for metric in METRICS:
                if scope == "source_equal_overall":
                    point = statistics.fmean(
                        statistics.fmean(
                            float(lookup[(source, left, campaign)][metric])
                            - float(lookup[(source, right, campaign)][metric])
                            for campaign in campaign_ids[source]
                        )
                        for source in SOURCES
                    )
                else:
                    point = statistics.fmean(
                        float(lookup[(scope, left, campaign)][metric])
                        - float(lookup[(scope, right, campaign)][metric])
                        for campaign in campaign_ids[scope]
                    )
                values = replicates[(scope, comparison, metric)]
                output.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "metric": metric,
                        "point_estimate": point,
                        "ci95_low": BASE.percentile(values, 0.025),
                        "ci95_high": BASE.percentile(values, 0.975),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return output


def stratified(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "transition_visibility": lambda row: row["transition_visibility"],
        "target_label_visibility": lambda row: row["target_label_visibility"],
        "text_length": lambda row: row["text_length_group"],
        "target_size": lambda row: str(row["target_size"]),
    }
    output = []
    for source in SOURCES:
        for method in ("T+B0", "B0", "T"):
            method_rows = [
                row
                for row in rows
                if row["held_out_source"] == source and row["method"] == method
            ]
            for name, getter in definitions.items():
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in method_rows:
                    groups[getter(row)].append(row)
                for group, values in sorted(groups.items()):
                    by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in values:
                        by_campaign[row["campaign_id"]].append(row)
                    output.append(
                        {
                            "held_out_source": source,
                            "method": method,
                            "stratum": name,
                            "group": group,
                            "rows": len(values),
                            "campaigns": len(by_campaign),
                            "inferentially_eligible": int(len(values) >= 20 and len(by_campaign) >= 5),
                            **{f"campaign_macro_{metric}": statistics.fmean(statistics.fmean(float(row[metric]) for row in items) for items in by_campaign.values()) for metric in METRICS},
                        }
                    )
    return output


def report(
    folds: Sequence[dict[str, Any]],
    selections: dict[str, tuple[float, float]],
    differences: Sequence[dict[str, Any]],
) -> str:
    lookup = {(row["method"], row["held_out_source"]): row for row in folds}
    diff = {(row["scope"], row["comparison"], row["metric"]): row for row in differences}
    lines = [
        "# Exploratory T + direct LLM rank fusion",
        "",
        "**Post-result exploratory analysis; prospective confirmation is required.**",
        "",
        "| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("A0", "A", "K", "T", "B0", "HM+B0", "T+B0"):
        values = [float(lookup[(method, source)]["campaign_macro_ndcg5"]) for source in SOURCES]
        lines.append(
            f"| {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | **{statistics.fmean(values):.4f}** |"
        )
    lines.extend(
        [
            "",
            "Selected weights: "
            + ", ".join(
                f"{source}=(T {selections[source][0]:.1f}, B0 {selections[source][1]:.1f})"
                for source in SOURCES
            ),
            "",
            "| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for comparison in ("T+B0-B0", "T+B0-T", "T+B0-HM+B0", "B0-T", "B0-K"):
        row = diff[("source_equal_overall", comparison, "ndcg5")]
        lines.append(
            f"| {comparison} | {row['point_estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Frozen T Top-20 and exact B0 Top-5 reproduction gates passed for all 784 rows.",
            "This result does not alter the negative preregistered HM+S conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = (
        "inner_selection.csv",
        "selected_weights.csv",
        "predictions.csv",
        "campaign_results.csv",
        "fold_results.csv",
        "paired_bootstrap_differences.csv",
        "stratified_results.csv",
        "report.md",
        "stdout.log",
        "results_manifest.json",
    )
    if any((output / name).exists() for name in managed):
        raise FileExistsError("refusing to overwrite T+B0 exploratory results")
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    candidate_tactics, tactic_by_label = BASE.parse_tactics()
    b0_scores, b0_rankings = DIRECT.load_b0(labels)
    frozen_t, frozen_tactic_lambdas, frozen_a = frozen_inputs()
    selections: dict[str, tuple[float, float]] = {}
    inner: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for held_out in SOURCES:
        tactic_weight, b0_weight, details = select_weights(
            held_out,
            rows,
            labels,
            label_index,
            candidate_tactics,
            tactic_by_label,
            b0_scores,
        )
        selections[held_out] = (tactic_weight, b0_weight)
        inner.extend(details)
        predictions.extend(
            outer_predictions(
                held_out,
                rows,
                labels,
                label_index,
                candidate_tactics,
                tactic_by_label,
                b0_scores,
                b0_rankings,
                tactic_weight,
                b0_weight,
                frozen_t,
                frozen_tactic_lambdas,
                frozen_a,
            )
        )
    predictions.sort(key=lambda row: (row["held_out_source"], row["sample_id"]))
    inner.sort(key=lambda row: (row["held_out_source"], row["inner_validation_source"], row["b0_lambda"], row["tactic_lambda"]))
    all_samples = [*predictions, *reference_rows(predictions)]
    campaigns = campaign_rows(all_samples)
    folds = fold_rows(campaigns)
    differences = bootstrap(campaigns)
    strata = stratified(all_samples)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "inner_selection.csv", inner, ("held_out_source", "inner_validation_source", "tactic_lambda", "b0_lambda", "campaign_macro_ndcg5", "two_source_mean_ndcg5", "selected", "training_rows", "validation_rows"))
    write_csv(output / "selected_weights.csv", [{"held_out_source": source, "selected_tactic_lambda": selections[source][0], "selected_b0_lambda": selections[source][1]} for source in SOURCES], ("held_out_source", "selected_tactic_lambda", "selected_b0_lambda"))
    write_csv(output / "predictions.csv", predictions, ("held_out_source", "method", "selected_tactic_lambda", "selected_b0_lambda", "sample_id", "campaign_id", "prefix_len", "target_parent_ids", "target_size", "transition_visibility", "target_label_visibility", "text_length_group", "top20_ids", "top20_scores", *METRICS))
    write_csv(output / "campaign_results.csv", campaigns, ("held_out_source", "method", "campaign_id", "rows", *METRICS))
    write_csv(output / "fold_results.csv", folds, ("held_out_source", "method", "campaigns", *[f"campaign_macro_{metric}" for metric in METRICS]))
    write_csv(output / "paired_bootstrap_differences.csv", differences, ("scope", "comparison", "metric", "point_estimate", "ci95_low", "ci95_high", "replicates", "seed"))
    write_csv(output / "stratified_results.csv", strata, ("held_out_source", "method", "stratum", "group", "rows", "campaigns", "inferentially_eligible", *[f"campaign_macro_{metric}" for metric in METRICS]))
    markdown = report(folds, selections, differences)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "stdout.log").write_text(markdown + "\n", encoding="utf-8")
    output_hashes = {name: sha256(output / name) for name in managed if name != "results_manifest.json" and (output / name).exists()}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post-result exploratory; prospective confirmation required",
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "inputs": {
            "nonsemantic_manifest_sha256": sha256(NONSEMANTIC_RESULTS / "results_manifest.json"),
            "b0_rankings_sha256": sha256(DIRECT.B0_RANKINGS),
            "hm_b0_manifest_sha256": sha256(HM_B0_RESULTS / "results_manifest.json"),
        },
        "parameters": {
            "joint_tactic_lambda_grid": list(GRID),
            "b0_lambda_grid": list(GRID),
            "b0_score": "1/rank for returned ranks 1..5; zero otherwise",
            "selected": {source: {"tactic_lambda": selections[source][0], "b0_lambda": selections[source][1]} for source in SOURCES},
            "tie_break": "smaller B0 lambda, then smaller tactic lambda",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "frozen_t_reproduction": "PASS all 784 outer rows",
        "b0_reproduction": "PASS all 784 outer rows",
        "outputs_sha256": output_hashes,
    }
    (output / "results_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
