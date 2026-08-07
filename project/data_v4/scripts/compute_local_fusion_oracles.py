#!/usr/bin/env python3
"""Compute target-aware upper bounds for the frozen zero-cost fusion search.

This script is deliberately oracle-only: targets are used to select the best
allowed action per sample. Its outputs are screening diagnostics, never model
features or deployable results. It performs no network operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/local_fusion_mechanism_search_v1.md"
BASE_RESULTS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
B0_PATH = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1/b0_rankings.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/local_fusion_search/oracles_v1"
SOURCES = ("ctid", "attack_flow", "stockpile")
EXPERTS = ("A", "T", "K")
K_GRID: tuple[int | str, ...] = (3, 5, 10, "all")
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = import_module(BASE_SCRIPT, "future3_nonsemantic_oracle_base")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def score(top5: Sequence[str], targets: Sequence[str]) -> float:
    return float(BASE.sample_metrics(top5, targets)["ndcg5"])


def target_first(candidates: Sequence[str], targets: Sequence[str]) -> list[str]:
    truth = set(targets)
    return sorted(candidates, key=lambda label: (label not in truth, candidates.index(label)))


def source_equal_campaign_macro(records: Sequence[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    campaign_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        campaign_values[(row["source"], row["campaign_id"])].append(float(row["ndcg5"]))
    source_values: dict[str, list[float]] = defaultdict(list)
    for (source, _campaign), values in campaign_values.items():
        source_values[source].append(sum(values) / len(values))
    by_source = {source: sum(source_values[source]) / len(source_values[source]) for source in SOURCES}
    return sum(by_source.values()) / len(SOURCES), by_source


def expert_reorder(b0: Sequence[str], expert: Sequence[str]) -> list[str]:
    ranks = {label: index for index, label in enumerate(expert)}
    original = {label: index for index, label in enumerate(b0)}
    return sorted(b0, key=lambda label: (ranks.get(label, 10_000), original[label]))


def expert_replace(b0: Sequence[str], expert: Sequence[str]) -> list[str]:
    existing = set(b0)
    replacement = next((label for label in expert if label not in existing), None)
    return list(b0) if replacement is None else [*b0[:4], replacement]


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def local_rankings(
    row: dict[str, Any],
    train: Sequence[dict[str, Any]],
    b0: Sequence[str],
    labels: Sequence[str],
    label_index: dict[str, int],
    tactic_by_label: dict[str, tuple[int, ...]],
) -> list[list[str]]:
    campaigns: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in train:
        campaigns[(item["source"], item["campaign_id"])].append(item)
    test_parent = set(row["history"])
    test_tactics = {index for label in test_parent for index in tactic_by_label.get(label, tuple())}
    similarities: list[tuple[float, str, tuple[str, str]]] = []
    for key, values in campaigns.items():
        parents = {label for item in values for label in item["history"]}
        tactics = {index for label in parents for index in tactic_by_label.get(label, tuple())}
        similarity = 0.7 * jaccard(test_parent, parents) + 0.3 * jaccard(test_tactics, tactics)
        tie = hashlib.sha256(f"{key[0]}::{key[1]}".encode()).hexdigest()
        similarities.append((-similarity, tie, key))
    similarities.sort()
    outputs: list[list[str]] = []
    for k in K_GRID:
        count = len(similarities) if k == "all" else min(int(k), len(similarities))
        chosen = {key for _similarity, _tie, key in similarities[:count]}
        selected = [item for key in chosen for item in campaigns[key]]
        model = BASE.RelevanceModel(
            selected,
            len(labels),
            lambda item: {label_index[label] for label in item["targets"]},
        )
        prior = model.unigram
        h1 = row["history"][-1]
        n1 = model.order1_counts.get(h1, 0)
        counts1 = model.order1_targets.get(h1, [0] * len(labels))
        h2 = tuple(row["history"][-2:]) if len(row["history"]) >= 2 else None
        n2 = model.order2_counts.get(h2, 0) if h2 is not None else 0
        counts2 = model.order2_targets.get(h2, [0] * len(labels)) if h2 is not None else [0] * len(labels)
        evidence1: dict[str, float] = {}
        evidence2: dict[str, float] = {}
        for label in b0:
            index = label_index[label]
            p0 = prior[index]
            p1 = (counts1[index] + BASE.ALPHA * p0) / (n1 + BASE.ALPHA) if n1 else p0
            p2 = (counts2[index] + BASE.ALPHA * p0) / (n2 + BASE.ALPHA) if n2 else p0
            evidence1[label] = max(-4.0, min(4.0, math.log((p1 + 1e-9) / (p0 + 1e-9))))
            evidence2[label] = max(-4.0, min(4.0, math.log((p2 + 1e-9) / (p0 + 1e-9))))
        for weight1 in WEIGHT_GRID:
            for weight2 in WEIGHT_GRID:
                values = {
                    label: -math.log(rank) + weight1 * evidence1[label] + weight2 * evidence2[label]
                    for rank, label in enumerate(b0, start=1)
                }
                original = {label: index for index, label in enumerate(b0)}
                outputs.append(sorted(b0, key=lambda label: (-values[label], original[label])))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    _candidate_tactics, tactic_by_label = BASE.parse_tactics()
    if len(rows) != 784:
        raise AssertionError("formal denominator changed")

    b0_rows = read_csv(B0_PATH)
    b0 = {item["sample_id"]: list(json.loads(item["predicted_next_ttps"])) for item in b0_rows}
    predictions = read_csv(BASE_RESULTS)
    expert: dict[tuple[str, str], list[str]] = {}
    for item in predictions:
        if item["method"] in {"A", "T", "K"}:
            expert[(item["sample_id"], item["method"])] = list(json.loads(item["top20_ids"]))
    if len(b0) != 784 or len(expert) != 784 * 3:
        raise AssertionError("ranking input denominator changed")

    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("B0", "F1_RR5_oracle", "F2_C1R_oracle", "F3_OSER_oracle", "F4_LCB_RS_oracle", "F5_LCTR5_oracle")}
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        base = b0[sample_id]
        rankings = {name: expert[(sample_id, name)] for name in EXPERTS}
        actions: dict[str, list[str]] = {"B0": list(base)}

        actions["F1_RR5_oracle"] = target_first(base, row["targets"])

        votes = Counter(label for name in EXPERTS for label in rankings[name][:10] if label not in set(base))
        eligible = sorted(label for label, count in votes.items() if count >= 2)
        f2_options = [list(base), *[[*base[:4], label] for label in eligible]]
        actions["F2_C1R_oracle"] = max(f2_options, key=lambda item: score(item, row["targets"]))

        f3_options = [list(base), *[rankings[name][:5] for name in EXPERTS]]
        actions["F3_OSER_oracle"] = max(f3_options, key=lambda item: score(item, row["targets"]))

        f4_options = [list(base)]
        for name in EXPERTS:
            f4_options.append(expert_reorder(base, rankings[name]))
        for name in EXPERTS:
            f4_options.append(expert_replace(base, rankings[name]))
        actions["F4_LCB_RS_oracle"] = max(f4_options, key=lambda item: score(item, row["targets"]))

        train = [item for item in rows if item["source"] != row["source"]]
        f5_options = local_rankings(row, train, base, labels, label_index, tactic_by_label)
        actions["F5_LCTR5_oracle"] = max(f5_options, key=lambda item: score(item, row["targets"]))

        for method, ranking in actions.items():
            value = score(ranking, row["targets"])
            records[method].append({"source": row["source"], "campaign_id": row["campaign_id"], "sample_id": sample_id, "ndcg5": value})
            if method != "B0":
                action_rows.append({"mechanism": method, "sample_id": sample_id, "source": row["source"], "campaign_id": row["campaign_id"], "top5": compact_json(ranking), "ndcg5": value, "b0_ndcg5": score(base, row["targets"])})

    summary: list[dict[str, Any]] = []
    b0_overall, b0_sources = source_equal_campaign_macro(records["B0"])
    for method in records:
        overall, by_source = source_equal_campaign_macro(records[method])
        summary.append({"mechanism": method, "ctid": by_source["ctid"], "attack_flow": by_source["attack_flow"], "stockpile": by_source["stockpile"], "source_equal_ndcg5": overall, "delta_vs_b0": overall - b0_overall, "passes_3pt_oracle_gate": int(method != "B0" and overall - b0_overall >= 0.03 - 1e-12)})

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "oracle_summary.csv", summary, ("mechanism", "ctid", "attack_flow", "stockpile", "source_equal_ndcg5", "delta_vs_b0", "passes_3pt_oracle_gate"))
    write_csv(output / "oracle_actions.csv", action_rows, ("mechanism", "sample_id", "source", "campaign_id", "top5", "ndcg5", "b0_ndcg5"))
    lines = ["# Local fusion mechanism oracle screen", "", "Target-aware upper bounds only; never inference features.", "", "| Mechanism | CTID | Attack Flow | Stockpile | Overall | Delta B0 | Gate |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for item in summary:
        gate = "--" if item["mechanism"] == "B0" else ("PASS" if item["passes_3pt_oracle_gate"] else "DROP")
        lines.append(f"| {item['mechanism']} | {item['ctid']:.4f} | {item['attack_flow']:.4f} | {item['stockpile']:.4f} | {item['source_equal_ndcg5']:.4f} | {item['delta_vs_b0']:+.4f} | {gate} |")
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    managed = ("oracle_summary.csv", "oracle_actions.csv", "report.md")
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "analysis_status": "target-aware oracle screening only", "network_calls": 0, "api_cost": 0, "protocol": {"path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(PROTOCOL)}, "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))}, "inputs": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in (BASE.SAMPLES_PATH, B0_PATH, BASE_RESULTS, BASE.VOCAB_PATH)}, "outputs": {name: sha256(output / name) for name in managed}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
