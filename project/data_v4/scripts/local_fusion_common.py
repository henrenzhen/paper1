"""Shared local-only utilities for the frozen zero-cost fusion search."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "data_v4/scripts/run_nonsemantic_future3_lodo.py"
B0_PATH = PROJECT_ROOT / "data_v4/semantic_summaries/deepseek_v4_flash_future3_v1/b0_rankings.csv"
BASE_RESULTS = PROJECT_ROOT / "data_v4/results/nonsemantic_future3_lodo_v1/predictions.csv"
SOURCES = ("ctid", "attack_flow", "stockpile")
EXPERTS = ("A", "T", "K")
VARIANTS = ("main", "permuted", "no_prior", "equal_capacity")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = import_module(BASE_SCRIPT, "local_fusion_nonsemantic_base")


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


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 50.0)))
    exp_value = math.exp(max(value, -50.0))
    return exp_value / (1.0 + exp_value)


def softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(min(50.0, value - maximum)) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def normalize_fit(vectors: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    dimension = len(vectors[0])
    means = [statistics.fmean(vector[index] for vector in vectors) for index in range(dimension)]
    scales = []
    for index, mean in enumerate(means):
        variance = statistics.fmean((vector[index] - mean) ** 2 for vector in vectors)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def normalize(vector: Sequence[float], means: Sequence[float], scales: Sequence[float]) -> list[float]:
    return [(value - means[index]) / scales[index] for index, value in enumerate(vector)]


def pseudo_value(sample_id: str, candidate: str, index: int) -> float:
    digest = hashlib.sha256(f"control-v1|{sample_id}|{candidate}|{index}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return 2.0 * integer / (2**64 - 1) - 1.0


def pseudo_ranking(sample_id: str, expert: str, labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(labels, key=lambda label: hashlib.sha256(f"control-v1|{sample_id}|{expert}|{label}".encode()).hexdigest()))


def load_inputs() -> tuple[list[dict[str, Any]], list[str], dict[str, int], list[tuple[int, ...]], dict[str, tuple[int, ...]], dict[str, tuple[str, ...]], dict[str, float]]:
    rows = BASE.parse_samples()
    labels, label_index = BASE.parse_vocabulary()
    candidate_tactics, tactic_by_label = BASE.parse_tactics()
    b0 = {row["sample_id"]: tuple(json.loads(row["predicted_next_ttps"])) for row in read_csv(B0_PATH)}
    if len(rows) != 784 or len(b0) != 784:
        raise AssertionError("formal denominator changed")
    lambda_values: dict[str, set[float]] = defaultdict(set)
    committed = read_csv(BASE_RESULTS)
    for row in committed:
        if row["method"] == "T":
            lambda_values[row["held_out_source"]].add(float(row["chosen_tactic_lambda"]))
    chosen = {source: next(iter(values)) for source, values in lambda_values.items() if len(values) == 1}
    if chosen != {"ctid": 0.0, "attack_flow": 0.0, "stockpile": 0.1}:
        raise AssertionError(f"unexpected frozen T lambdas: {chosen}")
    return rows, labels, label_index, candidate_tactics, tactic_by_label, b0, chosen


class EvidenceBuilder:
    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        labels: Sequence[str],
        label_index: dict[str, int],
        candidate_tactics: Sequence[tuple[int, ...]],
        tactic_by_label: dict[str, tuple[int, ...]],
        tactic_lambda: float,
    ) -> None:
        self.train = train
        self.labels = labels
        self.label_index = label_index
        self.candidate_tactics = candidate_tactics
        self.tactic_by_label = tactic_by_label
        self.tactic_lambda = tactic_lambda
        self.a_model = BASE.RelevanceModel(train, len(labels), lambda row: {label_index[label] for label in row["targets"]})
        self.t_model = BASE.RelevanceModel(train, 14, lambda row: BASE.tactic_target_indices(row, tactic_by_label))
        self.source_count = max(1, len({row["source"] for row in train}))
        self.pair1_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.pair2_sources: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)
        for row in train:
            h1 = row["history"][-1]
            h2 = tuple(row["history"][-2:]) if len(row["history"]) >= 2 else None
            for target in set(row["targets"]):
                self.pair1_sources[(h1, target)].add(row["source"])
                if h2 is not None:
                    self.pair2_sources[(h2, target)].add(row["source"])

    def build(self, row: dict[str, Any], b0: tuple[str, ...]) -> dict[str, Any]:
        history = row["history"]
        h1 = history[-1]
        h2 = tuple(history[-2:]) if len(history) >= 2 else None
        n1 = self.a_model.order1_counts.get(h1, 0)
        n2 = self.a_model.order2_counts.get(h2, 0) if h2 is not None else 0
        counts1 = self.a_model.order1_targets.get(h1, [0] * len(self.labels))
        counts2 = self.a_model.order2_targets.get(h2, [0] * len(self.labels)) if h2 is not None else [0] * len(self.labels)
        a_values = self.a_model.score(history)
        probabilities = [max(value, 0.0) for value in a_values]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities] if total else [1 / len(self.labels)] * len(self.labels)
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0) / math.log(len(self.labels))
        last_tactics = self.tactic_by_label.get(h1, tuple())
        compatible = [bool(last_tactics) and any(candidate >= last for candidate in indices for last in last_tactics) for indices in self.candidate_tactics]
        k_values = [int(flag) + value for flag, value in zip(compatible, a_values)]
        tactic_values = BASE.tactic_candidate_scores(self.t_model.score(history), self.candidate_tactics)
        t_values = BASE.fused_scores(a_values, tactic_values, self.tactic_lambda)
        rankings = {}
        for name, values in (("A", a_values), ("T", t_values), ("K", k_values)):
            ranked, _ = BASE.ranking(values, self.labels)
            rankings[name] = tuple(ranked)
        stats: dict[str, dict[str, float]] = {}
        for label in self.labels:
            index = self.label_index[label]
            p0 = self.a_model.unigram[index]
            p1 = (counts1[index] + BASE.ALPHA * p0) / (n1 + BASE.ALPHA) if n1 else p0
            p2 = (counts2[index] + BASE.ALPHA * p0) / (n2 + BASE.ALPHA) if n2 else p0
            stats[label] = {
                "k1": float(counts1[index]),
                "k2": float(counts2[index]),
                "s1": float(len(self.pair1_sources.get((h1, label), set()))),
                "s2": float(len(self.pair2_sources.get((h2, label), set()))) if h2 is not None else 0.0,
                "lr1": max(-4.0, min(4.0, math.log((p1 + 1e-9) / (p0 + 1e-9)))),
                "lr2": max(-4.0, min(4.0, math.log((p2 + 1e-9) / (p0 + 1e-9)))),
                "raw1": math.log(p1 + 1e-9),
                "raw2": math.log(p2 + 1e-9),
            }
        return {"row": row, "b0": b0, "evidence": {"n1": float(n1), "n2": float(n2), "entropy": entropy, "rankings": rankings, "stats": stats}}


def build_eval_bundles(
    train: Sequence[dict[str, Any]],
    evaluate: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    b0: dict[str, tuple[str, ...]],
    tactic_lambda: float,
) -> list[dict[str, Any]]:
    builder = EvidenceBuilder(train, labels, label_index, candidate_tactics, tactic_by_label, tactic_lambda)
    return [builder.build(row, b0[row["sample_id"]]) for row in evaluate]


def campaign_loo_bundles(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    label_index: dict[str, int],
    candidate_tactics: Sequence[tuple[int, ...]],
    tactic_by_label: dict[str, tuple[int, ...]],
    b0: dict[str, tuple[str, ...]],
    tactic_lambda: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["campaign_id"])].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        excluded_ids = {row["sample_id"] for row in groups[key]}
        train = [row for row in rows if row["sample_id"] not in excluded_ids]
        output.extend(build_eval_bundles(train, groups[key], labels, label_index, candidate_tactics, tactic_by_label, b0, tactic_lambda))
    return sorted(output, key=lambda bundle: bundle["row"]["sample_id"])


def permute_bundles(bundles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source_campaign: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for bundle in bundles:
        row = bundle["row"]
        by_source_campaign[row["source"]][row["campaign_id"]].append(bundle)
    output: list[dict[str, Any]] = []
    for source in sorted(by_source_campaign):
        campaigns = sorted(by_source_campaign[source])
        if len(campaigns) < 2:
            raise AssertionError(f"cannot rotate one campaign in {source}")
        for index, campaign in enumerate(campaigns):
            donor_campaign = campaigns[(index + 1) % len(campaigns)]
            recipients = sorted(by_source_campaign[source][campaign], key=lambda item: (item["row"]["prefix_len"], item["row"]["sample_id"]))
            donors = sorted(by_source_campaign[source][donor_campaign], key=lambda item: (item["row"]["prefix_len"], item["row"]["sample_id"]))
            for position, recipient in enumerate(recipients):
                donor = donors[min(len(donors) - 1, position * len(donors) // len(recipients))]
                output.append({"row": recipient["row"], "b0": recipient["b0"], "evidence": donor["evidence"]})
    return sorted(output, key=lambda bundle: bundle["row"]["sample_id"])


def variant_bundles(bundles: Sequence[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    return permute_bundles(bundles) if variant == "permuted" else list(bundles)


def rank_map(bundle: dict[str, Any], expert: str, labels: Sequence[str], variant: str) -> dict[str, int]:
    ranking = pseudo_ranking(bundle["row"]["sample_id"], expert, labels) if variant == "equal_capacity" else bundle["evidence"]["rankings"][expert]
    return {label: index + 1 for index, label in enumerate(ranking)}


def candidate_stat(bundle: dict[str, Any], label: str, name: str, variant: str, feature_index: int = 0) -> float:
    if variant == "equal_capacity":
        return pseudo_value(bundle["row"]["sample_id"], label, feature_index)
    if name == "e1":
        key = "raw1" if variant == "no_prior" else "lr1"
        return float(bundle["evidence"]["stats"][label][key])
    if name == "e2":
        key = "raw2" if variant == "no_prior" else "lr2"
        return float(bundle["evidence"]["stats"][label][key])
    return float(bundle["evidence"]["stats"][label][name])


def sample_metric(ranking: Sequence[str], row: dict[str, Any]) -> dict[str, float]:
    return BASE.sample_metrics(ranking[:5], row["targets"])


def prediction_row(method: str, bundle: dict[str, Any], ranking: Sequence[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = bundle["row"]
    return {"held_out_source": row["source"], "method": method, "sample_id": row["sample_id"], "campaign_id": row["campaign_id"], "prefix_len": row["prefix_len"], "target_parent_ids": compact_json(row["targets"]), "top5_ids": compact_json(list(ranking[:5])), **sample_metric(ranking, row), **(extra or {})}


def campaign_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["held_out_source"], row["method"], row["campaign_id"])].append(row)
    return [{"held_out_source": source, "method": method, "campaign_id": campaign, "rows": len(values), **{metric: statistics.fmean(float(row[metric]) for row in values) for metric in ("ndcg5", "hit5", "precision5", "recall5")}} for (source, method, campaign), values in sorted(grouped.items())]


def fold_rows(campaigns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in campaigns:
        grouped[(row["held_out_source"], row["method"])].append(row)
    return [{"held_out_source": source, "method": method, "campaigns": len(values), **{f"campaign_macro_{metric}": statistics.fmean(float(row[metric]) for row in values) for metric in ("ndcg5", "hit5", "precision5", "recall5")}} for (source, method), values in sorted(grouped.items())]


def source_equal(campaigns: Sequence[dict[str, Any]], method: str, metric: str = "ndcg5", excluded_ctid: str | None = None) -> float:
    values = []
    for source in SOURCES:
        rows = [row for row in campaigns if row["held_out_source"] == source and row["method"] == method and not (source == "ctid" and row["campaign_id"] == excluded_ctid)]
        values.append(statistics.fmean(float(row[metric]) for row in rows))
    return statistics.fmean(values)


def bootstrap(campaigns: Sequence[dict[str, Any]], comparisons: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    lookup = {(row["held_out_source"], row["method"], row["campaign_id"]): row for row in campaigns}
    ids = {source: sorted({row["campaign_id"] for row in campaigns if row["held_out_source"] == source}) for source in SOURCES}
    rng = random.Random(BOOTSTRAP_SEED)
    reps: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    metrics = ("ndcg5", "hit5", "precision5", "recall5")
    for _ in range(BOOTSTRAP_REPLICATES):
        draws = {source: [rng.choice(ids[source]) for _ in ids[source]] for source in SOURCES}
        for left, right in comparisons:
            for metric in metrics:
                source_deltas = [statistics.fmean(float(lookup[source, left, campaign][metric]) - float(lookup[source, right, campaign][metric]) for campaign in draws[source]) for source in SOURCES]
                reps[(left, right, metric)].append(statistics.fmean(source_deltas))
    output = []
    for left, right in comparisons:
        for metric in metrics:
            values = sorted(reps[(left, right, metric)])
            point = source_equal(campaigns, left, metric) - source_equal(campaigns, right, metric)
            output.append({"comparison": f"{left}-{right}", "metric": metric, "point_estimate": point, "ci95_low": values[int(0.025 * len(values))], "ci95_high": values[min(len(values)-1, int(0.975 * len(values)))], "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    return output


def ctid_influence(campaigns: Sequence[dict[str, Any]], main_methods: Sequence[str]) -> list[dict[str, Any]]:
    ctid_ids = sorted({row["campaign_id"] for row in campaigns if row["held_out_source"] == "ctid"})
    output = []
    for main in main_methods:
        full = source_equal(campaigns, main) - source_equal(campaigns, "B0")
        for campaign in ctid_ids:
            delta = source_equal(campaigns, main, excluded_ctid=campaign) - source_equal(campaigns, "B0", excluded_ctid=campaign)
            output.append({"method": main, "removed_ctid_campaign": campaign, "full_delta": full, "leave_one_out_delta": delta, "sign_reversed": int((full > 0 and delta <= 0) or (full < 0 and delta >= 0))})
    return output
