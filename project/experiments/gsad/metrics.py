from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd


def root_macro_mean(frame: pd.DataFrame, value_col: str, group_col: str = "root") -> float:
    if group_col not in frame or value_col not in frame:
        raise ValueError(f"missing {group_col} or {value_col}")
    if len(frame) == 0:
        return float("nan")
    values = pd.to_numeric(frame[value_col], errors="raise")
    grouped = pd.DataFrame({group_col: frame[group_col], value_col: values}).groupby(
        group_col, sort=True
    )[value_col]
    return float(grouped.mean().mean())


@dataclass(frozen=True)
class MetricBundle:
    row: dict[str, float]
    root_macro: dict[str, float]
    slices: dict[str, dict[str, float]]


def _reciprocal_rank(target: str, candidates: object) -> float:
    if isinstance(candidates, str):
        sequence = [part.strip() for part in candidates.split("||") if part.strip()]
    else:
        sequence = list(candidates)
    try:
        return 1.0 / (sequence.index(target) + 1)
    except ValueError:
        return 0.0


def _contains(container: object, target: str) -> bool:
    if isinstance(container, str):
        return target in {part.strip() for part in container.split("||") if part.strip()}
    return target in container


def _slice_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if len(frame) == 0:
        return {"n": 0, "top1_accuracy": float("nan"), "descendant_coverage": float("nan")}
    return {
        "n": int(len(frame)),
        "top1_accuracy": float(frame["_top1_correct"].mean()),
        "descendant_coverage": float(frame["_descendant_hit"].mean()),
    }


def evaluate_predictions(frame: pd.DataFrame) -> MetricBundle:
    required = {
        "root",
        "target",
        "top1_pred",
        "top5",
        "gamma",
        "descendants",
        "action_kind",
        "leaf_equivalent_size",
        "display_node_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    work = frame.copy()
    work["_top1_correct"] = work["top1_pred"].astype(str) == work["target"].astype(str)
    work["_hit5"] = [
        _contains(candidates, str(target))
        for candidates, target in zip(work["top5"], work["target"], strict=True)
    ]
    work["_mrr"] = [
        _reciprocal_rank(str(target), candidates)
        for candidates, target in zip(work["top5"], work["target"], strict=True)
    ]
    work["_leaf_hit"] = [
        _contains(gamma, str(target))
        for gamma, target in zip(work["gamma"], work["target"], strict=True)
    ]
    work["_descendant_hit"] = [
        _contains(descendants, str(target))
        for descendants, target in zip(
            work["descendants"], work["target"], strict=True
        )
    ]
    work["_exact"] = work["action_kind"].astype(str) == "exact"
    work["_abstain"] = work["action_kind"].astype(str) == "abstain"
    if "exact_label" in work:
        work["_exact_correct"] = (
            work["exact_label"].astype(str) == work["target"].astype(str)
        )
    else:
        work["_exact_correct"] = work["_top1_correct"]
    if "vocab_size" in work:
        work["_full_set"] = (
            work["leaf_equivalent_size"].astype(float)
            >= work["vocab_size"].astype(float)
        )
    else:
        work["_full_set"] = False

    exact_mask = work["_exact"].to_numpy(dtype=bool)
    exact_accuracy = (
        float(work.loc[exact_mask, "_exact_correct"].mean())
        if exact_mask.any()
        else float("nan")
    )
    row = {
        "n": int(len(work)),
        "top1_accuracy": float(work["_top1_correct"].mean()),
        "hit_at_5": float(work["_hit5"].mean()),
        "mrr": float(work["_mrr"].mean()),
        "leaf_coverage": float(work["_leaf_hit"].mean()),
        "descendant_coverage": float(work["_descendant_hit"].mean()),
        "mean_leaf_size": float(work["leaf_equivalent_size"].mean()),
        "mean_display_nodes": float(work["display_node_count"].mean()),
        "exact_coverage": float(work["_exact"].mean()),
        "exact_accuracy": exact_accuracy,
        "abstain_rate": float(work["_abstain"].mean()),
        "full_set_rate": float(work["_full_set"].mean()),
    }
    root_macro = {
        "top1_accuracy": root_macro_mean(work, "_top1_correct"),
        "hit_at_5": root_macro_mean(work, "_hit5"),
        "mrr": root_macro_mean(work, "_mrr"),
        "leaf_coverage": root_macro_mean(work, "_leaf_hit"),
        "descendant_coverage": root_macro_mean(work, "_descendant_hit"),
        "mean_leaf_size": root_macro_mean(work, "leaf_equivalent_size"),
        "exact_coverage": root_macro_mean(work, "_exact"),
        "abstain_rate": root_macro_mean(work, "_abstain"),
    }
    fit_seen = (
        work["fit_seen"].astype(bool)
        if "fit_seen" in work
        else pd.Series(True, index=work.index)
    )
    slices = {
        "closed_label": _slice_metrics(work.loc[fit_seen]),
        "open_label": _slice_metrics(work.loc[~fit_seen]),
        "overall": _slice_metrics(work),
    }
    return MetricBundle(row=row, root_macro=root_macro, slices=slices)


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    valid_replicates: int


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    group_col: str,
    n_boot: int,
    seed: int,
) -> Interval:
    if group_col not in frame or len(frame) == 0:
        raise ValueError("bootstrap frame is empty or lacks the group column")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    groups = np.asarray(sorted(frame[group_col].astype(str).unique()))
    group_values = frame[group_col].astype(str).to_numpy()
    indices_by_group = {
        group: np.flatnonzero(group_values == group) for group in groups
    }
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(int(n_boot)):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate(
            [indices_by_group[str(group)] for group in sampled_groups]
        )
        bootstrap_instances = np.concatenate(
            [
                np.full(len(indices_by_group[str(group)]), draw_index, dtype=int)
                for draw_index, group in enumerate(sampled_groups)
            ]
        )
        sample = frame.iloc[sampled_indices].copy()
        sample[group_col] = bootstrap_instances
        estimate = float(metric_fn(sample))
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError("no finite bootstrap replicates")
    values = np.asarray(estimates, dtype=float)
    return Interval(
        point=float(metric_fn(frame)),
        lower=float(np.quantile(values, 0.025)),
        upper=float(np.quantile(values, 0.975)),
        valid_replicates=len(values),
    )


def domain_root_bootstrap_difference(
    frame: pd.DataFrame,
    candidate_col: str,
    baseline_col: str,
    domain_col: str,
    root_col: str,
    n_boot: int,
    seed: int,
) -> Interval:
    required = {candidate_col, baseline_col, domain_col, root_col}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("bootstrap frame is empty or lacks required columns")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    work = frame.loc[:, list(required)].copy()
    work["_delta"] = pd.to_numeric(
        work[candidate_col], errors="raise"
    ) - pd.to_numeric(work[baseline_col], errors="raise")
    root_values = (
        work.groupby([domain_col, root_col], sort=True)["_delta"]
        .mean()
        .reset_index()
    )
    by_domain = {
        str(domain): group["_delta"].to_numpy(dtype=float)
        for domain, group in root_values.groupby(domain_col, sort=True)
    }
    if not by_domain or any(len(values) == 0 for values in by_domain.values()):
        raise ValueError("each domain must contain at least one root")
    point = float(np.mean([values.mean() for values in by_domain.values()]))
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(n_boot)):
        domain_means = []
        for values in by_domain.values():
            sampled = values[rng.integers(0, len(values), size=len(values))]
            domain_means.append(float(sampled.mean()))
        estimates.append(float(np.mean(domain_means)))
    bootstrap = np.asarray(estimates, dtype=float)
    return Interval(
        point=point,
        lower=float(np.quantile(bootstrap, 0.025)),
        upper=float(np.quantile(bootstrap, 0.975)),
        valid_replicates=len(bootstrap),
    )


def matched_cost_gain(candidate_curve: pd.DataFrame, baseline_curve: pd.DataFrame) -> float:
    for name, curve in (("candidate", candidate_curve), ("baseline", baseline_curve)):
        if not {"cost", "value"}.issubset(curve.columns) or len(curve) == 0:
            raise ValueError(f"{name} curve must contain nonempty cost/value columns")
    baseline = baseline_curve.sort_values("cost")
    costs = baseline["cost"].to_numpy(dtype=float)
    values = baseline["value"].to_numpy(dtype=float)
    candidate_costs = candidate_curve["cost"].to_numpy(dtype=float)
    if np.any(candidate_costs < costs.min()) or np.any(candidate_costs > costs.max()):
        raise ValueError("candidate cost lies outside baseline interpolation range")
    interpolated = np.interp(candidate_costs, costs, values)
    return float(
        np.mean(candidate_curve["value"].to_numpy(dtype=float) - interpolated)
    )


@dataclass(frozen=True)
class GateResult:
    passed: bool
    observed: float | None
    threshold: str
    lower: float | None
    reason: str


def _missing_gate(name: str, required: list[str]) -> GateResult:
    return GateResult(
        passed=False,
        observed=None,
        threshold=name,
        lower=None,
        reason=f"missing required metric or interval: {', '.join(required)}",
    )


def evaluate_gates(
    metrics: Mapping[str, float],
    intervals: Mapping[str, Interval],
    ablations: Mapping[str, object],
) -> dict[str, GateResult]:
    del ablations
    gates: dict[str, GateResult] = {}

    key_a = "coverage_gain_pp_matched_size"
    if key_a not in metrics or key_a not in intervals:
        gates["A"] = _missing_gate("gain >= 5pp and lower > 0", [key_a])
    else:
        value = float(metrics[key_a])
        lower = float(intervals[key_a].lower)
        gates["A"] = GateResult(
            value >= 5.0 and lower > 0,
            value,
            ">= 5 percentage points and CI lower > 0",
            lower,
            "coverage efficiency" if value >= 5.0 and lower > 0 else "coverage-efficiency threshold failed",
        )

    key_b = "exact_output_gain_relative"
    if key_b not in metrics or key_b not in intervals:
        gates["B"] = _missing_gate("relative gain >= 0.10 and lower > 0", [key_b])
    else:
        value = float(metrics[key_b])
        lower = float(intervals[key_b].lower)
        gates["B"] = GateResult(
            value >= 0.10 and lower > 0,
            value,
            ">= 0.10 relative and CI lower > 0",
            lower,
            "exact-output efficiency" if value >= 0.10 and lower > 0 else "exact-output threshold failed",
        )

    required_c = ["exact_coverage", "exact_accuracy_gain_pp"]
    if any(key not in metrics for key in required_c) or "exact_accuracy_gain_pp" not in intervals:
        gates["C"] = _missing_gate(
            "coverage >= 0.50, gain >= 5pp, lower > 0", required_c
        )
    else:
        coverage = float(metrics["exact_coverage"])
        gain = float(metrics["exact_accuracy_gain_pp"])
        lower = float(intervals["exact_accuracy_gain_pp"].lower)
        passed = coverage >= 0.50 and gain >= 5.0 and lower > 0
        gates["C"] = GateResult(
            passed,
            coverage,
            "exact coverage >= 0.50 and accuracy gain >= 5pp with CI lower > 0",
            lower,
            "nontrivial exact prediction" if passed else f"exact coverage={coverage:.4f}, gain={gain:.4f}",
        )

    if "abstain_rate" not in metrics:
        gates["D"] = _missing_gate("abstain <= 0.20", ["abstain_rate"])
    else:
        value = float(metrics["abstain_rate"])
        gates["D"] = GateResult(
            value <= 0.20,
            value,
            "<= 0.20",
            None,
            "nontrivial abstention" if value <= 0.20 else "abstention exceeds 20%",
        )

    required_e = [
        "mean_leaf_size",
        "baseline_mean_leaf_size",
        "full_set_rate",
        "baseline_full_set_rate",
    ]
    if any(key not in metrics for key in required_e):
        gates["E"] = _missing_gate("no wider than baseline", required_e)
    else:
        leaf_ok = float(metrics["mean_leaf_size"]) <= float(
            metrics["baseline_mean_leaf_size"]
        )
        full_ok = float(metrics["full_set_rate"]) <= float(
            metrics["baseline_full_set_rate"]
        )
        gates["E"] = GateResult(
            leaf_ok and full_ok,
            float(metrics["mean_leaf_size"]),
            "mean leaf size and full-set rate <= matched baseline",
            None,
            "information budget" if leaf_ok and full_ok else "wide-set escape detected",
        )

    required_f = ["row_coverage", "root_macro_coverage"]
    if any(key not in metrics for key in required_f):
        gates["F"] = _missing_gate("row coverage 0.88-0.92 and root gap <= 0.05", required_f)
    else:
        row_coverage = float(metrics["row_coverage"])
        root_coverage = float(metrics["root_macro_coverage"])
        passed = 0.88 <= row_coverage <= 0.92 and abs(row_coverage - root_coverage) <= 0.05
        gates["F"] = GateResult(
            passed,
            row_coverage,
            "0.88 <= row coverage <= 0.92 and |row-root| <= 0.05",
            None,
            "calibration" if passed else f"row={row_coverage:.4f}, root={root_coverage:.4f}",
        )

    if "ablation_wins" not in metrics:
        gates["G"] = _missing_gate("wins >= 2", ["ablation_wins"])
    else:
        value = float(metrics["ablation_wins"])
        gates["G"] = GateResult(
            value >= 2,
            value,
            ">= 2 of no-DAG/no-shift/no-root-balancing",
            None,
            "core attribution" if value >= 2 else "insufficient ablation attribution",
        )

    primary_passed = (gates["A"].passed or gates["B"].passed) and all(
        gates[name].passed for name in "CDEFG"
    )
    gates["PRIMARY"] = GateResult(
        primary_passed,
        None,
        "(A or B) and C through G",
        None,
        "all primary conditions passed" if primary_passed else "one or more primary conditions failed",
    )
    return gates
