"""Small-scale feasibility experiment for pairwise-regret candidate routing.

This module deliberately avoids the original deep-learning runtime.  It consumes
already-generated GRU and LLM predictions, aligns them by semantic keys, and
cross-fits a fixed linear pairwise candidate ranker over disjoint SIM roots.

The output is diagnostic evidence only: the original test pool supplies the
meta-ranker's cross-validation folds, so the results must not be presented as a
final untouched test-set result in a paper.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT_PATTERN = re.compile(r"_part\d+$", flags=re.IGNORECASE)
TECHNIQUE_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?", flags=re.IGNORECASE)
FEATURE_NAMES = [
    "gru_probability",
    "gru_reciprocal_rank",
    "llm_reciprocal_rank",
    "present_in_both",
    "is_gru_top1",
    "is_llm_top1",
    "llm_rank_x_gru_uncertainty",
    "gru_rank_x_gru_confidence",
    "llm_rank_x_prefix_length",
    "gru_rank_x_gru_margin",
    "candidate_prefix_frequency",
    "candidate_is_last_state",
    "strict_train_transition_log_probability",
    "strict_train_unigram_log_probability",
]


def sim_root(sequence_id: object) -> str:
    """Collapse SIM part identifiers to their source root."""

    return ROOT_PATTERN.sub("", str(sequence_id))


def normalize_state(value: object) -> tuple[str, ...]:
    """Normalize either whitespace- or ``||``-delimited ATT&CK states."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    parts = re.split(r"\s*\|\|\s*|\s+", text)
    return tuple(part.strip().upper() for part in parts if part.strip())


def parse_candidates(value: object) -> list[str]:
    """Parse a JSON/Python list or a delimiter-separated candidate string."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        raw = []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                raw = list(parsed)
        except (SyntaxError, ValueError):
            pass
        if not raw:
            raw = TECHNIQUE_PATTERN.findall(text)
        if not raw and "||" in text:
            raw = [item.strip() for item in text.split("||")]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        candidate = str(item).strip().upper()
        if TECHNIQUE_PATTERN.fullmatch(candidate) and "." in candidate:
            candidate = candidate.split(".", 1)[0]
        if candidate and candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return result


def parse_probabilities(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        probabilities = [float(item) for item in value]
    else:
        probabilities = [float(item.strip()) for item in str(value).split("||") if item.strip()]
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in probabilities):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return probabilities


def align_predictions(gru: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Strictly align shuffled GRU and LLM artifacts and validate each state."""

    gru_required = {
        "sequence_id",
        "prefix_len",
        "state",
        "true_label",
        "top1_pred",
        "top1_prob",
        "top5_labels",
        "top5_probs",
    }
    llm_required = {"sequence_id", "state", "true_label", "predicted_next_ttps"}
    missing_gru = gru_required - set(gru.columns)
    missing_llm = llm_required - set(llm.columns)
    if missing_gru or missing_llm:
        raise ValueError(
            f"missing required columns: GRU={sorted(missing_gru)}, LLM={sorted(missing_llm)}"
        )

    g = gru.copy().reset_index(drop=True)
    l = llm.copy().reset_index(drop=True)
    g["_row_order"] = np.arange(len(g))
    g["prefix_len"] = pd.to_numeric(g["prefix_len"], errors="raise").astype(int)
    g["_gru_state_tokens"] = g["state"].map(normalize_state)
    l["_llm_state_tokens"] = l["state"].map(normalize_state)
    l["prefix_len"] = l["_llm_state_tokens"].map(len).astype(int)

    key = ["sequence_id", "prefix_len", "true_label"]
    if g.duplicated(key).any() or l.duplicated(key).any():
        raise ValueError("duplicate alignment key detected")

    l_small = l[key + ["state", "_llm_state_tokens", "predicted_next_ttps"]].rename(
        columns={"state": "llm_state"}
    )
    aligned = g.merge(l_small, on=key, how="outer", indicator=True, validate="one_to_one")
    counts = aligned["_merge"].value_counts().to_dict()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(f"alignment incomplete: {counts}")
    aligned = aligned.sort_values("_row_order").reset_index(drop=True)
    aligned["state_match"] = aligned["_gru_state_tokens"] == aligned["_llm_state_tokens"]
    if not bool(aligned["state_match"].all()):
        bad = aligned.loc[~aligned["state_match"], key].head(5).to_dict("records")
        raise ValueError(f"state mismatch after key alignment: {bad}")

    aligned["state_tokens"] = aligned["_gru_state_tokens"]
    aligned["gru_candidates"] = aligned["top5_labels"].map(parse_candidates)
    aligned["gru_probabilities"] = aligned["top5_probs"].map(parse_probabilities)
    aligned["llm_candidates"] = aligned["predicted_next_ttps"].map(parse_candidates)
    for row in aligned.itertuples():
        if len(row.gru_candidates) != len(row.gru_probabilities):
            raise ValueError(f"GRU label/probability length mismatch at row {row.Index}")
    aligned["root"] = aligned["sequence_id"].map(sim_root)
    aligned["llm_parse_ok"] = aligned["llm_candidates"].map(bool)
    return aligned.drop(columns=["_merge", "_gru_state_tokens", "_llm_state_tokens"])


def assign_group_folds(sequence_ids: Sequence[object], n_splits: int = 5) -> np.ndarray:
    """Assign balanced deterministic folds while keeping each SIM root intact."""

    roots = pd.Series(sequence_ids).map(sim_root)
    unique_roots = roots.nunique()
    if n_splits < 2 or n_splits > unique_roots:
        raise ValueError(f"n_splits must be in [2, {unique_roots}]")
    counts = roots.value_counts().to_dict()
    ordered = sorted(counts, key=lambda root: (-counts[root], root))
    loads = [0] * n_splits
    mapping: dict[str, int] = {}
    for root in ordered:
        fold = min(range(n_splits), key=lambda idx: (loads[idx], idx))
        mapping[root] = fold
        loads[fold] += counts[root]
    return roots.map(mapping).to_numpy(dtype=int)


def fit_pairwise_ranker(
    features: np.ndarray,
    sample_ids: np.ndarray,
    is_correct: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    """Fit a fixed linear pairwise ranker using weighted ridge regression."""

    x = np.asarray(features, dtype=float)
    sample_ids = np.asarray(sample_ids)
    is_correct = np.asarray(is_correct, dtype=bool)
    if x.ndim != 2 or len(x) != len(sample_ids) or len(x) != len(is_correct):
        raise ValueError("features, sample_ids, and is_correct must have matching rows")
    differences: list[np.ndarray] = []
    pair_weights: list[float] = []
    for sample_id in np.unique(sample_ids):
        mask = sample_ids == sample_id
        positives = x[mask & is_correct]
        negatives = x[mask & ~is_correct]
        if len(positives) != 1 or len(negatives) == 0:
            continue
        sample_weight = 1.0 / len(negatives)
        for negative in negatives:
            differences.append(positives[0] - negative)
            pair_weights.append(sample_weight)
    if not differences:
        return np.zeros(x.shape[1], dtype=float)
    d = np.vstack(differences)
    w = np.asarray(pair_weights, dtype=float)
    lhs = d.T @ (d * w[:, None]) + float(l2) * np.eye(x.shape[1])
    rhs = d.T @ w
    return np.linalg.solve(lhs, rhs)


def risk_coverage(
    correct: np.ndarray,
    confidence: np.ndarray,
    coverages: Iterable[float],
) -> pd.DataFrame:
    """Evaluate selective accuracy after sorting examples by confidence."""

    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if len(correct) != len(confidence) or len(correct) == 0:
        raise ValueError("correct/confidence arrays must be non-empty and equal length")
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        accepted_n = min(len(correct), max(1, int(math.ceil(float(coverage) * len(correct)))))
        accepted = order[:accepted_n]
        accuracy = float(correct[accepted].mean())
        rows.append(
            {
                "target_coverage": float(coverage),
                "actual_coverage": accepted_n / len(correct),
                "accepted_n": accepted_n,
                "accuracy": accuracy,
                "risk": 1.0 - accuracy,
                "confidence_threshold": float(confidence[accepted[-1]]),
            }
        )
    return pd.DataFrame(rows)


def crossfit_quantile_acceptance(
    confidence: np.ndarray,
    folds: np.ndarray,
    target_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate a confidence threshold on other folds, then apply it out of fold."""

    confidence = np.asarray(confidence, dtype=float)
    folds = np.asarray(folds)
    if len(confidence) != len(folds) or len(confidence) == 0:
        raise ValueError("confidence/folds arrays must be non-empty and equal length")
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    accepted = np.zeros(len(confidence), dtype=bool)
    thresholds = np.full(len(confidence), np.nan, dtype=float)
    for fold in np.unique(folds):
        test_mask = folds == fold
        train_confidence = confidence[~test_mask]
        if len(train_confidence) == 0:
            raise ValueError("each fold requires at least one calibration observation")
        threshold = float(
            np.quantile(train_confidence, 1.0 - target_coverage, method="higher")
        )
        accepted[test_mask] = confidence[test_mask] >= threshold
        thresholds[test_mask] = threshold
    return accepted, thresholds


@dataclass(frozen=True)
class TransitionPriors:
    transition_counts: dict[str, Counter]
    transition_totals: Counter
    unigram_counts: Counter
    unigram_total: int
    vocabulary: tuple[str, ...]
    retained_rows: int
    excluded_rows: int

    def transition_log_probability(self, previous: str, candidate: str, alpha: float = 0.5) -> float:
        vocab_size = max(1, len(self.vocabulary))
        count = self.transition_counts.get(previous, Counter()).get(candidate, 0)
        total = self.transition_totals.get(previous, 0)
        return math.log((count + alpha) / (total + alpha * vocab_size))

    def unigram_log_probability(self, candidate: str, alpha: float = 0.5) -> float:
        vocab_size = max(1, len(self.vocabulary))
        return math.log(
            (self.unigram_counts.get(candidate, 0) + alpha)
            / (self.unigram_total + alpha * vocab_size)
        )


def build_strict_transition_priors(
    train: pd.DataFrame, excluded_roots: set[str]
) -> TransitionPriors:
    """Estimate simple priors after removing all roots used by the diagnostic pool."""

    required = {"sequence_id", "prefix_technique_ids_parent", "next_technique_id_parent"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"training CSV missing columns: {sorted(missing)}")
    roots = train["sequence_id"].map(sim_root)
    keep = ~roots.isin(excluded_roots)
    retained = train.loc[keep]
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    transition_totals: Counter = Counter()
    unigram_counts: Counter = Counter()
    vocabulary: set[str] = set()
    for row in retained.itertuples():
        prefix = normalize_state(row.prefix_technique_ids_parent)
        candidate = str(row.next_technique_id_parent).strip().upper()
        if not prefix or not TECHNIQUE_PATTERN.fullmatch(candidate):
            continue
        previous = prefix[-1]
        transition_counts[previous][candidate] += 1
        transition_totals[previous] += 1
        unigram_counts[candidate] += 1
        vocabulary.update(prefix)
        vocabulary.add(candidate)
    return TransitionPriors(
        transition_counts=dict(transition_counts),
        transition_totals=transition_totals,
        unigram_counts=unigram_counts,
        unigram_total=sum(unigram_counts.values()),
        vocabulary=tuple(sorted(vocabulary)),
        retained_rows=int(keep.sum()),
        excluded_rows=int((~keep).sum()),
    )


def build_candidate_table(aligned: pd.DataFrame, priors: TransitionPriors) -> pd.DataFrame:
    """Create one fixed-feature row per union candidate."""

    records: list[dict[str, object]] = []
    for sample_idx, row in aligned.iterrows():
        gru_candidates = list(row["gru_candidates"])
        gru_probs = list(row["gru_probabilities"])
        llm_candidates = list(row["llm_candidates"])
        union = list(dict.fromkeys(gru_candidates + llm_candidates))
        gru_rank = {candidate: rank + 1 for rank, candidate in enumerate(gru_candidates)}
        gru_prob = dict(zip(gru_candidates, gru_probs))
        llm_rank = {candidate: rank + 1 for rank, candidate in enumerate(llm_candidates)}
        state_tokens = tuple(row["state_tokens"])
        state_counts = Counter(state_tokens)
        prefix_length = max(1, len(state_tokens))
        prefix_scale = math.log1p(prefix_length) / math.log1p(100.0)
        top1_probability = float(gru_probs[0]) if gru_probs else float(row["top1_prob"])
        top2_probability = float(gru_probs[1]) if len(gru_probs) > 1 else 0.0
        margin = top1_probability - top2_probability
        previous = state_tokens[-1] if state_tokens else ""
        for candidate in union:
            g_rank = gru_rank.get(candidate, 0)
            l_rank = llm_rank.get(candidate, 0)
            g_rr = 1.0 / g_rank if g_rank else 0.0
            l_rr = 1.0 / l_rank if l_rank else 0.0
            features = [
                float(gru_prob.get(candidate, 0.0)),
                g_rr,
                l_rr,
                float(candidate in gru_rank and candidate in llm_rank),
                float(g_rank == 1),
                float(l_rank == 1),
                l_rr * (1.0 - top1_probability),
                g_rr * top1_probability,
                l_rr * prefix_scale,
                g_rr * margin,
                state_counts.get(candidate, 0) / prefix_length,
                float(bool(state_tokens) and candidate == previous),
                priors.transition_log_probability(previous, candidate),
                priors.unigram_log_probability(candidate),
            ]
            record: dict[str, object] = {
                "sample_idx": int(sample_idx),
                "candidate": candidate,
                "is_correct": candidate == str(row["true_label"]).upper(),
                "gru_rank": g_rank,
                "llm_rank": l_rank,
                "gru_probability": float(gru_prob.get(candidate, 0.0)),
            }
            record.update(dict(zip(FEATURE_NAMES, features)))
            records.append(record)
    return pd.DataFrame.from_records(records)


def _top_candidate(candidate_rows: pd.DataFrame, score_column: str) -> pd.DataFrame:
    ranked = candidate_rows.copy()
    ranked["_gru_tiebreak"] = ranked["gru_rank"].where(ranked["gru_rank"] > 0, np.inf)
    ranked["_llm_tiebreak"] = ranked["llm_rank"].where(ranked["llm_rank"] > 0, np.inf)
    ordered = ranked.sort_values(
        ["sample_idx", score_column, "_gru_tiebreak", "_llm_tiebreak", "candidate"],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    return (
        ordered.groupby("sample_idx", sort=False)
        .first()
        .reset_index()
        .drop(columns=["_gru_tiebreak", "_llm_tiebreak"])
    )


def crossfit_pairwise_ranker(
    aligned: pd.DataFrame,
    candidate_table: pd.DataFrame,
    n_splits: int = 5,
    l2: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Generate out-of-fold predictions over root-disjoint folds."""

    folds = assign_group_folds(aligned["sequence_id"], n_splits=n_splits)
    feature_columns = FEATURE_NAMES
    predictions: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []
    for fold in range(n_splits):
        test_samples = np.flatnonzero(folds == fold)
        train_samples = np.flatnonzero(folds != fold)
        train_roots = set(aligned.iloc[train_samples]["root"])
        test_roots = set(aligned.iloc[test_samples]["root"])
        overlap = train_roots & test_roots
        if overlap:
            raise AssertionError(f"root leakage in fold {fold}: {sorted(overlap)}")
        train_candidates = candidate_table[candidate_table["sample_idx"].isin(train_samples)].copy()
        test_candidates = candidate_table[candidate_table["sample_idx"].isin(test_samples)].copy()
        x_train_raw = train_candidates[feature_columns].to_numpy(dtype=float)
        mean = x_train_raw.mean(axis=0)
        scale = x_train_raw.std(axis=0)
        scale[scale < 1e-9] = 1.0
        x_train = (x_train_raw - mean) / scale
        weights = fit_pairwise_ranker(
            x_train,
            train_candidates["sample_idx"].to_numpy(),
            train_candidates["is_correct"].to_numpy(),
            l2=l2,
        )
        x_test = (test_candidates[feature_columns].to_numpy(dtype=float) - mean) / scale
        test_candidates["pairwise_score"] = x_test @ weights
        test_candidates["rrf_score"] = (
            test_candidates["gru_rank"].map(lambda rank: 1.0 / rank if rank else 0.0)
            + test_candidates["llm_rank"].map(lambda rank: 1.0 / rank if rank else 0.0)
        )
        pairwise_top = _top_candidate(test_candidates, "pairwise_score")
        rrf_top = _top_candidate(test_candidates, "rrf_score")

        margins = []
        for _, group in test_candidates.groupby("sample_idx", sort=False):
            scores = np.sort(group["pairwise_score"].to_numpy(dtype=float))[::-1]
            margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
            margins.append((int(group["sample_idx"].iloc[0]), float(1.0 / (1.0 + math.exp(-margin)))))
        confidence = pd.DataFrame(margins, columns=["sample_idx", "pairwise_confidence"])
        fold_prediction = pairwise_top[["sample_idx", "candidate", "pairwise_score"]].rename(
            columns={"candidate": "pairwise_pred"}
        )
        fold_prediction = fold_prediction.merge(
            rrf_top[["sample_idx", "candidate"]].rename(columns={"candidate": "rrf_pred"}),
            on="sample_idx",
            validate="one_to_one",
        ).merge(confidence, on="sample_idx", validate="one_to_one")
        fold_prediction["fold"] = fold
        predictions.append(fold_prediction)
        fold_audit.append(
            {
                "fold": fold,
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "train_roots": len(train_roots),
                "test_roots": len(test_roots),
                "root_overlap": len(overlap),
                "train_candidate_rows": len(train_candidates),
                "train_samples_with_true_candidate": int(
                    train_candidates.groupby("sample_idx")["is_correct"].any().sum()
                ),
                "weight_l2_norm": float(np.linalg.norm(weights)),
            }
        )
    oof = pd.concat(predictions, ignore_index=True).sort_values("sample_idx")
    if oof["sample_idx"].duplicated().any() or len(oof) != len(aligned):
        raise AssertionError("OOF predictions are not one-to-one with aligned samples")
    return oof.reset_index(drop=True), fold_audit


def _bootstrap_group_metric(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = float(metric(frame))
    groups = sorted(frame["root"].unique())
    group_frames = {group: frame[frame["root"] == group] for group in groups}
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat([group_frames[group] for group in sampled], ignore_index=True)
        values[idx] = metric(replicate)
    low, high = np.percentile(values, [2.5, 97.5])
    return estimate, float(low), float(high)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_external_ctid_selective(
    ctid_dir: Path,
    n_splits: int,
    n_boot: int,
    seed: int,
    target_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Stress-test GRU confidence on independent CTID organizations."""

    files = sorted(ctid_dir.glob("rl_*_predictions_top5.csv"))
    if not files:
        raise ValueError(f"no CTID prediction files found in {ctid_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"org_name", "true_label", "top1_pred", "top1_prob"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing CTID columns: {sorted(missing)}")
        frame["source_file"] = path.name
        frames.append(frame)
    external = pd.concat(frames, ignore_index=True)
    external["root"] = external["org_name"].astype(str)
    external["correct"] = external["top1_pred"] == external["true_label"]
    folds = assign_group_folds(external["org_name"].astype(str), n_splits=n_splits)
    accepted, thresholds = crossfit_quantile_acceptance(
        external["top1_prob"].to_numpy(dtype=float),
        folds,
        target_coverage=target_coverage,
    )
    external["fold"] = folds
    external["crossfit_accept_80"] = accepted
    external["crossfit_threshold_80"] = thresholds
    base_accuracy = float(external["correct"].mean())
    accepted_accuracy = float(external.loc[accepted, "correct"].mean())
    gain = accepted_accuracy - base_accuracy

    def external_selective_gain(frame: pd.DataFrame) -> float:
        selected = frame["crossfit_accept_80"]
        if not bool(selected.any()):
            return float("nan")
        return float(frame.loc[selected, "correct"].mean() - frame["correct"].mean())

    estimate, low, high = _bootstrap_group_metric(
        external,
        external_selective_gain,
        n_boot=n_boot,
        seed=seed,
    )
    summary = pd.DataFrame(
        [
            {"metric": "ctid_gru_top1_accuracy", "value": base_accuracy},
            {
                "metric": "ctid_crossfit_selective_actual_coverage",
                "value": float(accepted.mean()),
            },
            {"metric": "ctid_crossfit_selective_accuracy", "value": accepted_accuracy},
            {"metric": "ctid_crossfit_selective_accuracy_gain", "value": gain},
            {"metric": "ctid_selective_gain_ci95_low", "value": low},
            {"metric": "ctid_selective_gain_ci95_high", "value": high},
        ]
    )
    risk = risk_coverage(
        external["correct"].to_numpy(),
        external["top1_prob"].to_numpy(dtype=float),
        [1.0, 0.9, 0.8, 0.7, 0.5, 0.3],
    )
    risk["confidence_scope"] = "raw_GRU_confidence_external_descriptive_curve"
    audit = {
        "status": "exploratory_post_hoc_external_stress_test",
        "files": len(files),
        "input_hashes": {str(path): _sha256(path) for path in files},
        "rows": len(external),
        "organizations": int(external["org_name"].nunique()),
        "fold_org_overlap": 0,
        "base_accuracy": base_accuracy,
        "target_coverage": target_coverage,
        "actual_coverage": float(accepted.mean()),
        "accepted_accuracy": accepted_accuracy,
        "selective_gain": estimate,
        "selective_gain_ci95_low": low,
        "selective_gain_ci95_high": high,
        "inference_note": "Only 10 organization clusters; percentile interval is exploratory and coarse.",
    }
    return external, risk, summary, audit


def run_experiment(
    gru_path: Path,
    llm_path: Path,
    train_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 20260729,
    l2: float = 1.0,
    ctid_dir: Path | None = None,
) -> dict[str, object]:
    """Run the complete diagnostic and persist reproducible artifacts."""

    gru = pd.read_csv(gru_path)
    llm = pd.read_csv(llm_path)
    train = pd.read_csv(train_path)
    aligned = align_predictions(gru, llm)
    test_roots = set(aligned["root"])
    priors = build_strict_transition_priors(train, excluded_roots=test_roots)
    candidate_table = build_candidate_table(aligned, priors)
    oof, fold_audit = crossfit_pairwise_ranker(
        aligned, candidate_table, n_splits=n_splits, l2=l2
    )

    results = aligned.copy()
    results["sample_idx"] = np.arange(len(results))
    results = results.merge(oof, on="sample_idx", validate="one_to_one")
    results["gru_correct"] = results["top1_pred"] == results["true_label"]
    results["llm_top1_pred"] = results["llm_candidates"].map(
        lambda candidates: candidates[0] if candidates else ""
    )
    results["llm_correct"] = results["llm_top1_pred"] == results["true_label"]
    results["rrf_correct"] = results["rrf_pred"] == results["true_label"]
    results["pairwise_correct"] = results["pairwise_pred"] == results["true_label"]
    results["gru_top5_hit"] = results.apply(
        lambda row: row["true_label"] in row["gru_candidates"], axis=1
    )
    results["llm_top5_hit"] = results.apply(
        lambda row: row["true_label"] in row["llm_candidates"], axis=1
    )
    results["union_top5_hit"] = results["gru_top5_hit"] | results["llm_top5_hit"]
    results["expert_top1_oracle"] = results["gru_correct"] | results["llm_correct"]
    gru_accept_80, gru_threshold_80 = crossfit_quantile_acceptance(
        results["top1_prob"].to_numpy(dtype=float),
        results["fold"].to_numpy(),
        target_coverage=0.8,
    )
    results["gru_crossfit_accept_80"] = gru_accept_80
    results["gru_crossfit_threshold_80"] = gru_threshold_80
    pairwise_rescues = int((~results["gru_correct"] & results["pairwise_correct"]).sum())
    pairwise_harms = int((results["gru_correct"] & ~results["pairwise_correct"]).sum())

    metrics = {
        "gru_top1_accuracy": float(results["gru_correct"].mean()),
        "llm_top1_accuracy": float(results["llm_correct"].mean()),
        "equal_rrf_top1_accuracy": float(results["rrf_correct"].mean()),
        "pairwise_ranker_oof_top1_accuracy": float(results["pairwise_correct"].mean()),
        "top1_expert_oracle_accuracy": float(results["expert_top1_oracle"].mean()),
        "gru_top5_coverage": float(results["gru_top5_hit"].mean()),
        "llm_top5_coverage": float(results["llm_top5_hit"].mean()),
        "union_top5_oracle_coverage": float(results["union_top5_hit"].mean()),
        "llm_parse_success_rate": float(results["llm_parse_ok"].mean()),
        "gru_crossfit_selective_actual_coverage": float(results["gru_crossfit_accept_80"].mean()),
        "gru_crossfit_selective_accuracy": float(
            results.loc[results["gru_crossfit_accept_80"], "gru_correct"].mean()
        ),
    }
    accepted_n = int(results["gru_crossfit_accept_80"].sum())
    metric_denominators = {"gru_crossfit_selective_accuracy": accepted_n}
    metric_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
                "n": metric_denominators.get(metric, len(results)),
            }
            for metric, value in metrics.items()
        ]
    )

    coverages = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
    risk_tables = []
    for model, correct_col, confidence_col in [
        ("gru", "gru_correct", "top1_prob"),
        ("pairwise_ranker", "pairwise_correct", "pairwise_confidence"),
    ]:
        table = risk_coverage(
            results[correct_col].to_numpy(),
            results[confidence_col].to_numpy(dtype=float),
            coverages,
        )
        table.insert(0, "model", model)
        table["confidence_scope"] = (
            "raw_GRU_confidence_descriptive_curve"
            if model == "gru"
            else "uncalibrated_cross_fold_pairwise_margin_descriptive_only"
        )
        risk_tables.append(table)
    risk_table = pd.concat(risk_tables, ignore_index=True)

    def complementarity(frame: pd.DataFrame) -> float:
        return float(frame["union_top5_hit"].mean() - frame["gru_top5_hit"].mean())

    def routing_gain(frame: pd.DataFrame) -> float:
        return float(frame["pairwise_correct"].mean() - frame["gru_correct"].mean())

    def selective_gain(frame: pd.DataFrame) -> float:
        full = float(frame["gru_correct"].mean())
        accepted = frame["gru_crossfit_accept_80"]
        if not bool(accepted.any()):
            return float("nan")
        selective = float(frame.loc[accepted, "gru_correct"].mean())
        return float(selective - full)

    bootstrap_rows = []
    for offset, (name, metric) in enumerate(
        [
            ("union_top5_minus_gru_top5", complementarity),
            ("pairwise_top1_minus_gru_top1", routing_gain),
            ("gru_crossfit_selective_accuracy_gain_at_80pct_target_coverage", selective_gain),
        ]
    ):
        estimate, low, high = _bootstrap_group_metric(
            results, metric, n_boot=n_boot, seed=seed + offset
        )
        bootstrap_rows.append(
            {
                "metric": name,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_repetitions": n_boot,
                "resampling_unit": "SIM_root",
                "interval_type": "conditional_cluster_percentile_fixed_predictions_and_acceptance",
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    interval = bootstrap.set_index("metric")
    gates = pd.DataFrame(
        [
            {
                "gate": "A_complementarity",
                "threshold": ">=0.005 and CI95_low>0",
                "estimate": interval.loc["union_top5_minus_gru_top5", "estimate"],
                "ci95_low": interval.loc["union_top5_minus_gru_top5", "ci95_low"],
                "passed": bool(
                    interval.loc["union_top5_minus_gru_top5", "estimate"] >= 0.005
                    and interval.loc["union_top5_minus_gru_top5", "ci95_low"] > 0
                ),
            },
            {
                "gate": "B_usable_pairwise_routing",
                "threshold": ">=0.010 and CI95_low>0",
                "estimate": interval.loc["pairwise_top1_minus_gru_top1", "estimate"],
                "ci95_low": interval.loc["pairwise_top1_minus_gru_top1", "ci95_low"],
                "passed": bool(
                    interval.loc["pairwise_top1_minus_gru_top1", "estimate"] >= 0.010
                    and interval.loc["pairwise_top1_minus_gru_top1", "ci95_low"] > 0
                ),
            },
            {
                "gate": "C_selective_signal",
                "threshold": ">=0.050 and CI95_low>0",
                "estimate": interval.loc[
                    "gru_crossfit_selective_accuracy_gain_at_80pct_target_coverage", "estimate"
                ],
                "ci95_low": interval.loc[
                    "gru_crossfit_selective_accuracy_gain_at_80pct_target_coverage", "ci95_low"
                ],
                "passed": bool(
                    interval.loc[
                        "gru_crossfit_selective_accuracy_gain_at_80pct_target_coverage", "estimate"
                    ]
                    >= 0.050
                    and interval.loc[
                        "gru_crossfit_selective_accuracy_gain_at_80pct_target_coverage", "ci95_low"
                    ]
                    > 0
                ),
            },
        ]
    )
    gates["scope"] = "internal_root_contaminated_artifact_diagnostic"
    gates["publication_valid"] = False
    gates["evidence_for"] = gates["gate"].map(
        {
            "A_complementarity": "candidate_union_oracle_complementarity",
            "B_usable_pairwise_routing": "pairwise_routing",
            "C_selective_signal": "baseline_GRU_selectivity_not_pairwise_routing",
        }
    )

    external_audit: dict[str, object] | None = None
    if ctid_dir is not None:
        external, external_risk, external_summary, external_audit = evaluate_external_ctid_selective(
            ctid_dir=ctid_dir,
            n_splits=n_splits,
            n_boot=n_boot,
            seed=seed + 100,
            target_coverage=0.8,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = results[
        [
            "sample_idx",
            "sequence_id",
            "root",
            "prefix_len",
            "state",
            "true_label",
            "fold",
            "top1_pred",
            "top1_prob",
            "llm_top1_pred",
            "rrf_pred",
            "pairwise_pred",
            "pairwise_score",
            "pairwise_confidence",
            "gru_correct",
            "llm_correct",
            "rrf_correct",
            "pairwise_correct",
            "gru_top5_hit",
            "llm_top5_hit",
            "union_top5_hit",
            "gru_crossfit_accept_80",
            "gru_crossfit_threshold_80",
        ]
    ]
    serializable.to_csv(output_dir / "aligned_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig")
    risk_table.to_csv(output_dir / "risk_coverage.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "bootstrap_intervals.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(output_dir / "gates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_audit).to_csv(output_dir / "fold_audit.csv", index=False, encoding="utf-8-sig")
    if ctid_dir is not None and external_audit is not None:
        external.to_csv(output_dir / "ctid_external_predictions.csv", index=False, encoding="utf-8-sig")
        external_risk.to_csv(
            output_dir / "ctid_external_risk_coverage.csv", index=False, encoding="utf-8-sig"
        )
        external_summary.to_csv(
            output_dir / "ctid_external_selective_summary.csv", index=False, encoding="utf-8-sig"
        )

    generated_output_paths = [
        output_dir / "aligned_oof_predictions.csv",
        output_dir / "metric_summary.csv",
        output_dir / "risk_coverage.csv",
        output_dir / "bootstrap_intervals.csv",
        output_dir / "gates.csv",
        output_dir / "fold_audit.csv",
    ]
    if ctid_dir is not None and external_audit is not None:
        generated_output_paths.extend(
            [
                output_dir / "ctid_external_predictions.csv",
                output_dir / "ctid_external_risk_coverage.csv",
                output_dir / "ctid_external_selective_summary.csv",
            ]
        )

    original_train_roots = set(train["sequence_id"].map(sim_root))
    train_roots_after_exclusion = set(
        train.loc[~train["sequence_id"].map(sim_root).isin(test_roots), "sequence_id"].map(sim_root)
    )
    manifest = {
        "status": "diagnostic_root_contaminated_base_model_not_final_test",
        "inputs": {
            str(gru_path): _sha256(gru_path),
            str(llm_path): _sha256(llm_path),
            str(train_path): _sha256(train_path),
        },
        "parameters": {
            "n_splits": n_splits,
            "n_boot": n_boot,
            "seed": seed,
            "l2": l2,
            "features": FEATURE_NAMES,
            "gru_path": str(gru_path),
            "llm_path": str(llm_path),
            "train_path": str(train_path),
            "output_dir": str(output_dir),
            "ctid_dir": str(ctid_dir) if ctid_dir is not None else None,
            "planned_deviations": [
                "A sample-constant GRU entropy feature was omitted because it cancels in pairwise candidate differences; no post-hoc entropy interaction was added."
            ],
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "alignment": {
            "gru_rows": len(gru),
            "llm_rows": len(llm),
            "aligned_rows": len(aligned),
            "state_mismatches": int((~aligned["state_match"]).sum()),
            "duplicate_alignment_keys": 0,
        },
        "leakage_checks": {
            "diagnostic_roots": len(test_roots),
            "strict_prior_train_rows_retained": priors.retained_rows,
            "overlapping_prior_train_rows_excluded": priors.excluded_rows,
            "prior_train_roots_after_exclusion": len(train_roots_after_exclusion),
            "prior_train_vs_diagnostic_root_overlap": len(train_roots_after_exclusion & test_roots),
            "max_fold_root_overlap": max(item["root_overlap"] for item in fold_audit),
            "selective_threshold_calibration": "other_root-disjoint_folds_only",
            "base_gru_train_vs_diagnostic_root_overlap": len(original_train_roots & test_roots),
            "base_gru_train_vs_diagnostic_root_overlap_fraction": len(
                original_train_roots & test_roots
            )
            / len(test_roots),
            "base_gru_training_rows_from_diagnostic_roots": int(
                train["sequence_id"].map(sim_root).isin(test_roots).sum()
            ),
            "root_ood_publication_valid": False,
        },
        "gates": gates.to_dict("records"),
        "paired_comparison": {
            "pairwise_rescues_gru_errors": pairwise_rescues,
            "pairwise_harms_gru_correct": pairwise_harms,
        },
        "external_ctid_audit": external_audit,
        "output_hashes": {str(path): _sha256(path) for path in generated_output_paths},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "metrics": metrics,
        "bootstrap": bootstrap.to_dict("records"),
        "gates": gates.to_dict("records"),
        "manifest": manifest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--ctid-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment(
        gru_path=args.gru,
        llm_path=args.llm,
        train_path=args.train,
        output_dir=args.output_dir,
        n_splits=args.folds,
        n_boot=args.bootstrap,
        seed=args.seed,
        l2=args.l2,
        ctid_dir=args.ctid_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
