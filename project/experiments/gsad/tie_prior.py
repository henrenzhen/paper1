from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


class ReportCooccurrencePrior:
    """Report-balanced CTI technique co-occurrence prior.

    Frequencies within one CTI report are intentionally ignored: each report
    contributes one unit of association mass for every conditioning technique.
    This prevents a repeatedly mentioned technique in one report from acting
    like many independent reports.
    """

    def __init__(self, vocab: Sequence[str], alpha: float = 0.1) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if not np.isfinite(alpha) or float(alpha) < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.alpha = float(alpha)
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        self.global_prior_: np.ndarray | None = None
        self.conditionals_: dict[str, np.ndarray] = {}
        self.report_support_: dict[str, int] = {}

    def fit_json(self, path: Path) -> "ReportCooccurrencePrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        reports = payload.get("reports") if isinstance(payload, dict) else None
        if not isinstance(reports, list):
            raise ValueError("TIE JSON must contain a reports list")
        return self.fit_reports(reports)

    def fit_reports(
        self, reports: Sequence[Mapping[str, object]]
    ) -> "ReportCooccurrencePrior":
        if not reports:
            raise ValueError("at least one CTI report is required")
        report_label_sets: list[tuple[str, ...]] = []
        global_counts = np.zeros(len(self.vocab), dtype=float)
        for report in reports:
            raw = report.get("mitre_techniques", {})
            if not isinstance(raw, Mapping):
                continue
            labels = tuple(
                sorted({str(label) for label in raw if str(label) in self.label_to_index})
            )
            if not labels:
                continue
            report_label_sets.append(labels)
            for label in labels:
                global_counts[self.label_to_index[label]] += 1.0
        if not report_label_sets:
            raise ValueError("no TIE report contains an in-vocabulary technique")
        global_counts += self.alpha / len(self.vocab)
        self.global_prior_ = global_counts / global_counts.sum()

        accumulators = {
            label: np.zeros(len(self.vocab), dtype=float) for label in self.vocab
        }
        support = {label: 0 for label in self.vocab}
        for labels in report_label_sets:
            for source in labels:
                others = [label for label in labels if label != source]
                if not others:
                    continue
                contribution = 1.0 / len(others)
                for target in others:
                    accumulators[source][self.label_to_index[target]] += contribution
                support[source] += 1
        self.conditionals_ = {}
        self.report_support_ = {}
        for source, count in support.items():
            if count <= 0:
                continue
            probability = (
                accumulators[source] + self.alpha * self.global_prior_
            ) / (float(count) + self.alpha)
            probability /= probability.sum()
            self.conditionals_[source] = probability
            self.report_support_[source] = count
        return self

    def predict_proba_with_meta(
        self, prefixes: Iterable[Sequence[str]]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if self.global_prior_ is None:
            raise RuntimeError("prior is not fitted")
        rows: list[np.ndarray] = []
        metadata: list[dict[str, int | bool]] = []
        for prefix in prefixes:
            clean = tuple(str(item) for item in prefix)
            source = clean[-1] if clean else ""
            conditional = self.conditionals_.get(source)
            available = conditional is not None
            rows.append(
                conditional.copy() if conditional is not None else self.global_prior_.copy()
            )
            metadata.append(
                {
                    "report_support": int(self.report_support_.get(source, 0)),
                    "prior_available": available,
                }
            )
        matrix = np.stack(rows) if rows else np.empty((0, len(self.vocab)))
        return matrix, pd.DataFrame(metadata)


def support_adaptive_prior_pool(
    local_probabilities: np.ndarray,
    prior_probabilities: np.ndarray,
    local_support: np.ndarray,
    prior_available: np.ndarray,
    strength: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(local_probabilities, dtype=float)
    prior = np.asarray(prior_probabilities, dtype=float)
    support = np.asarray(local_support, dtype=float)
    available = np.asarray(prior_available, dtype=bool)
    if local.ndim != 2 or local.shape != prior.shape:
        raise ValueError("local and prior probability matrices must match")
    if support.shape != (len(local),) or available.shape != (len(local),):
        raise ValueError("support and availability must contain one value per row")
    if np.any(~np.isfinite(local)) or np.any(~np.isfinite(prior)):
        raise ValueError("probabilities must be finite")
    if np.any(local < 0) or np.any(prior < 0):
        raise ValueError("probabilities must be nonnegative")
    if not np.allclose(local.sum(axis=1), 1.0) or not np.allclose(
        prior.sum(axis=1), 1.0
    ):
        raise ValueError("probability rows must sum to one")
    if np.any(~np.isfinite(support)) or np.any(support < 0):
        raise ValueError("local support must be finite and nonnegative")
    if not np.isfinite(strength) or not 0 <= float(strength) <= 1:
        raise ValueError("strength must lie in [0, 1]")
    if not np.isfinite(kappa) or float(kappa) <= 0:
        raise ValueError("kappa must be positive and finite")
    weights = float(strength) * float(kappa) / (support + float(kappa))
    weights = np.where(available, weights, 0.0)
    pooled = (1.0 - weights[:, None]) * local + weights[:, None] * prior
    pooled /= pooled.sum(axis=1, keepdims=True)
    return pooled, weights
