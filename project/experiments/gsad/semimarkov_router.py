from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .multirelation_graph_residual import SelfCensoredMultiRelationGraphResidual


_LEVELS = ("parent", "parent_dwell", "parent_dwell_raw")
_DESTINATION_LEVELS = ("parent", "parent_raw", "parent_dwell_raw")


def _dwell_length(prefix: Sequence[str], maximum: int) -> int:
    if not prefix:
        return 0
    current = str(prefix[-1])
    length = 0
    for label in reversed(prefix):
        if str(label) != current:
            break
        length += 1
    return min(length, int(maximum))


class QuotientSemiMarkovRouter:
    """Factor parent persistence from the destination conditional.

    The exit hazard is estimated by equal-weighting roots at several nested
    past-only resolutions.  The destination expert has its self-transition
    removed, so it cannot obtain an advantage by copying the current parent.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        kappa: float = 5.0,
        max_dwell: int = 5,
        hazard_levels: Sequence[str] = _LEVELS,
        destination_kappa: float = 5.0,
        destination_levels: Sequence[str] = _DESTINATION_LEVELS,
    ) -> None:
        self.vocab = tuple(str(label) for label in vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary must be nonempty and unique")
        if not np.isfinite(kappa) or float(kappa) < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if int(max_dwell) < 1:
            raise ValueError("max_dwell must be positive")
        levels = tuple(str(level) for level in hazard_levels)
        if not levels or any(level not in _LEVELS for level in levels):
            raise ValueError(f"hazard levels must come from {_LEVELS}")
        expected = tuple(level for level in _LEVELS if level in levels)
        if levels != expected:
            raise ValueError("hazard levels must preserve the nested canonical order")
        if not np.isfinite(destination_kappa) or float(destination_kappa) < 0:
            raise ValueError("destination_kappa must be finite and nonnegative")
        destination_chain = tuple(str(level) for level in destination_levels)
        destination_expected = tuple(
            level for level in _DESTINATION_LEVELS if level in destination_chain
        )
        if not destination_chain or destination_chain != destination_expected:
            raise ValueError(
                "destination levels must preserve the nested canonical order"
            )
        self.kappa = float(kappa)
        self.max_dwell = int(max_dwell)
        self.hazard_levels = levels
        self.destination_kappa = float(destination_kappa)
        self.destination_levels = destination_chain
        self.label_to_index = {label: index for index, label in enumerate(self.vocab)}
        zeros = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        self.destination_model = SelfCensoredMultiRelationGraphResidual(
            self.vocab, zeros, zeros, transition_kappa=self.destination_kappa
        )
        self.hazard_tables_: dict[
            str, dict[tuple[str | int, ...], tuple[float, int]]
        ] = {}
        self.destination_tables_: dict[
            str, dict[tuple[str | int, ...], tuple[np.ndarray, int]]
        ] = {}
        self.fitted_ = False

    def _keys(
        self, parent_prefix: Sequence[str], raw_prefix: Sequence[str]
    ) -> dict[str, tuple[str | int, ...]]:
        current = str(parent_prefix[-1])
        dwell = _dwell_length(parent_prefix, self.max_dwell)
        raw = str(raw_prefix[-1])
        return {
            "parent": (current,),
            "parent_dwell": (current, dwell),
            "parent_dwell_raw": (current, dwell, raw),
            "parent_raw": (current, raw),
        }

    def fit(
        self,
        parent_prefixes: Sequence[Sequence[str]],
        raw_prefixes: Sequence[Sequence[str]],
        targets: Sequence[str],
        groups: Sequence[str],
        domains: Sequence[str] | None = None,
    ) -> "QuotientSemiMarkovRouter":
        if len(parent_prefixes) == 0 or not (
            len(parent_prefixes) == len(raw_prefixes) == len(targets) == len(groups)
        ):
            raise ValueError("training arrays must be nonempty and aligned")
        if domains is not None and len(domains) != len(parent_prefixes):
            raise ValueError("domains must align with training arrays")
        domain_balanced = domains is not None
        clean_domains = (
            [str(domain) for domain in domains]
            if domains is not None
            else ["__single_domain__"] * len(parent_prefixes)
        )
        cleaned_parent = [tuple(str(label) for label in prefix) for prefix in parent_prefixes]
        cleaned_raw = [tuple(str(label) for label in prefix) for prefix in raw_prefixes]
        if any(
            len(parent) != len(raw)
            for parent, raw in zip(cleaned_parent, cleaned_raw, strict=True)
        ):
            raise ValueError("raw and parent prefixes must align position by position")

        outcomes: dict[
            str, dict[tuple[tuple[str | int, ...], str, str], list[float]]
        ] = {level: defaultdict(list) for level in self.hazard_levels}
        destination_counts: dict[
            str, dict[tuple[tuple[str | int, ...], str, str], np.ndarray]
        ] = {level: {} for level in self.destination_levels}
        for parent, raw, target, group, domain in zip(
            cleaned_parent,
            cleaned_raw,
            targets,
            groups,
            clean_domains,
            strict=True,
        ):
            if not parent:
                continue
            keys = self._keys(parent, raw)
            changed = float(str(target) != str(parent[-1]))
            for level in self.hazard_levels:
                outcomes[level][(keys[level], domain, str(group))].append(changed)
            if changed:
                target_index = self.label_to_index.get(str(target))
                if target_index is None:
                    raise ValueError(f"target outside vocabulary: {target}")
                for level in self.destination_levels:
                    group_key = (keys[level], domain, str(group))
                    vector = destination_counts[level].setdefault(
                        group_key, np.zeros(len(self.vocab), dtype=float)
                    )
                    vector[target_index] += 1.0

        tables: dict[str, dict[tuple[str | int, ...], tuple[float, int]]] = {}
        for level in self.hazard_levels:
            domain_root_means: dict[
                tuple[str | int, ...], dict[str, list[float]]
            ] = defaultdict(lambda: defaultdict(list))
            for (key, domain, _), values in outcomes[level].items():
                domain_root_means[key][domain].append(float(np.mean(values)))
            table: dict[tuple[str | int, ...], tuple[float, int]] = {}
            for key, by_domain in domain_root_means.items():
                domain_means = [float(np.mean(values)) for values in by_domain.values()]
                support = sum(len(values) for values in by_domain.values())
                table[key] = (float(np.mean(domain_means)), support)
            tables[level] = table
        self.hazard_tables_ = tables
        destination_tables: dict[
            str, dict[tuple[str | int, ...], tuple[np.ndarray, int]]
        ] = {}
        for level in self.destination_levels:
            domain_root_rows: dict[
                tuple[str | int, ...], dict[str, list[np.ndarray]]
            ] = defaultdict(lambda: defaultdict(list))
            for (key, domain, _), vector in destination_counts[level].items():
                domain_root_rows[key][domain].append(vector / vector.sum())
            table: dict[tuple[str | int, ...], tuple[np.ndarray, int]] = {}
            for key, by_domain in domain_root_rows.items():
                domain_means = [np.mean(rows, axis=0) for rows in by_domain.values()]
                support = sum(len(rows) for rows in by_domain.values())
                table[key] = (np.mean(domain_means, axis=0), support)
            destination_tables[level] = table
        self.destination_tables_ = destination_tables
        self.destination_model.fit(cleaned_parent, targets, groups)
        self.fitted_ = True
        return self

    def predict_proba_with_meta(
        self,
        base_probabilities: np.ndarray,
        parent_prefixes: Sequence[Sequence[str]],
        raw_prefixes: Sequence[Sequence[str]],
        destination_fraction: float,
        hazard_fraction: float = 1.0,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        if len(parent_prefixes) != len(raw_prefixes):
            raise ValueError("raw and parent prefixes must be aligned")
        cleaned_parent = [tuple(str(label) for label in prefix) for prefix in parent_prefixes]
        cleaned_raw = [tuple(str(label) for label in prefix) for prefix in raw_prefixes]
        if any(
            len(parent) != len(raw)
            for parent, raw in zip(cleaned_parent, cleaned_raw, strict=True)
        ):
            raise ValueError("raw and parent prefixes must align position by position")
        base = np.asarray(base_probabilities, dtype=float)
        if base.shape != (len(cleaned_parent), len(self.vocab)):
            raise ValueError("base probabilities do not align with inputs/vocabulary")
        if not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base probabilities must be finite and nonnegative")
        if not np.allclose(base.sum(axis=1), np.ones(len(base))):
            raise ValueError("base probabilities must be normalized")
        if not 0 <= float(destination_fraction) <= 1:
            raise ValueError("destination_fraction must be in [0, 1]")
        if not 0 <= float(hazard_fraction) <= 1:
            raise ValueError("hazard_fraction must be in [0, 1]")

        output = base.copy()
        metadata: list[dict[str, float | int]] = []
        for row, (parent, raw) in enumerate(
            zip(cleaned_parent, cleaned_raw, strict=True)
        ):
            if not parent:
                metadata.append(
                    {
                        "baseline_exit_hazard": float("nan"),
                        "exit_hazard": float("nan"),
                        "hazard_levels_used": 0,
                        "hazard_max_root_support": 0,
                        "dwell_length": 0,
                    }
                )
                continue
            current_index = self.label_to_index.get(str(parent[-1]))
            if current_index is None:
                metadata.append(
                    {
                        "baseline_exit_hazard": float("nan"),
                        "exit_hazard": float("nan"),
                        "hazard_levels_used": 0,
                        "hazard_max_root_support": 0,
                        "dwell_length": _dwell_length(parent, self.max_dwell),
                    }
                )
                continue
            baseline_hazard = float(1.0 - base[row, current_index])
            hazard = baseline_hazard
            levels_used = 0
            max_support = 0
            keys = self._keys(parent, raw)
            for level in self.hazard_levels:
                estimate = self.hazard_tables_[level].get(keys[level])
                if estimate is None:
                    continue
                empirical, support = estimate
                weight = (
                    1.0 if self.kappa == 0 else support / (support + self.kappa)
                )
                hazard = weight * empirical + (1.0 - weight) * hazard
                levels_used += 1
                max_support = max(max_support, support)
            hazard = (
                float(hazard_fraction) * hazard
                + (1.0 - float(hazard_fraction)) * baseline_hazard
            )
            hazard = float(np.clip(hazard, 1e-8, 1.0 - 1e-8))

            nonself = np.ones(len(self.vocab), dtype=bool)
            nonself[current_index] = False
            base_total = float(base[row, nonself].sum())
            base_destination = (
                base[row, nonself] / base_total
                if base_total > 0
                else np.full(nonself.sum(), 1.0 / nonself.sum())
            )
            destination_full = np.zeros(len(self.vocab), dtype=float)
            destination_full[nonself] = base_destination
            destination_levels_used = 0
            destination_max_support = 0
            for level in self.destination_levels:
                estimate = self.destination_tables_[level].get(keys[level])
                if estimate is None:
                    continue
                empirical, support = estimate
                weight = (
                    1.0
                    if self.destination_kappa == 0
                    else support / (support + self.destination_kappa)
                )
                destination_full = (
                    weight * empirical + (1.0 - weight) * destination_full
                )
                destination_full[current_index] = 0.0
                destination_full /= destination_full.sum()
                destination_levels_used += 1
                destination_max_support = max(destination_max_support, support)
            learned_destination = destination_full[nonself]
            conditional = (
                (1.0 - float(destination_fraction)) * base_destination
                + float(destination_fraction) * learned_destination
            )
            output[row, current_index] = 1.0 - hazard
            output[row, nonself] = hazard * conditional
            output[row] /= output[row].sum()
            metadata.append(
                {
                    "baseline_exit_hazard": baseline_hazard,
                    "exit_hazard": hazard,
                    "hazard_levels_used": levels_used,
                    "hazard_max_root_support": max_support,
                    "dwell_length": _dwell_length(parent, self.max_dwell),
                    "destination_levels_used": destination_levels_used,
                    "destination_max_root_support": destination_max_support,
                }
            )
        return output, pd.DataFrame(metadata)
