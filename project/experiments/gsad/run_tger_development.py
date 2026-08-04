from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import write_canonical_json, write_manifest
from .data_protocol import TEST_ROOTS
from .metrics import Interval, cluster_bootstrap_difference, root_macro_mean
from .multirelation_graph_residual import (
    TypedGraphEscapeRedistributor,
    build_attack_relation_matrices,
)
from .run_development import (
    _select_probability_model,
    assign_balanced_root_folds,
    json_key,
    make_inner_roles,
)
from .run_mrct_development import _gain, _prediction_columns, load_multires_development


@dataclass(frozen=True)
class TGERConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    semantic_neighbors: int = 5
    max_context: int = 3
    relation_weights: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.75, 0.25, 0.0),
        (0.75, 0.0, 0.25),
        (0.5, 0.25, 0.25),
    )
    graph_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)

    def __post_init__(self) -> None:
        if self.bootstrap < 1 or self.n_splits < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if self.semantic_neighbors < 1 or self.max_context < 1:
            raise ValueError("neighbors and max_context must be positive")
        if not self.relation_weights or not self.graph_fractions:
            raise ValueError("selection grids must be nonempty")
        for weights in self.relation_weights:
            if len(weights) != 3 or min(weights) < 0 or not np.isclose(sum(weights), 1.0):
                raise ValueError("relation weights must be triples summing to one")
        if min(self.graph_fractions) < 0 or max(self.graph_fractions) > 1:
            raise ValueError("graph fractions must be in [0, 1]")


def _selection_score(
    probabilities: np.ndarray,
    validation: pd.DataFrame,
    vocab: Sequence[str],
) -> tuple[float, float, float]:
    targets = validation["target"].astype(str).to_numpy()
    top, reciprocal_rank, _ = _prediction_columns(probabilities, targets, vocab)
    work = pd.DataFrame(
        {
            "root": validation["root"].astype(str).to_numpy(),
            "correct": top == targets,
            "rr": reciprocal_rank,
            "is_self": [
                bool(prefix) and str(prefix[-1]) == target
                for prefix, target in zip(validation["prefix_ids"], targets, strict=True)
            ],
        }
    )
    nonself = work.loc[~work["is_self"]]
    return (
        root_macro_mean(work, "rr"),
        root_macro_mean(work, "correct"),
        root_macro_mean(nonself, "rr"),
    )


def select_tger_model(
    model: TypedGraphEscapeRedistributor,
    base_probabilities: np.ndarray,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    config: TGERConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    scored: list[tuple[Any, ...]] = []
    for weights in config.relation_weights:
        for fraction in config.graph_fractions:
            probabilities, _ = model.predict_proba_with_meta(
                base_probabilities,
                validation["prefix_ids"],
                weights,
                fraction,
            )
            mrr, top1, nonself_mrr = _selection_score(
                probabilities, validation, vocab
            )
            selected = {
                "relation_weights": [float(value) for value in weights],
                "graph_fraction": float(fraction),
            }
            scored.append(
                (-mrr, -top1, -nonself_mrr, json_key(selected), selected, probabilities)
            )
    _, _, _, _, selected, probabilities = min(scored, key=lambda row: row[:4])
    return selected, probabilities


def _evaluate_fold(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: object,
    semantic: np.ndarray,
    tactic: np.ndarray,
    config: TGERConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, baseline_config, _, _ = _select_probability_model(
        fit, validation, vocab, dag
    )
    model = TypedGraphEscapeRedistributor(
        vocab,
        semantic,
        tactic,
        max_context=config.max_context,
    ).fit(fit["prefix_ids"], fit["target"], fit["root"])
    validation_base, _ = baseline.predict_proba_with_meta(validation["prefix_ids"])
    selected, _ = select_tger_model(
        model, validation_base, validation, vocab, config
    )
    base, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    candidate, meta = model.predict_proba_with_meta(
        base,
        outer["prefix_ids"],
        tuple(selected["relation_weights"]),
        selected["graph_fraction"],
    )
    transition_only, _ = model.predict_proba_with_meta(
        base,
        outer["prefix_ids"],
        (1.0, 0.0, 0.0),
        selected["graph_fraction"],
    )
    targets = outer["target"].astype(str).to_numpy()
    base_top, base_rr, base_hit5 = _prediction_columns(base, targets, vocab)
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(candidate, targets, vocab)
    transition_top, transition_rr, _ = _prediction_columns(transition_only, targets, vocab)
    is_self = np.asarray(
        [
            bool(prefix) and str(prefix[-1]) == target
            for prefix, target in zip(outer["prefix_ids"], targets, strict=True)
        ],
        dtype=bool,
    )
    target_seen = np.asarray(
        [
            model.is_seen_successor(prefix, target)
            for prefix, target in zip(outer["prefix_ids"], targets, strict=True)
        ],
        dtype=bool,
    )
    predictions = pd.DataFrame(
        {
            "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
            "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
            "root": outer["root"].astype(str).to_numpy(),
            "target": targets,
            "baseline_top1": base_top,
            "candidate_top1": candidate_top,
            "transition_only_top1": transition_top,
            "baseline_correct": base_top == targets,
            "candidate_correct": candidate_top == targets,
            "transition_only_correct": transition_top == targets,
            "baseline_rr": base_rr,
            "candidate_rr": candidate_rr,
            "transition_only_rr": transition_rr,
            "baseline_hit5": base_hit5,
            "candidate_hit5": candidate_hit5,
            "is_self": is_self,
            "target_seen": target_seen,
        }
    )
    return pd.concat([predictions, meta.reset_index(drop=True)], axis=1), {
        "baseline": baseline_config,
        "tger": selected,
    }


def summarize_tger(
    predictions: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, Interval], dict[str, bool]]:
    predictions = predictions.copy()
    unseen = predictions.loc[~predictions["target_seen"].astype(bool)]
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
    fold_gains = [
        _gain(frame, "candidate_correct", "baseline_correct")
        for _, frame in predictions.groupby("fold", sort=True)
    ]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "candidate_top1": root_macro_mean(predictions, "candidate_correct"),
        "top1_gain_pp": 100.0 * _gain(predictions, "candidate_correct", "baseline_correct"),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "unseen_top1_gain_pp": 100.0 * _gain(unseen, "candidate_correct", "baseline_correct"),
        "unseen_mrr_gain": _gain(unseen, "candidate_rr", "baseline_rr"),
        "nonself_top1_gain_pp": 100.0 * _gain(nonself, "candidate_correct", "baseline_correct"),
        "nonself_mrr_gain": _gain(nonself, "candidate_rr", "baseline_rr"),
        "static_increment_top1_pp": 100.0 * _gain(
            unseen, "candidate_correct", "transition_only_correct"
        ),
        "static_increment_mrr": _gain(
            unseen, "candidate_rr", "transition_only_rr"
        ),
        "unseen_target_rate": float((~predictions["target_seen"].astype(bool)).mean()),
        "affected_row_rate": float((predictions["graph_shift_l1"] > 1e-12).mean()),
        "mean_escape_mass": float(predictions["escape_mass"].mean()),
        "positive_fold_count": float(sum(value > 0 for value in fold_gains)),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    specs = {
        "top1_gain_pp": (predictions, "candidate_correct", "baseline_correct", 100.0),
        "mrr_gain": (predictions, "candidate_rr", "baseline_rr", 1.0),
        "unseen_top1_gain_pp": (unseen, "candidate_correct", "baseline_correct", 100.0),
        "unseen_mrr_gain": (unseen, "candidate_rr", "baseline_rr", 1.0),
        "nonself_top1_gain_pp": (nonself, "candidate_correct", "baseline_correct", 100.0),
        "static_increment_top1_pp": (
            unseen,
            "candidate_correct",
            "transition_only_correct",
            100.0,
        ),
        "static_increment_mrr": (
            unseen,
            "candidate_rr",
            "transition_only_rr",
            1.0,
        ),
    }
    intervals: dict[str, Interval] = {}
    for offset, (name, (frame, candidate, baseline, scale)) in enumerate(specs.items()):
        intervals[name] = cluster_bootstrap_difference(
            frame,
            lambda sample, c=candidate, b=baseline, s=scale: s * _gain(sample, c, b),
            "root",
            n_boot,
            seed + offset,
        )
    gates = {
        "overall_top1": metrics["top1_gain_pp"] >= 2.0
        and intervals["top1_gain_pp"].lower > 0,
        "overall_mrr": metrics["mrr_gain"] >= 0.01
        and intervals["mrr_gain"].lower > 0,
        "hit5": metrics["hit5_gain"] >= -0.005,
        "unseen_mechanism": metrics["unseen_top1_gain_pp"] >= 5.0
        and intervals["unseen_top1_gain_pp"].lower > 0
        and metrics["unseen_mrr_gain"] >= 0.02
        and intervals["unseen_mrr_gain"].lower > 0,
        "nonself": metrics["nonself_top1_gain_pp"] >= 1.0
        and intervals["nonself_top1_gain_pp"].lower > 0
        and metrics["nonself_mrr_gain"] > 0,
        "fold_direction": metrics["positive_fold_count"] >= 4,
        "static_relation_increment": (
            metrics["static_increment_top1_pp"] >= 0.75
            and intervals["static_increment_top1_pp"].lower > 0
        )
        or (
            metrics["static_increment_mrr"] >= 0.005
            and intervals["static_increment_mrr"].lower > 0
        ),
    }
    gates["PRIMARY"] = all(gates.values())
    return metrics, intervals, gates


def run_development(
    config: TGERConfig,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Interval], dict[str, bool]]:
    root_path = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    frame, vocab, dag = load_multires_development(root_path)
    semantic, tactic, relation_audit = build_attack_relation_matrices(
        root_path / "data" / "enterprise-attack-18.1.json", vocab, config.semantic_neighbors
    )
    assignments = assign_balanced_root_folds(frame, config.n_splits)
    outputs: list[pd.DataFrame] = []
    configs: list[dict[str, Any]] = []
    for fold_id in range(config.n_splits):
        outer = frame.loc[assignments == fold_id].reset_index(drop=True)
        outer_training = frame.loc[assignments != fold_id].reset_index(drop=True)
        roles = make_inner_roles(outer_training, 20, 20)
        fit = outer_training.loc[outer_training["root"].isin(roles.fit_roots)].reset_index(drop=True)
        validation = outer_training.loc[
            outer_training["root"].isin(roles.validation_roots)
        ].reset_index(drop=True)
        predictions, selected = _evaluate_fold(
            fit, validation, outer, vocab, dag, semantic, tactic, config
        )
        predictions["fold"] = fold_id
        outputs.append(predictions)
        configs.append(selected)
    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != len(frame) or set(predictions["root"]) & set(TEST_ROOTS):
        raise AssertionError("OOF coverage or locked-root exclusion failed")
    metrics, intervals, gates = summarize_tger(predictions, config.bootstrap, config.seed)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [{"metric": name, **asdict(interval)} for name, interval in intervals.items()]
        ).to_csv(destination / "bootstrap_intervals.csv", index=False)
        write_canonical_json(destination / "gates.json", gates)
        write_canonical_json(destination / "model_configs.json", configs)
        write_canonical_json(destination / "relation_audit.json", relation_audit)
        write_manifest(
            destination / "run_manifest.json",
            inputs={"development_rows": len(predictions), "development_roots": predictions["root"].nunique()},
            config=asdict(config),
            split_audit={"test_roots_in_predictions": sorted(set(predictions["root"]) & set(TEST_ROOTS))},
        )
    return predictions, metrics, intervals, gates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = TGERConfig(seed=args.seed, bootstrap=args.bootstrap)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root / "experiments" / "gsad" / "results" / "development" / f"tger_seed{config.seed}"
    )
    _, metrics, _, gates = run_development(config, destination, project_root)
    print(json.dumps({"primary_passed": gates["PRIMARY"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
