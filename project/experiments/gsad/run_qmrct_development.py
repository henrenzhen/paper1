from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import write_canonical_json, write_manifest
from .data_protocol import TEST_ROOTS
from .metrics import Interval, cluster_bootstrap_difference, root_macro_mean
from .multires_context_tree import QuotientMultiResolutionContextTree
from .run_development import (
    _root_macro_mrr,
    _select_probability_model,
    assign_balanced_root_folds,
    json_key,
    make_inner_roles,
)
from .run_mrct_development import (
    _gain,
    _prediction_columns,
    load_multires_development,
)
from .run_racer_development import _root_top1


@dataclass(frozen=True)
class QMRCTConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    alpha: float = 0.1
    parent_contexts: tuple[int, ...] = (1, 2, 3)
    raw_contexts: tuple[int, ...] = (1, 2)
    parent_kappas: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0, 20.0)
    raw_kappas: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0, 20.0)

    def __post_init__(self) -> None:
        if self.bootstrap < 1 or self.n_splits < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if not self.parent_contexts or min(self.parent_contexts) < 1:
            raise ValueError("parent contexts must be positive")
        if not self.raw_contexts or min(self.raw_contexts) < 0:
            raise ValueError("raw contexts must be nonnegative")
        if min(self.parent_kappas) < 0 or min(self.raw_kappas) < 0:
            raise ValueError("kappas must be nonnegative")


def select_qmrct_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    config: QMRCTConfig,
) -> tuple[QuotientMultiResolutionContextTree, dict[str, Any], object]:
    scored: list[tuple[Any, ...]] = []
    for parent_context in config.parent_contexts:
        for raw_context in config.raw_contexts:
            for parent_kappa in config.parent_kappas:
                for raw_kappa in config.raw_kappas:
                    model = QuotientMultiResolutionContextTree(
                        vocab=vocab,
                        max_parent_context=int(parent_context),
                        max_raw_context=int(raw_context),
                        alpha=config.alpha,
                        parent_kappa=float(parent_kappa),
                        raw_kappa=float(raw_kappa),
                    ).fit(
                        fit["prefix_ids"],
                        fit["raw_prefix_ids"],
                        fit["target"],
                        fit["root"],
                    )
                    probabilities, _ = model.predict_proba_with_meta(
                        validation["prefix_ids"], validation["raw_prefix_ids"]
                    )
                    mrr = _root_macro_mrr(
                        probabilities,
                        validation["target"],
                        validation["root"],
                        vocab,
                    )
                    top1 = _root_top1(
                        probabilities,
                        validation["target"],
                        validation["root"],
                        vocab,
                    )
                    selected = {
                        "max_parent_context": int(parent_context),
                        "max_raw_context": int(raw_context),
                        "parent_kappa": float(parent_kappa),
                        "raw_kappa": float(raw_kappa),
                        "alpha": float(config.alpha),
                    }
                    scored.append(
                        (-mrr, -top1, json_key(selected), model, selected, probabilities)
                    )
    _, _, _, model, selected, probabilities = min(scored, key=lambda row: row[:3])
    return model, selected, probabilities


def _evaluate_fold(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: object,
    config: QMRCTConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, baseline_config, _, _ = _select_probability_model(
        fit, validation, vocab, dag
    )
    candidate, candidate_config, _ = select_qmrct_model(
        fit, validation, vocab, config
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    candidate_probabilities, meta = candidate.predict_proba_with_meta(
        outer["prefix_ids"], outer["raw_prefix_ids"]
    )
    parent_probabilities, _ = candidate.parent_tree.predict_proba_with_meta(
        outer["prefix_ids"]
    )
    targets = outer["target"].astype(str).to_numpy()
    baseline_top, baseline_rr, baseline_hit5 = _prediction_columns(
        baseline_probabilities, targets, vocab
    )
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(
        candidate_probabilities, targets, vocab
    )
    parent_top, parent_rr, parent_hit5 = _prediction_columns(
        parent_probabilities, targets, vocab
    )
    is_self = [
        bool(prefix) and str(prefix[-1]) == target
        for prefix, target in zip(outer["prefix_ids"], targets, strict=True)
    ]
    raw_changed = [
        bool(raw) and str(raw[-1]) != str(next_raw)
        for raw, next_raw in zip(
            outer["raw_prefix_ids"], outer["evaluation_next_raw_id"], strict=True
        )
    ]
    predictions = pd.DataFrame(
        {
            "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
            "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
            "root": outer["root"].astype(str).to_numpy(),
            "target": targets,
            "baseline_top1": baseline_top,
            "candidate_top1": candidate_top,
            "parent_only_top1": parent_top,
            "baseline_correct": baseline_top == targets,
            "candidate_correct": candidate_top == targets,
            "parent_only_correct": parent_top == targets,
            "baseline_rr": baseline_rr,
            "candidate_rr": candidate_rr,
            "parent_only_rr": parent_rr,
            "baseline_hit5": baseline_hit5,
            "candidate_hit5": candidate_hit5,
            "parent_only_hit5": parent_hit5,
            "is_self": is_self,
            "last_raw_differs_from_parent": raw_changed,
        }
    )
    return pd.concat([predictions, meta.reset_index(drop=True)], axis=1), {
        "baseline": baseline_config,
        "qmrct": candidate_config,
    }


def summarize(
    predictions: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, Interval], dict[str, bool]]:
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
    mechanism = predictions.loc[
        predictions["is_self"].astype(bool)
        & predictions["last_raw_differs_from_parent"].astype(bool)
    ]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "candidate_top1": root_macro_mean(predictions, "candidate_correct"),
        "top1_gain_pp": 100.0 * _gain(predictions, "candidate_correct", "baseline_correct"),
        "baseline_mrr": root_macro_mean(predictions, "baseline_rr"),
        "candidate_mrr": root_macro_mean(predictions, "candidate_rr"),
        "mrr_gain": _gain(predictions, "candidate_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "candidate_hit5", "baseline_hit5"),
        "nonself_top1_gain": _gain(nonself, "candidate_correct", "baseline_correct"),
        "nonself_mrr_gain": _gain(nonself, "candidate_rr", "baseline_rr"),
        "mechanism_top1_gain": _gain(mechanism, "candidate_correct", "baseline_correct"),
        "mechanism_mrr_gain": _gain(mechanism, "candidate_rr", "baseline_rr"),
        "raw_increment_top1": _gain(
            predictions, "candidate_correct", "parent_only_correct"
        ),
        "raw_increment_mrr": _gain(predictions, "candidate_rr", "parent_only_rr"),
        "raw_context_use_rate": float((predictions["raw_used_order"] > 0).mean()),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    intervals = {
        "top1_gain_pp": cluster_bootstrap_difference(
            predictions,
            lambda frame: 100.0 * _gain(frame, "candidate_correct", "baseline_correct"),
            "root",
            n_boot,
            seed,
        ),
        "mrr_gain": cluster_bootstrap_difference(
            predictions,
            lambda frame: _gain(frame, "candidate_rr", "baseline_rr"),
            "root",
            n_boot,
            seed + 1,
        ),
    }
    gates = {
        "top1": metrics["top1_gain_pp"] >= 2.0 and intervals["top1_gain_pp"].lower > 0,
        "mrr": metrics["mrr_gain"] >= 0.01 and intervals["mrr_gain"].lower > 0,
        "hit5": metrics["hit5_gain"] >= -0.005,
        "nonself": metrics["nonself_top1_gain"] >= 0.01 and metrics["nonself_mrr_gain"] > 0,
        "mechanism": metrics["mechanism_top1_gain"] >= 0.03,
        "raw_increment": metrics["raw_increment_top1"] > 0
        and metrics["raw_increment_mrr"] > 0,
    }
    gates["PRIMARY"] = all(gates.values())
    return metrics, intervals, gates


def run_development(
    config: QMRCTConfig,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Interval], dict[str, bool]]:
    frame, vocab, dag = load_multires_development(project_root)
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
            fit, validation, outer, vocab, dag, config
        )
        predictions["fold"] = fold_id
        outputs.append(predictions)
        configs.append(selected)
    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != len(frame) or set(predictions["root"]) & set(TEST_ROOTS):
        raise AssertionError("OOF coverage or locked-root exclusion failed")
    metrics, intervals, gates = summarize(
        predictions, n_boot=config.bootstrap, seed=config.seed
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [{"metric": key, **asdict(value)} for key, value in intervals.items()]
        ).to_csv(destination / "bootstrap_intervals.csv", index=False)
        write_canonical_json(destination / "gates.json", gates)
        write_canonical_json(destination / "model_configs.json", configs)
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
    config = QMRCTConfig(seed=args.seed, bootstrap=args.bootstrap)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / f"qmrct_seed{config.seed}"
    )
    _, metrics, _, gates = run_development(config, destination, project_root)
    print(json.dumps({"primary_passed": gates["PRIMARY"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
