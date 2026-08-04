from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
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
from .tie_prior import ReportCooccurrencePrior, support_adaptive_prior_pool


FROZEN_QMR_CONFIGS = (
    {
        "max_parent_context": 2,
        "max_raw_context": 2,
        "parent_kappa": 10.0,
        "raw_kappa": 10.0,
        "alpha": 0.1,
    },
    {
        "max_parent_context": 1,
        "max_raw_context": 2,
        "parent_kappa": 5.0,
        "raw_kappa": 10.0,
        "alpha": 0.1,
    },
    {
        "max_parent_context": 2,
        "max_raw_context": 2,
        "parent_kappa": 0.5,
        "raw_kappa": 10.0,
        "alpha": 0.1,
    },
    {
        "max_parent_context": 2,
        "max_raw_context": 1,
        "parent_kappa": 5.0,
        "raw_kappa": 5.0,
        "alpha": 0.1,
    },
    {
        "max_parent_context": 1,
        "max_raw_context": 2,
        "parent_kappa": 0.5,
        "raw_kappa": 10.0,
        "alpha": 0.1,
    },
)


@dataclass(frozen=True)
class EchoConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    strengths: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    kappas: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0)


def _local_support(meta: pd.DataFrame) -> np.ndarray:
    return np.where(
        meta["raw_used_order"].to_numpy(dtype=int) > 0,
        meta["raw_root_support"].to_numpy(dtype=float),
        meta["parent_root_support"].to_numpy(dtype=float),
    )


def select_prior_pool(
    local_probabilities: np.ndarray,
    prior_probabilities: np.ndarray,
    local_support: np.ndarray,
    prior_available: np.ndarray,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    strengths: Sequence[float],
    kappas: Sequence[float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    scored: list[tuple[Any, ...]] = []
    for strength in strengths:
        for kappa in kappas:
            pooled, weights = support_adaptive_prior_pool(
                local_probabilities,
                prior_probabilities,
                local_support,
                prior_available,
                strength=float(strength),
                kappa=float(kappa),
            )
            mrr = _root_macro_mrr(
                pooled, validation["target"], validation["root"], vocab
            )
            top1 = _root_top1(
                pooled, validation["target"], validation["root"], vocab
            )
            selected = {"strength": float(strength), "kappa": float(kappa)}
            scored.append((-mrr, -top1, json_key(selected), selected, pooled, weights))
    _, _, _, selected, pooled, weights = min(scored, key=lambda row: row[:3])
    return selected, pooled, weights


def _evaluate_fold(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: object,
    prior: ReportCooccurrencePrior,
    config: EchoConfig,
    fold_id: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, baseline_config, _, _ = _select_probability_model(
        fit, validation, vocab, dag
    )
    qmr_config = FROZEN_QMR_CONFIGS[fold_id]
    qmr = QuotientMultiResolutionContextTree(vocab=vocab, **qmr_config).fit(
        fit["prefix_ids"], fit["raw_prefix_ids"], fit["target"], fit["root"]
    )
    validation_local, validation_meta = qmr.predict_proba_with_meta(
        validation["prefix_ids"], validation["raw_prefix_ids"]
    )
    validation_prior, validation_prior_meta = prior.predict_proba_with_meta(
        validation["prefix_ids"]
    )
    pool_config, _, _ = select_prior_pool(
        local_probabilities=validation_local,
        prior_probabilities=validation_prior,
        local_support=_local_support(validation_meta),
        prior_available=validation_prior_meta["prior_available"].to_numpy(dtype=bool),
        validation=validation,
        vocab=vocab,
        strengths=config.strengths,
        kappas=config.kappas,
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    local_probabilities, local_meta = qmr.predict_proba_with_meta(
        outer["prefix_ids"], outer["raw_prefix_ids"]
    )
    prior_probabilities, prior_meta = prior.predict_proba_with_meta(outer["prefix_ids"])
    echo_probabilities, echo_weights = support_adaptive_prior_pool(
        local_probabilities,
        prior_probabilities,
        _local_support(local_meta),
        prior_meta["prior_available"].to_numpy(dtype=bool),
        strength=pool_config["strength"],
        kappa=pool_config["kappa"],
    )
    targets = outer["target"].astype(str).to_numpy()
    baseline_top, baseline_rr, baseline_hit5 = _prediction_columns(
        baseline_probabilities, targets, vocab
    )
    local_top, local_rr, local_hit5 = _prediction_columns(
        local_probabilities, targets, vocab
    )
    echo_top, echo_rr, echo_hit5 = _prediction_columns(
        echo_probabilities, targets, vocab
    )
    predictions = pd.DataFrame(
        {
            "sequence_id": outer["sequence_id"].astype(str).to_numpy(),
            "prefix_len": outer["prefix_len"].astype(int).to_numpy(),
            "root": outer["root"].astype(str).to_numpy(),
            "target": targets,
            "baseline_top1": baseline_top,
            "local_top1": local_top,
            "echo_top1": echo_top,
            "baseline_correct": baseline_top == targets,
            "local_correct": local_top == targets,
            "echo_correct": echo_top == targets,
            "baseline_rr": baseline_rr,
            "local_rr": local_rr,
            "echo_rr": echo_rr,
            "baseline_hit5": baseline_hit5,
            "local_hit5": local_hit5,
            "echo_hit5": echo_hit5,
            "is_self": [
                bool(prefix) and str(prefix[-1]) == target
                for prefix, target in zip(outer["prefix_ids"], targets, strict=True)
            ],
            "echo_prior_weight": echo_weights,
            "tie_report_support": prior_meta["report_support"].to_numpy(dtype=int),
        }
    )
    return pd.concat([predictions, local_meta.reset_index(drop=True)], axis=1), {
        "baseline": baseline_config,
        "qmr": qmr_config,
        "echo_pool": pool_config,
    }


def summarize(
    predictions: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, Interval], dict[str, bool]]:
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
    metrics = {
        "baseline_top1": root_macro_mean(predictions, "baseline_correct"),
        "echo_top1": root_macro_mean(predictions, "echo_correct"),
        "top1_gain_pp": 100.0 * _gain(predictions, "echo_correct", "baseline_correct"),
        "baseline_mrr": root_macro_mean(predictions, "baseline_rr"),
        "echo_mrr": root_macro_mean(predictions, "echo_rr"),
        "mrr_gain": _gain(predictions, "echo_rr", "baseline_rr"),
        "hit5_gain": _gain(predictions, "echo_hit5", "baseline_hit5"),
        "nonself_top1_gain": _gain(nonself, "echo_correct", "baseline_correct"),
        "nonself_mrr_gain": _gain(nonself, "echo_rr", "baseline_rr"),
        "prior_increment_top1": _gain(predictions, "echo_correct", "local_correct"),
        "prior_increment_mrr": _gain(predictions, "echo_rr", "local_rr"),
        "mean_prior_weight": float(predictions["echo_prior_weight"].mean()),
        "rows": float(len(predictions)),
        "roots": float(predictions["root"].nunique()),
    }
    intervals = {
        "top1_gain_pp": cluster_bootstrap_difference(
            predictions,
            lambda frame: 100.0 * _gain(frame, "echo_correct", "baseline_correct"),
            "root",
            n_boot,
            seed,
        ),
        "mrr_gain": cluster_bootstrap_difference(
            predictions,
            lambda frame: _gain(frame, "echo_rr", "baseline_rr"),
            "root",
            n_boot,
            seed + 1,
        ),
        "prior_increment_mrr": cluster_bootstrap_difference(
            predictions,
            lambda frame: _gain(frame, "echo_rr", "local_rr"),
            "root",
            n_boot,
            seed + 2,
        ),
    }
    fold_gains = predictions.assign(
        fold_gain=predictions["echo_correct"].astype(float)
        - predictions["baseline_correct"].astype(float)
    ).groupby("fold")["fold_gain"].mean()
    gates = {
        "top1": metrics["top1_gain_pp"] >= 2.0 and intervals["top1_gain_pp"].lower > 0,
        "mrr": metrics["mrr_gain"] >= 0.01 and intervals["mrr_gain"].lower > 0,
        "hit5": metrics["hit5_gain"] >= -0.005,
        "nonself": metrics["nonself_top1_gain"] >= 0.01 and metrics["nonself_mrr_gain"] > 0,
        "fold_stability": int((fold_gains > 0).sum()) >= 4,
        "prior_increment": metrics["prior_increment_mrr"] > 0,
    }
    gates["PRIMARY"] = all(gates.values())
    return metrics, intervals, gates


def run_development(
    config: EchoConfig,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Interval], dict[str, bool]]:
    root_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    frame, vocab, dag = load_multires_development(root_path)
    tie_path = root_path / "data_v2" / "external_tie" / "raw" / "combined_dataset_parent_only.json"
    prior = ReportCooccurrencePrior(vocab, alpha=0.1).fit_json(tie_path)
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
            fit, validation, outer, vocab, dag, prior, config, fold_id
        )
        predictions["fold"] = fold_id
        outputs.append(predictions)
        configs.append(selected)
    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != len(frame) or set(predictions["root"]) & set(TEST_ROOTS):
        raise AssertionError("OOF coverage or locked-root exclusion failed")
    metrics, intervals, gates = summarize(predictions, config.bootstrap, config.seed)
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
        write_manifest(
            destination / "run_manifest.json",
            inputs={
                "tie_sha256": sha256_file(tie_path),
                "development_rows": len(frame),
                "development_roots": frame["root"].nunique(),
            },
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
    config = EchoConfig(seed=args.seed, bootstrap=args.bootstrap)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / f"echo_qmr_seed{args.seed}"
    )
    _, metrics, _, gates = run_development(config, destination, project_root)
    print(json.dumps({"primary_passed": gates["PRIMARY"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
