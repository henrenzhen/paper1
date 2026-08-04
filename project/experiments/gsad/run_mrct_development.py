from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_canonical_json, write_manifest
from .attack_dag import AttackDAG
from .data_protocol import EXTERNAL_SIM_ROOTS, TEST_ROOTS, sim_root
from .metrics import Interval, cluster_bootstrap_difference, root_macro_mean
from .multires_context_tree import MultiResolutionContextTree
from .rank_conformal import stable_rank_matrix
from .run_development import (
    _root_macro_mrr,
    _select_probability_model,
    assign_balanced_root_folds,
    json_key,
    make_inner_roles,
)
from .run_racer_development import _root_top1


@dataclass(frozen=True)
class MRCTConfig:
    seed: int = 20260730
    bootstrap: int = 2000
    n_splits: int = 5
    alpha: float = 0.1
    parent_contexts: tuple[int, ...] = (1, 2, 3)
    raw_contexts: tuple[int, ...] = (0, 1, 2, 3)
    backoff_strengths: tuple[float, ...] = (1.0, 5.0, 20.0)
    raw_backoff_strengths: tuple[float, ...] = (1.0, 5.0, 20.0)

    def __post_init__(self) -> None:
        if self.bootstrap < 1 or self.n_splits < 2:
            raise ValueError("bootstrap must be positive and n_splits at least two")
        if not self.parent_contexts or min(self.parent_contexts) < 0:
            raise ValueError("parent context grid must be nonnegative")
        if not self.raw_contexts or min(self.raw_contexts) < 0:
            raise ValueError("raw context grid must be nonnegative")
        if min(self.backoff_strengths) < 0 or min(self.raw_backoff_strengths) < 0:
            raise ValueError("backoff strengths must be nonnegative")


def _parse_prefix(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split("||") if part.strip())


def _validate_development_cache(cache_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "development-only cache and its manifest are both required"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("development cache manifest is invalid JSON") from exc
    if manifest.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("development cache hash does not match manifest")
    if manifest.get("excluded_locked_roots") != sorted(TEST_ROOTS):
        raise ValueError("development cache manifest has the wrong locked-root boundary")
    if manifest.get("excluded_external_roots") != sorted(EXTERNAL_SIM_ROOTS):
        raise ValueError("development cache manifest has the wrong external-root boundary")
    if manifest.get("retained_rows") != 10555 or manifest.get("retained_roots") != 133:
        raise ValueError("development cache manifest has unexpected retained counts")
    return manifest


def load_multires_development(
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], AttackDAG]:
    root_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    core = root_path / "data_v2" / "core"
    source_columns = (
        "sequence_id",
        "prefix_len",
        "prefix_technique_ids",
        "prefix_technique_ids_parent",
        "next_technique_id_parent",
        "next_technique_id",
    )
    cache_path = core / "sim_development_multires_min3.csv"
    _validate_development_cache(
        cache_path, core / "sim_development_multires_min3.manifest.json"
    )
    combined = pd.read_csv(cache_path, usecols=list(source_columns))
    combined["sequence_id"] = combined["sequence_id"].astype(str)
    combined["prefix_len"] = combined["prefix_len"].astype(int)
    combined["root"] = combined["sequence_id"].map(sim_root)
    excluded = set(EXTERNAL_SIM_ROOTS) | set(TEST_ROOTS)
    combined = combined.loc[~combined["root"].isin(excluded)].copy()
    combined["prefix_ids"] = combined["prefix_technique_ids_parent"].map(
        _parse_prefix
    )
    combined["raw_prefix_ids"] = combined["prefix_technique_ids"].map(_parse_prefix)
    combined["target"] = (
        combined["next_technique_id_parent"].astype(str).str.strip()
    )
    combined["evaluation_next_raw_id"] = (
        combined["next_technique_id"].astype(str).str.strip()
    )
    if combined.duplicated(["sequence_id", "prefix_len"]).any():
        raise ValueError("duplicate sequence/prefix keys in multi-resolution data")
    for column in ("prefix_ids", "raw_prefix_ids"):
        if (combined[column].map(len) != combined["prefix_len"]).any():
            raise ValueError(f"{column} does not match prefix_len")
    combined = combined.sort_values(["root", "sequence_id", "prefix_len"]).reset_index(
        drop=True
    )
    vocab = tuple(
        pd.read_csv(core / "rl_label_vocab.csv")
        .sort_values("label_id")["technique_id_parent"]
        .astype(str)
    )
    dag = AttackDAG.from_stix(
        root_path / "data" / "enterprise-attack-18.1.json", vocab
    )
    return combined, vocab, dag


def select_mrct_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    vocab: Sequence[str],
    config: MRCTConfig,
) -> tuple[MultiResolutionContextTree, dict[str, Any], np.ndarray]:
    scored: list[tuple[Any, ...]] = []
    for parent_context in config.parent_contexts:
        for raw_context in config.raw_contexts:
            for backoff in config.backoff_strengths:
                for raw_backoff in config.raw_backoff_strengths:
                    model = MultiResolutionContextTree(
                        vocab=vocab,
                        max_parent_context=int(parent_context),
                        max_raw_context=int(raw_context),
                        alpha=config.alpha,
                        backoff_strength=float(backoff),
                        raw_backoff_strength=float(raw_backoff),
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
                        "backoff_strength": float(backoff),
                        "raw_backoff_strength": float(raw_backoff),
                        "alpha": float(config.alpha),
                    }
                    scored.append(
                        (-mrr, -top1, json_key(selected), model, selected, probabilities)
                    )
    _, _, _, model, selected, probabilities = min(scored, key=lambda row: row[:3])
    return model, selected, probabilities


def _prediction_columns(
    probabilities: np.ndarray, targets: Sequence[str], vocab: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = stable_rank_matrix(probabilities)
    label_to_index = {str(label): index for index, label in enumerate(vocab)}
    target_ranks = np.asarray(
        [
            ranks[row, label_to_index[str(target)]]
            if str(target) in label_to_index
            else len(vocab) + 1
            for row, target in enumerate(targets)
        ],
        dtype=int,
    )
    top_indices = np.argmin(ranks, axis=1)
    top_labels = np.asarray(vocab, dtype=object)[top_indices]
    return top_labels, 1.0 / target_ranks, target_ranks <= 5


def _evaluate_fold(
    inner_fit: pd.DataFrame,
    validation: pd.DataFrame,
    outer: pd.DataFrame,
    vocab: Sequence[str],
    dag: AttackDAG,
    config: MRCTConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline, baseline_config, _, _ = _select_probability_model(
        inner_fit, validation, vocab, dag
    )
    candidate, candidate_config, _ = select_mrct_model(
        inner_fit, validation, vocab, config
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(outer["prefix_ids"])
    candidate_probabilities, meta = candidate.predict_proba_with_meta(
        outer["prefix_ids"], outer["raw_prefix_ids"]
    )
    parent_only = MultiResolutionContextTree(
        vocab=vocab,
        max_parent_context=int(candidate_config["max_parent_context"]),
        max_raw_context=0,
        alpha=float(candidate_config["alpha"]),
        backoff_strength=float(candidate_config["backoff_strength"]),
        raw_backoff_strength=float(candidate_config["raw_backoff_strength"]),
    ).fit(
        inner_fit["prefix_ids"],
        inner_fit["raw_prefix_ids"],
        inner_fit["target"],
        inner_fit["root"],
    )
    parent_probabilities, _ = parent_only.predict_proba_with_meta(
        outer["prefix_ids"], outer["raw_prefix_ids"]
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
    is_self = np.asarray(
        [bool(prefix) and str(prefix[-1]) == target for prefix, target in zip(outer["prefix_ids"], targets, strict=True)]
    )
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
            "evaluation_next_raw_id": outer["evaluation_next_raw_id"].astype(str).to_numpy(),
            "last_raw_id": outer["raw_prefix_ids"].map(
                lambda prefix: str(prefix[-1]) if prefix else ""
            ).to_numpy(),
        }
    )
    predictions = pd.concat([predictions, meta.reset_index(drop=True)], axis=1)
    return predictions, {"baseline": baseline_config, "mrct": candidate_config}


def _gain(frame: pd.DataFrame, candidate: str, baseline: str) -> float:
    work = frame.copy()
    work["gain"] = work[candidate].astype(float) - work[baseline].astype(float)
    return root_macro_mean(work, "gain")


def summarize(
    predictions: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, Interval], dict[str, bool]]:
    nonself = predictions.loc[~predictions["is_self"].astype(bool)]
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
        "mrr": metrics["mrr_gain"] >= 0.015 and intervals["mrr_gain"].lower > 0,
        "hit5": metrics["hit5_gain"] >= -0.005,
        "nonself": metrics["nonself_top1_gain"] > 0 and metrics["nonself_mrr_gain"] > 0,
    }
    gates["PRIMARY"] = all(gates.values())
    return metrics, intervals, gates


def run_development(
    config: MRCTConfig,
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
        prediction, selected = _evaluate_fold(
            fit, validation, outer, vocab, dag, config
        )
        prediction["fold"] = fold_id
        outputs.append(prediction)
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
            inputs={"development_rows": len(frame), "development_roots": frame["root"].nunique()},
            config=asdict(config),
            split_audit={"test_roots_in_predictions": sorted(set(predictions["root"]) & set(TEST_ROOTS))},
        )
    return predictions, metrics, intervals, gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MRCTConfig(seed=args.seed, bootstrap=args.bootstrap)
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / f"mrct_seed{config.seed}"
    )
    _, metrics, _, gates = run_development(
        config=config, output_dir=output_dir, project_root=project_root
    )
    print(json.dumps({"primary_passed": gates["PRIMARY"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
