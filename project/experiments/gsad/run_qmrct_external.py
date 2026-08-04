from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifacts import write_canonical_json, write_manifest
from .data_protocol import normalize_ctid_actor
from .metrics import cluster_bootstrap_difference, root_macro_mean
from .multires_context_tree import QuotientMultiResolutionContextTree
from .probability_models import InterpolatedNGram
from .run_mrct_development import _gain, _prediction_columns, load_multires_development


PLAN_NAMES = (
    "apt29",
    "carbanak",
    "fin6",
    "fin7",
    "menu_pass",
    "oilrig",
    "sandworm",
    "turla_carbon",
    "turla_snake",
    "wizard_spider",
)


def _json_list(value: object) -> list[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON list")
    return [str(item).strip() for item in parsed]


def _raw_for_parent(raw_ids: Sequence[str], parent: str) -> str:
    matches = [raw for raw in raw_ids if raw.split(".")[0] == str(parent)]
    if not matches:
        raise ValueError(f"no raw technique maps to parent {parent}")
    return sorted(matches)[0]


def load_ctid_external(project_root: Path | None = None) -> pd.DataFrame:
    root_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    parsed = root_path / "data_v2" / "external_ctid" / "parsed"
    outputs: list[pd.DataFrame] = []
    for plan in PLAN_NAMES:
        evaluation = pd.read_csv(parsed / f"ctid_eval_parent_{plan}_in184.csv")
        steps = pd.read_csv(parsed / f"ctid_steps_long_{plan}.csv")
        raw_by_step = {
            str(row.step_id): _json_list(row.attack_technique_ids_raw)
            for row in steps.itertuples(index=False)
        }
        parent_prefixes: list[tuple[str, ...]] = []
        raw_prefixes: list[tuple[str, ...]] = []
        target_raw_ids: list[str] = []
        for row in evaluation.itertuples(index=False):
            parents = tuple(_json_list(row.prefix_technique_ids_parent))
            step_ids = _json_list(row.prefix_source_step_ids)
            if len(parents) != len(step_ids):
                raise ValueError("CTID prefix parents and source steps are misaligned")
            raw = tuple(
                _raw_for_parent(raw_by_step[step_id], parent)
                for step_id, parent in zip(step_ids, parents, strict=True)
            )
            target_parent = str(row.next_technique_id_parent).strip()
            target_raw = _raw_for_parent(
                raw_by_step[str(row.target_source_step_id)], target_parent
            )
            parent_prefixes.append(parents)
            raw_prefixes.append(raw)
            target_raw_ids.append(target_raw)
        work = evaluation.copy()
        work["prefix_ids"] = parent_prefixes
        work["raw_prefix_ids"] = raw_prefixes
        work["target"] = work["next_technique_id_parent"].astype(str).str.strip()
        work["evaluation_next_raw_id"] = target_raw_ids
        work["actor"] = work["org_name"].map(normalize_ctid_actor)
        outputs.append(work)
    frame = pd.concat(outputs, ignore_index=True)
    if frame.duplicated("sample_id").any():
        raise ValueError("duplicate CTID sample IDs")
    for column in ("prefix_ids", "raw_prefix_ids"):
        if (frame[column].map(len) != frame["prefix_len"].astype(int)).any():
            raise ValueError(f"CTID {column} does not match prefix_len")
    return frame.sort_values(["actor", "org_name", "sample_id"]).reset_index(drop=True)


def run_external(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    bootstrap: int = 2000,
    seed: int = 20260730,
) -> tuple[pd.DataFrame, dict[str, float]]:
    root_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    train, vocab, _ = load_multires_development(root_path)
    external = load_ctid_external(root_path)

    # Frozen from the five development folds before external labels are scored:
    # median QMR-CT configuration and modal strong-baseline configuration.
    qmrct_config = {
        "max_parent_context": 2,
        "max_raw_context": 2,
        "alpha": 0.1,
        "parent_kappa": 5.0,
        "raw_kappa": 10.0,
    }
    qmrct = QuotientMultiResolutionContextTree(vocab=vocab, **qmrct_config).fit(
        train["prefix_ids"],
        train["raw_prefix_ids"],
        train["target"],
        train["root"],
    )
    baseline_config = {
        "order": 3,
        "alpha": 0.1,
        "interpolation": (0.4, 0.3, 0.3),
    }
    baseline = InterpolatedNGram(vocab=vocab, **baseline_config).fit(
        train["prefix_ids"], train["target"], train["root"]
    )
    baseline_probabilities, _ = baseline.predict_proba_with_meta(
        external["prefix_ids"]
    )
    candidate_probabilities, meta = qmrct.predict_proba_with_meta(
        external["prefix_ids"], external["raw_prefix_ids"]
    )
    parent_probabilities, _ = qmrct.parent_tree.predict_proba_with_meta(
        external["prefix_ids"]
    )
    targets = external["target"].astype(str).to_numpy()
    baseline_top, baseline_rr, baseline_hit5 = _prediction_columns(
        baseline_probabilities, targets, vocab
    )
    candidate_top, candidate_rr, candidate_hit5 = _prediction_columns(
        candidate_probabilities, targets, vocab
    )
    parent_top, parent_rr, parent_hit5 = _prediction_columns(
        parent_probabilities, targets, vocab
    )
    predictions = pd.DataFrame(
        {
            "sample_id": external["sample_id"].astype(str).to_numpy(),
            "org_name": external["org_name"].astype(str).to_numpy(),
            "actor": external["actor"].astype(str).to_numpy(),
            "target": targets,
            "baseline_top1": baseline_top,
            "parent_only_top1": parent_top,
            "candidate_top1": candidate_top,
            "baseline_correct": baseline_top == targets,
            "parent_only_correct": parent_top == targets,
            "candidate_correct": candidate_top == targets,
            "baseline_rr": baseline_rr,
            "parent_only_rr": parent_rr,
            "candidate_rr": candidate_rr,
            "baseline_hit5": baseline_hit5,
            "parent_only_hit5": parent_hit5,
            "candidate_hit5": candidate_hit5,
        }
    )
    predictions = pd.concat([predictions, meta.reset_index(drop=True)], axis=1)
    metrics = {
        "baseline_actor_macro_top1": root_macro_mean(
            predictions, "baseline_correct", "actor"
        ),
        "candidate_actor_macro_top1": root_macro_mean(
            predictions, "candidate_correct", "actor"
        ),
        "top1_gain_pp": 100.0
        * _gain(
            predictions.rename(columns={"actor": "root"}),
            "candidate_correct",
            "baseline_correct",
        ),
        "baseline_actor_macro_mrr": root_macro_mean(
            predictions, "baseline_rr", "actor"
        ),
        "candidate_actor_macro_mrr": root_macro_mean(
            predictions, "candidate_rr", "actor"
        ),
        "mrr_gain": _gain(
            predictions.rename(columns={"actor": "root"}),
            "candidate_rr",
            "baseline_rr",
        ),
        "raw_increment_top1": _gain(
            predictions.rename(columns={"actor": "root"}),
            "candidate_correct",
            "parent_only_correct",
        ),
        "raw_increment_mrr": _gain(
            predictions.rename(columns={"actor": "root"}),
            "candidate_rr",
            "parent_only_rr",
        ),
        "rows": float(len(predictions)),
        "actors": float(predictions["actor"].nunique()),
    }
    bootstrap_frame = predictions.rename(columns={"actor": "root"})
    intervals = {
        "top1_gain_pp": cluster_bootstrap_difference(
            bootstrap_frame,
            lambda frame: 100.0
            * _gain(frame, "candidate_correct", "baseline_correct"),
            "root",
            bootstrap,
            seed,
        ),
        "mrr_gain": cluster_bootstrap_difference(
            bootstrap_frame,
            lambda frame: _gain(frame, "candidate_rr", "baseline_rr"),
            "root",
            bootstrap,
            seed + 1,
        ),
    }
    actor_metrics = (
        predictions.assign(
            top1_gain=predictions["candidate_correct"].astype(float)
            - predictions["baseline_correct"].astype(float),
            mrr_gain=predictions["candidate_rr"] - predictions["baseline_rr"],
        )
        .groupby("actor", sort=True)
        .agg(
            rows=("sample_id", "size"),
            baseline_top1=("baseline_correct", "mean"),
            candidate_top1=("candidate_correct", "mean"),
            top1_gain=("top1_gain", "mean"),
            baseline_mrr=("baseline_rr", "mean"),
            candidate_mrr=("candidate_rr", "mean"),
            mrr_gain=("mrr_gain", "mean"),
        )
        .reset_index()
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [{"metric": name, **asdict(interval)} for name, interval in intervals.items()]
        ).to_csv(destination / "bootstrap_intervals.csv", index=False)
        actor_metrics.to_csv(destination / "actor_metrics.csv", index=False)
        write_canonical_json(
            destination / "frozen_config.json",
            {"qmrct": qmrct_config, "baseline": baseline_config},
        )
        write_manifest(
            destination / "run_manifest.json",
            inputs={"train_rows": len(train), "external_rows": len(external)},
            config={"seed": seed, "bootstrap": bootstrap},
            split_audit={
                "train_roots": train["root"].nunique(),
                "external_actors": external["actor"].nunique(),
            },
        )
    return predictions, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    destination = args.output_dir or (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "external"
        / f"qmrct_ctid_seed{args.seed}"
    )
    _, metrics = run_external(
        project_root=project_root,
        output_dir=destination,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
