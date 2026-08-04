from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifacts import (
    FreezeToken,
    _payload_digest,
    _validate_token,
    claim_locked_run,
    sha256_file,
    write_canonical_json,
    write_manifest,
)
from .attack_dag import AttackDAG
from .data_protocol import FrozenSplit
from .run_racer_development import (
    RacerConfig,
    RacerFoldResult,
    RacerSummary,
    _load_default_experiment,
    evaluate_racer_outer_fold,
    frozen_file_hashes,
    summarize_racer_predictions,
)


@dataclass(frozen=True)
class LockedRacerResult:
    predictions: pd.DataFrame
    summary: RacerSummary
    fold_result: RacerFoldResult
    results_dir: Path


def load_freeze_token(path: Path) -> FreezeToken:
    token_path = Path(path)
    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
        token = FreezeToken(
            digest=str(payload["digest"]),
            config_digest=str(payload["config_digest"]),
            gate_digest=str(payload["gate_digest"]),
            token_path=token_path,
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("freeze token file is missing or invalid") from exc
    _validate_token(token)
    return token


def verify_development_bundle(token: FreezeToken) -> RacerConfig:
    directory = token.token_path.parent
    try:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        gates = json.loads((directory / "gates.json").read_text(encoding="utf-8"))
        config_payload = {
            "candidate": "racer",
            "development_config": manifest["config"],
            "manifest_digest": manifest["manifest_digest"],
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("frozen development bundle is incomplete") from exc
    if _payload_digest(config_payload) != token.config_digest:
        raise ValueError("development config no longer matches the freeze token")
    if _payload_digest(gates) != token.gate_digest:
        raise ValueError("development gates no longer match the freeze token")
    manifest_core = {
        "inputs": manifest["inputs"],
        "config": manifest["config"],
        "split_audit": manifest["split_audit"],
    }
    if _payload_digest(manifest_core) != manifest["manifest_digest"]:
        raise ValueError("development manifest digest is invalid")
    recorded_files = manifest.get("inputs", {}).get("files")
    if recorded_files != frozen_file_hashes():
        raise ValueError("frozen source or input file hash changed")
    if not bool(gates.get("PRIMARY", {}).get("passed", False)):
        raise ValueError("frozen development PRIMARY did not pass")
    return RacerConfig(**manifest["config"])


def run_locked_evaluation(
    config: RacerConfig,
    split: FrozenSplit,
    vocab: Sequence[str],
    dag: AttackDAG,
    token: FreezeToken,
    results_dir: Path,
) -> LockedRacerResult:
    _validate_token(token)
    destination = Path(results_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"locked result directory is not empty: {destination}")
    claim_locked_run(token, destination)
    return _evaluate_claimed_locked_split(
        config=config,
        split=split,
        vocab=vocab,
        dag=dag,
        token=token,
        destination=destination,
    )


def _evaluate_claimed_locked_split(
    config: RacerConfig,
    split: FrozenSplit,
    vocab: Sequence[str],
    dag: AttackDAG,
    token: FreezeToken,
    destination: Path,
) -> LockedRacerResult:
    fold_result = evaluate_racer_outer_fold(
        inner_fit=split.fit,
        validation=split.validation,
        calibration=split.calibration,
        outer=split.test,
        vocab=vocab,
        dag=dag,
        config=config,
        fold_id=999,
    )
    summary = summarize_racer_predictions(
        fold_result.predictions, n_boot=config.bootstrap, seed=config.seed + 50000
    )
    result = LockedRacerResult(
        predictions=fold_result.predictions,
        summary=summary,
        fold_result=fold_result,
        results_dir=destination,
    )
    _write_locked_artifacts(result, config, split, dag, token)
    return result


def run_frozen_locked_evaluation(
    token: FreezeToken, results_dir: Path
) -> LockedRacerResult:
    """Consume the token before hashing or loading any locked input file."""
    _validate_token(token)
    destination = Path(results_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"locked result directory is not empty: {destination}")
    claim_locked_run(token, destination)
    config = verify_development_bundle(token)
    split, vocab, dag = _load_default_experiment()
    return _evaluate_claimed_locked_split(
        config=config,
        split=split,
        vocab=vocab,
        dag=dag,
        token=token,
        destination=destination,
    )


def _write_locked_artifacts(
    result: LockedRacerResult,
    config: RacerConfig,
    split: FrozenSplit,
    dag: AttackDAG,
    token: FreezeToken,
) -> None:
    output_dir = result.results_dir
    serializable = result.predictions.copy()
    for column in ("baseline_set", "racer_set"):
        serializable[column] = serializable[column].map(
            lambda value: " || ".join(sorted(str(item) for item in value))
        )
    serializable.to_csv(
        output_dir / "locked_predictions.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([result.summary.metrics]).to_csv(
        output_dir / "locked_metrics.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"metric": metric, **asdict(interval)}
            for metric, interval in result.summary.intervals.items()
        ]
    ).to_csv(
        output_dir / "locked_bootstrap_intervals.csv", index=False, encoding="utf-8"
    )
    write_canonical_json(
        output_dir / "locked_gates.json",
        {name: asdict(gate) for name, gate in result.summary.gates.items()},
    )
    write_canonical_json(output_dir / "locked_fold_audit.json", result.fold_result.audit)
    write_canonical_json(
        output_dir / "locked_model_config.json", result.fold_result.model_config
    )
    write_canonical_json(
        output_dir / "locked_data_audit.json",
        {
            "freeze_digest": token.digest,
            "partition_roots": {
                "fit": int(split.fit["root"].nunique()),
                "validation": int(split.validation["root"].nunique()),
                "calibration": int(split.calibration["root"].nunique()),
                "test": int(split.test["root"].nunique()),
            },
            "partition_rows": {
                "fit": len(split.fit),
                "validation": len(split.validation),
                "calibration": len(split.calibration),
                "test": len(split.test),
            },
            "test_predictions": len(result.predictions),
            "attack_dag": dag.mapping_audit,
        },
    )
    source_dir = Path(__file__).resolve().parent
    sources = (
        "context_tree.py",
        "rank_conformal.py",
        "opinion_pool.py",
        "run_racer_development.py",
        "run_racer_locked.py",
    )
    write_manifest(
        output_dir / "locked_manifest.json",
        inputs={
            "freeze_digest": token.digest,
            "source_hashes": {
                name: sha256_file(source_dir / name) for name in sources
            },
        },
        config=asdict(config),
        split_audit={"locked_fold": result.fold_result.audit},
    )
    lines = [
        "# RACER 一次性锁定测试摘要",
        "",
        f"- 测试：{len(result.predictions)} 行，{result.predictions['root'].nunique()} 个 roots。",
        f"- PRIMARY：{'通过' if result.summary.gates['PRIMARY'].passed else '失败'}。",
        f"- Top-1 增益：{result.summary.metrics['top1_gain_pp']:.3f} pp。",
        f"- MRR 增益：{result.summary.metrics['mrr_gain']:.5f}。",
        f"- 集合缩减：{100 * result.summary.metrics['set_reduction_relative']:.3f}%。",
        "- 该冻结令牌的锁定评估已经消费，不得重复运行。",
        "",
    ]
    (output_dir / "locked_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    development_dir = (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "development"
        / "racer_op_v2_seed20260730"
    )
    token = load_freeze_token(development_dir / "freeze_token.json")
    locked_dir = (
        project_root
        / "experiments"
        / "gsad"
        / "results"
        / "locked"
        / "racer_op_v2_seed20260730"
    )
    result = run_frozen_locked_evaluation(token=token, results_dir=locked_dir)
    print(
        json.dumps(
            {
                "candidate": "racer_op",
                "primary_passed": result.summary.gates["PRIMARY"].passed,
                "rows": len(result.predictions),
                "roots": int(result.predictions["root"].nunique()),
                "results_dir": str(result.results_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
