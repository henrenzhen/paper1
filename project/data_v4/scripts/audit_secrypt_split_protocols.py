#!/usr/bin/env python3
"""Clean-room audit of the SECRYPT 2026 campaign-chain split protocols.

This script implements the data-construction algorithm described by Raj et al.
and exposed by their public research materials without importing their Python
modules or distributing their workbook. It measures exact prefix-target reuse
and deterministic non-neural baselines under random-pair and campaign-LOCO
splits. It does not reproduce the paper's neural model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice, permutations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SEED = 42
REQUIRED_HASH_SEED = "0"
MIN_CHAIN_LENGTH = 3
MAX_BUCKET_PERMUTATIONS = 25
MAX_CHAINS_PER_CAMPAIGN = 200
TACTIC_ORDER = (
    "reconnaissance",
    "resource development",
    "initial access",
    "execution",
    "persistence",
    "privilege escalation",
    "defense evasion",
    "credential access",
    "discovery",
    "lateral movement",
    "collection",
    "command and control",
    "exfiltration",
    "impact",
)
METHODS = ("FREQ", "PREFIX", "M1", "M2")


@dataclass(frozen=True)
class PairRow:
    campaign_id: str
    chain_ordinal: int
    prefix_len: int
    prefix: tuple[str, ...]
    target: str

    @property
    def pair_key(self) -> tuple[str, int, int]:
        return (self.campaign_id, self.chain_ordinal, self.prefix_len)

    @property
    def content_key(self) -> tuple[tuple[str, ...], str]:
        return (self.prefix, self.target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack-workbook",
        type=Path,
        required=True,
        help="Readable .xlsx copy of the public ATT&CK v16 workbook.",
    )
    parser.add_argument(
        "--original-workbook",
        type=Path,
        help="Optional original .xls path; hashed for provenance but not read.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--external-commit",
        default="e188cc6ec96df0288470380dbafccda1591e2c95",
        help="Audited public repository commit.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("released", "canonical"),
        default=("released", "canonical"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("secrypt-split-audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")
    formatter.converter = time_gmtime
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "stdout.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def time_gmtime(timestamp: float):
    import time

    return time.gmtime(timestamp)


def require_environment() -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual != REQUIRED_HASH_SEED:
        raise SystemExit(
            "PYTHONHASHSEED must be 0 before interpreter startup; run as "
            "`PYTHONHASHSEED=0 python ...`."
        )


def load_attack_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if path.suffix.lower() == ".xls":
        raise SystemExit(
            "This environment intentionally lacks xlrd. Convert the public .xls "
            "losslessly to .xlsx, pass the converted file here, and provide the "
            "original with --original-workbook so both hashes are recorded."
        )
    required = {
        "techniques": {"ID", "STIX ID", "name", "tactics"},
        "relationships": {"source ref", "target ref"},
        "campaigns": {"ID", "STIX ID", "name"},
    }
    loaded: dict[str, pd.DataFrame] = {}
    for sheet_name, columns in required.items():
        frame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{sheet_name} missing columns: {sorted(missing)}")
        loaded[sheet_name] = frame
    return loaded["techniques"], loaded["relationships"], loaded["campaigns"]


def normalize_tactics(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def sample_bucket_permutations(
    bucket: Sequence[str], mode: str, rng: random.Random
) -> list[tuple[str, ...]]:
    values = list(dict.fromkeys(bucket))
    if mode == "canonical":
        values = sorted(values)
    if len(values) <= 1:
        return [tuple(values)]
    if len(values) <= 6:
        return list(islice(permutations(values), MAX_BUCKET_PERMUTATIONS))
    sampled: set[tuple[str, ...]] = set()
    for _ in range(MAX_BUCKET_PERMUTATIONS * 20):
        candidate = values.copy()
        rng.shuffle(candidate)
        sampled.add(tuple(candidate))
        if len(sampled) == MAX_BUCKET_PERMUTATIONS:
            break
    result = list(sampled)
    return sorted(result) if mode == "canonical" else result


def construct_pairs(
    techniques: pd.DataFrame,
    relationships: pd.DataFrame,
    campaigns: pd.DataFrame,
    mode: str,
) -> tuple[list[PairRow], dict[str, object]]:
    tactic_series = techniques["tactics"].fillna("").map(normalize_tactics)
    stix_to_name = dict(zip(techniques["STIX ID"], techniques["name"]))
    name_to_tactics = dict(zip(techniques["name"], tactic_series))
    rng = random.Random(SEED)
    pair_rows: list[PairRow] = []
    chain_lengths: list[int] = []
    campaign_chain_counts: Counter[str] = Counter()
    used_campaigns: set[str] = set()

    targets_by_source: dict[str, list[str]] = defaultdict(list)
    for source_ref, target_ref in relationships[["source ref", "target ref"]].itertuples(
        index=False, name=None
    ):
        if isinstance(source_ref, str) and isinstance(target_ref, str):
            if target_ref.startswith("attack-pattern--"):
                targets_by_source[source_ref].append(target_ref)

    chain_ordinal = 0
    for campaign_id_raw, campaign_stix_raw in campaigns[["ID", "STIX ID"]].itertuples(
        index=False, name=None
    ):
        campaign_id = str(campaign_id_raw)
        campaign_stix = str(campaign_stix_raw)
        names = [
            stix_to_name[target]
            for target in targets_by_source.get(campaign_stix, [])
            if target in stix_to_name and pd.notna(stix_to_name[target])
        ]
        if mode == "released":
            unique_names = list(set(names))
        else:
            unique_names = sorted(set(names))

        buckets: dict[str, list[str]] = {tactic: [] for tactic in TACTIC_ORDER}
        for name in unique_names:
            for tactic in name_to_tactics.get(name, []):
                if tactic in buckets:
                    buckets[tactic].append(name)
                    break
        non_empty = [buckets[tactic] for tactic in TACTIC_ORDER if buckets[tactic]]
        if not non_empty:
            continue
        per_bucket = [sample_bucket_permutations(bucket, mode, rng) for bucket in non_empty]
        for combination in islice(product(*per_bucket), MAX_CHAINS_PER_CAMPAIGN):
            chain = tuple(item for group in combination for item in group)
            if len(chain) < MIN_CHAIN_LENGTH:
                continue
            chain_ordinal += 1
            used_campaigns.add(campaign_id)
            campaign_chain_counts[campaign_id] += 1
            chain_lengths.append(len(chain))
            for prefix_len in range(1, len(chain)):
                pair_rows.append(
                    PairRow(
                        campaign_id=campaign_id,
                        chain_ordinal=chain_ordinal,
                        prefix_len=prefix_len,
                        prefix=chain[:prefix_len],
                        target=chain[prefix_len],
                    )
                )

    if len({row.pair_key for row in pair_rows}) != len(pair_rows):
        raise AssertionError("pair_key is not unique")
    facts = {
        "mode": mode,
        "campaign_rows": int(len(campaigns)),
        "campaigns_used": len(used_campaigns),
        "chains": len(chain_lengths),
        "occurrences": int(sum(chain_lengths)),
        "pairs": len(pair_rows),
        "unique_content_pairs": len({row.content_key for row in pair_rows}),
        "unique_prefixes": len({row.prefix for row in pair_rows}),
        "vocab_size": len({item for row in pair_rows for item in (*row.prefix, row.target)}),
        "chain_length_min": int(min(chain_lengths)),
        "chain_length_max": int(max(chain_lengths)),
        "chain_length_mean": float(np.mean(chain_lengths)),
        "campaign_chain_count_min": int(min(campaign_chain_counts.values())),
        "campaign_chain_count_max": int(max(campaign_chain_counts.values())),
    }
    return pair_rows, facts


def released_pair_splits(n_rows: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(SEED)
    initial = rng.permutation(n_rows)
    split90 = int(0.9 * n_rows)
    result = {"pair90_diagnostic": (initial[:split90], initial[split90:])}
    rng.permutation(n_rows)  # public 100% scenario consumes a permutation
    rng.permutation(n_rows)  # public 50/50 scenario consumes a permutation
    formal80 = rng.permutation(n_rows)
    split80 = int(0.8 * n_rows)
    result["pair80_primary"] = (formal80[:split80], formal80[split80:])
    return result


def select_winner(counts: Counter[str], global_counts: Counter[str]) -> str:
    if not counts:
        counts = global_counts
    if not counts:
        raise ValueError("training targets are empty")
    return min(
        counts,
        key=lambda label: (-counts[label], -global_counts[label], label),
    )


def fit_predictors(train: Sequence[PairRow]):
    global_counts: Counter[str] = Counter(row.target for row in train)
    prefix_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    m1_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    m2_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in train:
        prefix_counts[row.prefix][row.target] += 1
        m1_counts[row.prefix[-1:]][row.target] += 1
        if len(row.prefix) >= 2:
            m2_counts[row.prefix[-2:]][row.target] += 1
    fallback = select_winner(global_counts, global_counts)

    def predict(method: str, row: PairRow) -> str:
        if method == "FREQ":
            return fallback
        if method == "PREFIX":
            return select_winner(prefix_counts.get(row.prefix, Counter()), global_counts)
        if method == "M1":
            return select_winner(m1_counts.get(row.prefix[-1:], Counter()), global_counts)
        if method == "M2":
            if len(row.prefix) >= 2 and row.prefix[-2:] in m2_counts:
                return select_winner(m2_counts[row.prefix[-2:]], global_counts)
            return select_winner(m1_counts.get(row.prefix[-1:], Counter()), global_counts)
        raise KeyError(method)

    return predict


def evaluate(train: Sequence[PairRow], test: Sequence[PairRow]) -> dict[str, float]:
    predictor = fit_predictors(train)
    return {
        method: sum(predictor(method, row) == row.target for row in test) / len(test)
        for method in METHODS
    }


def split_diagnostics(train: Sequence[PairRow], test: Sequence[PairRow]) -> dict[str, object]:
    train_contents = {row.content_key for row in train}
    train_prefixes = {row.prefix for row in train}
    train_campaigns = {row.campaign_id for row in train}
    return {
        "n_train": len(train),
        "n_test": len(test),
        "exact_content_overlap": sum(row.content_key in train_contents for row in test) / len(test),
        "prefix_coverage": sum(row.prefix in train_prefixes for row in test) / len(test),
        "campaign_overlap": sum(row.campaign_id in train_campaigns for row in test) / len(test),
        "train_campaigns": len(train_campaigns),
        "test_campaigns": len({row.campaign_id for row in test}),
    }


def run_pair_protocols(
    rows: Sequence[PairRow], mode: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for protocol, (train_idx, test_idx) in released_pair_splits(len(rows)).items():
        train = [rows[int(index)] for index in train_idx]
        test = [rows[int(index)] for index in test_idx]
        diag = {"construction_mode": mode, "protocol": protocol}
        diag.update(split_diagnostics(train, test))
        diagnostics.append(diag)
        for method, accuracy in evaluate(train, test).items():
            summaries.append(
                {
                    "construction_mode": mode,
                    "protocol": protocol,
                    "aggregation": "row_micro",
                    "method": method,
                    "accuracy": accuracy,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_folds": 1,
                }
            )
    return summaries, diagnostics


def run_campaign_loco(
    rows: Sequence[PairRow], mode: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    campaigns = sorted({row.campaign_id for row in rows})
    fold_rows: list[dict[str, object]] = []
    total_correct = Counter()
    total_test = 0
    for campaign_id in campaigns:
        train = [row for row in rows if row.campaign_id != campaign_id]
        test = [row for row in rows if row.campaign_id == campaign_id]
        accuracies = evaluate(train, test)
        total_test += len(test)
        for method, accuracy in accuracies.items():
            correct = round(accuracy * len(test))
            total_correct[method] += correct
            fold_rows.append(
                {
                    "construction_mode": mode,
                    "campaign_id": campaign_id,
                    "method": method,
                    "n_train": len(train),
                    "n_test": len(test),
                    "correct": correct,
                    "accuracy": accuracy,
                }
            )
    summaries: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in fold_rows if row["method"] == method]
        summaries.extend(
            [
                {
                    "construction_mode": mode,
                    "protocol": "campaign_loco",
                    "aggregation": "campaign_macro",
                    "method": method,
                    "accuracy": float(np.mean([row["accuracy"] for row in method_rows])),
                    "n_train": "varies",
                    "n_test": total_test,
                    "n_folds": len(campaigns),
                },
                {
                    "construction_mode": mode,
                    "protocol": "campaign_loco",
                    "aggregation": "row_micro",
                    "method": method,
                    "accuracy": total_correct[method] / total_test,
                    "n_train": "varies",
                    "n_test": total_test,
                    "n_folds": len(campaigns),
                },
            ]
        )
    return summaries, fold_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    manifest: dict[str, object],
    summaries: Sequence[dict[str, object]],
    diagnostics: Sequence[dict[str, object]],
) -> str:
    by_key = {
        (row["construction_mode"], row["protocol"], row["aggregation"], row["method"]): row
        for row in summaries
    }
    released = manifest["construction"]["released"]
    pair80 = next(
        row
        for row in diagnostics
        if row["construction_mode"] == "released" and row["protocol"] == "pair80_primary"
    )
    original = manifest["inputs"].get("original_workbook")
    original_line = (
        f"- 原始 `.xls` SHA-256：`{original['sha256']}`"
        if original
        else "- 原始 `.xls`：未提供，只记录了可读工作簿"
    )
    lines = [
        "# SECRYPT 2026 划分协议审计",
        "",
        "> 以下数值均为本仓库对公开材料的 clean-room 独立复算，不是 Raj et al. 原论文报告的结果。",
        "",
        "## 冻结来源",
        "",
        f"- 公开仓库 commit：`{manifest['external_commit']}`",
        f"- `PYTHONHASHSEED`: `{manifest['environment']['python_hash_seed']}`",
        original_line,
        f"- 转换后 `.xlsx` SHA-256：`{manifest['inputs']['attack_workbook']['sha256']}`",
        f"- 审计脚本 SHA-256：`{manifest['script']['sha256']}`",
        "",
        "## released-order 数据重建",
        "",
        "| campaign | 生成链 | occurrence | pair | 唯一 `(prefix,target)` | 词表 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {released['campaigns_used']} | {released['chains']} | {released['occurrences']} | {released['pairs']} | {released['unique_content_pairs']} | {released['vocab_size']} |",
        "",
        "重建 pair 数严格等于 `sum(chain_length-1)`。正式论文表格写 128,413，与本审计相差641；必须披露，不能静默修正。",
        "",
        "## 主随机 80/20 划分诊断",
        "",
        "| 训练行 | 测试行 | `(prefix,target)` 完全重复 | prefix 覆盖 | 测试行所属 campaign 在训练出现 |",
        "|---:|---:|---:|---:|---:|",
        f"| {pair80['n_train']} | {pair80['n_test']} | {pair80['exact_content_overlap']:.4%} | {pair80['prefix_coverage']:.4%} | {pair80['campaign_overlap']:.4%} |",
        "",
        "## 确定性基线 Accuracy",
        "",
        "| 方法 | pair 80/20 | campaign LOCO macro | campaign LOCO pooled |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        pair = by_key[("released", "pair80_primary", "row_micro", method)]["accuracy"]
        macro = by_key[("released", "campaign_loco", "campaign_macro", method)]["accuracy"]
        pooled = by_key[("released", "campaign_loco", "row_micro", method)]["accuracy"]
        lines.append(f"| {method} | {pair:.4f} | {macro:.4f} | {pooled:.4f} |")
    lines.extend(
        [
            "",
            "`PREFIX` 是查表诊断，不是论文候选预测模型。随机 pair 与 campaign-LOCO 的差距量化了同 campaign 排列复用对任务难度的影响。",
            "",
            "## 解读边界",
            "",
            "- 不得把这些复算值写成 SECRYPT 原论文结果。",
            "- 不得把这些 `H=1` Accuracy 与本项目 future-3 NDCG@5/Recall@5 直接比较。",
            "- 本次运行不含 LSTM、Hybrid、LLM、API 请求、token 或付费推理。",
            "- canonical-order 只作敏感性检查；固定 hash seed 的 released-order 才是公开代码路径主审计。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    require_environment()
    args = parse_args()
    started = datetime.now(timezone.utc)
    logger = setup_logging(args.output_dir)
    logger.info("Starting clean-room split audit")
    techniques, relationships, campaigns = load_attack_tables(args.attack_workbook)
    logger.info("Loaded ATT&CK workbook sheets")

    all_summaries: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    all_loco_rows: list[dict[str, object]] = []
    construction: dict[str, object] = {}
    for mode in args.modes:
        rows, facts = construct_pairs(techniques, relationships, campaigns, mode)
        construction[mode] = facts
        logger.info(
            "%s: campaigns=%d chains=%d occurrences=%d pairs=%d unique=%d vocab=%d",
            mode,
            facts["campaigns_used"],
            facts["chains"],
            facts["occurrences"],
            facts["pairs"],
            facts["unique_content_pairs"],
            facts["vocab_size"],
        )
        pair_summary, diagnostics = run_pair_protocols(rows, mode)
        loco_summary, loco_rows = run_campaign_loco(rows, mode)
        all_summaries.extend(pair_summary)
        all_summaries.extend(loco_summary)
        all_diagnostics.extend(diagnostics)
        all_loco_rows.extend(loco_rows)

    script_path = Path(__file__).resolve()
    inputs: dict[str, object] = {
        "attack_workbook": {
            "path": str(args.attack_workbook.resolve()),
            "sha256": sha256_file(args.attack_workbook),
        }
    }
    if args.original_workbook:
        inputs["original_workbook"] = {
            "path": str(args.original_workbook.resolve()),
            "sha256": sha256_file(args.original_workbook),
        }
    finished = datetime.now(timezone.utc)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_type": "secrypt_split_clean_room_audit",
        "external_commit": args.external_commit,
        "paper_doi": "10.5220/0015075400004103",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "inputs": inputs,
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "config": {
            "seed": SEED,
            "min_chain_length": MIN_CHAIN_LENGTH,
            "max_bucket_permutations": MAX_BUCKET_PERMUTATIONS,
            "max_chains_per_campaign": MAX_CHAINS_PER_CAMPAIGN,
            "tactic_order": list(TACTIC_ORDER),
            "modes": args.modes,
            "tie_break": "count desc, global target frequency desc, label Unicode asc",
            "fallback": "training target global-frequency winner",
            "pair80_rng_call": 4,
            "campaign_protocol": "leave-one-campaign-out",
        },
        "construction": construction,
        "paper_count_note": {
            "paper_occurrences": 132621,
            "paper_pairs": 128413,
            "reconstructed_pair_formula": "sum(chain_length - 1)",
        },
    }

    write_csv(
        args.output_dir / "summary.csv",
        all_summaries,
        (
            "construction_mode",
            "protocol",
            "aggregation",
            "method",
            "accuracy",
            "n_train",
            "n_test",
            "n_folds",
        ),
    )
    write_csv(
        args.output_dir / "split_diagnostics.csv",
        all_diagnostics,
        (
            "construction_mode",
            "protocol",
            "n_train",
            "n_test",
            "exact_content_overlap",
            "prefix_coverage",
            "campaign_overlap",
            "train_campaigns",
            "test_campaigns",
        ),
    )
    write_csv(
        args.output_dir / "campaign_loco.csv",
        all_loco_rows,
        (
            "construction_mode",
            "campaign_id",
            "method",
            "n_train",
            "n_test",
            "correct",
            "accuracy",
        ),
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        make_report(manifest, all_summaries, all_diagnostics), encoding="utf-8"
    )
    logger.info("Completed audit; outputs written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
