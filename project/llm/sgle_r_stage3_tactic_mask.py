from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd

from sgle_r_common import (
    DATA_DIR,
    PROJECT_ROOT,
    MITRE_TACTIC_ENUM,
    load_attack_parent_lookup,
    load_split_csv,
    normalize_tactic_list,
    parse_lookup_tactic_ids,
)

LLM_DIR = PROJECT_ROOT / "llm"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_pipe_list(raw_value) -> List[str]:
    if raw_value is None:
        return []
    s = str(raw_value).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split("||") if x.strip()]


def build_parent_to_tactics(lookup_df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        tactics = parse_lookup_tactic_ids(row["tactic_ids"])
        out[tid] = tactics
    return out


def build_transition_matrix(
    train_df: pd.DataFrame,
    parent_to_tactics: Dict[str, List[str]],
    alpha: float = 1.0,
) -> pd.DataFrame:
    counts = pd.DataFrame(
        0.0,
        index=MITRE_TACTIC_ENUM,
        columns=MITRE_TACTIC_ENUM,
    )

    for _, row in train_df.iterrows():
        prefix_tactics = normalize_tactic_list(parse_pipe_list(row["prefix_tactics"]))
        gold_parent = str(row["next_technique_id_parent"]).strip()
        next_tactics = parent_to_tactics.get(gold_parent, [])

        if not prefix_tactics or not next_tactics:
            continue

        for src_t in prefix_tactics:
            for dst_t in next_tactics:
                counts.loc[src_t, dst_t] += 1.0

    probs = counts.copy()
    for src_t in MITRE_TACTIC_ENUM:
        row_sum = probs.loc[src_t].sum()
        probs.loc[src_t] = (probs.loc[src_t] + alpha) / (row_sum + alpha * len(MITRE_TACTIC_ENUM))

    return probs


def aggregate_mask_score(
    predicted_tactics: Sequence[str],
    candidate_parent_tactics: Sequence[str],
    transition_probs: pd.DataFrame,
) -> float:
    pred_ts = [t for t in predicted_tactics if t in MITRE_TACTIC_ENUM]
    cand_ts = [t for t in candidate_parent_tactics if t in MITRE_TACTIC_ENUM]

    if not pred_ts or not cand_ts:
        return 1.0

    vals: List[float] = []
    for src_t in pred_ts:
        for dst_t in cand_ts:
            vals.append(float(transition_probs.loc[src_t, dst_t]))

    if not vals:
        return 1.0

    return sum(vals) / len(vals)


def load_stage1_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(DATA_DIR / "sim_train_parent_min3.csv"),
    )
    parser.add_argument(
        "--stage1_jsonl",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage1_test.jsonl"),
    )
    parser.add_argument(
        "--lookup_csv",
        type=str,
        default=str(DATA_DIR / "attack_parent_lookup_for_llm.csv"),
    )
    parser.add_argument(
        "--transition_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_tactic_transition_matrix.csv"),
    )
    parser.add_argument(
        "--mask_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_label_mask_test.csv"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all stage1 rows",
    )
    args = parser.parse_args()

    ensure_dir(Path(args.transition_csv).parent)
    ensure_dir(Path(args.mask_csv).parent)

    train_df = load_split_csv(args.train_csv)
    lookup_df = load_attack_parent_lookup(args.lookup_csv)
    stage1_rows = load_stage1_jsonl(args.stage1_jsonl)
    if args.limit > 0:
        stage1_rows = stage1_rows[: args.limit]

    parent_to_tactics = build_parent_to_tactics(lookup_df)
    transition_probs = build_transition_matrix(
        train_df=train_df,
        parent_to_tactics=parent_to_tactics,
        alpha=args.alpha,
    )
    transition_probs.to_csv(args.transition_csv)

    all_parent_tids = sorted(parent_to_tactics.keys())
    records = []

    for row_idx, row in enumerate(stage1_rows):
        sample_id = row.get("sample_id")
        instance_id = f"{sample_id}__row{row_idx}"
        gold_label = row.get("gold_label")
        predicted_tactics = normalize_tactic_list(row.get("predicted_tactics", []))

        for cand_tid in all_parent_tids:
            cand_tactics = parent_to_tactics.get(cand_tid, [])
            mask_score = aggregate_mask_score(
                predicted_tactics=predicted_tactics,
                candidate_parent_tactics=cand_tactics,
                transition_probs=transition_probs,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "gold_label": gold_label,
                    "candidate_tid": cand_tid,
                    "predicted_tactics": " || ".join(predicted_tactics),
                    "candidate_tactics": " || ".join(cand_tactics),
                    "mask_score": mask_score,
                }
            )

        if row_idx < 3:
            print(
                f"[{row_idx}] instance_id={instance_id} "
                f"gold_label={gold_label} "
                f"predicted_tactics={predicted_tactics}"
            )

    pd.DataFrame(records).to_csv(args.mask_csv, index=False)

    print("\nSaved:", args.transition_csv)
    print("Saved:", args.mask_csv)


if __name__ == "__main__":
    main()