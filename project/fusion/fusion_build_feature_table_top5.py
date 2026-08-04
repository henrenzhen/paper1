from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path("/root/project")
RL_DIR = PROJECT_ROOT / "rl"
LLM_DIR = PROJECT_ROOT / "llm"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
FUSION_DIR = PROJECT_ROOT / "fusion"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_pipe_list(raw) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split("||") if x.strip()]


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parent_to_tactics(lookup_df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        tactics = parse_pipe_list(row["tactic_ids"])
        out[tid] = tactics
    return out


def build_parent_to_name(lookup_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        name = str(row["technique_name_parent"]).strip()
        out[tid] = name
    return out


def build_parent_to_member_names(lookup_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        txt = ""
        if "member_technique_names" in row and pd.notna(row["member_technique_names"]):
            txt = str(row["member_technique_names"]).strip()
        out[tid] = txt
    return out


def build_parent_to_subtechnique_names(lookup_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        txt = ""
        if "subtechnique_names" in row and pd.notna(row["subtechnique_names"]):
            txt = str(row["subtechnique_names"]).strip()
        out[tid] = txt
    return out


def tactic_match_score(predicted_tactics: List[str], candidate_tactics: List[str]) -> float:
    pred = set(predicted_tactics)
    cand = set(candidate_tactics)
    if not pred or not cand:
        return 0.0
    return len(pred & cand) / len(cand)


def normalize_text_tokens(text: str) -> List[str]:
    s = str(text).lower()
    for ch in [",", ".", ";", ":", "|", "/", "\\", "(", ")", "[", "]", "{", "}", "-", "_"]:
        s = s.replace(ch, " ")
    toks = [t.strip() for t in s.split() if t.strip()]
    return toks


def artifact_keyword_overlap(artifact_text: str, verb_text: str, candidate_text: str) -> float:
    av_text = f"{artifact_text} {verb_text}"
    av_tokens = set(normalize_text_tokens(av_text))
    cand_tokens = set(normalize_text_tokens(candidate_text))

    if not av_tokens or not cand_tokens:
        return 0.0

    return len(av_tokens & cand_tokens) / len(av_tokens)


def build_rl_candidate_rows(rl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row_idx, row in rl_df.iterrows():
        sample_id = str(row["sequence_id"]).strip()
        instance_id = f"{sample_id}__row{row_idx}"
        gold_label = str(row["true_label"]).strip()

        top5_labels = parse_pipe_list(row["top5_labels"])
        top5_probs = [float(x) for x in parse_pipe_list(row["top5_probs"])]

        if len(top5_labels) != len(top5_probs):
            raise ValueError(f"Mismatch top5 labels/probs at row {row_idx}")

        rl_top1_pred = str(row["top1_pred"]).strip()
        rl_top1_prob = float(row["top1_prob"])
        rl_margin_top1_top2 = rl_top1_prob - top5_probs[1] if len(top5_probs) > 1 else rl_top1_prob

        for rank, (cand_tid, cand_prob) in enumerate(zip(top5_labels, top5_probs), start=1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "gold_label": gold_label,
                    "candidate_tid": cand_tid,
                    "label": int(cand_tid == gold_label),
                    "rl_rank": rank,
                    "rl_prob": float(cand_prob),
                    "rl_top1_pred": rl_top1_pred,
                    "rl_top1_prob": rl_top1_prob,
                    "rl_margin_top1_top2": rl_margin_top1_top2,
                    "rl_is_top1": int(rank == 1),
                    "rl_true_rank_global": int(row["true_rank"]),
                }
            )

    return pd.DataFrame(rows)


def join_nonempty(parts: List[str], sep: str = " | ") -> str:
    xs = [str(x).strip() for x in parts if str(x).strip() and str(x).strip().lower() != "nan"]
    return sep.join(xs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_top5_csv",
        type=str,
        default=str(RL_DIR / "rl_v2_test_predictions_top5.csv"),
    )
    parser.add_argument(
        "--llm_rerank_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_rerank_rl_top5.csv"),
    )
    parser.add_argument(
        "--stage1_jsonl",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage1_test.jsonl"),
    )
    parser.add_argument(
        "--micro_jsonl",
        type=str,
        default=str(LOGS_DIR / "sgle_r_action_profile_micro_30.jsonl"),
    )
    parser.add_argument(
        "--lookup_csv",
        type=str,
        default=str(DATA_DIR / "attack_parent_lookup_for_llm.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_features_top5.csv"),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="0 means all rows",
    )
    args = parser.parse_args()

    ensure_dir(Path(args.output_csv).parent)

    rl_df = pd.read_csv(args.rl_top5_csv)
    llm_df = pd.read_csv(args.llm_rerank_csv)
    stage1_rows = load_jsonl(args.stage1_jsonl)
    micro_rows = load_jsonl(args.micro_jsonl)
    lookup_df = pd.read_csv(args.lookup_csv)

    if args.num_samples > 0:
        rl_df = rl_df.head(args.num_samples).copy()
        stage1_rows = stage1_rows[: args.num_samples]
        micro_rows = micro_rows[: args.num_samples]

    parent_to_tactics = build_parent_to_tactics(lookup_df)
    parent_to_name = build_parent_to_name(lookup_df)
    parent_to_member_names = build_parent_to_member_names(lookup_df)
    parent_to_sub_names = build_parent_to_subtechnique_names(lookup_df)

    base_df = build_rl_candidate_rows(rl_df)

    llm_keep = llm_df[
        [
            "instance_id",
            "candidate_tid",
            "length_norm_score",
            "rerank_rank",
        ]
    ].copy()
    llm_keep = llm_keep.rename(
        columns={
            "length_norm_score": "llm_seq_score",
            "rerank_rank": "llm_seq_rank",
        }
    )

    merged = base_df.merge(
        llm_keep,
        on=["instance_id", "candidate_tid"],
        how="left",
    )

    llm_top1_map: Dict[str, str] = {}
    llm_margin_map: Dict[str, float] = {}

    for instance_id, g in llm_df.groupby("instance_id", sort=False):
        g = g.sort_values("rerank_rank", ascending=True).copy()
        llm_top1_map[instance_id] = str(g.iloc[0]["candidate_tid"]).strip()
        if len(g) >= 2:
            margin = float(g.iloc[0]["length_norm_score"]) - float(g.iloc[1]["length_norm_score"])
        else:
            margin = 0.0
        llm_margin_map[instance_id] = margin

    merged["llm_top1_tid"] = merged["instance_id"].map(llm_top1_map)
    merged["llm_seq_margin_top1_top2"] = merged["instance_id"].map(llm_margin_map)
    merged["llm_is_top1"] = (merged["llm_seq_rank"] == 1).astype(int)
    merged["llm_agree_with_rl_top1"] = (
        (merged["llm_top1_tid"] == merged["rl_top1_pred"]).astype(int)
    )

    stage1_map: Dict[str, dict] = {}
    for row_idx, row in enumerate(stage1_rows):
        sample_id = row.get("sample_id")
        instance_id = f"{sample_id}__row{row_idx}"
        stage1_map[instance_id] = row

    micro_map: Dict[str, dict] = {}
    for row_idx, row in enumerate(micro_rows):
        sample_id = row.get("sample_id")
        instance_id = f"{sample_id}__row{row_idx}"
        micro_map[instance_id] = row

    predicted_tactics_col = []
    tactic_match_col = []
    candidate_tactics_col = []
    artifact_verb_text_col = []
    artifact_keyword_overlap_col = []
    artifact_query_text_col = []
    candidate_semantic_text_col = []

    for _, row in merged.iterrows():
        instance_id = row["instance_id"]
        candidate_tid = row["candidate_tid"]

        s1 = stage1_map.get(instance_id, {})
        predicted_tactics = s1.get("predicted_tactics", [])
        if not isinstance(predicted_tactics, list):
            predicted_tactics = []

        candidate_tactics = parent_to_tactics.get(candidate_tid, [])
        score = tactic_match_score(predicted_tactics, candidate_tactics)

        predicted_tactics_col.append(" || ".join(predicted_tactics))
        tactic_match_col.append(score)
        candidate_tactics_col.append(" || ".join(candidate_tactics))

        mrow = micro_map.get(instance_id, {})
        next_target_artifact = str(mrow.get("next_target_artifact", "")).strip()
        next_micro_verb = str(mrow.get("next_micro_verb", "")).strip()

        artifact_verb_text = f"artifact: {next_target_artifact} | verb: {next_micro_verb}".strip()
        artifact_query_text = join_nonempty(
            [
                f"artifact {next_target_artifact}",
                f"verb {next_micro_verb}",
            ]
        )

        cand_name = parent_to_name.get(candidate_tid, "")
        member_names = parent_to_member_names.get(candidate_tid, "")
        sub_names = parent_to_sub_names.get(candidate_tid, "")

        candidate_text = f"{candidate_tid} - {cand_name}" if cand_name else candidate_tid
        candidate_semantic_text = join_nonempty(
            [
                    f"{candidate_tid} - {cand_name}" if cand_name else candidate_tid,
                        member_names,
                        sub_names,
            ]
            )

        kw_overlap = artifact_keyword_overlap(
            artifact_text=next_target_artifact,
            verb_text=next_micro_verb,
            candidate_text=candidate_semantic_text,
        )

        artifact_verb_text_col.append(artifact_verb_text)
        artifact_query_text_col.append(artifact_query_text)
        candidate_semantic_text_col.append(candidate_semantic_text)
        artifact_keyword_overlap_col.append(kw_overlap)

    merged["stage1_predicted_tactics"] = predicted_tactics_col
    merged["tactic_match_score"] = tactic_match_col
    merged["candidate_tactics"] = candidate_tactics_col
    merged["artifact_verb_text"] = artifact_verb_text_col
    merged["artifact_query_text"] = artifact_query_text_col
    merged["candidate_semantic_text"] = candidate_semantic_text_col
    merged["artifact_keyword_overlap"] = artifact_keyword_overlap_col

    if merged["llm_seq_score"].isna().any():
        bad = merged[merged["llm_seq_score"].isna()][["instance_id", "candidate_tid"]].head(10)
        raise ValueError(
            f"Missing llm_seq_score after merge. Examples:\n{bad.to_string(index=False)}"
        )

    merged = merged.sort_values(
        ["instance_id", "rl_rank", "candidate_tid"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    merged.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print("shape:", merged.shape)
    print("columns:", merged.columns.tolist())
    print("\nPreview:")
    print(
        merged[
            [
                "instance_id",
                "candidate_tid",
                "artifact_query_text",
                "candidate_semantic_text",
                "artifact_keyword_overlap",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()