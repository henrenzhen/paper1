from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from transformers import AutoTokenizer

from sgle_r_common import (
    DATA_DIR,
    PROJECT_ROOT,
    load_attack_parent_lookup,
)

LLM_DIR = PROJECT_ROOT / "llm"
LOGS_DIR = PROJECT_ROOT / "logs"
RL_DIR = PROJECT_ROOT / "rl"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_stage1_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stage1 jsonl not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Empty stage1 jsonl: {path}")
    return rows


def build_parent_tid_to_name(lookup_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in lookup_df.iterrows():
        tid = str(row["technique_id_parent"]).strip()
        name = str(row["technique_name_parent"]).strip()
        out[tid] = name
    return out


def build_stage2_prompt(
    semantic_context: str,
    reasoning: str,
    predicted_tactics: Sequence[str],
) -> str:
    tactics_str = ", ".join(predicted_tactics) if predicted_tactics else "none"

    parts = [
        "You are performing immediate next-step ATT&CK parent-technique inference.",
        "",
        "Task definition:",
        "Given the prior observed ATT&CK parent techniques, infer the single most likely immediate next parent technique.",
        "",
        "Important constraints:",
        "1. Predict the next local step only.",
        "2. Do not predict broad downstream phases or long-term attacker goals.",
        "3. Focus on the most direct parent-technique transition implied by the latest attack state.",
        "4. Use the semantic meaning of the observed techniques, not surface string matching.",
        "5. Prefer the candidate that best matches the immediate functional need after the current prefix.",
        "",
        "Observed prior ATT&CK parent techniques:",
        semantic_context.strip(),
        "",
        "Predicted next-step tactics:",
        tactics_str,
        "",
        "Reasoning summary:",
        reasoning.strip() if reasoning.strip() else "None.",
        "",
        "The single most likely immediate next ATT&CK parent technique is:",
    ]
    return "\n".join(parts)


def tokenize_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def boundary_diagnostics(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    candidate_text: str,
) -> Dict[str, Any]:
    prompt_len = tokenize_len(tokenizer, prompt_text)
    cand_len = tokenize_len(tokenizer, candidate_text)
    full_len = tokenize_len(tokenizer, prompt_text + candidate_text)

    return {
        "prompt_token_len": prompt_len,
        "candidate_token_len_alone": cand_len,
        "full_token_len": full_len,
        "candidate_token_len_in_context": full_len - prompt_len,
        "boundary_merged": (full_len - prompt_len) != cand_len,
    }


def extract_prompt_logprob_value(item: Any) -> Optional[float]:
    if item is None:
        return None

    if isinstance(item, dict):
        if "logprob" in item and isinstance(item["logprob"], (int, float)):
            return float(item["logprob"])
        for v in item.values():
            if isinstance(v, dict) and "logprob" in v and isinstance(v["logprob"], (int, float)):
                return float(v["logprob"])

    if isinstance(item, list):
        for v in item:
            x = extract_prompt_logprob_value(v)
            if x is not None:
                return x

    return None


def score_full_sequence_vllm(
    api_base: str,
    api_key: str,
    model_name: str,
    full_text: str,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "prompt": full_text,
        "max_tokens": 0,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 0,
        "extra_body": {
            "prompt_logprobs": 1,
            "add_special_tokens": False,
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    prompt_logprobs = choice.get("prompt_logprobs", None)
    if prompt_logprobs is None:
        raise ValueError("vLLM response missing prompt_logprobs")

    return {
        "prompt_logprobs": prompt_logprobs,
    }


def sum_label_prompt_logprobs(
    prompt_logprobs: Sequence[Any],
    label_token_len_in_context: int,
) -> Tuple[float, int]:
    if label_token_len_in_context <= 0:
        raise ValueError("label_token_len_in_context must be > 0")

    values: List[float] = []
    tail = list(prompt_logprobs)[-label_token_len_in_context:]

    for item in tail:
        x = extract_prompt_logprob_value(item)
        if x is not None:
            values.append(x)

    if not values:
        raise ValueError("No usable prompt logprobs extracted for label tail")

    return float(sum(values)), len(values)


def parse_topk_labels(raw: str) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("||") if x.strip()]


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true: List[str] = []
    ranked_preds: List[List[str]] = []

    for instance_id, g in df.groupby("instance_id", sort=False):
        g = g.sort_values("rerank_rank", ascending=True)
        gold = str(g["gold_label"].iloc[0]).strip()
        preds = g["candidate_tid"].astype(str).tolist()
        y_true.append(gold)
        ranked_preds.append(preds)

    def topk_acc(y_true: Sequence[str], ranked_preds: Sequence[Sequence[str]], k: int) -> float:
        hit = 0
        for y, preds in zip(y_true, ranked_preds):
            if y in list(preds)[:k]:
                hit += 1
        return hit / len(y_true) if y_true else 0.0

    def mrr(y_true: Sequence[str], ranked_preds: Sequence[Sequence[str]]) -> float:
        s = 0.0
        for y, preds in zip(y_true, ranked_preds):
            rr = 0.0
            for i, p in enumerate(preds, start=1):
                if p == y:
                    rr = 1.0 / i
                    break
            s += rr
        return s / len(y_true) if y_true else 0.0

    return {
        "num_instances": len(y_true),
        "top1": topk_acc(y_true, ranked_preds, 1),
        "top5": topk_acc(y_true, ranked_preds, 5),
        "mrr": mrr(y_true, ranked_preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_top5_csv",
        type=str,
        default=str(RL_DIR / "rl_v2_test_predictions_top5.csv"),
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
        "--output_csv",
        type=str,
        default=str(LLM_DIR / "sgle_r_rerank_rl_top5.csv"),
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=str(LLM_DIR / "sgle_r_rerank_rl_top5_metrics.json"),
    )
    parser.add_argument(
        "--boundary_txt",
        type=str,
        default=str(LOGS_DIR / "sgle_r_rerank_rl_top5_boundary.txt"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/model-storage/model/Qwen3.5-35B-A3B-FP8",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="0 means all rows",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    ensure_dir(Path(args.output_csv).parent)
    ensure_dir(Path(args.metrics_json).parent)
    ensure_dir(Path(args.boundary_txt).parent)

    rl_df = pd.read_csv(args.rl_top5_csv)
    stage1_rows = load_stage1_jsonl(args.stage1_jsonl)
    lookup_df = load_attack_parent_lookup(args.lookup_csv)
    parent_tid_to_name = build_parent_tid_to_name(lookup_df)

    if args.num_samples > 0:
        rl_df = rl_df.head(args.num_samples).copy()
        stage1_rows = stage1_rows[: args.num_samples]

    if len(rl_df) != len(stage1_rows):
        raise ValueError(
            f"Row count mismatch: rl_df={len(rl_df)} vs stage1_rows={len(stage1_rows)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    records: List[Dict[str, Any]] = []
    boundary_lines: List[str] = []

    for row_idx, (rl_row, s1_row) in enumerate(zip(rl_df.itertuples(index=False), stage1_rows)):
        sample_id = s1_row.get("sample_id")
        instance_id = f"{sample_id}__row{row_idx}"
        gold_label = str(getattr(rl_row, "true_label")).strip()

        semantic_context = s1_row.get("semantic_context", "")
        reasoning = s1_row.get("reasoning", "")
        predicted_tactics = s1_row.get("predicted_tactics", [])

        prompt_text = build_stage2_prompt(
            semantic_context=semantic_context,
            reasoning=reasoning,
            predicted_tactics=predicted_tactics,
        )

        candidates = parse_topk_labels(getattr(rl_row, "top5_labels"))
        sample_scores: List[Tuple[str, float]] = []

        for cand_tid in candidates:
            cand_name = parent_tid_to_name.get(cand_tid, "")
            candidate_text = f"{cand_tid} - {cand_name}" if cand_name else cand_tid

            diag = boundary_diagnostics(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                candidate_text=candidate_text,
            )

            full_text = prompt_text + candidate_text
            scored = score_full_sequence_vllm(
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                full_text=full_text,
            )

            seq_logprob_sum, used_token_count = sum_label_prompt_logprobs(
                prompt_logprobs=scored["prompt_logprobs"],
                label_token_len_in_context=diag["candidate_token_len_in_context"],
            )

            length_norm_score = seq_logprob_sum / (used_token_count ** args.beta)
            sample_scores.append((cand_tid, length_norm_score))

            records.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "gold_label": gold_label,
                    "candidate_tid": cand_tid,
                    "candidate_text": candidate_text,
                    "seq_logprob_sum": seq_logprob_sum,
                    "used_token_count": used_token_count,
                    "boundary_merged": diag["boundary_merged"],
                    "length_norm_score": length_norm_score,
                    "rl_true_rank": int(getattr(rl_row, "true_rank")),
                    "rl_top1_pred": str(getattr(rl_row, "top1_pred")).strip(),
                }
            )

            boundary_lines.append(
                f"instance_id={instance_id} | candidate={candidate_text} | "
                f"prompt_len={diag['prompt_token_len']} | "
                f"cand_len_alone={diag['candidate_token_len_alone']} | "
                f"cand_len_in_context={diag['candidate_token_len_in_context']} | "
                f"boundary_merged={diag['boundary_merged']}"
            )

        sample_scores = sorted(sample_scores, key=lambda x: x[1], reverse=True)
        print("=" * 100)
        print(f"instance_id={instance_id} gold_label={gold_label} rl_true_rank={getattr(rl_row, 'true_rank')}")
        for rank, (cand_tid, score) in enumerate(sample_scores, start=1):
            cand_name = parent_tid_to_name.get(cand_tid, "")
            candidate_text = f"{cand_tid} - {cand_name}" if cand_name else cand_tid
            print(f"rank={rank} candidate={candidate_text} score={score:.6f}")

    out_df = pd.DataFrame(records)
    out_df["rerank_rank"] = (
        out_df.groupby("instance_id")["length_norm_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out_df = out_df.sort_values(
        ["instance_id", "rerank_rank", "candidate_tid"],
        ascending=[True, True, True],
    )
    out_df.to_csv(args.output_csv, index=False)

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        metrics = compute_metrics(out_df)
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(args.boundary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(boundary_lines) + "\n")

    print("\nSaved:", args.output_csv)
    print("Saved:", args.metrics_json)
    print("Saved:", args.boundary_txt)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()