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
    get_all_parent_tids,
    load_label_vocab,
)

LLM_DIR = PROJECT_ROOT / "llm"
PROMPTS_DIR = LLM_DIR / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"


def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


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
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Empty stage1 jsonl: {path}")
    return rows


def build_stage2_prompt(
    semantic_context: str,
    reasoning: str,
    predicted_tactics: Sequence[str],
    suffix_text: str,
) -> str:
    tactics_str = ", ".join(predicted_tactics) if predicted_tactics else "none"

    parts = [
        "Observed prior ATT&CK parent techniques:",
        semantic_context.strip(),
        "",
        "Predicted next-step tactics:",
        tactics_str,
        "",
        "Reasoning:",
        reasoning.strip() if reasoning.strip() else "None.",
        "",
        suffix_text.strip(),
    ]
    return "\n".join(parts)


def choose_candidate_tids(
    all_tids: Sequence[str],
    gold_tid: Optional[str],
    k: int,
) -> List[str]:
    picked: List[str] = []

    if gold_tid is not None and gold_tid in all_tids:
        picked.append(gold_tid)

    for tid in all_tids:
        if tid in picked:
            continue
        picked.append(tid)
        if len(picked) >= k:
            break

    return picked[:k]


def tokenize_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def boundary_diagnostics(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    candidate_tid: str,
) -> Dict[str, Any]:
    prompt_len = tokenize_len(tokenizer, prompt_text)
    cand_len = tokenize_len(tokenizer, candidate_tid)
    full_len = tokenize_len(tokenizer, prompt_text + candidate_tid)

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
        "response_json": data,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1_jsonl",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage1_smoke.jsonl"),
    )
    parser.add_argument(
        "--label_vocab_csv",
        type=str,
        default=str(DATA_DIR / "rl_label_vocab.csv"),
    )
    parser.add_argument(
        "--suffix_path",
        type=str,
        default=str(PROMPTS_DIR / "sgle_r_stage2_suffix.txt"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage2_smoke_scores.csv"),
    )
    parser.add_argument(
        "--boundary_txt",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage2_tokenizer_boundary.txt"),
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
        default=3,
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    ensure_dir(Path(args.output_csv).parent)
    ensure_dir(Path(args.boundary_txt).parent)

    stage1_rows = load_stage1_jsonl(args.stage1_jsonl)[: args.num_samples]
    label_vocab_df = load_label_vocab(args.label_vocab_csv)
    all_tids = get_all_parent_tids(label_vocab_df)

    suffix_path = Path(args.suffix_path)
    if suffix_path.exists():
        suffix_text = read_text(suffix_path)
    else:
        suffix_text = "综合上述深度分析，攻击者下一步最可能执行的单项 ATT&CK 父技术编号是："

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    records: List[Dict[str, Any]] = []
    boundary_lines: List[str] = []

    for row_idx, row in enumerate(stage1_rows):
        sample_id = row.get("sample_id")
        instance_id = f"{sample_id}__row{row_idx}"
        gold_label = row.get("gold_label")
        semantic_context = row.get("semantic_context", "")
        reasoning = row.get("reasoning", "")
        predicted_tactics = row.get("predicted_tactics", [])

        prompt_text = build_stage2_prompt(
            semantic_context=semantic_context,
            reasoning=reasoning,
            predicted_tactics=predicted_tactics,
            suffix_text=suffix_text,
        )

        candidates = choose_candidate_tids(
            all_tids=all_tids,
            gold_tid=gold_label,
            k=args.num_candidates,
        )

        sample_scores: List[Tuple[str, float]] = []

        for cand_tid in candidates:
            diag = boundary_diagnostics(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                candidate_tid=cand_tid,
            )

            full_text = prompt_text + cand_tid
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
                    "seq_logprob_sum": seq_logprob_sum,
                    "label_token_len_alone": diag["candidate_token_len_alone"],
                    "label_token_len_in_context": diag["candidate_token_len_in_context"],
                    "used_token_count": used_token_count,
                    "boundary_merged": diag["boundary_merged"],
                    "length_norm_score": length_norm_score,
                }
            )

            boundary_lines.append(
                f"instance_id={instance_id} | sample_id={sample_id} | candidate={cand_tid} | "
                f"prompt_len={diag['prompt_token_len']} | "
                f"cand_len_alone={diag['candidate_token_len_alone']} | "
                f"cand_len_in_context={diag['candidate_token_len_in_context']} | "
                f"boundary_merged={diag['boundary_merged']}"
            )

        sample_scores = sorted(sample_scores, key=lambda x: x[1], reverse=True)
        print("=" * 100)
        print(f"instance_id={instance_id} sample_id={sample_id} gold_label={gold_label}")
        for rank, (cand_tid, score) in enumerate(sample_scores[:5], start=1):
            print(f"rank={rank} candidate={cand_tid} score={score:.6f}")

    out_df = pd.DataFrame(records)
    out_df["rank"] = (
        out_df.groupby("instance_id")["length_norm_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out_df = out_df.sort_values(
        ["instance_id", "rank", "candidate_tid"],
        ascending=[True, True, True],
    )
    out_df.to_csv(args.output_csv, index=False)

    with open(args.boundary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(boundary_lines) + "\n")

    print("\nSaved:", args.output_csv)
    print("Saved:", args.boundary_txt)


if __name__ == "__main__":
    main()