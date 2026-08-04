from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

from sgle_r_common import (
    DATA_DIR,
    PROJECT_ROOT,
    are_all_tactics_valid,
    build_parent_semantic_context,
    load_attack_parent_lookup,
    load_split_csv,
    normalize_tactic_list,
)


LLM_DIR = PROJECT_ROOT / "llm"
PROMPTS_DIR = LLM_DIR / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"


def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def infer_split_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = df.columns.tolist()

    id_col = None
    for c in ["sequence_id", "sample_id", "id", "row_id", "case_id"]:
        if c in cols:
            id_col = c
            break

    prefix_col = None
    for c in [
        "prefix_techniques",
        "prefix_technique_names",
        "prefix",
        "prefix_text",
        "history_text",
        "input_text",
        "source_text",
        "sequence_text",
    ]:
        if c in cols:
            prefix_col = c
            break

    gold_col = None
    for c in [
        "next_technique_id_parent",
        "next_parent_tid",
        "label",
        "target",
        "gold_label",
        "technique_id_parent_next",
        "y_label",
    ]:
        if c in cols:
            gold_col = c
            break

    prior_tid_col = None
    for c in [
        "prefix_technique_ids_parent",
        "prefix_parent_tids",
        "prior_parent_tids",
        "history_parent_tids",
        "observed_parent_tids",
        "parent_tid_seq",
        "technique_id_parent_seq",
    ]:
        if c in cols:
            prior_tid_col = c
            break

    return {
        "id_col": id_col,
        "prefix_col": prefix_col,
        "gold_col": gold_col,
        "prior_tid_col": prior_tid_col,
    }


def parse_tid_sequence(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []

    s = str(raw_value).strip()
    if not s or s.lower() == "nan":
        return []

    for sep in ["||", ";", ",", "->", "|"]:
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]

    return [s]


def build_semantic_context_for_row(
    row: pd.Series,
    split_cols: Dict[str, Optional[str]],
    lookup_df: pd.DataFrame,
) -> str:
    prior_tid_col = split_cols["prior_tid_col"]
    prefix_col = split_cols["prefix_col"]

    parent_tids: List[str] = []
    if prior_tid_col is not None:
        parent_tids = parse_tid_sequence(row[prior_tid_col])

    if parent_tids:
        return build_parent_semantic_context(parent_tids=parent_tids, lookup_df=lookup_df)

    prefix_text = ""
    if prefix_col is not None:
        prefix_text = str(row[prefix_col]).strip()

    if prefix_text:
        return prefix_text

    return "No prior parent techniques available."


def coerce_stage1_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    reasoning = obj.get("reasoning", "")
    predicted_tactics = obj.get("predicted_tactics", [])

    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    if isinstance(predicted_tactics, str):
        predicted_tactics = [predicted_tactics]
    elif not isinstance(predicted_tactics, list):
        predicted_tactics = []

    predicted_tactics = [str(x) for x in predicted_tactics]
    predicted_tactics = normalize_tactic_list(predicted_tactics)

    return {
        "reasoning": reasoning.strip(),
        "predicted_tactics": predicted_tactics,
    }


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate = text[left : right + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def call_stage1_once(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "predicted_tactics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "reconnaissance",
                        "resource-development",
                        "initial-access",
                        "execution",
                        "persistence",
                        "privilege-escalation",
                        "defense-evasion",
                        "credential-access",
                        "discovery",
                        "lateral-movement",
                        "collection",
                        "command-and-control",
                        "exfiltration",
                        "impact",
                    ],
                },
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["reasoning", "predicted_tactics"],
        "additionalProperties": False,
    }

    resp = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "stage1_tactic_prediction",
                "schema": schema,
            },
        },
    )

    content = resp.choices[0].message.content
    return {
        "raw_text": content if content is not None else "",
    }


def run_stage1(
    split_csv: str | Path,
    output_jsonl: str | Path,
    model_name: str,
    system_prompt_path: str | Path,
    user_prompt_path: str | Path,
    lookup_csv: str | Path = DATA_DIR / "attack_parent_lookup_for_llm.csv",
    api_base: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    limit: Optional[int] = None,
    sleep_sec: float = 0.0,
) -> None:
    ensure_dir(LOGS_DIR)
    output_jsonl = Path(output_jsonl)
    ensure_dir(output_jsonl.parent)

    split_df = load_split_csv(split_csv)
    if limit is not None and limit > 0:
        split_df = split_df.head(limit).copy()

    lookup_df = load_attack_parent_lookup(lookup_csv)
    split_cols = infer_split_columns(split_df)

    system_prompt = read_text(system_prompt_path)
    user_template = read_text(user_prompt_path)

    client = OpenAI(base_url=api_base, api_key=api_key)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in split_df.iterrows():
            sample_id = (
                row[split_cols["id_col"]]
                if split_cols["id_col"] is not None
                else idx
            )
            gold_label = (
                str(row[split_cols["gold_col"]]).strip()
                if split_cols["gold_col"] is not None
                else None
            )

            semantic_context = build_semantic_context_for_row(
                row=row,
                split_cols=split_cols,
                lookup_df=lookup_df,
            )
            user_prompt = user_template.format(semantic_context=semantic_context)

            result = call_stage1_once(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            raw_text = result["raw_text"]
            parsed = try_parse_json(raw_text)

            json_valid = parsed is not None
            if parsed is None:
                parsed = {"reasoning": "", "predicted_tactics": []}

            parsed = coerce_stage1_json(parsed)
            predicted_tactics = parsed["predicted_tactics"]
            all_tactics_in_enum = are_all_tactics_valid(predicted_tactics)

            record = {
                "sample_id": sample_id,
                "gold_label": gold_label,
                "semantic_context": semantic_context,
                "reasoning": parsed["reasoning"],
                "predicted_tactics": predicted_tactics,
                "raw_response": raw_text,
                "json_valid": json_valid,
                "all_tactics_in_enum": all_tactics_in_enum,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{idx}] sample_id={sample_id} "
                f"gold_label={gold_label} "
                f"json_valid={json_valid} "
                f"n_tactics={len(predicted_tactics)} "
                f"tactics={predicted_tactics}"
            )

            if sleep_sec > 0:
                time.sleep(sleep_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default=str(DATA_DIR / "sim_test_parent_min3.csv"),
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=str(LOGS_DIR / "sgle_r_stage1_smoke.jsonl"),
    )
    parser.add_argument(
        "--lookup_csv",
        type=str,
        default=str(DATA_DIR / "attack_parent_lookup_for_llm.csv"),
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=str(PROMPTS_DIR / "sgle_r_stage1_system.txt"),
    )
    parser.add_argument(
        "--user_prompt_path",
        type=str,
        default=str(PROMPTS_DIR / "sgle_r_stage1_user.txt"),
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
        "--limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    run_stage1(
        split_csv=args.split_csv,
        output_jsonl=args.output_jsonl,
        model_name=args.model_name,
        system_prompt_path=args.system_prompt_path,
        user_prompt_path=args.user_prompt_path,
        lookup_csv=args.lookup_csv,
        api_base=args.api_base,
        api_key=args.api_key,
        limit=args.limit,
        sleep_sec=args.sleep_sec,
    )


if __name__ == "__main__":
    main()