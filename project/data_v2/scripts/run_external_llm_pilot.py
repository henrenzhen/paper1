#!/usr/bin/env python3
"""Run the frozen 30-row DeepSeek external-semantics pilot.

The API key is accepted only through DEEPSEEK_API_KEY. Every HTTP attempt is
appended to an immutable JSONL audit stream before the final per-sample CSV is
written. A new timestamped run directory is required for every invocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import ssl
import sys
import threading
import time
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)
import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data_v4/external_reasoning/pilot/pilot_sample_30.csv"
PILOT_MANIFEST_PATH = (
    PROJECT_ROOT / "data_v4/external_reasoning/pilot/pilot_sample_manifest.json"
)
PREFLIGHT_PATH = (
    PROJECT_ROOT / "data_v4/external_reasoning/pilot/generation_preflight.json"
)
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
PROTOCOL_PATH = (
    PROJECT_ROOT / "data_v4/protocols/LLM_semantic_external_validation_v4.2.md"
)
RUNS_DIR = PROJECT_ROOT / "data_v4/external_reasoning/pilot/runs"

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
THINKING = {"type": "disabled"}
RESPONSE_FORMAT = {"type": "json_object"}
CONCURRENCY = 50
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 300

SYSTEM_PROMPT = """你是一个高级 APT 威胁狩猎专家与 ATT&CK 攻击图分析师。
你的任务是：基于攻击者已执行的 ATT&CK 技术序列（Prefix）以及相关的知识图谱上下文（KG Context），推断攻击者当前的【阶段性操作状态】，并直接预测下一步最可能执行的 5 个 ATT&CK Parent Technique（父技术）。
输入是真实攻击活动的技术序列。你必须遵守以下严格限制：
1. 绝对不要凭空捏造微观动作。推理必须完全基于传入的 Prefix ID 及 KG Context 进行逻辑推演。
2. 预测结果必须是纯粹的父技术 ID。
请以 json 格式输出，在 `_thinking_process` 字段中写下你的推理过程，按以下三步进行思考：
[战术阶段评估]：分析 Prefix 中最后两步，它们处于什么战术阶段？
[已获资产推演]：基于前缀技术，攻击者目前掌握了什么级别的粗粒度资产或权限？
[意图图谱映射]：结合 KG Context，前缀的最后几步操作最可能为后续攻击开启了什么逻辑攻击面？
推理完成后，请在 `predicted_next_ttps` 数组中输出恰好 5 个最可能的下一步 ATT&CK 父技术 ID。
输出格式示例：
{"_thinking_process":"[战术阶段评估]...[已获资产推演]...[意图图谱映射]...","predicted_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}"""

USER_PROMPT_TEMPLATE = """### 攻击前缀序列 (Prefix) ###
{prefix}
(重点关注最后两步：{recent_ids})

### 相关的知识图谱上下文 (KG Context) ###
{kg_context}

### 任务要求 ###
请先在 `_thinking_process` 字段推演，随后在 `predicted_next_ttps` 数组输出 5 个预测的父技术 ID。"""

JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["_thinking_process", "predicted_next_ttps"],
    "additionalProperties": True,
    "properties": {
        "_thinking_process": {"type": "string"},
        "predicted_next_ttps": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^T\\d{4}$"},
        },
    },
}

SECTION_TITLES = ("[战术阶段评估]", "[已获资产推演]", "[意图图谱映射]")
PARENT_ID_RE = re.compile(r"^T\d{4}$")

# Official rates retrieved before the pilot from DeepSeek's pricing page.
PRICE_USD_PER_MILLION = {
    "prompt_cache_hit": 0.0028,
    "prompt_cache_miss": 0.14,
    "completion": 0.28,
}
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_json_schema(value: Any) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "root_not_object"
    if not isinstance(value.get("_thinking_process"), str):
        return False, "thinking_process_not_string"
    predictions = value.get("predicted_next_ttps")
    if not isinstance(predictions, list):
        return False, "predicted_next_ttps_not_array"
    if len(predictions) != 5:
        return False, "predicted_next_ttps_length_not_5"
    if not all(isinstance(item, str) and PARENT_ID_RE.fullmatch(item) for item in predictions):
        return False, "predicted_next_ttps_invalid_parent_id"
    if len(set(predictions)) != 5:
        return False, "predicted_next_ttps_not_unique"
    return True, ""


def sections_present(reasoning: str) -> bool:
    return all(title in reasoning for title in SECTION_TITLES)


def sections_nonempty(reasoning: str) -> bool:
    if not sections_present(reasoning):
        return False
    for index, title in enumerate(SECTION_TITLES):
        start = reasoning.find(title) + len(title)
        end = (
            reasoning.find(SECTION_TITLES[index + 1], start)
            if index + 1 < len(SECTION_TITLES)
            else len(reasoning)
        )
        if end < start or not reasoning[start:end].strip("：: \n\t"):
            return False
    return True


def build_user_prompt(row: dict[str, str]) -> str:
    prefix = json.loads(row["prefix"])
    if not isinstance(prefix, list) or not prefix:
        raise ValueError("pilot prefix must be a non-empty JSON array")
    recent = prefix[-2:]
    return USER_PROMPT_TEMPLATE.format(
        prefix=json.dumps(prefix, ensure_ascii=False, separators=(",", ":")),
        recent_ids=json.dumps(recent, ensure_ascii=False, separators=(",", ":")),
        kg_context=row["kg_context"],
    )


def safe_response_headers(response: requests.Response) -> dict[str, str]:
    allowed = {
        "content-type",
        "date",
        "server",
        "x-request-id",
        "request-id",
        "cf-ray",
    }
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in allowed
    }


class AuditWriter:
    def __init__(self, attempts_path: Path, log_path: Path) -> None:
        self._lock = threading.Lock()
        self._attempts = attempts_path.open("x", encoding="utf-8")
        self._log = log_path.open("x", encoding="utf-8")

    def attempt(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._attempts.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._attempts.flush()

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        with self._lock:
            self._log.write(line + "\n")
            self._log.flush()
        print(line, flush=True)

    def close(self) -> None:
        with self._lock:
            self._attempts.close()
            self._log.close()


def response_request_id(response: requests.Response) -> str:
    return response.headers.get("x-request-id", response.headers.get("request-id", ""))


def deterministic_backoff(row: dict[str, str], attempt: int) -> float:
    key = f"{row['source']}|{row['campaign_id']}|{row['prefix_len']}|{attempt}"
    jitter = int(sha256_text(key)[:8], 16) / 0xFFFFFFFF
    return min(60.0, (2 ** (attempt - 1)) + jitter)


def empty_result(row: dict[str, str]) -> dict[str, Any]:
    return {
        "source": row["source"],
        "campaign_id": row["campaign_id"],
        "prefix_len": row["prefix_len"],
        "prefix": row["prefix"],
        "true_label": row["true_label"],
        "llm_thinking_process": "",
        "predicted_next_ttps": "[]",
        "raw_output": "",
        "generation_status": "api_error",
        "has_reasoning": False,
        "valid_reasoning": False,
        "schema_valid": False,
        "schema_error": "",
        "valid_top5": False,
        "three_sections_present": False,
        "sections_nonempty": False,
        "finish_reason": "",
        "model_returned": "",
        "completion_id": "",
        "system_fingerprint": "",
        "request_id": "",
        "reasoning_content_len": 0,
        "attempt": 0,
        "latency_ms": 0,
        "http_status": 0,
        "error_type": "",
        "error_message": "",
        "kg_snippet_ids": row["kg_snippet_ids"],
        "kg_chars_before": row["kg_chars_before"],
        "kg_chars_after": row["kg_chars_after"],
        "kg_empty_reason": row["kg_empty_reason"],
        "generated_at": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def token_cost(usage: dict[str, Any]) -> float:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = int(usage.get("prompt_cache_miss_tokens") or max(0, prompt_tokens - hit))
    completion = int(usage.get("completion_tokens") or 0)
    return (
        hit * PRICE_USD_PER_MILLION["prompt_cache_hit"]
        + miss * PRICE_USD_PER_MILLION["prompt_cache_miss"]
        + completion * PRICE_USD_PER_MILLION["completion"]
    ) / 1_000_000


def run_one(
    row_index: int,
    row: dict[str, str],
    api_key: str,
    vocabulary: set[str],
    audit: AuditWriter,
) -> tuple[int, dict[str, Any]]:
    user_prompt = build_user_prompt(row)
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "response_format": RESPONSE_FORMAT,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    final = empty_result(row)
    total_attempts = 1 + MAX_RETRIES
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0
    cumulative_cache_hit_tokens = 0
    cumulative_cache_miss_tokens = 0
    cumulative_cost_usd = 0.0

    for attempt in range(1, total_attempts + 1):
        started = time.monotonic()
        response: requests.Response | None = None
        response_json: dict[str, Any] | None = None
        raw_text = ""
        error_type = ""
        error_message = ""
        status_code = 0
        request_id = ""
        response_headers: dict[str, str] = {}
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            status_code = response.status_code
            request_id = response_request_id(response)
            response_headers = safe_response_headers(response)
            raw_text = response.text
            try:
                parsed_response = response.json()
                if isinstance(parsed_response, dict):
                    response_json = parsed_response
                else:
                    error_type = "response_not_object"
            except ValueError as exc:
                error_type = "http_json_decode_error"
                error_message = str(exc)
        except requests.RequestException as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

        latency_ms = round((time.monotonic() - started) * 1000)
        audit.attempt(
            {
                "recorded_at": utc_now(),
                "row_index": row_index,
                "key": [row["source"], row["campaign_id"], int(row["prefix_len"])],
                "attempt": attempt,
                "http_status": status_code,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "response_headers": response_headers,
                "response_json": response_json,
                "response_text_if_non_json": raw_text if response_json is None else "",
                "error_type": error_type,
                "error_message": error_message,
            }
        )

        final.update(
            {
                "attempt": attempt,
                "latency_ms": latency_ms,
                "http_status": status_code,
                "request_id": request_id,
                "raw_output": raw_text,
                "error_type": error_type,
                "error_message": error_message,
                "generated_at": utc_now(),
                "finish_reason": "",
                "model_returned": "",
                "completion_id": "",
                "system_fingerprint": "",
                "reasoning_content_len": 0,
                "schema_valid": False,
                "schema_error": "",
            }
        )

        retry = False
        if response is None or status_code == 429 or 500 <= status_code <= 599:
            final["generation_status"] = "api_error"
            if not error_type and response_json:
                api_error = response_json.get("error", {})
                final["error_type"] = str(api_error.get("type", "api_error"))
                final["error_message"] = str(api_error.get("message", ""))
            retry = True
        elif status_code < 200 or status_code >= 300:
            final["generation_status"] = "api_error"
            if response_json:
                api_error = response_json.get("error", {})
                final["error_type"] = str(api_error.get("type", "api_error"))
                final["error_message"] = str(api_error.get("message", ""))
            # Non-429 4xx requests are deterministic and must not be retried.
            retry = False
        elif response_json is None:
            final["generation_status"] = "json_parse_failed"
            retry = True
        else:
            choices = response_json.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            message = message if isinstance(message, dict) else {}
            content = message.get("content")
            content = content if isinstance(content, str) else ""
            reasoning_content = message.get("reasoning_content")
            reasoning_content = (
                reasoning_content if isinstance(reasoning_content, str) else ""
            )
            finish_reason = str(choice.get("finish_reason") or "")
            usage = response_json.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)
            cache_miss_tokens = int(
                usage.get("prompt_cache_miss_tokens")
                or max(0, prompt_tokens - cache_hit_tokens)
            )
            completion_tokens = int(usage.get("completion_tokens") or 0)
            cumulative_input_tokens += prompt_tokens
            cumulative_output_tokens += completion_tokens
            cumulative_cache_hit_tokens += cache_hit_tokens
            cumulative_cache_miss_tokens += cache_miss_tokens
            cumulative_cost_usd += token_cost(usage)
            final.update(
                {
                    "raw_output": content,
                    "finish_reason": finish_reason,
                    "model_returned": str(response_json.get("model") or ""),
                    "completion_id": str(response_json.get("id") or ""),
                    "system_fingerprint": str(
                        response_json.get("system_fingerprint") or ""
                    ),
                    "reasoning_content_len": len(reasoning_content),
                    "input_tokens": cumulative_input_tokens,
                    "output_tokens": cumulative_output_tokens,
                    "prompt_cache_hit_tokens": cumulative_cache_hit_tokens,
                    "prompt_cache_miss_tokens": cumulative_cache_miss_tokens,
                    "estimated_cost_usd": cumulative_cost_usd,
                }
            )
            if finish_reason == "length":
                final["generation_status"] = "truncated"
                retry = True
            elif not content.strip():
                final["generation_status"] = "empty_content"
                retry = True
            else:
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    final["generation_status"] = "json_parse_failed"
                    final["schema_error"] = f"json_decode_error:{exc}"
                    retry = True
                else:
                    schema_valid, schema_error = validate_json_schema(payload)
                    final["schema_valid"] = schema_valid
                    final["schema_error"] = schema_error
                    if not schema_valid:
                        final["generation_status"] = "json_parse_failed"
                        retry = True
                    else:
                        reasoning = payload["_thinking_process"].strip()
                        predictions = payload["predicted_next_ttps"]
                        final.update(
                            {
                                "llm_thinking_process": reasoning,
                                "predicted_next_ttps": json.dumps(
                                    predictions, ensure_ascii=False
                                ),
                                "generation_status": "ok",
                                "has_reasoning": bool(reasoning),
                                "valid_reasoning": bool(reasoning),
                                "valid_top5": (
                                    len(predictions) == 5
                                    and len(set(predictions)) == 5
                                    and all(item in vocabulary for item in predictions)
                                ),
                                "three_sections_present": sections_present(reasoning),
                                "sections_nonempty": sections_nonempty(reasoning),
                            }
                        )
                        retry = False

        if not retry or attempt == total_attempts:
            break
        delay = deterministic_backoff(row, attempt)
        audit.log(
            f"retry key={row['source']}|{row['campaign_id']}|{row['prefix_len']} "
            f"after_status={final['generation_status']} delay={delay:.3f}s"
        )
        time.sleep(delay)

    audit.log(
        f"completed {row_index + 1}/30 key={row['source']}|{row['campaign_id']}|"
        f"{row['prefix_len']} status={final['generation_status']} attempts={final['attempt']}"
    )
    return row_index, final


def get_models(api_key: str) -> tuple[requests.Response, dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("/models response is not an object")
    model_ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    if MODEL not in model_ids:
        raise ValueError(f"required model {MODEL!r} absent from /models: {sorted(model_ids)}")
    return response, payload


def validate_inputs() -> tuple[list[str], list[dict[str, str]], set[str]]:
    missing = [
        str(path)
        for path in (INPUT_PATH, PILOT_MANIFEST_PATH, VOCAB_PATH, PROTOCOL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing generation input(s): {missing}")
    input_fields, rows = read_csv(INPUT_PATH)
    if len(rows) != 30:
        raise ValueError(f"expected 30 pilot rows, found {len(rows)}")
    keys = [
        (row["source"], row["campaign_id"], int(row["prefix_len"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot keys are not unique")
    _, vocab_rows = read_csv(VOCAB_PATH)
    vocabulary = {row["technique_id_parent"].strip() for row in vocab_rows}
    if len(vocabulary) != 184:
        raise ValueError(f"expected 184 vocabulary labels, found {len(vocabulary)}")
    for row in rows:
        if row["true_label"] not in vocabulary:
            raise ValueError(f"pilot contains OOV target: {keys}")
        if int(row["kg_chars_after"]) > 700:
            raise ValueError("pilot KG exceeds 700 characters")
        build_user_prompt(row)
    return input_fields, rows, vocabulary


def config_snapshot() -> dict[str, Any]:
    return {
        "base_url": BASE_URL,
        "model_requested": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "response_format": RESPONSE_FORMAT,
        "stream": False,
        "concurrency": CONCURRENCY,
        "max_retries_after_initial_attempt": MAX_RETRIES,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "user_prompt_template_sha256": sha256_text(USER_PROMPT_TEMPLATE),
        "json_schema_sha256": sha256_text(canonical_json(JSON_SCHEMA)),
        "json_schema": JSON_SCHEMA,
        "price_usd_per_million_tokens": PRICE_USD_PER_MILLION,
        "price_source": PRICE_SOURCE,
        "dependencies": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "ssl": ssl.OPENSSL_VERSION,
            "requests": requests.__version__,
            "urllib3": urllib3.__version__,
            "executable": sys.executable,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all frozen inputs and print hashes without using the network",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_fields, rows, vocabulary = validate_inputs()
    offline = {
        "validated_at": utc_now(),
        "input_rows": len(rows),
        "unique_keys": len(
            {(row["source"], row["campaign_id"], row["prefix_len"]) for row in rows}
        ),
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in (
                INPUT_PATH,
                PILOT_MANIFEST_PATH,
                VOCAB_PATH,
                PROTOCOL_PATH,
            )
        },
        "script_sha256": sha256_file(Path(__file__)),
        "config": config_snapshot(),
    }
    if args.validate_only:
        PREFLIGHT_PATH.write_text(
            json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True))
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set; the key must be supplied via the environment"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_path = run_dir / "raw_attempts.jsonl"
    log_path = run_dir / "stdout.log"
    audit = AuditWriter(attempts_path, log_path)
    started_at = utc_now()
    audit.log(f"run_id={run_id} input_rows={len(rows)} concurrency={CONCURRENCY}")

    try:
        models_response, models_payload = get_models(api_key)
        models_record = {
            "recorded_at": utc_now(),
            "http_status": models_response.status_code,
            "request_id": response_request_id(models_response),
            "response_headers": safe_response_headers(models_response),
            "response": models_payload,
        }
        (run_dir / "models_response.json").write_text(
            json.dumps(models_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit.log(f"/models confirmed model={MODEL}")

        results: list[dict[str, Any] | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {
                executor.submit(run_one, index, row, api_key, vocabulary, audit): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
        final_rows = [result for result in results if result is not None]
        if len(final_rows) != 30:
            raise RuntimeError(f"only {len(final_rows)} final pilot rows were produced")

        final_path = run_dir / "pilot_raw_results.csv"
        final_fields = list(final_rows[0])
        write_csv(final_path, final_fields, final_rows)
        ended_at = utc_now()
        status_counts = Counter(str(row["generation_status"]) for row in final_rows)
        total_input = sum(int(row["input_tokens"]) for row in final_rows)
        total_output = sum(int(row["output_tokens"]) for row in final_rows)
        total_cost = sum(float(row["estimated_cost_usd"]) for row in final_rows)
        audit.log(
            f"finished statuses={dict(status_counts)} input_tokens={total_input} "
            f"output_tokens={total_output} estimated_cost_usd={total_cost:.8f}"
        )
        manifest = {
            **offline,
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "models_response_sha256": sha256_file(run_dir / "models_response.json"),
            "model_ids_returned": [
                item.get("id")
                for item in models_payload.get("data", [])
                if isinstance(item, dict)
            ],
            "status_counts": dict(sorted(status_counts.items())),
            "tokens": {"input": total_input, "output": total_output},
            "estimated_cost_usd": total_cost,
            "outputs": {
                "raw_attempts.jsonl": sha256_file(attempts_path),
                "stdout.log": sha256_file(log_path),
                "pilot_raw_results.csv": sha256_file(final_path),
            },
        }
        manifest_path = run_dir / "generation_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        audit.close()


if __name__ == "__main__":
    main()
