#!/usr/bin/env python3
"""Validate and run the authorized 30-row future-3 DeepSeek pilot."""

from __future__ import annotations

import argparse
import copy
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

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")
import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_DIR = PROJECT_ROOT / "data_v4/semantic_preflight/future3_dev_prompts_v1"
REQUESTS_PATH = PREFLIGHT_DIR / "development_request_preflight.jsonl"
PROMPT_MANIFEST_PATH = PREFLIGHT_DIR / "prompt_manifest.json"
PROMPT_INDEX_PATH = PREFLIGHT_DIR / "development_prompt_index.csv"
SAMPLES_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
STEPS_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/step_text_alignment.csv"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
PROTOCOL_PATH = PROJECT_ROOT / "data_v4/protocols/future3_deepseek_pilot_v1.md"
V8_PATH = PROJECT_ROOT / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.md"
ADDENDUM_PATH = PROJECT_ROOT / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.1_addendum.md"
EXECUTION_PREFLIGHT_PATH = PREFLIGHT_DIR / "execution_preflight.json"
RUNS_DIR = PROJECT_ROOT / "data_v4/external_reasoning/future3/pilot/runs"

BASE_URL = "https://api.deepseek.com"
MODEL_FAMILY = "deepseek-v4-flash"
CONCURRENCY = 30
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 300
PARENT_ID_RE = re.compile(r"^T\d{4}$")
ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
FORBIDDEN_KEY_FRAGMENTS = ("source", "campaign", "target", "future", "file", "path", "audit", "sample_id", "development_slot")
SUMMARY_FIELDS = ("stage_assessment", "observed_capabilities", "likely_next_intents")
EXPECTED_RESPONSE_KEYS = set(SUMMARY_FIELDS) | {"predicted_next_ttps"}
PRICE_USD_PER_MILLION = {
    "prompt_cache_hit": 0.0028,
    "prompt_cache_miss": 0.14,
    "completion": 0.28,
}
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_RETRIEVED_DATE = "2026-08-07"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_response_headers(response: requests.Response) -> dict[str, str]:
    allowed = {"content-type", "date", "server", "x-request-id", "request-id", "cf-ray"}
    return {key: value for key, value in response.headers.items() if key.lower() in allowed}


def response_request_id(response: requests.Response) -> str:
    return response.headers.get("x-request-id", response.headers.get("request-id", ""))


def recursively_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                found.append(f"{path}.{key}")
            found.extend(recursively_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(recursively_forbidden_keys(item, f"{path}[{index}]"))
    return found


def redact_exact(text: str, value: str, replacement: str) -> tuple[str, int]:
    if not value:
        return text, 0
    return re.subn(re.escape(value), replacement, text, flags=re.IGNORECASE)


def build_transmitted_body(
    request_payload: dict[str, Any], sample: dict[str, str], model_id: str
) -> tuple[dict[str, Any], dict[str, int]]:
    body = copy.deepcopy(request_payload)
    extra = body.pop("extra_body", None)
    if extra != {"thinking": {"type": "disabled"}}:
        raise AssertionError("frozen extra_body/thinking configuration changed")
    body["thinking"] = extra["thinking"]
    body["model"] = model_id
    body["stream"] = False
    redactions = {"source": 0, "campaign": 0}
    for message in body.get("messages", []):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise AssertionError("invalid frozen message structure")
        content, count = redact_exact(message["content"], sample["source"], "[REDACTED_SOURCE]")
        redactions["source"] += count
        content, count = redact_exact(content, sample["campaign_id"], "[REDACTED_CAMPAIGN_ENTITY]")
        redactions["campaign"] += count
        message["content"] = content
    return body, redactions


def validate_response_payload(payload: Any, vocabulary: set[str]) -> dict[str, Any]:
    json_object = isinstance(payload, dict)
    exact_keys = json_object and set(payload) == EXPECTED_RESPONSE_KEYS
    summaries = [payload.get(field) if json_object else None for field in SUMMARY_FIELDS]
    valid_summary = exact_keys and all(isinstance(value, str) and value.strip() for value in summaries)
    summary_text = "\n".join(value for value in summaries if isinstance(value, str))
    summary_attack_id = bool(ATTACK_ID_RE.search(summary_text))
    predictions = payload.get("predicted_next_ttps") if json_object else None
    prediction_strings = (
        isinstance(predictions, list)
        and all(isinstance(item, str) for item in predictions)
    )
    valid_top5 = (
        prediction_strings
        and len(predictions) == 5
        and len(set(predictions)) == 5
        and all(PARENT_ID_RE.fullmatch(item) and item in vocabulary for item in predictions)
    )
    return {
        "json_object": bool(json_object),
        "exact_schema_keys": bool(exact_keys),
        "valid_summary": bool(valid_summary),
        "summary_attack_id": summary_attack_id,
        "valid_top5": bool(valid_top5),
        "summary_fields": {field: payload.get(field, "") if json_object else "" for field in SUMMARY_FIELDS},
        "predicted_next_ttps": predictions if isinstance(predictions, list) else [],
    }


def usage_numbers(usage: dict[str, Any]) -> dict[str, int | float]:
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = int(usage.get("prompt_cache_miss_tokens") or max(0, prompt - hit))
    completion = int(usage.get("completion_tokens") or 0)
    cost = (
        hit * PRICE_USD_PER_MILLION["prompt_cache_hit"]
        + miss * PRICE_USD_PER_MILLION["prompt_cache_miss"]
        + completion * PRICE_USD_PER_MILLION["completion"]
    ) / 1_000_000
    return {"prompt_tokens": prompt, "cache_hit_tokens": hit, "cache_miss_tokens": miss, "completion_tokens": completion, "estimated_cost_usd": cost}


def deterministic_backoff(sample_id: str, attempt: int) -> float:
    jitter = int(sha256_text(f"{sample_id}|{attempt}")[:8], 16) / 0xFFFFFFFF
    return min(30.0, 2 ** (attempt - 1) + jitter)


class AuditWriter:
    def __init__(self, attempts_path: Path, log_path: Path) -> None:
        self._lock = threading.Lock()
        self._attempts = attempts_path.open("x", encoding="utf-8")
        self._log = log_path.open("x", encoding="utf-8")

    def attempt(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._attempts.write(compact_json(record) + "\n")
            self._attempts.flush()

    def log(self, value: str) -> None:
        line = f"{utc_now()} {value}"
        with self._lock:
            self._log.write(line + "\n")
            self._log.flush()
        print(line, flush=True)

    def close(self) -> None:
        with self._lock:
            self._attempts.close()
            self._log.close()


def validate_inputs(model_id: str = MODEL_FAMILY) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], set[str], dict[str, Any]]:
    required = [REQUESTS_PATH, PROMPT_MANIFEST_PATH, PROMPT_INDEX_PATH, SAMPLES_PATH, STEPS_PATH, VOCAB_PATH, PROTOCOL_PATH, V8_PATH, ADDENDUM_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    records = read_jsonl(REQUESTS_PATH)
    samples = {row["sample_id"]: row for row in read_csv(SAMPLES_PATH)}
    steps = {row["stable_step_id"]: row for row in read_csv(STEPS_PATH)}
    vocab_rows = read_csv(VOCAB_PATH)
    vocabulary = {row["technique_id_parent"].strip() for row in vocab_rows}
    if len(records) != 30 or len(vocabulary) != 184:
        raise AssertionError(f"frozen count changed: requests={len(records)} vocab={len(vocabulary)}")
    slots: set[str] = set()
    sample_ids: set[str] = set()
    row_checks: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    total_redactions: Counter[str] = Counter()
    for record in records:
        audit = record.get("audit_key_not_sent")
        payload = record.get("request_payload")
        if not isinstance(audit, dict) or not isinstance(payload, dict) or record.get("network_status") != "NOT_SENT":
            raise AssertionError("invalid preflight record envelope")
        sample_id = str(audit["sample_id"])
        slot = str(audit["development_slot"])
        sample = samples.get(sample_id)
        if sample is None or sample["is_development"] != "1" or sample["development_slot"] != slot:
            raise AssertionError(f"development mapping mismatch: {sample_id}")
        if sample_id in sample_ids or slot in slots:
            raise AssertionError("duplicate sample ID or development slot")
        sample_ids.add(sample_id)
        slots.add(slot)
        source_counts[sample["source"]] += 1
        if sha256_text(compact_json(payload)) != record.get("request_payload_sha256"):
            raise AssertionError(f"preflight payload hash mismatch: {sample_id}")
        observed = json.loads(sample["observed_step_ids"])
        targets = json.loads(sample["target_step_ids"])
        if observed != audit.get("observed_step_ids") or targets != audit.get("target_step_ids_excluded"):
            raise AssertionError(f"step provenance mismatch: {sample_id}")
        if set(observed) & set(targets) or len(observed) != int(sample["prefix_len"]):
            raise AssertionError(f"history/target temporal gate failed: {sample_id}")
        if any(step_id not in steps for step_id in observed + targets):
            raise AssertionError(f"unknown step ID: {sample_id}")
        body, redactions = build_transmitted_body(payload, sample, model_id)
        total_redactions.update(redactions)
        body_text = compact_json(body, sort_keys=True)
        forbidden_keys = recursively_forbidden_keys(body)
        forbidden_literals = [
            value
            for value in (sample_id, slot, sample["source"], sample["campaign_id"], *targets)
            if value and value.casefold() in body_text.casefold()
        ]
        if forbidden_keys or forbidden_literals:
            raise AssertionError(f"pre-send leakage gate failed for {sample_id}: keys={forbidden_keys} literals={forbidden_literals}")
        if body.get("thinking") != {"type": "disabled"} or "extra_body" in body:
            raise AssertionError("thinking transmission transform failed")
        row_checks.append({
            "development_slot": slot,
            "sample_id": sample_id,
            "source_audit_only": sample["source"],
            "campaign_audit_only": sample["campaign_id"],
            "observed_step_count": len(observed),
            "target_step_count_excluded": len(targets),
            "source_literal_redactions": redactions["source"],
            "campaign_literal_redactions": redactions["campaign"],
            "transmitted_body_sha256_with_placeholder_model": sha256_text(body_text),
            "leakage_assertions_passed": True,
        })
    if source_counts != Counter({"ctid": 10, "attack_flow": 10, "stockpile": 10}):
        raise AssertionError(f"source balance changed: {source_counts}")
    offline = {
        "validated_at": utc_now(),
        "network_requests_performed": 0,
        "authorization_scope": "user-confirmed in Codex task on 2026-08-07 for /models and 30 billed future-3 development requests only",
        "input_rows": len(records),
        "source_counts_audit_only": dict(sorted(source_counts.items())),
        "redactions_required": dict(total_redactions),
        "all_leakage_assertions_passed": all(row["leakage_assertions_passed"] for row in row_checks),
        "row_checks": sorted(row_checks, key=lambda row: row["development_slot"]),
        "inputs": {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in required},
        "script_sha256": sha256_file(Path(__file__)),
        "configuration": config_snapshot(model_id),
    }
    return records, samples, vocabulary, offline


def config_snapshot(model_id: str) -> dict[str, Any]:
    return {
        "base_url": BASE_URL,
        "model_family_requested": MODEL_FAMILY,
        "model_id_for_snapshot": model_id,
        "temperature": 0.0,
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "stream": False,
        "concurrency": CONCURRENCY,
        "max_retries_after_initial_attempt": MAX_RETRIES,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "price_usd_per_million_tokens": PRICE_USD_PER_MILLION,
        "price_source": PRICE_SOURCE,
        "price_retrieved_date": PRICE_RETRIEVED_DATE,
        "dependencies": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "ssl": ssl.OPENSSL_VERSION,
            "requests": requests.__version__,
            "urllib3": urllib3.__version__,
            "executable": sys.executable,
        },
    }


def choose_model(models_payload: dict[str, Any]) -> tuple[str, str]:
    ids = sorted({str(item.get("id")) for item in models_payload.get("data", []) if isinstance(item, dict) and item.get("id")})
    if MODEL_FAMILY in ids:
        return MODEL_FAMILY, "exact_match"
    casefold = [item for item in ids if item.casefold() == MODEL_FAMILY.casefold()]
    if casefold:
        return casefold[0], "case_insensitive_exact_match"
    normalized_family = MODEL_FAMILY.casefold().replace("_", "-")
    compatible = [item for item in ids if normalized_family in item.casefold().replace("_", "-")]
    if compatible:
        return compatible[0], "lexicographically_first_family_match"
    raise RuntimeError(f"no available model matches {MODEL_FAMILY!r}; returned IDs={ids}")


def get_models(api_key: str) -> tuple[requests.Response, dict[str, Any]]:
    response = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    try:
        value = response.json()
    except ValueError:
        value = {"non_json_response": response.text}
    return response, value if isinstance(value, dict) else {"response": value}


def empty_result(record: dict[str, Any], sample: dict[str, str], redactions: dict[str, int], body_sha: str) -> dict[str, Any]:
    audit = record["audit_key_not_sent"]
    return {
        "development_slot": audit["development_slot"],
        "sample_id": audit["sample_id"],
        "source_audit_only": sample["source"],
        "campaign_id_audit_only": sample["campaign_id"],
        "prefix_len": sample["prefix_len"],
        "observed_step_ids_audit_only": compact_json(audit["observed_step_ids"]),
        "target_step_ids_excluded_audit_only": compact_json(audit["target_step_ids_excluded"]),
        "transmitted_body_sha256": body_sha,
        "source_literal_redactions": redactions["source"],
        "campaign_literal_redactions": redactions["campaign"],
        "leakage_assertions_passed": True,
        "generation_status": "api_error",
        "json_object": False,
        "exact_schema_keys": False,
        "valid_summary": False,
        "summary_attack_id": False,
        "valid_top5": False,
        "stage_assessment": "",
        "observed_capabilities": "",
        "likely_next_intents": "",
        "predicted_next_ttps": "[]",
        "raw_output": "",
        "reasoning_content_len": 0,
        "finish_reason": "",
        "model_returned": "",
        "completion_id": "",
        "request_id": "",
        "attempts": 0,
        "last_http_status": 0,
        "last_latency_ms": 0,
        "error_type": "",
        "error_message": "",
        "input_tokens_all_attempts": 0,
        "cache_hit_tokens_all_attempts": 0,
        "cache_miss_tokens_all_attempts": 0,
        "output_tokens_all_attempts": 0,
        "estimated_cost_usd_all_attempts": 0.0,
        "generated_at": "",
    }


def run_one(index: int, record: dict[str, Any], sample: dict[str, str], model_id: str, api_key: str, vocabulary: set[str], audit_writer: AuditWriter) -> tuple[int, dict[str, Any]]:
    body, redactions = build_transmitted_body(record["request_payload"], sample, model_id)
    body_text = compact_json(body, sort_keys=True)
    body_sha = sha256_text(body_text)
    result = empty_result(record, sample, redactions, body_sha)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    totals = {"prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0}
    for attempt in range(1, MAX_RETRIES + 2):
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
            response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
            status_code = response.status_code
            request_id = response_request_id(response)
            response_headers = safe_response_headers(response)
            raw_text = response.text
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    response_json = parsed
                else:
                    error_type = "response_not_object"
            except ValueError as exc:
                error_type, error_message = "http_json_decode_error", str(exc)
        except requests.RequestException as exc:
            error_type, error_message = type(exc).__name__, str(exc)
        latency_ms = round((time.monotonic() - started) * 1000)
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        numbers = usage_numbers(usage)
        for key in totals:
            totals[key] += numbers[key]
        audit_writer.attempt({
            "recorded_at": utc_now(),
            "audit_key_not_sent": {"development_slot": result["development_slot"], "sample_id": result["sample_id"]},
            "attempt": attempt,
            "transmitted_body_sha256": body_sha,
            "transmitted_body": body,
            "http_status": status_code,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "response_headers": response_headers,
            "response_json": response_json,
            "response_text_if_non_json": raw_text if response_json is None else "",
            "error_type": error_type,
            "error_message": error_message,
            "usage_cost": numbers,
        })
        result.update({
            "attempts": attempt,
            "last_http_status": status_code,
            "last_latency_ms": latency_ms,
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "generated_at": utc_now(),
            "input_tokens_all_attempts": int(totals["prompt_tokens"]),
            "cache_hit_tokens_all_attempts": int(totals["cache_hit_tokens"]),
            "cache_miss_tokens_all_attempts": int(totals["cache_miss_tokens"]),
            "output_tokens_all_attempts": int(totals["completion_tokens"]),
            "estimated_cost_usd_all_attempts": float(totals["estimated_cost_usd"]),
        })
        retry = False
        if response is None or status_code == 429 or 500 <= status_code <= 599:
            result["generation_status"] = "api_error"
            retry = True
        elif status_code < 200 or status_code >= 300:
            result["generation_status"] = "api_error"
            if response_json:
                api_error = response_json.get("error", {})
                if isinstance(api_error, dict):
                    result["error_type"] = str(api_error.get("type") or "api_error")
                    result["error_message"] = str(api_error.get("message") or "")
        elif response_json is None:
            result["generation_status"] = "http_json_parse_failed"
            retry = True
        else:
            choices = response_json.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            message = message if isinstance(message, dict) else {}
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            reasoning_content = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
            finish_reason = str(choice.get("finish_reason") or "")
            result.update({
                "raw_output": content,
                "reasoning_content_len": len(reasoning_content),
                "finish_reason": finish_reason,
                "model_returned": str(response_json.get("model") or ""),
                "completion_id": str(response_json.get("id") or ""),
            })
            if finish_reason == "length":
                result["generation_status"] = "truncated"
                retry = True
            elif reasoning_content:
                result["generation_status"] = "thinking_not_disabled"
                retry = True
            elif not content.strip():
                result["generation_status"] = "empty_content"
                retry = True
            else:
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    result["generation_status"] = "content_json_parse_failed"
                    result["error_message"] = str(exc)
                    retry = True
                else:
                    checks = validate_response_payload(payload, vocabulary)
                    result.update({
                        **{key: checks[key] for key in ("json_object", "exact_schema_keys", "valid_summary", "summary_attack_id", "valid_top5")},
                        **checks["summary_fields"],
                        "predicted_next_ttps": compact_json(checks["predicted_next_ttps"]),
                    })
                    if not checks["valid_summary"]:
                        result["generation_status"] = "invalid_summary"
                        retry = True
                    elif checks["summary_attack_id"]:
                        result["generation_status"] = "summary_contains_attack_id"
                        retry = True
                    elif not checks["valid_top5"]:
                        result["generation_status"] = "invalid_top5"
                        retry = True
                    else:
                        result["generation_status"] = "ok"
        if not retry or attempt == MAX_RETRIES + 1:
            break
        delay = deterministic_backoff(result["sample_id"], attempt)
        audit_writer.log(f"retry slot={result['development_slot']} status={result['generation_status']} delay={delay:.3f}s")
        time.sleep(delay)
    audit_writer.log(f"completed {index + 1}/30 slot={result['development_slot']} status={result['generation_status']} attempts={result['attempts']}")
    return index, result


def gate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    metrics = {
        "json_success_rate": sum(bool(row["json_object"]) for row in rows) / n,
        "reasoning_content_empty_rate": sum(int(row["reasoning_content_len"]) == 0 for row in rows) / n,
        "finish_reason_length_rate": sum(row["finish_reason"] == "length" for row in rows) / n,
        "valid_summary_rate": sum(bool(row["valid_summary"]) for row in rows) / n,
        "summary_attack_id_rate": sum(bool(row["summary_attack_id"]) for row in rows) / n,
        "valid_top5_rate": sum(bool(row["valid_top5"]) for row in rows) / n,
        "leakage_assertions_rate": sum(bool(row["leakage_assertions_passed"]) for row in rows) / n,
    }
    pass_by_gate = {
        "json_success_rate": metrics["json_success_rate"] >= 0.95,
        "reasoning_content_empty_rate": metrics["reasoning_content_empty_rate"] == 1.0,
        "finish_reason_length_rate": metrics["finish_reason_length_rate"] <= 0.02,
        "valid_summary_rate": metrics["valid_summary_rate"] >= 0.95,
        "summary_attack_id_rate": metrics["summary_attack_id_rate"] == 0.0,
        "valid_top5_rate": metrics["valid_top5_rate"] >= 0.90,
        "leakage_assertions_rate": metrics["leakage_assertions_rate"] == 1.0,
    }
    return {"rows": n, "metrics": metrics, "passed_by_gate": pass_by_gate, "all_gates_passed": all(pass_by_gate.values())}


def attempt_totals(path: Path) -> dict[str, int | float]:
    totals: dict[str, int | float] = {"attempts": 0, "prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0}
    if not path.exists():
        return totals
    for record in read_jsonl(path):
        totals["attempts"] = int(totals["attempts"]) + 1
        cost = record.get("usage_cost", {})
        for key in ("prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens"):
            totals[key] = int(totals[key]) + int(cost.get(key) or 0)
        totals["estimated_cost_usd"] = float(totals["estimated_cost_usd"]) + float(cost.get("estimated_cost_usd") or 0)
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="run all local gates and write no network request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, samples, vocabulary, offline = validate_inputs()
    if args.validate_only:
        EXECUTION_PREFLIGHT_PATH.write_text(json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: offline[key] for key in ("validated_at", "network_requests_performed", "input_rows", "source_counts_audit_only", "redactions_required", "all_leakage_assertions_passed", "script_sha256")}, ensure_ascii=False, indent=2, sort_keys=True))
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; it must be supplied only through the environment")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_path = run_dir / "raw_attempts.jsonl"
    log_path = run_dir / "stdout.log"
    audit_writer = AuditWriter(attempts_path, log_path)
    started_at = utc_now()
    failure_stage = "models"
    audit_writer.log(f"run_id={run_id} rows=30 concurrency={CONCURRENCY}")
    try:
        models_response, models_payload = get_models(api_key)
        models_record = {
            "recorded_at": utc_now(),
            "http_status": models_response.status_code,
            "request_id": response_request_id(models_response),
            "response_headers": safe_response_headers(models_response),
            "response": models_payload,
        }
        write_json_exclusive(run_dir / "models_response.json", models_record)
        if models_response.status_code != 200:
            raise RuntimeError(f"/models returned HTTP {models_response.status_code}; no completions sent")
        model_id, selection_rule = choose_model(models_payload)
        records, samples, vocabulary, offline = validate_inputs(model_id)
        audit_writer.log(f"/models selected model={model_id} rule={selection_rule}")
        failure_stage = "generation"
        results: list[dict[str, Any] | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {}
            for index, record in enumerate(records):
                sample = samples[record["audit_key_not_sent"]["sample_id"]]
                future = executor.submit(run_one, index, record, sample, model_id, api_key, vocabulary, audit_writer)
                futures[future] = index
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
        final_rows = [row for row in results if row is not None]
        if len(final_rows) != 30:
            raise RuntimeError(f"only {len(final_rows)} final rows produced")
        final_path = run_dir / "pilot_raw_results.csv"
        write_csv_exclusive(final_path, final_rows)
        gates = gate_report(final_rows)
        write_json_exclusive(run_dir / "pilot_gate_report.json", gates)
        totals = attempt_totals(attempts_path)
        status_counts = Counter(str(row["generation_status"]) for row in final_rows)
        audit_writer.log(f"finished statuses={dict(status_counts)} attempts={totals['attempts']} cost_usd={float(totals['estimated_cost_usd']):.8f} all_gates_passed={gates['all_gates_passed']}")
        manifest = {
            **offline,
            "network_requests_performed": 1 + int(totals["attempts"]),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "model_id_selected": model_id,
            "model_selection_rule": selection_rule,
            "model_ids_returned": [item.get("id") for item in models_payload.get("data", []) if isinstance(item, dict)],
            "models_response_sha256": sha256_file(run_dir / "models_response.json"),
            "status_counts": dict(sorted(status_counts.items())),
            "billed_attempt_totals": totals,
            "gates": gates,
            "outputs": {
                "raw_attempts.jsonl": sha256_file(attempts_path),
                "stdout.log": sha256_file(log_path),
                "pilot_raw_results.csv": sha256_file(final_path),
                "pilot_gate_report.json": sha256_file(run_dir / "pilot_gate_report.json"),
            },
            "stop_rule": "pilot complete; no full-corpus generation without separate user approval",
        }
        write_json_exclusive(run_dir / "generation_manifest.json", manifest)
        print(compact_json({"run_dir": str(run_dir), "model": model_id, "statuses": dict(status_counts), "tokens_and_cost": totals, "gates": gates}, sort_keys=True))
    except Exception as exc:
        audit_writer.log(f"run_failed stage={failure_stage} error_type={type(exc).__name__} message={exc}")
        totals = attempt_totals(attempts_path)
        write_json_exclusive(run_dir / "failure_manifest.json", {
            **offline,
            "run_id": run_id,
            "started_at": started_at,
            "failed_at": utc_now(),
            "failure_stage": failure_stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "billed_attempt_totals": totals,
            "existing_outputs": {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file()},
        })
        raise
    finally:
        audit_writer.close()


if __name__ == "__main__":
    main()
