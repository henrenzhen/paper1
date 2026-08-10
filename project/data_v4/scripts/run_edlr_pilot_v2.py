#!/usr/bin/env python3
"""Validate or execute the explicitly authorized 120-completion EDLR pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import threading
import time
import uuid
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")
import requests

import run_future3_deepseek_pilot as P


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = PROJECT_ROOT / "data_v4/semantic_preflight/edlr_pilot_v2"
REQUESTS = PREFLIGHT / "request_preflight.jsonl"
INDEX = PREFLIGHT / "request_index.csv"
PREFLIGHT_MANIFEST = PREFLIGHT / "preflight_manifest.json"
SAMPLES = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.md"
ADDENDUM = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2_implementation.md"
RUNS = PROJECT_ROOT / "data_v4/external_reasoning/edlr_pilot_v2/runs"

BASE_URL = "https://api.deepseek.com"
MODEL_FAMILY = "deepseek-v4-flash"
ARMS = ("EA_TOP5", "UNION_LLM", "EDLR", "EDLR_SHUFFLE")
CONCURRENCY = 30
MAX_RETRIES = 3
TIMEOUT_SECONDS = 300
PARENT_ID = re.compile(r"^T\d{4}$")
EXPECTED_RESPONSE_KEYS = {"evidence_summary", "reranked_next_ttps"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def body_for_network(record: dict[str, Any], model_id: str) -> dict[str, Any]:
    body = copy.deepcopy(record["request_payload"])
    extra = body.pop("extra_body", None)
    if extra != {"thinking": {"type": "disabled"}}:
        raise AssertionError("frozen thinking configuration changed")
    body["thinking"] = extra["thinking"]
    body["model"] = model_id
    body["stream"] = False
    return body


def validate_response(value: Any, candidates: set[str]) -> dict[str, Any]:
    is_object = isinstance(value, dict)
    exact_keys = is_object and set(value) == EXPECTED_RESPONSE_KEYS
    summary = value.get("evidence_summary") if is_object else None
    valid_summary = exact_keys and isinstance(summary, str) and 1 <= len(summary.strip()) <= 120
    predictions = value.get("reranked_next_ttps") if is_object else None
    valid_top5 = (
        isinstance(predictions, list)
        and len(predictions) == 5
        and len(set(predictions)) == 5
        and all(isinstance(item, str) and PARENT_ID.fullmatch(item) and item in candidates for item in predictions)
    )
    return {
        "json_object": bool(is_object),
        "exact_schema_keys": bool(exact_keys),
        "valid_summary": bool(valid_summary),
        "valid_top5_in_candidates": bool(valid_top5),
        "evidence_summary": summary if isinstance(summary, str) else "",
        "reranked_next_ttps": predictions if isinstance(predictions, list) else [],
    }


def validate_inputs(model_id: str = MODEL_FAMILY) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    required = (REQUESTS, INDEX, PREFLIGHT_MANIFEST, SAMPLES, PROTOCOL, ADDENDUM)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    records = read_jsonl(REQUESTS)
    index = read_csv(INDEX)
    samples = {row["sample_id"]: row for row in read_csv(SAMPLES) if row["is_development"] == "1"}
    manifest = read_json(PREFLIGHT_MANIFEST)
    if P.sha256_file(REQUESTS) != manifest["outputs"]["request_preflight.jsonl"]:
        raise AssertionError("request preflight hash differs from manifest")
    if P.sha256_file(INDEX) != manifest["outputs"]["request_index.csv"]:
        raise AssertionError("request index hash differs from manifest")
    if len(records) != 120 or len(index) != 120 or len(samples) != 30:
        raise AssertionError("frozen 120-request/30-row denominator changed")

    keys: set[tuple[str, str]] = set()
    arm_counts: Counter[str] = Counter()
    source_arm_counts: Counter[tuple[str, str]] = Counter()
    checks: list[dict[str, Any]] = []
    for record in records:
        audit = record.get("audit_key_not_sent")
        payload = record.get("request_payload")
        arm = record.get("arm")
        if not isinstance(audit, dict) or not isinstance(payload, dict) or arm not in ARMS:
            raise AssertionError("invalid preflight record envelope")
        if record.get("network_status") != "NOT_SENT":
            raise AssertionError("preflight record is not NOT_SENT")
        sample_id = str(audit["sample_id"])
        sample = samples.get(sample_id)
        if sample is None or sample["development_slot"] != audit["development_slot"]:
            raise AssertionError(f"development mapping mismatch: {sample_id}")
        key = (sample_id, arm)
        if key in keys:
            raise AssertionError(f"duplicate sample/arm: {key}")
        keys.add(key)
        arm_counts[arm] += 1
        source_arm_counts[(sample["source"], arm)] += 1
        if P.sha256_text(P.compact_json(payload, sort_keys=True)) != record["request_payload_sha256"]:
            raise AssertionError(f"payload hash mismatch: {key}")
        candidate_set = audit.get("candidate_set")
        if not isinstance(candidate_set, list) or len(candidate_set) < 5 or len(set(candidate_set)) != len(candidate_set):
            raise AssertionError(f"invalid candidate set: {key}")
        if arm == "EA_TOP5" and candidate_set != audit.get("b0_top5"):
            raise AssertionError(f"EA candidate set differs from B0: {key}")
        if arm != "EA_TOP5" and not 5 <= len(candidate_set) <= 20:
            raise AssertionError(f"union candidate size invalid: {key}")
        body = body_for_network(record, model_id)
        body_text = P.compact_json(body, sort_keys=True)
        forbidden = [
            value
            for value in (
                sample_id,
                sample["source"],
                sample["campaign_id"],
                sample["development_slot"],
                *audit["target_step_ids_excluded"],
            )
            if value and value.casefold() in body_text.casefold()
        ]
        if forbidden:
            raise AssertionError(f"pre-send literal leakage {key}: {forbidden}")
        if body.get("thinking") != {"type": "disabled"} or body.get("stream") is not False:
            raise AssertionError(f"network transform changed: {key}")
        checks.append(
            {
                "sample_id_audit_only": sample_id,
                "development_slot_audit_only": sample["development_slot"],
                "source_audit_only": sample["source"],
                "arm": arm,
                "candidate_count": len(candidate_set),
                "transmitted_body_sha256_placeholder_model": P.sha256_text(body_text),
                "leakage_gate_passed": True,
            }
        )
    if arm_counts != Counter({arm: 30 for arm in ARMS}):
        raise AssertionError(f"arm counts changed: {arm_counts}")
    expected_source_arm = Counter({(source, arm): 10 for source in ("ctid", "attack_flow", "stockpile") for arm in ARMS})
    if source_arm_counts != expected_source_arm:
        raise AssertionError(f"source-arm balance changed: {source_arm_counts}")
    offline = {
        "validated_at": P.utc_now(),
        "network_requests_performed": 0,
        "authorization_scope": "NOT_YET_GRANTED_FOR_V2_AT_VALIDATION_TIME",
        "completion_records": len(records),
        "development_rows": len(samples),
        "arm_counts": dict(arm_counts),
        "source_arm_counts": {f"{source}:{arm}": source_arm_counts[(source, arm)] for source, arm in sorted(source_arm_counts)},
        "all_leakage_assertions_passed": True,
        "row_checks": checks,
        "inputs": {path.relative_to(PROJECT_ROOT).as_posix(): P.sha256_file(path) for path in required},
        "script_sha256": P.sha256_file(Path(__file__)),
        "network_configuration": {
            "base_url": BASE_URL,
            "model_family": MODEL_FAMILY,
            "model_id_for_snapshot": model_id,
            "temperature": 0.0,
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
            "concurrency": CONCURRENCY,
            "initial_plus_retry_ceiling": 1 + MAX_RETRIES,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    }
    return records, samples, offline


class AuditWriter:
    def __init__(self, attempts: Path, log: Path) -> None:
        self.lock = threading.Lock()
        self.attempts = attempts.open("x", encoding="utf-8")
        self.log_file = log.open("x", encoding="utf-8")

    def write_attempt(self, value: dict[str, Any]) -> None:
        with self.lock:
            self.attempts.write(P.compact_json(value) + "\n")
            self.attempts.flush()

    def log(self, message: str) -> None:
        line = f"{P.utc_now()} {message}"
        with self.lock:
            self.log_file.write(line + "\n")
            self.log_file.flush()
        print(line, flush=True)

    def close(self) -> None:
        with self.lock:
            self.attempts.close()
            self.log_file.close()


def base_result(record: dict[str, Any], sample: dict[str, str], body_sha: str) -> dict[str, Any]:
    audit = record["audit_key_not_sent"]
    return {
        "development_slot": audit["development_slot"],
        "sample_id": audit["sample_id"],
        "source_audit_only": sample["source"],
        "campaign_id_audit_only": sample["campaign_id"],
        "arm": record["arm"],
        "candidate_set_audit_only": P.compact_json(audit["candidate_set"]),
        "b0_top5_audit_only": P.compact_json(audit["b0_top5"]),
        "target_parent_ids_excluded_audit_only": P.compact_json(audit["target_parent_ids_excluded"]),
        "transmitted_body_sha256": body_sha,
        "leakage_assertions_passed": True,
        "generation_status": "api_error",
        "json_object": False,
        "exact_schema_keys": False,
        "valid_summary": False,
        "valid_top5_in_candidates": False,
        "evidence_summary": "",
        "reranked_next_ttps": "[]",
        "changed_from_b0": "",
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


def run_one(
    index: int,
    record: dict[str, Any],
    sample: dict[str, str],
    model_id: str,
    api_key: str,
    audit_writer: AuditWriter,
) -> tuple[int, dict[str, Any]]:
    body = body_for_network(record, model_id)
    body_sha = P.sha256_text(P.compact_json(body, sort_keys=True))
    result = base_result(record, sample, body_sha)
    candidates = set(record["audit_key_not_sent"]["candidate_set"])
    b0 = record["audit_key_not_sent"]["b0_top5"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    totals: dict[str, float] = defaultdict(float)
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(1, MAX_RETRIES + 2):
        started = time.monotonic()
        response = None
        response_json = None
        response_text = ""
        status = 0
        request_id = ""
        error_type = ""
        error_message = ""
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=body,
                timeout=TIMEOUT_SECONDS,
            )
            status = response.status_code
            request_id = P.response_request_id(response)
            response_text = response.text
            try:
                parsed = response.json()
                response_json = parsed if isinstance(parsed, dict) else None
            except ValueError as exc:
                error_type, error_message = "http_json_decode_error", str(exc)
        except requests.RequestException as exc:
            error_type, error_message = type(exc).__name__, str(exc)
        latency_ms = round((time.monotonic() - started) * 1000)
        usage_raw = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        usage = P.usage_numbers(usage_raw if isinstance(usage_raw, dict) else {})
        for key, value in usage.items():
            totals[key] += float(value)
        audit_writer.write_attempt(
            {
                "recorded_at": P.utc_now(),
                "audit_key_not_sent": {
                    "development_slot": result["development_slot"],
                    "sample_id": result["sample_id"],
                    "arm": result["arm"],
                },
                "attempt": attempt,
                "transmitted_body_sha256": body_sha,
                "transmitted_body": body,
                "http_status": status,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "response_headers": P.safe_response_headers(response) if response is not None else {},
                "response_json": response_json,
                "response_text_if_non_json": response_text if response_json is None else "",
                "error_type": error_type,
                "error_message": error_message,
                "usage_cost": usage,
            }
        )
        result.update(
            {
                "attempts": attempt,
                "last_http_status": status,
                "last_latency_ms": latency_ms,
                "request_id": request_id,
                "error_type": error_type,
                "error_message": error_message,
                "input_tokens_all_attempts": int(totals["prompt_tokens"]),
                "cache_hit_tokens_all_attempts": int(totals["cache_hit_tokens"]),
                "cache_miss_tokens_all_attempts": int(totals["cache_miss_tokens"]),
                "output_tokens_all_attempts": int(totals["completion_tokens"]),
                "estimated_cost_usd_all_attempts": totals["estimated_cost_usd"],
                "generated_at": P.utc_now(),
            }
        )
        retry = False
        if response is None or status in retryable:
            result["generation_status"] = "api_error"
            retry = True
        elif not 200 <= status < 300:
            result["generation_status"] = "api_error"
        elif response_json is None:
            result["generation_status"] = "http_json_parse_failed"
            retry = True
        else:
            choices = response_json.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
            finish = str(choice.get("finish_reason") or "")
            result.update(
                {
                    "raw_output": content,
                    "reasoning_content_len": len(reasoning),
                    "finish_reason": finish,
                    "model_returned": str(response_json.get("model") or ""),
                    "completion_id": str(response_json.get("id") or ""),
                }
            )
            if finish == "length":
                result["generation_status"] = "truncated"
                retry = True
            elif reasoning:
                result["generation_status"] = "thinking_not_disabled"
                retry = True
            elif not content.strip():
                result["generation_status"] = "empty_content"
                retry = True
            else:
                try:
                    response_value = json.loads(content)
                except json.JSONDecodeError as exc:
                    result["generation_status"] = "content_json_parse_failed"
                    result["error_message"] = str(exc)
                    retry = True
                else:
                    checked = validate_response(response_value, candidates)
                    reranked = checked["reranked_next_ttps"]
                    result.update(
                        {
                            **{key: checked[key] for key in ("json_object", "exact_schema_keys", "valid_summary", "valid_top5_in_candidates", "evidence_summary")},
                            "reranked_next_ttps": P.compact_json(reranked),
                            "changed_from_b0": int(reranked != b0) if checked["valid_top5_in_candidates"] else "",
                        }
                    )
                    if not checked["valid_summary"]:
                        result["generation_status"] = "invalid_summary"
                        retry = True
                    elif not checked["valid_top5_in_candidates"]:
                        result["generation_status"] = "invalid_top5"
                        retry = True
                    else:
                        result["generation_status"] = "ok"
        if not retry or attempt == MAX_RETRIES + 1:
            break
        delay = P.deterministic_backoff(f"{result['sample_id']}|{result['arm']}", attempt)
        audit_writer.log(f"retry slot={result['development_slot']} arm={result['arm']} status={result['generation_status']} delay={delay:.3f}s")
        time.sleep(delay)
    audit_writer.log(
        f"completed {index + 1}/120 slot={result['development_slot']} arm={result['arm']} "
        f"status={result['generation_status']} attempts={result['attempts']}"
    )
    return index, result


def mechanical_gates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    output: dict[str, Any] = {"by_arm": {}}
    all_pass = True
    for arm in ARMS:
        values = by_arm[arm]
        n = len(values)
        metrics = {
            "rows": n,
            "json_object_rate": sum(bool(row["json_object"]) for row in values) / n,
            "valid_summary_rate": sum(bool(row["valid_summary"]) for row in values) / n,
            "valid_top5_in_candidates_rate": sum(bool(row["valid_top5_in_candidates"]) for row in values) / n,
            "reasoning_content_empty_rate": sum(int(row["reasoning_content_len"]) == 0 for row in values) / n,
            "leakage_assertions_rate": sum(bool(row["leakage_assertions_passed"]) for row in values) / n,
        }
        passed = {
            "rows": n == 30,
            "json_object_rate": metrics["json_object_rate"] >= 0.95,
            "valid_summary_rate": metrics["valid_summary_rate"] >= 0.97,
            "valid_top5_in_candidates_rate": metrics["valid_top5_in_candidates_rate"] >= 0.95,
            "reasoning_content_empty_rate": metrics["reasoning_content_empty_rate"] == 1.0,
            "leakage_assertions_rate": metrics["leakage_assertions_rate"] == 1.0,
        }
        if arm == "EA_TOP5":
            valid = [row for row in values if row["valid_top5_in_candidates"]]
            changed = sum(int(row["changed_from_b0"]) for row in valid) / len(valid) if valid else 0.0
            metrics["changed_from_b0_rate"] = changed
            passed["changed_from_b0_rate"] = 0.10 <= changed <= 0.90
        output["by_arm"][arm] = {"metrics": metrics, "passed": passed, "all_passed": all(passed.values())}
        all_pass = all_pass and all(passed.values())
    output["all_mechanical_gates_passed"] = all_pass
    return output


def attempt_totals(path: Path) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    attempts = read_jsonl(path) if path.exists() else []
    for item in attempts:
        for key, value in item.get("usage_cost", {}).items():
            totals[key] += float(value)
    return {
        "attempts": len(attempts),
        "prompt_tokens": int(totals["prompt_tokens"]),
        "cache_hit_tokens": int(totals["cache_hit_tokens"]),
        "cache_miss_tokens": int(totals["cache_miss_tokens"]),
        "completion_tokens": int(totals["completion_tokens"]),
        "estimated_cost_usd": totals["estimated_cost_usd"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--authorized-120-requests",
        action="store_true",
        help="required only after the user explicitly authorizes /models and all 120 billed completions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, samples, offline = validate_inputs()
    if args.validate_only:
        output = PREFLIGHT / "execution_preflight.json"
        output.write_text(json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(P.compact_json({key: offline[key] for key in ("validated_at", "network_requests_performed", "completion_records", "development_rows", "arm_counts", "all_leakage_assertions_passed", "script_sha256")}, sort_keys=True))
        return
    if not args.authorized_120_requests:
        raise RuntimeError("network stage requires explicit user authorization and --authorized-120-requests")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; it must remain environment-only")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_path = run_dir / "raw_attempts.jsonl"
    log_path = run_dir / "stdout.log"
    writer = AuditWriter(attempts_path, log_path)
    started_at = P.utc_now()
    failure_stage = "models"
    writer.log(f"run_id={run_id} completion_records=120 concurrency={CONCURRENCY}")
    try:
        models_response, models_payload = P.get_models(api_key)
        write_json_exclusive(
            run_dir / "models_response.json",
            {
                "recorded_at": P.utc_now(),
                "http_status": models_response.status_code,
                "request_id": P.response_request_id(models_response),
                "response_headers": P.safe_response_headers(models_response),
                "response": models_payload,
            },
        )
        if models_response.status_code != 200:
            raise RuntimeError(f"/models returned HTTP {models_response.status_code}; no completions sent")
        model_id, selection_rule = P.choose_model(models_payload)
        records, samples, offline = validate_inputs(model_id)
        offline["authorization_scope"] = "explicit user authorization for /models and 120 billed EDLR pilot v2 completions in the active Codex task"
        writer.log(f"/models selected model={model_id} rule={selection_rule}")
        failure_stage = "generation"
        results: list[Any] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {
                executor.submit(
                    run_one,
                    index,
                    record,
                    samples[record["audit_key_not_sent"]["sample_id"]],
                    model_id,
                    api_key,
                    writer,
                ): index
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
        final = [row for row in results if row is not None]
        if len(final) != 120:
            raise RuntimeError(f"only {len(final)} final rows produced")
        write_csv_exclusive(run_dir / "pilot_raw_results.csv", final)
        gates = mechanical_gates(final)
        write_json_exclusive(run_dir / "mechanical_gate_report.json", gates)
        totals = attempt_totals(attempts_path)
        statuses = Counter(str(row["generation_status"]) for row in final)
        writer.log(
            f"finished statuses={dict(statuses)} attempts={totals['attempts']} "
            f"estimated_cost_usd={totals['estimated_cost_usd']:.8f} "
            f"all_gates_passed={gates['all_mechanical_gates_passed']}"
        )
        manifest = {
            **offline,
            "network_requests_performed": 1 + totals["attempts"],
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": P.utc_now(),
            "model_id_selected": model_id,
            "model_selection_rule": selection_rule,
            "status_counts": dict(statuses),
            "billed_attempt_totals": totals,
            "mechanical_gates": gates,
            "interpretation": "mechanical/development pilot only; no formal effectiveness claim",
            "outputs": {
                name: P.sha256_file(run_dir / name)
                for name in ("models_response.json", "raw_attempts.jsonl", "stdout.log", "pilot_raw_results.csv", "mechanical_gate_report.json")
            },
            "stop_rule": "no 784-row requests without separate explicit user authorization",
        }
        write_json_exclusive(run_dir / "generation_manifest.json", manifest)
        print(P.compact_json({"run_dir": str(run_dir), "model": model_id, "statuses": dict(statuses), "tokens_and_cost": totals, "mechanical_gates": gates}, sort_keys=True))
    except Exception as exc:
        writer.log(f"run_failed stage={failure_stage} error_type={type(exc).__name__} message={exc}")
        totals = attempt_totals(attempts_path)
        write_json_exclusive(
            run_dir / "failure_manifest.json",
            {
                **offline,
                "run_id": run_id,
                "started_at": started_at,
                "failed_at": P.utc_now(),
                "failure_stage": failure_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "billed_attempt_totals": totals,
                "existing_outputs": {path.name: P.sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file()},
            },
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
