#!/usr/bin/env python3
"""Validate or run the separately authorized 784-row DeepSeek generation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_future3_deepseek_pilot import (
    ADDENDUM_PATH,
    BASE_URL,
    CONCURRENCY,
    MODEL_FAMILY,
    AuditWriter,
    attempt_totals,
    build_transmitted_body,
    choose_model,
    compact_json,
    config_snapshot,
    gate_report,
    get_models,
    read_csv,
    read_jsonl,
    recursively_forbidden_keys,
    response_request_id,
    run_one,
    safe_response_headers,
    sha256_file,
    sha256_text,
    utc_now,
    write_csv_exclusive,
    write_json_exclusive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_DIR = PROJECT_ROOT / "data_v4/semantic_preflight/future3_full_prompts_v1"
REQUESTS_PATH = PREFLIGHT_DIR / "full_request_preflight.jsonl"
PREFLIGHT_MANIFEST_PATH = PREFLIGHT_DIR / "full_prompt_manifest.json"
INDEX_PATH = PREFLIGHT_DIR / "full_prompt_index.csv"
SAMPLES_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
STEPS_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/step_text_alignment.csv"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
PROTOCOL_PATH = PROJECT_ROOT / "data_v4/protocols/future3_deepseek_full_generation_v1.md"
V8_PATH = PROJECT_ROOT / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.md"
EXECUTION_PREFLIGHT_PATH = PREFLIGHT_DIR / "full_execution_preflight.json"
RUNS_DIR = PROJECT_ROOT / "data_v4/external_reasoning/future3/full/runs"


class FullRunAuditAdapter:
    """Reuse the pilot worker while correcting its local progress denominator."""

    def __init__(self, writer: AuditWriter) -> None:
        self._writer = writer

    def attempt(self, record: dict[str, Any]) -> None:
        self._writer.attempt(record)

    def log(self, value: str) -> None:
        if value.startswith("completed "):
            value = value.replace("/30 ", "/784 ", 1)
        self._writer.log(value)


def validate_full_inputs(
    model_id: str = MODEL_FAMILY,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], set[str], dict[str, Any]]:
    required = (
        REQUESTS_PATH,
        PREFLIGHT_MANIFEST_PATH,
        INDEX_PATH,
        SAMPLES_PATH,
        STEPS_PATH,
        VOCAB_PATH,
        PROTOCOL_PATH,
        V8_PATH,
        ADDENDUM_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen full-generation input(s): {missing}")
    preflight_manifest = json.loads(PREFLIGHT_MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, expected in preflight_manifest["outputs"].items():
        actual = sha256_file(PREFLIGHT_DIR / name)
        if actual != expected:
            raise AssertionError(f"preflight output hash changed: {name}")
    records = read_jsonl(REQUESTS_PATH)
    samples = {row["sample_id"]: row for row in read_csv(SAMPLES_PATH)}
    steps = {row["stable_step_id"]: row for row in read_csv(STEPS_PATH)}
    vocabulary = {
        row["technique_id_parent"].strip() for row in read_csv(VOCAB_PATH)
    }
    if len(records) != 784 or len(vocabulary) != 184:
        raise AssertionError(
            f"frozen full counts changed: requests={len(records)} vocab={len(vocabulary)}"
        )
    source_counts: Counter[str] = Counter()
    campaigns: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    executable_records: list[dict[str, Any]] = []
    row_checks: list[dict[str, Any]] = []
    for index, original in enumerate(records):
        record = copy.deepcopy(original)
        audit = record.get("audit_key_not_sent")
        payload = record.get("request_payload")
        if (
            not isinstance(audit, dict)
            or not isinstance(payload, dict)
            or record.get("network_status") != "NOT_SENT"
        ):
            raise AssertionError("invalid full preflight envelope")
        sample_id = str(audit["sample_id"])
        sample = samples.get(sample_id)
        if sample is None or sample["is_development"] != "0":
            raise AssertionError(f"formal/development separation failed: {sample_id}")
        if sample_id in sample_ids:
            raise AssertionError(f"duplicate formal sample: {sample_id}")
        sample_ids.add(sample_id)
        if sha256_text(compact_json(payload, sort_keys=True)) != record.get(
            "request_payload_sha256"
        ):
            raise AssertionError(f"formal payload hash failed: {sample_id}")
        observed = json.loads(sample["observed_step_ids"])
        targets = json.loads(sample["target_step_ids"])
        if observed != audit.get("observed_step_ids") or targets != audit.get(
            "target_step_ids_excluded"
        ):
            raise AssertionError(f"formal step provenance failed: {sample_id}")
        if (
            len(observed) != int(sample["prefix_len"])
            or set(observed) & set(targets)
            or any(step_id not in steps for step_id in observed + targets)
        ):
            raise AssertionError(f"formal temporal gate failed: {sample_id}")
        body, redactions = build_transmitted_body(payload, sample, model_id)
        if redactions != {"source": 0, "campaign": 0}:
            raise AssertionError(
                f"preflight failed to remove metadata literal: {sample_id} {redactions}"
            )
        body_text = compact_json(body, sort_keys=True)
        forbidden_keys = recursively_forbidden_keys(body)
        forbidden_literals = [
            value
            for value in (
                sample_id,
                sample["source"],
                sample["campaign_id"],
                *targets,
            )
            if value and value.casefold() in body_text.casefold()
        ]
        if forbidden_keys or forbidden_literals:
            raise AssertionError(
                f"full execution leakage gate failed: {sample_id}; "
                f"keys={forbidden_keys} literals={forbidden_literals}"
            )
        source_counts[sample["source"]] += 1
        campaigns.setdefault(sample["source"], set()).add(sample["campaign_id"])
        # The imported, already audited request worker uses this local display
        # key. It is appended only to the non-transmitted audit envelope.
        audit["development_slot"] = f"formal:{index + 1:04d}"
        executable_records.append(record)
        row_checks.append(
            {
                "formal_slot": audit["development_slot"],
                "sample_id": sample_id,
                "source_audit_only": sample["source"],
                "campaign_audit_only": sample["campaign_id"],
                "observed_steps": len(observed),
                "target_steps_excluded": len(targets),
                "transmitted_body_sha256": sha256_text(body_text),
                "leakage_assertions_passed": True,
            }
        )
    expected_rows = Counter({"ctid": 263, "attack_flow": 412, "stockpile": 109})
    expected_campaigns = {"ctid": 10, "attack_flow": 35, "stockpile": 27}
    campaign_counts = {key: len(value) for key, value in campaigns.items()}
    if source_counts != expected_rows or campaign_counts != expected_campaigns:
        raise AssertionError(
            f"formal denominator changed: rows={source_counts} campaigns={campaign_counts}"
        )
    offline = {
        "validated_at": utc_now(),
        "network_requests_performed": 0,
        "authorization_status": "separate explicit full-generation authorization still required before network execution",
        "input_rows": len(executable_records),
        "source_counts_audit_only": dict(sorted(source_counts.items())),
        "campaign_counts_audit_only": dict(sorted(campaign_counts.items())),
        "all_leakage_assertions_passed": True,
        "row_checks": row_checks,
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in required
        },
        "script_sha256": sha256_file(Path(__file__)),
        "configuration": config_snapshot(model_id),
        "preflight_cost_estimate": preflight_manifest["cost_estimate"],
    }
    return executable_records, samples, vocabulary, offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run all local gates and perform no network request",
    )
    parser.add_argument(
        "--authorized-full-generation",
        action="store_true",
        help="required only after separate user authorization for all 784 requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, samples, vocabulary, offline = validate_full_inputs()
    if args.validate_only:
        EXECUTION_PREFLIGHT_PATH.write_text(
            json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    key: offline[key]
                    for key in (
                        "validated_at",
                        "network_requests_performed",
                        "input_rows",
                        "source_counts_audit_only",
                        "campaign_counts_audit_only",
                        "all_leakage_assertions_passed",
                        "script_sha256",
                        "preflight_cost_estimate",
                    )
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.authorized_full_generation:
        raise RuntimeError(
            "full generation is blocked: obtain separate explicit user authorization "
            "and then pass --authorized-full-generation"
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_path = run_dir / "raw_attempts.jsonl"
    log_path = run_dir / "stdout.log"
    audit_writer = AuditWriter(attempts_path, log_path)
    worker_audit = FullRunAuditAdapter(audit_writer)
    started_at = utc_now()
    failure_stage = "models"
    audit_writer.log(f"run_id={run_id} rows=784 concurrency={CONCURRENCY}")
    try:
        models_response, models_payload = get_models(api_key)
        write_json_exclusive(
            run_dir / "models_response.json",
            {
                "recorded_at": utc_now(),
                "http_status": models_response.status_code,
                "request_id": response_request_id(models_response),
                "response_headers": safe_response_headers(models_response),
                "response": models_payload,
            },
        )
        if models_response.status_code != 200:
            raise RuntimeError(
                f"/models returned HTTP {models_response.status_code}; no completions sent"
            )
        model_id, selection_rule = choose_model(models_payload)
        records, samples, vocabulary, offline = validate_full_inputs(model_id)
        offline["authorization_status"] = (
            "user separately confirmed the 784-row token-billed DeepSeek scope; "
            "runner authorization flag present"
        )
        audit_writer.log(f"/models selected model={model_id} rule={selection_rule}")
        failure_stage = "generation"
        results: list[dict[str, Any] | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {}
            for index, record in enumerate(records):
                sample = samples[record["audit_key_not_sent"]["sample_id"]]
                future = executor.submit(
                    run_one,
                    index,
                    record,
                    sample,
                    model_id,
                    api_key,
                    vocabulary,
                    worker_audit,
                )
                futures[future] = index
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
        final_rows = [row for row in results if row is not None]
        if len(final_rows) != 784:
            raise RuntimeError(f"only {len(final_rows)} formal rows produced")
        final_path = run_dir / "full_raw_results.csv"
        write_csv_exclusive(final_path, final_rows)
        gates = gate_report(final_rows)
        write_json_exclusive(run_dir / "full_generation_gate_report.json", gates)
        totals = attempt_totals(attempts_path)
        status_counts = Counter(str(row["generation_status"]) for row in final_rows)
        audit_writer.log(
            f"finished statuses={dict(status_counts)} attempts={totals['attempts']} "
            f"cost_usd={float(totals['estimated_cost_usd']):.8f} "
            f"all_gates_passed={gates['all_gates_passed']}"
        )
        manifest = {
            **offline,
            "network_requests_performed": 1 + int(totals["attempts"]),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "model_id_selected": model_id,
            "model_selection_rule": selection_rule,
            "model_ids_returned": [
                item.get("id")
                for item in models_payload.get("data", [])
                if isinstance(item, dict)
            ],
            "models_response_sha256": sha256_file(
                run_dir / "models_response.json"
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "billed_attempt_totals": totals,
            "gates": gates,
            "outputs": {
                "raw_attempts.jsonl": sha256_file(attempts_path),
                "stdout.log": sha256_file(log_path),
                "full_raw_results.csv": sha256_file(final_path),
                "full_generation_gate_report.json": sha256_file(
                    run_dir / "full_generation_gate_report.json"
                ),
            },
            "next_step": "commit raw run before any embedding or training",
        }
        write_json_exclusive(run_dir / "generation_manifest.json", manifest)
        print(
            compact_json(
                {
                    "run_dir": str(run_dir),
                    "model": model_id,
                    "statuses": dict(status_counts),
                    "tokens_and_cost": totals,
                    "gates": gates,
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        audit_writer.log(
            f"run_failed stage={failure_stage} error_type={type(exc).__name__} "
            f"message={exc}"
        )
        totals = attempt_totals(attempts_path)
        write_json_exclusive(
            run_dir / "failure_manifest.json",
            {
                **offline,
                "run_id": run_id,
                "started_at": started_at,
                "failed_at": utc_now(),
                "failure_stage": failure_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "billed_attempt_totals": totals,
                "existing_outputs": {
                    path.name: sha256_file(path)
                    for path in sorted(run_dir.iterdir())
                    if path.is_file()
                },
            },
        )
        raise
    finally:
        audit_writer.close()


if __name__ == "__main__":
    main()
