#!/usr/bin/env python3
"""Validate or run the explicitly authorized three-row EDLR v2.1 repair."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_edlr_pilot_v2 as R
import run_future3_deepseek_pilot as P


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = PROJECT_ROOT / "data_v4/semantic_preflight/edlr_pilot_v2_1_repair"
REQUESTS = PREFLIGHT / "repair_request_preflight.jsonl"
MANIFEST = PREFLIGHT / "repair_manifest.json"
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.1_repair.md"
SAMPLES = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
V2_RUN = PROJECT_ROOT / "data_v4/external_reasoning/edlr_pilot_v2/runs/20260810T031013Z_1338c419"
V2_RESULTS = V2_RUN / "pilot_raw_results.csv"
RUNS = PROJECT_ROOT / "data_v4/external_reasoning/edlr_pilot_v2_1_repair/runs"
EXPECTED_KEYS = {
    ("ctid:long:1", "EDLR"),
    ("ctid:medium:3", "UNION_LLM"),
    ("ctid:medium:3", "EDLR"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_inputs(model_id: str = R.MODEL_FAMILY) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    repair_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = read_jsonl(REQUESTS)
    samples = {row["sample_id"]: row for row in read_csv(SAMPLES) if row["is_development"] == "1"}
    if P.sha256_file(REQUESTS) != repair_manifest["outputs"]["repair_request_preflight.jsonl"]:
        raise AssertionError("repair request hash differs from manifest")
    if len(records) != 3:
        raise AssertionError("repair request denominator changed")
    keys = {(item["audit_key_not_sent"]["development_slot"], item["arm"]) for item in records}
    if keys != EXPECTED_KEYS:
        raise AssertionError(f"repair key set changed: {keys}")
    checks = []
    for record in records:
        audit = record["audit_key_not_sent"]
        sample = samples[audit["sample_id"]]
        if record["network_status"] != "NOT_SENT":
            raise AssertionError("repair record is not NOT_SENT")
        if P.sha256_text(P.compact_json(record["request_payload"], sort_keys=True)) != record["request_payload_sha256"]:
            raise AssertionError("repair payload hash mismatch")
        body = R.body_for_network(record, model_id)
        text = P.compact_json(body, sort_keys=True)
        forbidden = [
            value
            for value in (
                audit["sample_id"],
                audit["development_slot"],
                sample["source"],
                sample["campaign_id"],
                *audit["target_step_ids_excluded"],
            )
            if value and value.casefold() in text.casefold()
        ]
        if forbidden:
            raise AssertionError(f"repair pre-send leakage: {forbidden}")
        checks.append(
            {
                "sample_id_audit_only": audit["sample_id"],
                "development_slot_audit_only": audit["development_slot"],
                "arm": record["arm"],
                "candidate_count": len(audit["candidate_set"]),
                "transmitted_body_sha256_placeholder_model": P.sha256_text(text),
                "leakage_gate_passed": True,
            }
        )
    offline = {
        "validated_at": P.utc_now(),
        "network_requests_performed": 0,
        "authorization_scope": "NOT_YET_GRANTED_FOR_V2_1_AT_VALIDATION_TIME",
        "repair_records": 3,
        "all_leakage_assertions_passed": True,
        "row_checks": checks,
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): P.sha256_file(path)
            for path in (REQUESTS, MANIFEST, PROTOCOL, SAMPLES, V2_RESULTS)
        },
        "script_sha256": P.sha256_file(Path(__file__)),
    }
    return records, samples, offline


def normalize_csv_row(row: dict[str, str]) -> dict[str, Any]:
    value: dict[str, Any] = dict(row)
    for key in (
        "leakage_assertions_passed",
        "json_object",
        "exact_schema_keys",
        "valid_summary",
        "valid_top5_in_candidates",
    ):
        value[key] = row[key].casefold() == "true"
    value["reasoning_content_len"] = int(row["reasoning_content_len"])
    if row["changed_from_b0"] != "":
        value["changed_from_b0"] = int(row["changed_from_b0"])
    return value


def merge_results(repairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = [normalize_csv_row(row) for row in read_csv(V2_RESULTS)]
    replacement = {(row["development_slot"], row["arm"]): row for row in repairs}
    if set(replacement) != EXPECTED_KEYS:
        raise AssertionError("repair output key set changed")
    merged = []
    provenance = []
    for row in original:
        key = (row["development_slot"], row["arm"])
        if key in replacement:
            new = replacement[key]
            provenance.append(
                {
                    "development_slot": key[0],
                    "arm": key[1],
                    "v2_status": row["generation_status"],
                    "v2_1_status": new["generation_status"],
                    "v2_transmitted_body_sha256": row["transmitted_body_sha256"],
                    "v2_1_transmitted_body_sha256": new["transmitted_body_sha256"],
                }
            )
            merged.append(new)
        else:
            merged.append(row)
    if len(merged) != 120:
        raise AssertionError("merged result denominator changed")
    return merged, {"replacements": provenance, "mechanical_gates": R.mechanical_gates(merged)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--authorized-3-requests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, samples, offline = validate_inputs()
    if args.validate_only:
        (PREFLIGHT / "execution_preflight.json").write_text(
            json.dumps(offline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(P.compact_json({"repair_records": 3, "network_requests_performed": 0, "all_leakage_assertions_passed": True, "script_sha256": offline["script_sha256"]}, sort_keys=True))
        return
    if not args.authorized_3_requests:
        raise RuntimeError("network stage requires new explicit authorization and --authorized-3-requests")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; it must remain environment-only")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts = run_dir / "raw_attempts.jsonl"
    log = run_dir / "stdout.log"
    writer = R.AuditWriter(attempts, log)
    started = P.utc_now()
    stage = "models"
    writer.log(f"run_id={run_id} repair_records=3 concurrency=3")
    try:
        models_response, models_payload = P.get_models(api_key)
        R.write_json_exclusive(
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
        model_id, selection = P.choose_model(models_payload)
        records, samples, offline = validate_inputs(model_id)
        offline["authorization_scope"] = "explicit user authorization for /models and three billed v2.1 repair completions in the active Codex task"
        writer.log(f"/models selected model={model_id} rule={selection}")
        stage = "generation"
        results: list[Any] = [None] * 3
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    R.run_one,
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
        repaired = [row for row in results if row is not None]
        if len(repaired) != 3:
            raise RuntimeError("repair output denominator changed")
        R.write_csv_exclusive(run_dir / "repair_raw_results.csv", repaired)
        merged, gate = merge_results(repaired)
        R.write_csv_exclusive(run_dir / "merged_pilot_results.csv", merged)
        R.write_json_exclusive(run_dir / "merged_mechanical_gate_report.json", gate)
        totals = R.attempt_totals(attempts)
        statuses = Counter(str(row["generation_status"]) for row in repaired)
        writer.log(
            f"finished statuses={dict(statuses)} attempts={totals['attempts']} "
            f"estimated_cost_usd={totals['estimated_cost_usd']:.8f} "
            f"merged_gates={gate['mechanical_gates']['all_mechanical_gates_passed']}"
        )
        manifest = {
            **offline,
            "network_requests_performed": 1 + totals["attempts"],
            "run_id": run_id,
            "started_at": started,
            "ended_at": P.utc_now(),
            "model_id_selected": model_id,
            "model_selection_rule": selection,
            "repair_status_counts": dict(statuses),
            "billed_attempt_totals": totals,
            "merged_mechanical_gates": gate,
            "targets_opened_for_effectiveness": False,
            "outputs": {
                name: P.sha256_file(run_dir / name)
                for name in (
                    "models_response.json",
                    "raw_attempts.jsonl",
                    "stdout.log",
                    "repair_raw_results.csv",
                    "merged_pilot_results.csv",
                    "merged_mechanical_gate_report.json",
                )
            },
        }
        R.write_json_exclusive(run_dir / "generation_manifest.json", manifest)
        print(P.compact_json({"run_dir": str(run_dir), "statuses": dict(statuses), "tokens_and_cost": totals, "merged_mechanical_gates": gate["mechanical_gates"]}, sort_keys=True))
    except Exception as exc:
        writer.log(f"run_failed stage={stage} error_type={type(exc).__name__} message={exc}")
        R.write_json_exclusive(
            run_dir / "failure_manifest.json",
            {
                **offline,
                "run_id": run_id,
                "started_at": started,
                "failed_at": P.utc_now(),
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "billed_attempt_totals": R.attempt_totals(attempts),
            },
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
