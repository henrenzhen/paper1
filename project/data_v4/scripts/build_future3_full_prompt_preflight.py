#!/usr/bin/env python3
"""Build the 784-row future-3 full-generation preflight without networking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_future3_prompt_preflight import SYSTEM_PROMPT, USER_TEMPLATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
RAW_INPUTS_PATH = (
    PROJECT_ROOT
    / "data_v4/semantic_preflight/future3_dev_prompts_v1/raw_semantic_inputs.jsonl"
)
PILOT_RUN = (
    PROJECT_ROOT
    / "data_v4/external_reasoning/future3/pilot/runs/20260807T073425Z_1ec8fb8d"
)
PILOT_ATTEMPTS = PILOT_RUN / "raw_attempts.jsonl"
PILOT_MANIFEST = PILOT_RUN / "generation_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_preflight/future3_full_prompts_v1"

FORBIDDEN_KEY_FRAGMENTS = (
    "source",
    "campaign",
    "target",
    "future",
    "file",
    "path",
    "audit",
    "sample_id",
    "development_slot",
)
PRICE_USD_PER_MILLION = {
    "prompt_cache_hit": 0.0028,
    "prompt_cache_miss": 0.14,
    "completion": 0.28,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":")
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def pilot_usage() -> dict[str, int | float]:
    attempts = read_jsonl(PILOT_ATTEMPTS)
    if len(attempts) != 30:
        raise AssertionError("the frozen pilot no longer has exactly 30 attempts")
    totals: dict[str, int | float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "request_characters": 0,
        "estimated_cost_usd": 0.0,
    }
    for attempt in attempts:
        usage = attempt["usage_cost"]
        totals["prompt_tokens"] = int(totals["prompt_tokens"]) + int(
            usage["prompt_tokens"]
        )
        totals["completion_tokens"] = int(totals["completion_tokens"]) + int(
            usage["completion_tokens"]
        )
        totals["estimated_cost_usd"] = float(
            totals["estimated_cost_usd"]
        ) + float(usage["estimated_cost_usd"])
        totals["request_characters"] = int(
            totals["request_characters"]
        ) + len(compact_json(attempt["transmitted_body"], sort_keys=True))
    return totals


def build_cost_estimate(
    pilot: dict[str, int | float], full_request_characters: int, rows: int
) -> dict[str, Any]:
    input_tokens = round(
        int(pilot["prompt_tokens"])
        * full_request_characters
        / int(pilot["request_characters"])
    )
    output_tokens = round(int(pilot["completion_tokens"]) * rows / 30)
    no_retry_cost = (
        input_tokens * PRICE_USD_PER_MILLION["prompt_cache_miss"]
        + output_tokens * PRICE_USD_PER_MILLION["completion"]
    ) / 1_000_000
    return {
        "method": (
            "input tokens scaled by total canonical request characters from the "
            "successful 30-row pilot; output tokens scaled by row count"
        ),
        "assumes_cache_miss": True,
        "estimated_prompt_tokens": input_tokens,
        "estimated_completion_tokens": output_tokens,
        "estimated_cost_usd_no_retry": no_retry_cost,
        "mechanical_four_attempt_ceiling_usd": no_retry_cost * 4,
        "warning": "estimate only; actual billing follows API usage fields",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = (
        "full_request_preflight.jsonl",
        "full_prompt_index.csv",
        "full_prompt_manifest.json",
        "report.md",
    )
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite full preflight: {existing}")

    samples = {row["sample_id"]: row for row in read_csv(SAMPLES_PATH)}
    raw_inputs = {
        record["audit_key"]["sample_id"]: record for record in read_jsonl(RAW_INPUTS_PATH)
    }
    main_samples = [row for row in samples.values() if row["is_development"] == "0"]
    if len(samples) != 814 or len(raw_inputs) != 814 or len(main_samples) != 784:
        raise AssertionError(
            f"frozen counts changed: samples={len(samples)} raw={len(raw_inputs)} "
            f"main={len(main_samples)}"
        )

    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    campaign_counts: dict[str, set[str]] = {}
    redaction_counts: Counter[str] = Counter()
    full_request_characters = 0
    for sample in sorted(main_samples, key=lambda row: row["sample_id"]):
        sample_id = sample["sample_id"]
        raw = raw_inputs[sample_id]
        if raw["audit_key"]["is_development"]:
            raise AssertionError(f"main row marked development in raw inputs: {sample_id}")
        observed = json.loads(sample["observed_step_ids"])
        targets = json.loads(sample["target_step_ids"])
        if len(observed) != int(sample["prefix_len"]) or set(observed) & set(targets):
            raise AssertionError(f"temporal gate failed: {sample_id}")
        events = raw["model_input"]["events"]
        if len(events) != len(observed):
            raise AssertionError(f"event provenance count failed: {sample_id}")
        serialized = raw["model_input"]["serialized_observed_events"]
        if sha256_text(serialized) != raw["model_input_sha256"]:
            raise AssertionError(f"raw semantic input hash failed: {sample_id}")
        user_prompt = USER_TEMPLATE.format(serialized_observed_events=serialized)
        user_prompt, source_redactions = redact_exact(
            user_prompt, sample["source"], "[REDACTED_SOURCE]"
        )
        user_prompt, campaign_redactions = redact_exact(
            user_prompt,
            sample["campaign_id"],
            "[REDACTED_CAMPAIGN_ENTITY]",
        )
        payload = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload_text = compact_json(payload, sort_keys=True)
        forbidden_keys = recursively_forbidden_keys(payload)
        forbidden_literals = [
            value
            for value in (
                sample_id,
                sample["source"],
                sample["campaign_id"],
                *targets,
            )
            if value and value.casefold() in payload_text.casefold()
        ]
        if forbidden_keys or forbidden_literals:
            raise AssertionError(
                f"full pre-send leakage gate failed: {sample_id}; "
                f"keys={forbidden_keys} literals={forbidden_literals}"
            )
        full_request_characters += len(payload_text)
        source_counts[sample["source"]] += 1
        campaign_counts.setdefault(sample["source"], set()).add(sample["campaign_id"])
        redaction_counts["source"] += source_redactions
        redaction_counts["campaign"] += campaign_redactions
        request_sha = sha256_text(payload_text)
        requests.append(
            {
                "audit_key_not_sent": {
                    "sample_id": sample_id,
                    "observed_step_ids": observed,
                    "target_step_ids_excluded": targets,
                },
                "request_payload": payload,
                "request_payload_sha256": request_sha,
                "network_status": "NOT_SENT",
            }
        )
        index_rows.append(
            {
                "sample_id": sample_id,
                "source_audit_only": sample["source"],
                "campaign_id_audit_only": sample["campaign_id"],
                "prefix_len": sample["prefix_len"],
                "observed_events": len(events),
                "target_steps_excluded": len(targets),
                "source_literal_redactions": source_redactions,
                "campaign_literal_redactions": campaign_redactions,
                "request_payload_sha256": request_sha,
                "network_status": "NOT_SENT",
            }
        )

    expected_sources = Counter({"ctid": 263, "attack_flow": 412, "stockpile": 109})
    expected_campaigns = {"ctid": 10, "attack_flow": 35, "stockpile": 27}
    actual_campaigns = {key: len(value) for key, value in campaign_counts.items()}
    if source_counts != expected_sources or actual_campaigns != expected_campaigns:
        raise AssertionError(
            f"formal denominator changed: rows={source_counts} campaigns={actual_campaigns}"
        )

    output.mkdir(parents=True, exist_ok=False)
    with (output / "full_request_preflight.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for record in requests:
            handle.write(compact_json(record) + "\n")
    with (output / "full_prompt_index.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)

    pilot = pilot_usage()
    estimate = build_cost_estimate(pilot, full_request_characters, len(requests))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network_requests_performed": 0,
        "api_authorization_for_full_generation": False,
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in (
                SAMPLES_PATH,
                RAW_INPUTS_PATH,
                PILOT_ATTEMPTS,
                PILOT_MANIFEST,
            )
        },
        "counts": {
            "requests_prepared_not_sent": len(requests),
            "source_rows_audit_only": dict(sorted(source_counts.items())),
            "source_campaigns_audit_only": dict(sorted(actual_campaigns.items())),
            "source_literal_redactions": redaction_counts["source"],
            "campaign_literal_redactions": redaction_counts["campaign"],
        },
        "leakage_gates": {
            "passed": True,
            "rows_checked": len(requests),
            "observed_target_step_disjoint": len(requests),
            "request_payload_contains_forbidden_key": 0,
            "request_payload_contains_source_literal": 0,
            "request_payload_contains_campaign_literal": 0,
            "request_payload_contains_target_step_id": 0,
        },
        "pilot_basis": pilot,
        "cost_estimate": estimate,
        "authorization_required_before_network": {
            "rows": 784,
            "fields": [
                "observed parent techniques",
                "possible tactics",
                "cleaned historical descriptions",
                "frozen prompt instructions",
            ],
            "excluded": [
                "source",
                "campaign",
                "audit/sample identifiers",
                "future labels",
                "future descriptions",
            ],
            "model": "actual available deepseek-v4-flash",
            "billing": "token-billed DeepSeek requests",
        },
    }
    write_json(output / "full_prompt_manifest.json", manifest)
    report = "\n".join(
        [
            "# Future-3 full-generation preflight",
            "",
            "- Prepared formal requests: **784** (CTID 263, Attack Flow 412, Stockpile 109).",
            "- Preserved campaigns: **10 / 35 / 27**.",
            "- Network requests performed: **0**.",
            f"- Exact campaign literal redactions: **{redaction_counts['campaign']}**; source literal redactions: **{redaction_counts['source']}**.",
            "- All 784 provenance, temporal, structured-key, and literal leakage gates passed.",
            f"- Empirical no-retry cost estimate: **USD {estimate['estimated_cost_usd_no_retry']:.6f}**.",
            f"- Mechanical four-attempt ceiling: **USD {estimate['mechanical_four_attempt_ceiling_usd']:.6f}**.",
            "",
            "No full-generation API authorization is inferred from the 30-row pilot or from this offline build.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
