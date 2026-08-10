#!/usr/bin/env python3
"""Prepare the three frozen EDLR v2.1 format-repair requests offline."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.1_repair.md"
V2_PREFLIGHT = PROJECT_ROOT / "data_v4/semantic_preflight/edlr_pilot_v2/request_preflight.jsonl"
V2_RUN = PROJECT_ROOT / "data_v4/external_reasoning/edlr_pilot_v2/runs/20260810T031013Z_1338c419"
V2_RESULTS = V2_RUN / "pilot_raw_results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_preflight/edlr_pilot_v2_1_repair"

EXPECTED_FAILURES = {
    ("ctid:long:1", "EDLR"): "invalid_summary",
    ("ctid:medium:3", "UNION_LLM"): "invalid_top5",
    ("ctid:medium:3", "EDLR"): "invalid_top5",
}
EXPECTED_V2_RESULTS_SHA256 = "687d3351c7c767d3f03a07458849bd026c801080864bf68b05787a1958d7f630"
OUTPUT_MARKER = "\n### 输出\n"
REPAIRED_OUTPUT = """### 输出格式（硬限制）
只输出一个JSON对象，且只能有两个键：
- evidence_summary：1至80个中文字符；
- reranked_next_ttps：恰好5个互不重复的ID，必须逐字复制自上面的候选表。
输出前检查数组中每个ID均在候选表内。不要输出示例ID，不要输出JSON之外的文本。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_json(value: Any, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    output = DEFAULT_OUTPUT
    managed = ("repair_request_preflight.jsonl", "repair_manifest.json", "report.md")
    if any((output / name).exists() for name in managed):
        raise FileExistsError("refusing to overwrite v2.1 repair preflight")
    if sha256_file(V2_RESULTS) != EXPECTED_V2_RESULTS_SHA256:
        raise AssertionError("immutable v2 result hash changed")
    v2_results = read_csv(V2_RESULTS)
    actual_failures = {
        (row["development_slot"], row["arm"]): row["generation_status"]
        for row in v2_results
        if row["generation_status"] != "ok"
    }
    if actual_failures != EXPECTED_FAILURES:
        raise AssertionError(f"v2 terminal failure set changed: {actual_failures}")
    v2_requests = {
        (item["audit_key_not_sent"]["development_slot"], item["arm"]): item
        for item in read_jsonl(V2_PREFLIGHT)
    }
    prepared: list[dict[str, Any]] = []
    for key in EXPECTED_FAILURES:
        original = v2_requests[key]
        payload = json.loads(json.dumps(original["request_payload"], ensure_ascii=False))
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise AssertionError(f"unexpected v2 messages: {key}")
        user = messages[1].get("content")
        if not isinstance(user, str) or user.count(OUTPUT_MARKER) != 1:
            raise AssertionError(f"output marker changed: {key}")
        prefix, old_output = user.split(OUTPUT_MARKER, 1)
        if "T1059" not in old_output or "reranked_next_ttps" not in old_output:
            raise AssertionError(f"fixed-ID example not found: {key}")
        messages[1]["content"] = prefix + "\n\n" + REPAIRED_OUTPUT
        if messages[0] != original["request_payload"]["messages"][0]:
            raise AssertionError("system prompt changed during repair")
        if prefix != original["request_payload"]["messages"][1]["content"].split(OUTPUT_MARKER, 1)[0]:
            raise AssertionError("semantic/evidence prefix changed during repair")
        payload_text = compact_json(payload, sort_keys=True)
        prepared.append(
            {
                "audit_key_not_sent": original["audit_key_not_sent"],
                "arm": original["arm"],
                "repair_of_v2_status": EXPECTED_FAILURES[key],
                "repair_scope": "output_format_only_remove_fixed_ids_and_require_1_to_80_char_summary",
                "request_payload": payload,
                "request_payload_sha256": sha256_text(payload_text),
                "network_status": "NOT_SENT",
            }
        )
    if len(prepared) != 3:
        raise AssertionError("repair denominator is not three")
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "repair_request_preflight.jsonl"
    with request_path.open("x", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(compact_json(item) + "\n")
    report = "\n".join(
        [
            "# EDLR pilot v2.1 repair preflight",
            "",
            "- Frozen terminal failures: 3.",
            "- Prompt change: output-format section only; fixed example IDs removed; summary requested at 1--80 characters.",
            "- Semantic history, candidates, evidence, system prompt, and API settings unchanged.",
            "- Network calls: **0**; cost: **0**; all three payloads are `NOT_SENT`.",
            "",
            "A separate authorization is required for `/models` and three billed repair completions (up to three retries each).",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "offline_repair_preflight_complete_not_sent",
        "network_calls": 0,
        "api_cost": 0,
        "requests": 3,
        "authorization_for_network_stage": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            V2_PREFLIGHT.relative_to(PROJECT_ROOT).as_posix(): sha256_file(V2_PREFLIGHT),
            V2_RESULTS.relative_to(PROJECT_ROOT).as_posix(): sha256_file(V2_RESULTS),
        },
        "failure_set": [
            {"development_slot": slot, "arm": arm, "v2_status": status}
            for (slot, arm), status in EXPECTED_FAILURES.items()
        ],
        "gates": {
            "only_frozen_three_failures": True,
            "system_prompt_unchanged": True,
            "semantic_candidate_evidence_prefix_unchanged": True,
            "api_settings_unchanged": True,
            "fixed_id_example_removed": True,
            "summary_instruction_stricter_not_relaxed": True,
            "all_network_status_not_sent": True,
        },
        "outputs": {
            "repair_request_preflight.jsonl": sha256_file(request_path),
            "report.md": sha256_file(output / "report.md"),
        },
    }
    write_json(output / "repair_manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
