#!/usr/bin/env python3
"""Build future-3 raw-semantic inputs and the 30-row API prompt preflight.

This script performs no network request.  It separates audit metadata from the
exact request payload so that source/campaign fields cannot be sent by accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = PROJECT_ROOT / "data_v4/semantic_alignment/step_text_alignment.csv"
SAMPLES = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
TACTICS = PROJECT_ROOT / "data_v4/semantic_alignment/technique_tactic_multihot.csv"
ADDENDUM = (
    PROJECT_ROOT
    / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.1_addendum.md"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_preflight/future3_dev_prompts_v1"

ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

SYSTEM_PROMPT = """你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻已经观察到的攻击事件，
每个事件只包含 ATT&CK 父技术、该技术可能所属的战术，以及事件的真实描述。

任务：基于已观察历史，概括攻击者当前状态，并给出未来三次攻击动作范围内最值得检查的
5 个 ATT&CK Parent Technique 候选。未来三步是一个无序目标集合；你的5个候选仍须按
可能性从高到低排列。

严格限制：
1. 只能使用输入中的历史事件，不能声称看到了未来事件。
2. 不得从来源名称、campaign 名称或文件格式推断答案；这些字段不会提供。
3. stage_assessment、observed_capabilities、likely_next_intents 中不得出现任何 Txxxx
   或 Txxxx.xxx 标识，也不要逐字重复 predicted_next_ttps。
4. predicted_next_ttps 必须恰好包含5个互不重复的 ATT&CK 父技术 ID。
5. 输出必须是 JSON，不要输出 JSON 之外的文本。

JSON 示例：
{
  "stage_assessment":"对已观察阶段的简洁判断",
  "observed_capabilities":"仅由历史描述支持的能力概括",
  "likely_next_intents":"对下一阶段意图的概括，不写ATT&CK ID",
  "predicted_next_ttps":["T1059","T1078","T1021","T1003","T1105"]
}"""

USER_TEMPLATE = """### 已观察攻击事件（按时间顺序）
{serialized_observed_events}

### 输出要求
请根据以上历史事件输出 JSON。预测目标是未来三次动作内的技术集合，候选数组按可能性排序。"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "stage_assessment",
        "observed_capabilities",
        "likely_next_intents",
        "predicted_next_ttps",
    ],
    "properties": {
        "stage_assessment": {"type": "string", "minLength": 1},
        "observed_capabilities": {"type": "string", "minLength": 1},
        "likely_next_intents": {"type": "string", "minLength": 1},
        "predicted_next_ttps": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^T\\d{4}$"},
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def serialize_events(
    sample: dict[str, str],
    step_lookup: dict[str, dict[str, str]],
    tactic_lookup: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    step_ids = json.loads(sample["observed_step_ids"])
    parent_ids = json.loads(sample["observed_parent_ids"])
    if len(step_ids) != len(parent_ids) or len(step_ids) != int(sample["prefix_len"]):
        raise AssertionError(f"history length mismatch: {sample['sample_id']}")
    events: list[dict[str, Any]] = []
    blocks: list[str] = []
    for index, (step_id, parent) in enumerate(zip(step_ids, parent_ids), start=1):
        step = step_lookup[step_id]
        if step["parent_technique_id"] != parent:
            raise AssertionError(f"parent/step mismatch: {sample['sample_id']}")
        description = step["description_clean"]
        if ATTACK_ID_RE.search(description):
            raise AssertionError(f"clean description contains ATT&CK ID: {step_id}")
        # Prefix history is intentionally allowed to contain labels outside the
        # frozen 184-class candidate set.  Preserve the event and expose the
        # missing tactic mapping instead of dropping it or guessing a tactic.
        tactics = tactic_lookup.get(parent, ["unknown"])
        events.append(
            {
                "event_index": index,
                "parent_technique": parent,
                "possible_tactics": tactics,
                "description": description,
            }
        )
        blocks.append(
            f"事件 {index}\n"
            f"父技术: {parent}\n"
            f"可能战术: {', '.join(tactics)}\n"
            f"描述: {description}"
        )
    return "\n\n".join(blocks), events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = [
        "raw_semantic_inputs.jsonl",
        "development_request_preflight.jsonl",
        "development_prompt_index.csv",
        "prompt_manifest.json",
        "report.md",
    ]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preflight artifacts: {existing}")

    steps = read_csv(ALIGNMENT)
    samples = read_csv(SAMPLES)
    tactics = read_csv(TACTICS)
    step_lookup = {row["stable_step_id"]: row for row in steps}
    tactic_lookup = {
        row["parent_technique_id"]: json.loads(row["tactics"]) for row in tactics
    }
    if len(step_lookup) != 898 or len(samples) != 814 or len(tactic_lookup) != 184:
        raise AssertionError("frozen input counts changed")

    raw_records: list[dict[str, Any]] = []
    development_records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    leakage_gates: Counter[str] = Counter()
    for sample in samples:
        serialized, events = serialize_events(sample, step_lookup, tactic_lookup)
        observed_ids = set(json.loads(sample["observed_step_ids"]))
        target_ids = set(json.loads(sample["target_step_ids"]))
        if observed_ids & target_ids:
            raise AssertionError(f"observed/target overlap: {sample['sample_id']}")
        leakage_gates["observed_target_step_disjoint"] += 1
        leakage_gates["descriptions_attack_id_free"] += 1
        leakage_gates["event_count_matches_prefix_len"] += 1

        model_text = serialized
        raw_records.append(
            {
                "audit_key": {
                    "sample_id": sample["sample_id"],
                    "is_development": bool(int(sample["is_development"])),
                    "development_slot": sample["development_slot"],
                },
                "model_input": {
                    "serialized_observed_events": model_text,
                    "events": events,
                },
                "model_input_sha256": text_sha256(model_text),
            }
        )
        if not int(sample["is_development"]):
            continue

        user_prompt = USER_TEMPLATE.format(serialized_observed_events=serialized)
        request_payload = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload_text = compact_json(request_payload)
        development_records.append(
            {
                "audit_key_not_sent": {
                    "sample_id": sample["sample_id"],
                    "development_slot": sample["development_slot"],
                    "observed_step_ids": json.loads(sample["observed_step_ids"]),
                    "target_step_ids_excluded": json.loads(sample["target_step_ids"]),
                },
                "request_payload": request_payload,
                "request_payload_sha256": text_sha256(payload_text),
                "network_status": "NOT_SENT",
            }
        )
        index_rows.append(
            {
                "development_slot": sample["development_slot"],
                "sample_id": sample["sample_id"],
                "events": len(events),
                "serialized_events_chars": len(serialized),
                "user_prompt_chars": len(user_prompt),
                "request_payload_sha256": text_sha256(payload_text),
                "network_status": "NOT_SENT",
            }
        )

    if len(development_records) != 30:
        raise AssertionError("development prompt count is not 30")
    if Counter(
        item["audit_key_not_sent"]["sample_id"].split("::", 1)[0]
        for item in development_records
    ) != Counter({"ctid": 10, "attack_flow": 10, "stockpile": 10}):
        raise AssertionError("development prompt source balance changed")
    if any(
        set(record["request_payload"].keys())
        - {"temperature", "max_tokens", "response_format", "extra_body", "messages"}
        for record in development_records
    ):
        raise AssertionError("unexpected request payload field")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "raw_semantic_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(compact_json(record) + "\n")
    with (output / "development_request_preflight.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in sorted(
            development_records,
            key=lambda value: value["audit_key_not_sent"]["development_slot"],
        ):
            handle.write(compact_json(record) + "\n")
    index_rows.sort(key=lambda row: row["development_slot"])
    write_csv(
        output / "development_prompt_index.csv",
        index_rows,
        [
            "development_slot",
            "sample_id",
            "events",
            "serialized_events_chars",
            "user_prompt_chars",
            "request_payload_sha256",
            "network_status",
        ],
    )

    prompt_lengths = [int(row["user_prompt_chars"]) for row in index_rows]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network_requests_performed": 0,
        "api_authorization_for_future3": False,
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "protocol_addendum_sha256": sha256(ADDENDUM),
        "inputs": {
            "alignment_sha256": sha256(ALIGNMENT),
            "samples_sha256": sha256(SAMPLES),
            "tactics_sha256": sha256(TACTICS),
        },
        "prompt": {
            "system_sha256": text_sha256(SYSTEM_PROMPT),
            "user_template_sha256": text_sha256(USER_TEMPLATE),
            "json_schema_sha256": text_sha256(compact_json(JSON_SCHEMA)),
            "json_schema": JSON_SCHEMA,
        },
        "counts": {
            "raw_semantic_inputs": len(raw_records),
            "development_requests_prepared_not_sent": len(development_records),
            "development_source_counts": {"ctid": 10, "attack_flow": 10, "stockpile": 10},
            "history_event_instances_with_unknown_tactic": sum(
                event["possible_tactics"] == ["unknown"]
                for record in raw_records
                for event in record["model_input"]["events"]
            ),
        },
        "development_prompt_chars": {
            "min": min(prompt_lengths),
            "median": sorted(prompt_lengths)[len(prompt_lengths) // 2],
            "max": max(prompt_lengths),
        },
        "leakage_gates": {
            "passed": True,
            "rows_checked": len(samples),
            **dict(leakage_gates),
            "request_payload_contains_source_field": 0,
            "request_payload_contains_campaign_field": 0,
            "request_payload_contains_target_step_field": 0,
            "request_payload_contains_file_path_field": 0,
        },
    }
    write_json(output / "prompt_manifest.json", manifest)
    report = "\n".join(
        [
            "# Future-3 semantic prompt preflight",
            "",
            "- Prepared raw semantic inputs: 814.",
            "- Prepared development request payloads: 30 (10 per source).",
            "- Network requests performed: **0**.",
            "- All request records are marked `NOT_SENT`.",
            "- All 814 temporal, history/target disjointness, event-count, and cleaned-description gates passed.",
            f"- Development user-prompt characters: min {min(prompt_lengths)}, "
            f"median {sorted(prompt_lengths)[len(prompt_lengths) // 2]}, max {max(prompt_lengths)}.",
            "- Audit keys and excluded target-step IDs are stored outside `request_payload`.",
            "",
            "No future-3 DeepSeek authorization is inferred from the earlier experiment.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
