#!/usr/bin/env python3
"""Build the frozen four-arm EDLR pilot payloads without networking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import local_fusion_common as C


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.md"
ADDENDUM = PROJECT_ROOT / "data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2_implementation.md"
SAMPLES = PROJECT_ROOT / "data_v4/semantic_alignment/future3_samples.csv"
RAW_INPUTS = PROJECT_ROOT / "data_v4/semantic_preflight/future3_dev_prompts_v1/raw_semantic_inputs.jsonl"
PILOT_RESULTS = PROJECT_ROOT / "data_v4/external_reasoning/future3/pilot/runs/20260807T073425Z_1ec8fb8d/pilot_raw_results.csv"
TACTICS = PROJECT_ROOT / "data_v4/semantic_alignment/technique_tactic_multihot.csv"
LOOKUP = PROJECT_ROOT / "data_v2/core/attack_lookup_dedup.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_preflight/edlr_pilot_v2"

ARMS = ("EA_TOP5", "UNION_LLM", "EDLR", "EDLR_SHUFFLE")
EXPECTED_LAMBDAS = {"ctid": 0.0, "attack_flow": 0.0, "stockpile": 0.1}
EXPECTED_INPUT_HASHES = {
    "future3_samples.csv": "af7d9a6b6358939697730c5de206c51f1b63f7c16954d5dce5a0a4006bbca724",
    "technique_tactic_multihot.csv": "dfe0aa810207eac80a91a8f37452c475445e3f8a323139e090f7a343fd85e0b4",
    "attack_lookup_dedup.csv": "a8368252e37ba485a8dd7429fd4544935343bd96062ea78d71befb782b5dd1b6",
    "b0_rankings.csv": "f4b03b95db516a46db231f4fec9e505019c41cbb60b7372b52c984f1077b9216",
    "predictions.csv": "75b53e71e166139c1e42d29ab40a2252d8fa6bf80879ec625cc91e1d825d640c",
    "development_prompt_index.csv": "f8b860839d5ad0a43b339b1f1356374c29d6df3d22d341019a44c45bbcdb5dd6",
    "raw_semantic_inputs.jsonl": "00820b1d401ac61be04a72658e904d8c3f67911a4f663ad1807ecf341e00d876",
    "pilot_raw_results.csv": "41bb231e1e82c53be5e4799255e444cd15bf645f9d2d61db93fbf0c090408315",
}

EA_SYSTEM = """你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻的已观察攻击事件，以及同一模型首次分析
给出的5个候选。请重新核对历史语义，只能重新排列这5个候选，输出未来三次攻击动作范围内最
值得检查的5个 ATT&CK Parent Technique。

限制：
1. 历史描述是唯一新证据；候选原始排名只是先验，不保证正确。
2. 不得新增、删除或重复候选。
3. 不得使用或猜测来源、campaign、未来事件或真实答案。
4. 输出必须是JSON，不输出内部思维链或JSON之外的文本。"""

EA_USER = """### 已观察攻击事件（按时间顺序）
{history}

### 首次候选（按原始顺序）
{candidates}

### 输出
{{"evidence_summary":"不超过120个中文字符","reranked_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}}"""

UNION_SYSTEM = """你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻的已观察攻击事件、一个候选技术表，以及
可能提供的训练语料序列统计。任务是在候选表内选出未来三次攻击动作范围内最值得检查的5个
ATT&CK Parent Technique，并按可能性排序。

判断规则：
1. 历史事件的行为描述是语义主证据；B0排名是首次语义分析形成的先验。
2. 仅当候选表提供了序列统计时才使用它。出现次数、上下文分母和跨训练源支持必须结合判断；
   0次观察只表示训练语料未见，不表示不可能。
3. 不得使用或猜测来源、campaign、未来事件或真实答案。
4. 最终数组必须恰好包含5个互不重复的候选表内父技术ID，不得新增候选。
5. 输出必须是JSON，不输出内部思维链或JSON之外的文本。"""

UNION_USER = """### 已观察攻击事件（按时间顺序）
{history}

### 候选技术表（按技术ID排序）
{candidate_table}

### 输出
{{"evidence_summary":"不超过120个中文字符","reranked_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}}"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evidence_summary", "reranked_next_ttps"],
    "properties": {
        "evidence_summary": {"type": "string", "minLength": 1, "maxLength": 120},
        "reranked_next_ttps": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^T\\d{4}$"},
        },
    },
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def redact_exact(text: str, value: str, replacement: str) -> tuple[str, int]:
    if not value:
        return text, 0
    return re.subn(re.escape(value), replacement, text, flags=re.IGNORECASE)


def parse_development_samples() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    original: dict[str, dict[str, str]] = {}
    safe: list[dict[str, Any]] = []
    for row in read_csv(SAMPLES):
        if row["is_development"] != "1":
            continue
        original[row["sample_id"]] = row
        # Deliberately construct a target-free object for every scoring/ranking call.
        safe.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "campaign_id": row["campaign_id"],
                "prefix_len": int(row["prefix_len"]),
                "history": tuple(json.loads(row["observed_parent_ids"])),
                "development_slot": row["development_slot"],
            }
        )
    if len(safe) != 30 or Counter(row["source"] for row in safe) != Counter(
        {"ctid": 10, "attack_flow": 10, "stockpile": 10}
    ):
        raise AssertionError("frozen 30-row development denominator changed")
    return safe, original


def load_pilot_b0(labels: set[str]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in read_csv(PILOT_RESULTS):
        if row["generation_status"] != "ok" or row["valid_top5"].casefold() != "true":
            raise AssertionError(f"invalid frozen pilot B0 row: {row['sample_id']}")
        ranking = tuple(json.loads(row["predicted_next_ttps"]))
        if len(ranking) != 5 or len(set(ranking)) != 5 or not set(ranking) <= labels:
            raise AssertionError(f"invalid frozen pilot candidate set: {row['sample_id']}")
        if row["sample_id"] in result:
            raise AssertionError(f"duplicate pilot B0 row: {row['sample_id']}")
        result[row["sample_id"]] = ranking
    if len(result) != 30:
        raise AssertionError(f"expected 30 frozen pilot rankings, found {len(result)}")
    return result


def load_names(labels: Sequence[str]) -> dict[str, str]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(LOOKUP):
        if row["technique_id"] in labels:
            values[row["technique_id"]].add(row["technique_name"])
    names = {label: next(iter(values[label])) for label in labels if len(values[label]) == 1}
    if len(names) != len(labels):
        bad = {label: sorted(values[label]) for label in labels if len(values[label]) != 1}
        raise AssertionError(f"parent name mapping is not unique/complete: {bad}")
    return names


def load_tactic_names(labels: Sequence[str]) -> dict[str, list[str]]:
    tactics = {
        row["parent_technique_id"]: list(json.loads(row["tactics"]))
        for row in read_csv(TACTICS)
    }
    if set(tactics) != set(labels) or any(not values for values in tactics.values()):
        raise AssertionError("tactic-name mapping changed")
    return tactics


def rank5(value: str, ranking: Sequence[str]) -> Any:
    try:
        position = ranking.index(value) + 1
    except ValueError:
        return "not_top5"
    return position if position <= 5 else "not_top5"


def round8(value: float) -> float:
    return float(f"{value:.8f}")


def build_candidate_records(
    row: dict[str, Any],
    b0: tuple[str, ...],
    builder: C.EvidenceBuilder,
    names: dict[str, str],
    tactic_names: dict[str, list[str]],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, tuple[str, ...]]]:
    # EvidenceBuilder.build reads only history/source/sample metadata from this
    # target-free development row; targets are not present by construction.
    bundle = builder.build(row, b0)
    rankings = bundle["evidence"]["rankings"]
    union = sorted(set(b0).union(*(set(rankings[name][:5]) for name in C.EXPERTS)))
    if not 5 <= len(union) <= 20:
        raise AssertionError(f"candidate union outside 5--20: {row['sample_id']}={len(union)}")

    history = row["history"]
    h1 = history[-1]
    h2 = tuple(history[-2:]) if len(history) >= 2 else None
    n1 = int(builder.a_model.order1_counts.get(h1, 0))
    n2 = int(builder.a_model.order2_counts.get(h2, 0)) if h2 is not None else 0
    counts1 = builder.a_model.order1_targets.get(h1, [0] * len(builder.labels))
    counts2 = (
        builder.a_model.order2_targets.get(h2, [0] * len(builder.labels))
        if h2 is not None
        else [0] * len(builder.labels)
    )
    a_values = builder.a_model.score(history)
    records: dict[str, dict[str, Any]] = {}
    for label in union:
        index = builder.label_index[label]
        p0 = builder.a_model.unigram[index]
        p1 = (counts1[index] + C.BASE.ALPHA * p0) / (n1 + C.BASE.ALPHA) if n1 else p0
        p2 = (counts2[index] + C.BASE.ALPHA * p0) / (n2 + C.BASE.ALPHA) if n2 else p0
        evidence: dict[str, Any] = {
            "a_rank": rank5(label, rankings["A"]),
            "t_rank": rank5(label, rankings["T"]),
            "k_rank": rank5(label, rankings["K"]),
            "unigram_target_count": int(builder.a_model.target_counts[index]),
            "unigram_relevance": round8(p0),
            "a_smoothed_relevance": round8(a_values[index]),
            "order1": {
                "target_count": int(counts1[index]),
                "context_total": n1,
                "conditional_relevance": round8(p1),
                "supporting_training_sources": len(builder.pair1_sources.get((h1, label), set())),
            },
            "order2": "context_unavailable",
        }
        if h2 is not None:
            evidence["order2"] = {
                "target_count": int(counts2[index]),
                "context_total": n2,
                "conditional_relevance": round8(p2),
                "supporting_training_sources": len(builder.pair2_sources.get((h2, label), set())),
            }
        records[label] = {
            "candidate_id": label,
            "name": names[label],
            "possible_tactics": tactic_names[label],
            "b0_rank": rank5(label, b0),
            **evidence,
        }
    return union, records, rankings


def serialize_ea_candidates(
    b0: Sequence[str], names: dict[str, str], tactic_names: dict[str, list[str]]
) -> str:
    return "\n".join(
        f"候选 {rank}: {label} | 名称: {names[label]} | 可能战术: {', '.join(tactic_names[label])}"
        for rank, label in enumerate(b0, start=1)
    )


def common_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "name": record["name"],
        "possible_tactics": record["possible_tactics"],
        "b0_rank": record["b0_rank"],
    }


def evidence_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "a_rank",
            "t_rank",
            "k_rank",
            "unigram_target_count",
            "unigram_relevance",
            "a_smoothed_relevance",
            "order1",
            "order2",
        )
    }


def serialize_union_table(
    union: Sequence[str], records: dict[str, dict[str, Any]], arm: str
) -> tuple[str, dict[str, str]]:
    donor_map: dict[str, str] = {}
    lines: list[str] = []
    shift = 37 % len(union)
    for index, label in enumerate(union):
        item = common_record(records[label])
        if arm == "UNION_LLM":
            item["sequence_evidence"] = "not_provided"
            donor_map[label] = "not_provided"
        else:
            donor = label if arm == "EDLR" else union[(index + shift) % len(union)]
            donor_map[label] = donor
            item.update(evidence_record(records[donor]))
        lines.append(compact_json(item))
    if arm == "EDLR_SHUFFLE" and any(label == donor for label, donor in donor_map.items()):
        raise AssertionError("shuffle unexpectedly contains identity assignments")
    return "\n".join(lines), donor_map


def payload(system: str, user: str) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "disabled"}},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = ("request_preflight.jsonl", "request_index.csv", "evidence_audit.csv", "preflight_manifest.json", "report.md")
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v2 preflight: {existing}")

    anchored_paths = {
        "future3_samples.csv": SAMPLES,
        "technique_tactic_multihot.csv": TACTICS,
        "attack_lookup_dedup.csv": LOOKUP,
        "b0_rankings.csv": C.B0_PATH,
        "predictions.csv": C.BASE_RESULTS,
        "development_prompt_index.csv": PROJECT_ROOT / "data_v4/semantic_preflight/future3_dev_prompts_v1/development_prompt_index.csv",
        "raw_semantic_inputs.jsonl": RAW_INPUTS,
        "pilot_raw_results.csv": PILOT_RESULTS,
    }
    actual_hashes = {name: sha256_file(path) for name, path in anchored_paths.items()}
    if actual_hashes != EXPECTED_INPUT_HASHES:
        changed = {
            name: {"expected": EXPECTED_INPUT_HASHES[name], "actual": actual_hashes[name]}
            for name in EXPECTED_INPUT_HASHES
            if EXPECTED_INPUT_HASHES[name] != actual_hashes[name]
        }
        raise AssertionError(f"frozen v1 input hash changed: {changed}")

    formal, labels, label_index, candidate_tactics, tactic_by_label, _formal_b0, lambdas = C.load_inputs()
    if lambdas != EXPECTED_LAMBDAS:
        raise AssertionError(f"frozen tactic lambdas changed: {lambdas}")
    development, original = parse_development_samples()
    b0 = load_pilot_b0(set(labels))
    raw_inputs = {
        item["audit_key"]["sample_id"]: item["model_input"]["serialized_observed_events"]
        for item in read_jsonl(RAW_INPUTS)
        if item["audit_key"]["is_development"]
    }
    if set(raw_inputs) != {row["sample_id"] for row in development} or set(b0) != set(raw_inputs):
        raise AssertionError("development/B0/raw-input key alignment failed")
    names = load_names(labels)
    tactic_names = load_tactic_names(labels)

    builders = {
        source: C.EvidenceBuilder(
            [row for row in formal if row["source"] != source],
            labels,
            label_index,
            candidate_tactics,
            tactic_by_label,
            lambdas[source],
        )
        for source in C.SOURCES
    }

    prepared: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    union_sizes: list[int] = []
    for row in sorted(development, key=lambda item: item["development_slot"]):
        sample_id = row["sample_id"]
        source = row["source"]
        history = raw_inputs[sample_id]
        union, records, rankings = build_candidate_records(
            row, b0[sample_id], builders[source], names, tactic_names
        )
        union_sizes.append(len(union))
        source_counts[source] += 1

        arm_users: dict[str, str] = {
            "EA_TOP5": EA_USER.format(
                history=history,
                candidates=serialize_ea_candidates(b0[sample_id], names, tactic_names),
            )
        }
        donor_maps: dict[str, dict[str, str]] = {}
        for arm in ("UNION_LLM", "EDLR", "EDLR_SHUFFLE"):
            table, donor_map = serialize_union_table(union, records, arm)
            donor_maps[arm] = donor_map
            arm_users[arm] = UNION_USER.format(history=history, candidate_table=table)

        sample_original = original[sample_id]
        for arm in ARMS:
            user_text = arm_users[arm]
            user_text, source_redactions = redact_exact(user_text, source, "[REDACTED_SOURCE]")
            user_text, campaign_redactions = redact_exact(
                user_text, row["campaign_id"], "[REDACTED_CAMPAIGN_ENTITY]"
            )
            redactions["source"] += source_redactions
            redactions["campaign"] += campaign_redactions
            system = EA_SYSTEM if arm == "EA_TOP5" else UNION_SYSTEM
            body = payload(system, user_text)
            body_text = compact_json(body, sort_keys=True)

            forbidden_literals = [
                value
                for value in (
                    sample_id,
                    source,
                    row["campaign_id"],
                    row["development_slot"],
                    *json.loads(sample_original["target_step_ids"]),
                )
                if value and value.casefold() in body_text.casefold()
            ]
            if forbidden_literals:
                raise AssertionError(f"payload literal leakage {sample_id}/{arm}: {forbidden_literals}")
            if set(body) != {"temperature", "max_tokens", "response_format", "extra_body", "messages"}:
                raise AssertionError("unexpected payload envelope")

            candidate_set = list(b0[sample_id]) if arm == "EA_TOP5" else list(union)
            audit_key = {
                "development_slot": row["development_slot"],
                "sample_id": sample_id,
                "source": source,
                "campaign_id": row["campaign_id"],
                "allowed_training_sources": [item for item in C.SOURCES if item != source],
                "target_parent_ids_excluded": json.loads(sample_original["target_parent_ids"]),
                "target_step_ids_excluded": json.loads(sample_original["target_step_ids"]),
                "candidate_set": candidate_set,
                "b0_top5": list(b0[sample_id]),
                "donor_map_not_sent": donor_maps.get(arm, {}),
            }
            prepared.append(
                {
                    "audit_key_not_sent": audit_key,
                    "arm": arm,
                    "request_payload": body,
                    "request_payload_sha256": sha256_text(body_text),
                    "network_status": "NOT_SENT",
                }
            )
            arm_counts[arm] += 1
            index_rows.append(
                {
                    "development_slot": row["development_slot"],
                    "sample_id": sample_id,
                    "source_audit_only": source,
                    "arm": arm,
                    "candidate_count": len(candidate_set),
                    "b0_reproduced": 1,
                    "allowed_training_sources": compact_json(audit_key["allowed_training_sources"]),
                    "user_prompt_chars": len(user_text),
                    "request_payload_sha256": sha256_text(body_text),
                    "source_literal_redactions": source_redactions,
                    "campaign_literal_redactions": campaign_redactions,
                    "network_status": "NOT_SENT",
                }
            )

        evidence_rows.append(
            {
                "development_slot": row["development_slot"],
                "sample_id": sample_id,
                "source_audit_only": source,
                "prefix_len": row["prefix_len"],
                "b0_top5": compact_json(b0[sample_id]),
                "a_top5": compact_json(rankings["A"][:5]),
                "t_top5": compact_json(rankings["T"][:5]),
                "k_top5": compact_json(rankings["K"][:5]),
                "candidate_union": compact_json(union),
                "candidate_union_size": len(union),
                "shuffle_donor_map": compact_json(donor_maps["EDLR_SHUFFLE"], sort_keys=True),
                "target_fields_used_by_builder": 0,
            }
        )

    if len(prepared) != 120 or arm_counts != Counter({arm: 30 for arm in ARMS}):
        raise AssertionError(f"four-arm request denominator failed: {len(prepared)} {arm_counts}")
    if source_counts != Counter({"ctid": 10, "attack_flow": 10, "stockpile": 10}):
        raise AssertionError(f"source denominator failed: {source_counts}")
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        by_slot[item["audit_key_not_sent"]["development_slot"]].append(item)
    for slot, items in by_slot.items():
        if {item["arm"] for item in items} != set(ARMS):
            raise AssertionError(f"missing arm for {slot}")
        unions = [item["audit_key_not_sent"]["candidate_set"] for item in items if item["arm"] != "EA_TOP5"]
        if not all(value == unions[0] for value in unions[1:]):
            raise AssertionError(f"union mismatch across arms: {slot}")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "request_preflight.jsonl").open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(compact_json(item) + "\n")
    write_csv(
        output / "request_index.csv",
        index_rows,
        (
            "development_slot",
            "sample_id",
            "source_audit_only",
            "arm",
            "candidate_count",
            "b0_reproduced",
            "allowed_training_sources",
            "user_prompt_chars",
            "request_payload_sha256",
            "source_literal_redactions",
            "campaign_literal_redactions",
            "network_status",
        ),
    )
    write_csv(
        output / "evidence_audit.csv",
        evidence_rows,
        (
            "development_slot",
            "sample_id",
            "source_audit_only",
            "prefix_len",
            "b0_top5",
            "a_top5",
            "t_top5",
            "k_top5",
            "candidate_union",
            "candidate_union_size",
            "shuffle_donor_map",
            "target_fields_used_by_builder",
        ),
    )

    prompt_hashes = {
        "ea_system": sha256_text(EA_SYSTEM),
        "ea_user_template": sha256_text(EA_USER),
        "union_system": sha256_text(UNION_SYSTEM),
        "union_user_template": sha256_text(UNION_USER),
        "json_schema": sha256_text(compact_json(JSON_SCHEMA, sort_keys=True)),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "offline_preflight_complete_not_sent",
        "network_calls": 0,
        "completion_requests_prepared": len(prepared),
        "models_request_prepared": 1,
        "api_cost": 0,
        "authorization_for_network_stage": False,
        "protocol": {"path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(PROTOCOL)},
        "implementation_addendum": {"path": ADDENDUM.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(ADDENDUM)},
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(Path(__file__))},
        "imported_common_script_sha256": sha256_file(Path(C.__file__)),
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in (*anchored_paths.values(), C.BASE.VOCAB_PATH)
        },
        "prompt_hashes": prompt_hashes,
        "json_schema": JSON_SCHEMA,
        "counts": {
            "development_rows": len(development),
            "source_rows": dict(sorted(source_counts.items())),
            "arms": dict(arm_counts),
            "candidate_union_min": min(union_sizes),
            "candidate_union_max": max(union_sizes),
            "candidate_union_mean": sum(union_sizes) / len(union_sizes),
            "source_literal_redactions": redactions["source"],
            "campaign_literal_redactions": redactions["campaign"],
        },
        "gates": {
            "target_free_development_objects_used_for_scoring": True,
            "b0_reproduced_rows": 30,
            "frozen_vocab_candidates_only": True,
            "union_identical_across_union_arms": True,
            "shuffle_non_identity_all_rows": True,
            "forbidden_literal_occurrences_after_redaction": 0,
            "all_network_status_not_sent": True,
        },
    }
    managed_without_manifest = ("request_preflight.jsonl", "request_index.csv", "evidence_audit.csv")
    manifest["outputs"] = {name: sha256_file(output / name) for name in managed_without_manifest}
    write_json(output / "preflight_manifest.json", manifest)
    report = "\n".join(
        [
            "# EDLR pilot v2 offline preflight",
            "",
            "- Development rows: 30 (10 per source).",
            "- Arms: EA_TOP5, UNION_LLM, EDLR, EDLR_SHUFFLE.",
            "- Prepared completion payloads: 120.",
            "- Network/API calls: **0**; cost: **0**; every payload is `NOT_SENT`.",
            f"- Candidate-union size: min {min(union_sizes)}, mean {sum(union_sizes)/len(union_sizes):.2f}, max {max(union_sizes)}.",
            "- B0 reproduction, target-free construction, outer-training-only evidence, union identity, shuffle non-identity, vocabulary, and literal leakage gates: PASS.",
            "",
            "A new explicit authorization is required before `/models` or any of the 120 billed completions.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest["outputs"]["report.md"] = sha256_file(output / "report.md")
    # Rewrite once so the report hash is also covered by the manifest.
    write_json(output / "preflight_manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
