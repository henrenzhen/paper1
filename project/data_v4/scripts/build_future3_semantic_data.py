#!/usr/bin/env python3
"""Build the audited three-source future-3 semantic dataset.

This script intentionally has no model-training or prediction code.  It freezes
the source-aligned logical steps, the 184-class future-3 samples, a deterministic
mapping of the already disclosed 30-row pilot to a development-only subset, and
the ATT&CK tactic multi-hot mapping used by later experiments.

CTID is parsed from *top-level* YAML ability records.  The older recursive
loader is used only as an audited legacy input: it recursively treated nested
``technique`` dictionaries as new events and extracted every ATT&CK-looking ID
from free text.  Reusing that representation would create non-existent steps
and false multi-technique labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUILD_PATH = PROJECT_ROOT / "data_v2/scripts/rebuild_cumulative_prefixes.py"
VOCAB_PATH = PROJECT_ROOT / "data_v2/core/rl_label_vocab.csv"
TACTIC_PATH = PROJECT_ROOT / "data/attack_lookup_dedup.csv"
CTID_RAW = PROJECT_ROOT / "data_v2/repro_external/ctid/raw_yaml"
CTID_LEGACY_PARSED = PROJECT_ROOT / "data_v2/repro_external/ctid/parsed"
ATTACK_FLOW_CORPUS = PROJECT_ROOT / "data_v2/repro_external/attack_flow/corpus"
STOCKPILE_ROOT = PROJECT_ROOT / "data_v2/repro_external/stockpile"
ATTACK_FLOW_CUMULATIVE = (
    PROJECT_ROOT / "data_v2/repro_external/cumulative/attack_flow_cumulative.csv"
)
STOCKPILE_CUMULATIVE = (
    PROJECT_ROOT / "data_v2/repro_external/cumulative/stockpile_cumulative.csv"
)
PILOT_PATH = PROJECT_ROOT / "data_v4/external_reasoning/pilot/pilot_sample_30.csv"
PROTOCOL_PATH = (
    PROJECT_ROOT / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.md"
)
ADDENDUM_PATH = (
    PROJECT_ROOT
    / "data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.1_addendum.md"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/semantic_alignment"

VOCAB_SHA256 = "9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738"
DEV_HASH_SALT = "v7-dev-20260806"
ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
EXPECTED_SOURCE_STEPS = {"ctid": 283, "attack_flow": 466, "stockpile": 149}
EXPECTED_FUTURE3 = {"ctid": 273, "attack_flow": 422, "stockpile": 119}
EXPECTED_MAIN = {"ctid": 263, "attack_flow": 412, "stockpile": 109}
EXPECTED_CAMPAIGNS = {"ctid": 10, "attack_flow": 35, "stockpile": 27}

TACTIC_ORDER = (
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
)
TACTIC_INDEX = {name: index for index, name in enumerate(TACTIC_ORDER)}


def load_rebuild_module() -> Any:
    spec = importlib.util.spec_from_file_location("rebuild_cumulative_prefixes", REBUILD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REBUILD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REBUILD = load_rebuild_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parent_id(attack_id: str) -> str:
    return str(attack_id).strip().split(".", 1)[0]


def clean_description(value: Any) -> tuple[str, int, bool, bool, bool]:
    raw = "" if value is None else str(value)
    normalized = unicodedata.normalize("NFKC", raw)
    contained_attack_id = bool(ATTACK_ID_RE.search(normalized))
    normalized = ATTACK_ID_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", normalized)
    truncated = len(normalized) > 2000
    normalized = normalized[:2000].strip()
    has_description = bool(normalized)
    if not has_description:
        normalized = "[NO_DESCRIPTION]"
    return normalized, (len(normalized) if has_description else 0), has_description, truncated, contained_attack_id


def source_step(
    *,
    source: str,
    campaign_id: str,
    step_index: int,
    stable_step_id: str,
    raw_id: str,
    name: Any,
    description: Any,
    source_file: Path,
) -> dict[str, Any]:
    clean, chars, has_description, truncated, contained_attack_id = clean_description(description)
    audit_name = "" if name is None else " ".join(str(name).split())
    audit_description = "" if description is None else "\n".join(
        line.rstrip() for line in str(description).strip().splitlines()
    )
    return {
        "source": source,
        "campaign_id": campaign_id,
        "step_index": step_index,
        "stable_step_id": stable_step_id,
        "raw_technique_id": str(raw_id),
        "parent_technique_id": parent_id(raw_id),
        "step_name_audit_only": audit_name,
        "description_raw_audit_only": audit_description,
        "description_clean": clean,
        "description_chars": chars,
        "has_description": int(has_description),
        "description_truncated": int(truncated),
        "raw_description_had_attack_id": int(contained_attack_id),
        "source_file_audit_only": source_file.relative_to(PROJECT_ROOT).as_posix(),
        "source_file_sha256": sha256(source_file),
    }


def collapse_consecutive(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if output and output[-1]["parent_technique_id"] == row["parent_technique_id"]:
            continue
        output.append(dict(row))
    for index, row in enumerate(output):
        row["step_index"] = index
    return output


def parse_ctid() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_campaign: dict[str, Any] = {}
    ignored_invalid: list[dict[str, str]] = []
    false_multitech_legacy: list[dict[str, Any]] = []

    legacy_multitech = 0
    for legacy_path in sorted(CTID_LEGACY_PARSED.glob("ctid_steps_long_*.csv")):
        for legacy in read_csv(legacy_path):
            parents = json.loads(legacy["attack_technique_ids_parent"])
            if len(parents) > 1:
                legacy_multitech += 1
                false_multitech_legacy.append(
                    {
                        "legacy_step_id": legacy["step_id"],
                        "legacy_parent_ids": parents,
                        "step_title": legacy["step_title"],
                    }
                )

    for path in sorted(CTID_RAW.glob("*.yaml")):
        campaign = path.stem
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"CTID root is not a list: {path}")
        logical: list[dict[str, Any]] = []
        top_level_objects = 0
        for item_index, item in enumerate(payload):
            if not isinstance(item, dict) or "emulation_plan_details" in item:
                continue
            top_level_objects += 1
            technique = item.get("technique")
            attack_id = technique.get("attack_id") if isinstance(technique, dict) else None
            if not isinstance(attack_id, str) or not re.fullmatch(r"T\d{4}(?:\.\d{3})?", attack_id):
                ignored_invalid.append(
                    {
                        "campaign_id": f"{campaign}::unknown",
                        "top_level_item_index": str(item_index),
                        "item_id": str(item.get("id", "")),
                        "attack_id": "" if attack_id is None else str(attack_id),
                    }
                )
                continue
            campaign_id = f"{campaign}::unknown"
            stable = f"ctid::{campaign}::{item.get('id', item_index)}"
            logical.append(
                source_step(
                    source="ctid",
                    campaign_id=campaign_id,
                    step_index=len(logical),
                    stable_step_id=stable,
                    raw_id=attack_id,
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    source_file=path,
                )
            )
        collapsed = collapse_consecutive(logical)
        rows.extend(collapsed)
        per_campaign[f"{campaign}::unknown"] = {
            "top_level_objects": top_level_objects,
            "valid_structured_technique_steps": len(logical),
            "steps_after_consecutive_parent_dedup": len(collapsed),
        }

    return rows, {
        "parser": "top-level YAML records; label only technique.attack_id",
        "legacy_recursive_rows": sum(
            len(read_csv(path))
            for path in sorted(CTID_LEGACY_PARSED.glob("ctid_steps_long_*.csv"))
        ),
        "legacy_false_multitech_rows": legacy_multitech,
        "legacy_false_multitech_details": false_multitech_legacy,
        "ignored_top_level_records_without_valid_structured_attack_id": ignored_invalid,
        "per_campaign": per_campaign,
    }


def attack_flow_selected_nodes(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    nodes = [
        obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("instance") and obj.get("id") != "dynamic_line"
    ]
    by_instance = {str(obj["instance"]): obj for obj in nodes}
    node_order = {instance: index for index, instance in enumerate(by_instance)}
    actions: dict[str, str] = {}
    for instance, obj in by_instance.items():
        if obj.get("id") != "action":
            continue
        technique = str(REBUILD.properties(obj).get("technique_id", "")).strip()
        if re.fullmatch(r"T\d{4}(?:\.\d{3})?", technique):
            actions[instance] = technique

    anchor_to_node: dict[str, str] = {}
    for instance, obj in by_instance.items():
        anchors = obj.get("anchors", {})
        if isinstance(anchors, dict):
            for anchor in anchors.values():
                anchor_to_node[str(anchor)] = instance
    latch_to_node: dict[str, str] = {}
    for anchor_instance, node_instance in anchor_to_node.items():
        anchor = by_instance.get(anchor_instance, {})
        for latch in anchor.get("latches", []) or []:
            latch_to_node[str(latch)] = node_instance

    adjacency: dict[str, list[str]] = {instance: [] for instance in by_instance}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("id") != "dynamic_line":
            continue
        source = latch_to_node.get(str(obj.get("source", "")))
        target = latch_to_node.get(str(obj.get("target", "")))
        if source not in adjacency or target not in by_instance:
            continue
        if target not in adjacency[source]:
            adjacency[source].append(target)
    for source in adjacency:
        adjacency[source].sort(key=node_order.__getitem__)

    back_edges = REBUILD.deterministic_dfs_back_edges(adjacency, node_order)
    dag = {
        source: [target for target in targets if (source, target) not in back_edges]
        for source, targets in adjacency.items()
    }
    if REBUILD.cyclic_strongly_connected_components(dag, node_order):
        raise ValueError(f"cycle remained after frozen policy: {path}")
    indegree = {instance: 0 for instance in by_instance}
    for targets in dag.values():
        for target in targets:
            indegree[target] += 1
    roots = sorted((node for node, degree in indegree.items() if degree == 0), key=node_order.__getitem__)
    memo: dict[str, tuple[str, ...]] = {}

    def longest_from(node: str) -> tuple[str, ...]:
        if node in memo:
            return memo[node]
        best = (node,)
        for target in dag[node]:
            candidate = (node,) + longest_from(target)
            best = REBUILD.better_path(best, candidate, actions, node_order)
        memo[node] = best
        return best

    best: tuple[str, ...] = ()
    for root in roots:
        candidate = longest_from(root)
        best = candidate if not best else REBUILD.better_path(best, candidate, actions, node_order)
    selected_actions = [node for node in best if node in actions]

    authoritative, audit = REBUILD.parse_attack_flow(path)
    authoritative_ids = tuple() if authoritative is None else authoritative.raw_sequence
    selected_ids = tuple(actions[node] for node in selected_actions)
    if selected_ids != authoritative_ids:
        raise AssertionError(f"Attack Flow path reconstruction differs for {path.name}")
    return selected_actions, {"by_instance": by_instance, "authoritative_audit": audit}


def cumulative_sequences(path: Path) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["campaign_id"]].append(row)
    result: dict[str, tuple[str, ...]] = {}
    for campaign, rows in grouped.items():
        rows.sort(key=lambda row: int(row["prefix_len"]))
        last = rows[-1]
        raw_prefix = json.loads(last["raw_prefix"])
        result[campaign] = tuple(raw_prefix + [last["target_raw_id"]])
    return result


def parse_attack_flow() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = cumulative_sequences(ATTACK_FLOW_CUMULATIVE)
    rows: list[dict[str, Any]] = []
    per_campaign: dict[str, Any] = {}
    for path in sorted(ATTACK_FLOW_CORPUS.glob("*.afb")):
        if path.name in REBUILD.ATTACK_FLOW_OVERLAPS:
            continue
        selected, detail = attack_flow_selected_nodes(path)
        if not selected:
            continue
        by_instance = detail["by_instance"]
        campaign_rows: list[dict[str, Any]] = []
        for index, instance in enumerate(selected):
            obj = by_instance[instance]
            props = REBUILD.properties(obj)
            campaign_rows.append(
                source_step(
                    source="attack_flow",
                    campaign_id=path.stem,
                    step_index=index,
                    stable_step_id=f"attack_flow::{path.stem}::{instance}",
                    raw_id=str(props["technique_id"]),
                    name=props.get("name", ""),
                    description=props.get("description", ""),
                    source_file=path,
                )
            )
        sequence = tuple(row["raw_technique_id"] for row in campaign_rows)
        if sequence != expected.get(path.stem):
            raise AssertionError(f"Attack Flow cumulative mismatch: {path.stem}")
        rows.extend(campaign_rows)
        per_campaign[path.stem] = {
            "steps": len(campaign_rows),
            "authoritative_audit": detail["authoritative_audit"],
        }
    if set(expected) != set(per_campaign):
        raise AssertionError("Attack Flow campaign set differs from frozen cumulative CSV")
    return rows, {"per_campaign": per_campaign, "cumulative_sha256": sha256(ATTACK_FLOW_CUMULATIVE)}


def yaml_items(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def stockpile_ability_metadata() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, str]] = []
    files = sorted((STOCKPILE_ROOT / "data/abilities").rglob("*.yml"))
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        for item in yaml_items(payload):
            ability_id = str(item.get("id", path.stem)).lower()
            technique = item.get("technique")
            attack_id = technique.get("attack_id") if isinstance(technique, dict) else item.get("technique_id")
            if not isinstance(attack_id, str) or not re.fullmatch(r"T\d{4}(?:\.\d{3})?", attack_id):
                continue
            value = {
                "raw_technique_id": attack_id,
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "source_file": path,
            }
            existing = metadata.get(ability_id)
            if existing and existing["raw_technique_id"] != value["raw_technique_id"]:
                conflicts.append(
                    {
                        "ability_id": ability_id,
                        "first": existing["raw_technique_id"],
                        "second": value["raw_technique_id"],
                    }
                )
                continue
            if existing is None or (not existing["description"] and value["description"]):
                metadata[ability_id] = value
    if conflicts:
        raise ValueError(f"Stockpile ability conflicts: {conflicts[:3]}")
    return metadata, {"ability_yaml_files": len(files), "mapped_ability_metadata": len(metadata)}


def parse_stockpile() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = cumulative_sequences(STOCKPILE_CUMULATIVE)
    ability_map, ability_audit = REBUILD.parse_ability_map(STOCKPILE_ROOT / "data/abilities")
    metadata, metadata_audit = stockpile_ability_metadata()
    rows: list[dict[str, Any]] = []
    per_campaign: dict[str, Any] = {}
    missing_description_metadata: list[str] = []
    for path in sorted((STOCKPILE_ROOT / "data/adversaries").rglob("*.yml")):
        profile_id, profile_name, ordering = REBUILD.parse_profile(path)
        campaign_rows: list[dict[str, Any]] = []
        for source_position, ability_id in enumerate(ordering):
            attack_id = ability_map.get(ability_id)
            if attack_id is None:
                continue
            meta = metadata.get(ability_id)
            if meta is None:
                missing_description_metadata.append(ability_id)
                meta = {
                    "name": "",
                    "description": "",
                    "source_file": path,
                    "raw_technique_id": attack_id,
                }
            if parent_id(meta["raw_technique_id"]) != parent_id(attack_id):
                raise AssertionError(f"Stockpile metadata technique mismatch: {ability_id}")
            campaign_rows.append(
                source_step(
                    source="stockpile",
                    campaign_id=profile_id,
                    step_index=len(campaign_rows),
                    stable_step_id=(
                        f"stockpile::{profile_id}::{source_position:04d}::{ability_id}"
                    ),
                    raw_id=attack_id,
                    name=meta["name"],
                    description=meta["description"],
                    source_file=meta["source_file"],
                )
            )
        if not campaign_rows:
            continue
        sequence = tuple(row["raw_technique_id"] for row in campaign_rows)
        if sequence != expected.get(profile_id):
            raise AssertionError(f"Stockpile cumulative mismatch: {profile_id}")
        rows.extend(campaign_rows)
        per_campaign[profile_id] = {
            "profile_name_audit_only": profile_name,
            "profile_file_audit_only": path.relative_to(PROJECT_ROOT).as_posix(),
            "steps": len(campaign_rows),
        }
    if set(expected) != set(per_campaign):
        raise AssertionError("Stockpile campaign set differs from frozen cumulative CSV")
    return rows, {
        **ability_audit,
        **metadata_audit,
        "missing_description_metadata_instances": missing_description_metadata,
        "per_campaign": per_campaign,
        "cumulative_sha256": sha256(STOCKPILE_CUMULATIVE),
    }


def tactic_mapping(vocabulary: Sequence[str]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    tactic_sets: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(TACTIC_PATH):
        technique = parent_id(row.get("technique_id", ""))
        tactic = row.get("tactic", "").strip()
        if technique.startswith("T") and tactic in TACTIC_INDEX:
            tactic_sets[technique].add(tactic)
    mapping: dict[str, list[str]] = {
        technique: sorted(values, key=TACTIC_INDEX.__getitem__)
        for technique, values in tactic_sets.items()
    }
    rows: list[dict[str, Any]] = []
    for label_index, technique in enumerate(vocabulary):
        tactics = mapping.get(technique, [])
        vector = [int(name in tactics) for name in TACTIC_ORDER]
        rows.append(
            {
                "label_index": label_index,
                "parent_technique_id": technique,
                "tactics": compact_json(tactics),
                "tactic_indices": compact_json([TACTIC_INDEX[name] for name in tactics]),
                "multi_hot_14": compact_json(vector),
                "mapping_missing": int(not tactics),
            }
        )
    return mapping, rows


def group_steps(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["campaign_id"])].append(row)
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["step_index"]))
        if [int(row["step_index"]) for row in values] != list(range(len(values))):
            raise AssertionError(f"non-contiguous step index: {key}")
    return grouped


def build_future3(
    aligned: Sequence[dict[str, Any]],
    vocabulary: Sequence[str],
    tactic_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vocab = set(vocabulary)
    samples: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for (source, campaign), steps in sorted(group_steps(aligned).items()):
        for endpoint in range(len(steps) - 1):
            future = steps[endpoint + 1 : endpoint + 4]
            targets: list[str] = []
            for row in future:
                label = row["parent_technique_id"]
                if label not in targets:
                    targets.append(label)
            oov = [label for label in targets if label not in vocab]
            if oov:
                dropped.append(
                    {
                        "source": source,
                        "campaign_id": campaign,
                        "prefix_len": endpoint + 1,
                        "prefix_endpoint_step_id": steps[endpoint]["stable_step_id"],
                        "target_parent_ids": compact_json(targets),
                        "oov_target_parent_ids": compact_json(oov),
                    }
                )
                continue
            observed = steps[: endpoint + 1]
            tactic_vectors = []
            tactic_missing = []
            for row in observed:
                tactics = tactic_map.get(row["parent_technique_id"], [])
                tactic_vectors.append([int(name in tactics) for name in TACTIC_ORDER])
                if not tactics:
                    tactic_missing.append(row["parent_technique_id"])
            samples.append(
                {
                    "sample_id": f"{source}::{campaign}::{endpoint + 1:04d}",
                    "source": source,
                    "campaign_id": campaign,
                    "prefix_len": endpoint + 1,
                    "prefix_endpoint_step_id": steps[endpoint]["stable_step_id"],
                    "observed_parent_ids": compact_json(
                        [row["parent_technique_id"] for row in observed]
                    ),
                    "observed_step_ids": compact_json(
                        [row["stable_step_id"] for row in observed]
                    ),
                    "observed_tactic_multihot": compact_json(tactic_vectors),
                    "history_tactic_mapping_missing": compact_json(sorted(set(tactic_missing))),
                    "target_parent_ids": compact_json(targets),
                    "target_step_ids": compact_json([row["stable_step_id"] for row in future]),
                    "target_size": len(targets),
                    "last_description_chars": steps[endpoint]["description_chars"],
                    "text_length_group": (
                        "lt40" if int(steps[endpoint]["description_chars"]) < 40 else "ge40"
                    ),
                    "is_development": 0,
                    "development_slot": "",
                }
            )
    return samples, dropped


def quantile_cutoffs(values: Sequence[int]) -> tuple[int, int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile input")
    q1 = ordered[max(0, (len(ordered) - 1) // 3)]
    q2 = ordered[max(0, (2 * (len(ordered) - 1)) // 3)]
    return q1, q2


def length_stratum(value: int, cutoffs: tuple[int, int]) -> str:
    q1, q2 = cutoffs
    if value <= q1:
        return "short"
    if value <= q2:
        return "medium"
    return "long"


def deterministic_hash(source: str, campaign: str, prefix_len: int) -> str:
    value = f"{DEV_HASH_SALT}||{source}||{campaign}||{prefix_len}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def map_development(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pilot = read_csv(PILOT_PATH)
    if len(pilot) != 30 or Counter(row["source"] for row in pilot) != Counter(
        {"ctid": 10, "attack_flow": 10, "stockpile": 10}
    ):
        raise AssertionError("legacy pilot is not the frozen 10-per-source 30 rows")

    sample_by_key = {
        (row["source"], row["campaign_id"], int(row["prefix_len"])): row
        for row in samples
    }
    campaign_sample_counts = Counter(
        (row["source"], row["campaign_id"]) for row in samples
    )
    cutoffs = {
        source: quantile_cutoffs(
            [int(row["prefix_len"]) for row in samples if row["source"] == source]
        )
        for source in EXPECTED_SOURCE_STEPS
    }
    selected_ids: set[str] = set()
    used_campaigns: dict[str, set[str]] = defaultdict(set)
    mappings: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, str], str]] = []

    for old in pilot:
        key = (old["source"], old["campaign_id"], int(old["prefix_len"]))
        candidate = sample_by_key.get(key)
        old_prefix = json.loads(old["prefix"])
        reason = ""
        if candidate is None:
            reason = "future3_sample_not_available_after_closed_set_filter"
        elif json.loads(candidate["observed_parent_ids"]) != old_prefix:
            reason = "legacy_recursive_prefix_differs_from_corrected_logical_sequence"
        elif campaign_sample_counts[(candidate["source"], candidate["campaign_id"])] <= 1:
            reason = "direct_mapping_would_remove_campaign_from_main_evaluation"
        elif candidate["sample_id"] in selected_ids:
            reason = "duplicate_corrected_endpoint"
        if reason:
            pending.append((old, reason))
            continue
        candidate["is_development"] = 1
        candidate["development_slot"] = old["pilot_slot"]
        selected_ids.add(candidate["sample_id"])
        used_campaigns[old["source"]].add(candidate["campaign_id"])
        mappings.append(
            {
                "pilot_slot": old["pilot_slot"],
                "source": old["source"],
                "old_campaign_id": old["campaign_id"],
                "old_prefix_len": old["prefix_len"],
                "old_prefix": old["prefix"],
                "mapping_status": "direct_prefix_match",
                "mapping_reason": "",
                "new_sample_id": candidate["sample_id"],
                "new_campaign_id": candidate["campaign_id"],
                "new_prefix_len": candidate["prefix_len"],
                "new_prefix_endpoint_step_id": candidate["prefix_endpoint_step_id"],
                "length_stratum": old["length_stratum"],
                "selection_sha256": deterministic_hash(
                    candidate["source"], candidate["campaign_id"], int(candidate["prefix_len"])
                ),
            }
        )

    for old, reason in pending:
        source = old["source"]
        stratum = old["length_stratum"]
        candidates = [
            row
            for row in samples
            if row["source"] == source
            and row["sample_id"] not in selected_ids
            and campaign_sample_counts[(row["source"], row["campaign_id"])] > 1
            and length_stratum(int(row["prefix_len"]), cutoffs[source]) == stratum
        ]
        candidates.sort(
            key=lambda row: (
                row["campaign_id"] in used_campaigns[source],
                deterministic_hash(source, row["campaign_id"], int(row["prefix_len"])),
            )
        )
        if not candidates:
            raise AssertionError(f"no deterministic development replacement: {old['pilot_slot']}")
        candidate = candidates[0]
        candidate["is_development"] = 1
        candidate["development_slot"] = old["pilot_slot"]
        selected_ids.add(candidate["sample_id"])
        used_campaigns[source].add(candidate["campaign_id"])
        mappings.append(
            {
                "pilot_slot": old["pilot_slot"],
                "source": source,
                "old_campaign_id": old["campaign_id"],
                "old_prefix_len": old["prefix_len"],
                "old_prefix": old["prefix"],
                "mapping_status": "deterministic_stratum_replacement",
                "mapping_reason": reason,
                "new_sample_id": candidate["sample_id"],
                "new_campaign_id": candidate["campaign_id"],
                "new_prefix_len": candidate["prefix_len"],
                "new_prefix_endpoint_step_id": candidate["prefix_endpoint_step_id"],
                "length_stratum": stratum,
                "selection_sha256": deterministic_hash(
                    source, candidate["campaign_id"], int(candidate["prefix_len"])
                ),
            }
        )

    mappings.sort(key=lambda row: row["pilot_slot"])
    return mappings


def description_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in EXPECTED_SOURCE_STEPS:
        subset = [row for row in rows if row["source"] == source]
        lengths = [int(row["description_chars"]) for row in subset]
        result[source] = {
            "steps": len(subset),
            "has_description": sum(int(row["has_description"]) for row in subset),
            "missing_description": sum(not int(row["has_description"]) for row in subset),
            "median_clean_chars": statistics.median(lengths),
            "lt40": sum(value < 40 for value in lengths),
            "ge40": sum(value >= 40 for value in lengths),
            "raw_description_had_attack_id": sum(
                int(row["raw_description_had_attack_id"]) for row in subset
            ),
            "truncated": sum(int(row["description_truncated"]) for row in subset),
        }
    return result


def assert_gates(
    aligned: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    development: Sequence[dict[str, Any]],
    tactic_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    step_counts = Counter(row["source"] for row in aligned)
    future_counts = Counter(row["source"] for row in samples)
    main = [row for row in samples if not int(row["is_development"])]
    main_counts = Counter(row["source"] for row in main)
    campaign_counts = {
        source: len({row["campaign_id"] for row in main if row["source"] == source})
        for source in EXPECTED_SOURCE_STEPS
    }
    if dict(step_counts) != EXPECTED_SOURCE_STEPS:
        raise AssertionError(f"step count gate: {dict(step_counts)}")
    if dict(future_counts) != EXPECTED_FUTURE3:
        raise AssertionError(f"future3 count gate: {dict(future_counts)}")
    if dict(main_counts) != EXPECTED_MAIN:
        raise AssertionError(f"main count gate: {dict(main_counts)}")
    if campaign_counts != EXPECTED_CAMPAIGNS:
        raise AssertionError(f"campaign gate: {campaign_counts}")
    if len(development) != 30 or Counter(row["source"] for row in development) != Counter(
        {"ctid": 10, "attack_flow": 10, "stockpile": 10}
    ):
        raise AssertionError("development gate")
    if len({row["new_sample_id"] for row in development}) != 30:
        raise AssertionError("development rows are not unique")
    if len(tactic_rows) != 184:
        raise AssertionError("tactic mapping is not 184 rows")
    if len({(row["source"], row["campaign_id"], row["step_index"]) for row in aligned}) != len(aligned):
        raise AssertionError("step alignment composite key collision")
    if len({(row["source"], row["stable_step_id"]) for row in aligned}) != len(aligned):
        raise AssertionError("stable step key collision")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise AssertionError("future3 sample ID collision")
    if any(ATTACK_ID_RE.search(row["description_clean"]) for row in aligned):
        raise AssertionError("clean description contains an ATT&CK ID")

    step_lookup = {row["stable_step_id"]: row for row in aligned}
    for sample in samples:
        observed = json.loads(sample["observed_step_ids"])
        target = json.loads(sample["target_step_ids"])
        if set(observed) & set(target):
            raise AssertionError(f"history/target step overlap: {sample['sample_id']}")
        endpoint = int(step_lookup[sample["prefix_endpoint_step_id"]]["step_index"])
        if any(int(step_lookup[value]["step_index"]) > endpoint for value in observed):
            raise AssertionError(f"future in observed: {sample['sample_id']}")
        if any(int(step_lookup[value]["step_index"]) <= endpoint for value in target):
            raise AssertionError(f"non-future target: {sample['sample_id']}")

    return {
        "passed": True,
        "step_counts": dict(step_counts),
        "future3_counts": dict(future_counts),
        "development_counts": dict(Counter(row["source"] for row in development)),
        "main_counts": dict(main_counts),
        "main_campaign_counts": campaign_counts,
        "clean_description_attack_id_matches": 0,
        "unique_step_keys": len(aligned),
        "unique_sample_ids": len(samples),
    }


def markdown_report(
    manifest: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    development: Sequence[dict[str, Any]],
) -> str:
    desc = manifest["description_summary"]
    gates = manifest["gates"]
    target_sizes = Counter(int(row["target_size"]) for row in samples)
    direct = sum(row["mapping_status"] == "direct_prefix_match" for row in development)
    replacement = len(development) - direct
    lines = [
        "# Future-3 semantic dataset audit",
        "",
        "All count, key, closed-set, text-cleaning, and temporal leakage gates passed.",
        "",
        "## Frozen counts",
        "",
        "| Source | Logical steps | Future-3 rows | Development-only | Main rows | Main campaigns |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in ("ctid", "attack_flow", "stockpile"):
        lines.append(
            f"| {source} | {gates['step_counts'][source]} | {gates['future3_counts'][source]} "
            f"| {gates['development_counts'][source]} | {gates['main_counts'][source]} "
            f"| {gates['main_campaign_counts'][source]} |"
        )
    lines.extend(
        [
            f"| **Total** | **{sum(gates['step_counts'].values())}** | "
            f"**{sum(gates['future3_counts'].values())}** | **30** | "
            f"**{sum(gates['main_counts'].values())}** | **72** |",
            "",
            "## Text coverage",
            "",
            "| Source | Has description | Missing | Median cleaned chars | <40 | >=40 | Raw text with removed ATT&CK ID |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("ctid", "attack_flow", "stockpile"):
        item = desc[source]
        lines.append(
            f"| {source} | {item['has_description']} | {item['missing_description']} | "
            f"{item['median_clean_chars']} | {item['lt40']} | {item['ge40']} | "
            f"{item['raw_description_had_attack_id']} |"
        )
    lines.extend(
        [
            "",
            "## Target and development diagnostics",
            "",
            f"- Target-set cardinalities: {dict(sorted(target_sizes.items()))}.",
            f"- Old pilot rows mapped directly by exact corrected prefix: {direct}/30.",
            f"- Deterministic same-source/same-length-stratum replacements: {replacement}/30.",
            "- Development selection never uses a target label.",
            "",
            "## CTID correction",
            "",
            "The legacy CTID loader recursively visited nested YAML dictionaries and regex-extracted "
            "all ATT&CK-looking IDs from the entire object. This created child `technique` objects as "
            "extra events and made 12 ordinary single-label abilities appear multi-label when their "
            "name or description mentioned an older ID. The corrected representation uses one top-level "
            "ability as one logical event and reads only `technique.attack_id`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    managed = [
        "step_text_alignment.csv",
        "future3_samples.csv",
        "future3_dropped_oov.csv",
        "development_mapping.csv",
        "technique_tactic_multihot.csv",
        "step_text_alignment_manifest.json",
        "audit_report.md",
    ]
    existing = [name for name in managed if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    if sha256(VOCAB_PATH) != VOCAB_SHA256:
        raise AssertionError("frozen vocabulary SHA-256 changed")
    vocabulary = [row["technique_id_parent"] for row in read_csv(VOCAB_PATH)]
    if len(vocabulary) != 184 or len(set(vocabulary)) != 184:
        raise AssertionError("frozen vocabulary must contain 184 unique labels")

    ctid, ctid_audit = parse_ctid()
    attack_flow, attack_flow_audit = parse_attack_flow()
    stockpile, stockpile_audit = parse_stockpile()
    aligned = sorted(
        ctid + attack_flow + stockpile,
        key=lambda row: (row["source"], row["campaign_id"], int(row["step_index"])),
    )
    tactic_map, tactic_rows = tactic_mapping(vocabulary)
    samples, dropped = build_future3(aligned, vocabulary, tactic_map)
    development = map_development(samples)
    gates = assert_gates(aligned, samples, development, tactic_rows)

    step_columns = [
        "source",
        "campaign_id",
        "step_index",
        "stable_step_id",
        "raw_technique_id",
        "parent_technique_id",
        "description_clean",
        "description_chars",
        "has_description",
        "description_truncated",
        "raw_description_had_attack_id",
        "step_name_audit_only",
        "description_raw_audit_only",
        "source_file_audit_only",
        "source_file_sha256",
    ]
    sample_columns = [
        "sample_id",
        "source",
        "campaign_id",
        "prefix_len",
        "prefix_endpoint_step_id",
        "observed_parent_ids",
        "observed_step_ids",
        "observed_tactic_multihot",
        "history_tactic_mapping_missing",
        "target_parent_ids",
        "target_step_ids",
        "target_size",
        "last_description_chars",
        "text_length_group",
        "is_development",
        "development_slot",
    ]
    development_columns = [
        "pilot_slot",
        "source",
        "old_campaign_id",
        "old_prefix_len",
        "old_prefix",
        "mapping_status",
        "mapping_reason",
        "new_sample_id",
        "new_campaign_id",
        "new_prefix_len",
        "new_prefix_endpoint_step_id",
        "length_stratum",
        "selection_sha256",
    ]
    tactic_columns = [
        "label_index",
        "parent_technique_id",
        "tactics",
        "tactic_indices",
        "multi_hot_14",
        "mapping_missing",
    ]

    write_csv(output / "step_text_alignment.csv", aligned, step_columns)
    write_csv(output / "future3_samples.csv", samples, sample_columns)
    write_csv(
        output / "future3_dropped_oov.csv",
        dropped,
        [
            "source",
            "campaign_id",
            "prefix_len",
            "prefix_endpoint_step_id",
            "target_parent_ids",
            "oov_target_parent_ids",
        ],
    )
    write_csv(output / "development_mapping.csv", development, development_columns)
    write_csv(output / "technique_tactic_multihot.csv", tactic_rows, tactic_columns)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(PROTOCOL_PATH),
            "addendum_path": ADDENDUM_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "addendum_sha256": sha256(ADDENDUM_PATH),
            "pre_result_correction": "v8.1 CTID logical-step correction",
        },
        "inputs": {
            "vocabulary": {"path": VOCAB_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(VOCAB_PATH)},
            "tactic_lookup": {"path": TACTIC_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(TACTIC_PATH)},
            "pilot": {"path": PILOT_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(PILOT_PATH)},
            "ctid_raw_yaml": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
                for path in sorted(CTID_RAW.glob("*.yaml"))
            },
            "attack_flow_files": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
                for path in sorted(ATTACK_FLOW_CORPUS.glob("*.afb"))
                if path.name not in REBUILD.ATTACK_FLOW_OVERLAPS
            },
            "stockpile_ability_and_profile_tree_sha256": hashlib.sha256(
                "\n".join(
                    f"{path.relative_to(PROJECT_ROOT).as_posix()} {sha256(path)}"
                    for path in sorted(
                        list((STOCKPILE_ROOT / "data/abilities").rglob("*.yml"))
                        + list((STOCKPILE_ROOT / "data/adversaries").rglob("*.yml"))
                    )
                ).encode("utf-8")
            ).hexdigest(),
        },
        "ctid_audit": ctid_audit,
        "attack_flow_audit": attack_flow_audit,
        "stockpile_audit": stockpile_audit,
        "tactic_mapping": {
            "tactic_order": list(TACTIC_ORDER),
            "candidate_labels": len(tactic_rows),
            "missing_candidate_mappings": sum(int(row["mapping_missing"]) for row in tactic_rows),
            "provenance": "repository attack_lookup_dedup.csv; layer-source metadata identifies ATT&CK v18",
        },
        "description_summary": description_summary(aligned),
        "future3_target_size_counts": dict(
            sorted(Counter(int(row["target_size"]) for row in samples).items())
        ),
        "dropped_oov_future3_rows": len(dropped),
        "development_mapping_status_counts": dict(
            Counter(row["mapping_status"] for row in development)
        ),
        "gates": gates,
    }
    write_json(output / "step_text_alignment_manifest.json", manifest)
    (output / "audit_report.md").write_text(
        markdown_report(manifest, samples, development), encoding="utf-8"
    )

    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print(f"wrote audited artifacts to {output}")


if __name__ == "__main__":
    main()
