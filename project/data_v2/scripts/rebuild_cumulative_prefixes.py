from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


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

ATTACK_FLOW_OVERLAPS = {
    "Turla - Carbon Emulation Plan.afb",
    "Turla - Snake Emulation Plan.afb",
}

REQUIRED_COLUMNS = (
    "sample_id",
    "source",
    "campaign_id",
    "prefix_len",
    "prefix",
    "raw_prefix",
    "true_label",
    "target_raw_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_id(attack_id: str) -> str:
    return str(attack_id).strip().split(".", 1)[0]


def json_array(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def properties(obj: dict[str, Any]) -> dict[str, Any]:
    values = obj.get("properties", [])
    if not isinstance(values, list):
        return {}
    return {str(key): value for key, value in values}


def read_vocab(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = tuple(str(row["technique_id_parent"]).strip() for row in rows)
    if len(labels) != 184 or len(set(labels)) != 184:
        raise ValueError(f"expected 184 unique labels, found {len(labels)}/{len(set(labels))}")
    return labels


def read_tactic_map(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            technique = parent_id(row.get("technique_id", ""))
            tactic = str(row.get("tactic", "")).strip()
            if not technique.startswith("T") or tactic not in TACTIC_INDEX:
                continue
            mapping[technique] = min(mapping.get(technique, len(TACTIC_ORDER)), TACTIC_INDEX[tactic])
    return mapping


@dataclass(frozen=True)
class SequenceRecord:
    source: str
    campaign_id: str
    raw_sequence: tuple[str, ...]
    metadata: dict[str, str]


def monotonicity(
    records: Sequence[SequenceRecord], tactic_map: dict[str, int]
) -> dict[str, int | float]:
    numerator = 0
    denominator = 0
    unknown_pairs = 0
    for record in records:
        parents = tuple(parent_id(value) for value in record.raw_sequence)
        for left, right in zip(parents, parents[1:]):
            denominator += 1
            left_tactic = tactic_map.get(left)
            right_tactic = tactic_map.get(right)
            if left_tactic is None or right_tactic is None:
                unknown_pairs += 1
                continue
            numerator += int(right_tactic >= left_tactic)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / denominator if denominator else 0.0,
        "unknown_tactic_pairs_counted_non_monotonic": unknown_pairs,
    }


def better_path(
    left: tuple[str, ...],
    right: tuple[str, ...],
    actions: dict[str, str],
    order: dict[str, int],
) -> tuple[str, ...]:
    left_score = (sum(node in actions for node in left), len(left))
    right_score = (sum(node in actions for node in right), len(right))
    if left_score != right_score:
        return left if left_score > right_score else right
    left_order = tuple(order[node] for node in left)
    right_order = tuple(order[node] for node in right)
    return left if left_order < right_order else right


def cyclic_strongly_connected_components(
    adjacency: dict[str, list[str]], order: dict[str, int]
) -> list[tuple[str, ...]]:
    """Return deterministic cyclic SCCs for auditing non-DAG Attack Flow files."""
    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)

        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        component.sort(key=order.__getitem__)
        is_self_loop = len(component) == 1 and component[0] in adjacency[component[0]]
        if len(component) > 1 or is_self_loop:
            components.append(tuple(component))

    for node in sorted(adjacency, key=order.__getitem__):
        if node not in indices:
            visit(node)
    components.sort(key=lambda component: tuple(order[node] for node in component))
    return components


def deterministic_dfs_back_edges(
    adjacency: dict[str, list[str]], order: dict[str, int]
) -> set[tuple[str, str]]:
    """Freeze cycle breaking by source-order DFS and remove gray-ancestor edges."""
    color = {node: 0 for node in adjacency}  # 0 white, 1 gray, 2 black
    back_edges: set[tuple[str, str]] = set()

    def visit(node: str) -> None:
        color[node] = 1
        for target in adjacency[node]:
            if color[target] == 0:
                visit(target)
            elif color[target] == 1:
                back_edges.add((node, target))
        color[node] = 2

    for node in sorted(adjacency, key=order.__getitem__):
        if color[node] == 0:
            visit(node)
    return back_edges


def parse_attack_flow(path: Path) -> tuple[SequenceRecord | None, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"{path} has no objects list")

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
        technique = str(properties(obj).get("technique_id", "")).strip()
        if re.fullmatch(r"T\d{4}(?:\.\d{3})?", technique):
            actions[instance] = technique

    anchor_to_node: dict[str, str] = {}
    for instance, obj in by_instance.items():
        anchors = obj.get("anchors", {})
        if not isinstance(anchors, dict):
            continue
        for anchor in anchors.values():
            anchor_to_node[str(anchor)] = instance

    latch_to_node: dict[str, str] = {}
    for anchor_instance, node_instance in anchor_to_node.items():
        anchor = by_instance.get(anchor_instance, {})
        for latch in anchor.get("latches", []) or []:
            latch_to_node[str(latch)] = node_instance

    adjacency: dict[str, list[str]] = {instance: [] for instance in by_instance}
    dynamic_lines = 0
    unresolved_lines = 0
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("id") != "dynamic_line":
            continue
        dynamic_lines += 1
        source = latch_to_node.get(str(obj.get("source", "")))
        target = latch_to_node.get(str(obj.get("target", "")))
        if source not in adjacency or target not in by_instance:
            unresolved_lines += 1
            continue
        if target not in adjacency[source]:
            adjacency[source].append(target)
    for source in adjacency:
        adjacency[source].sort(key=node_order.__getitem__)

    cyclic_components = cyclic_strongly_connected_components(adjacency, node_order)
    back_edges = deterministic_dfs_back_edges(adjacency, node_order)
    dag_adjacency = {
        source: [
            target
            for target in targets
            if (source, target) not in back_edges
        ]
        for source, targets in adjacency.items()
    }
    remaining_cycles = cyclic_strongly_connected_components(dag_adjacency, node_order)
    if remaining_cycles:
        raise ValueError(f"deterministic back-edge removal left a cycle in {path}")

    dag_indegree = {instance: 0 for instance in by_instance}
    for targets in dag_adjacency.values():
        for target in targets:
            dag_indegree[target] += 1

    roots = sorted(
        (instance for instance in by_instance if dag_indegree[instance] == 0),
        key=node_order.__getitem__,
    )
    if not roots and by_instance:
        raise ValueError(f"cycle-broken graph unexpectedly has no roots: {path}")

    memo: dict[str, tuple[str, ...]] = {}

    def longest_from(node: str) -> tuple[str, ...]:
        if node in memo:
            return memo[node]
        best = (node,)
        for target in dag_adjacency[node]:
            tail = longest_from(target)
            if tail:
                best = better_path(best, (node,) + tail, actions, node_order)
        memo[node] = best
        return best

    best_path: tuple[str, ...] = ()
    for root in roots:
        candidate = longest_from(root)
        best_path = (
            candidate
            if not best_path
            else better_path(best_path, candidate, actions, node_order)
        )

    raw_sequence = tuple(actions[node] for node in best_path if node in actions)
    selected_nodes = set(best_path)
    audit = {
        "actions": len(actions),
        "graph_nodes": len(by_instance),
        "dynamic_lines": dynamic_lines,
        "unresolved_lines": unresolved_lines,
        "cycle_edges_skipped": len(back_edges),
        "cycle_policy": (
            "source-order DFS; remove edges pointing to gray ancestors; then solve "
            "longest path on the resulting DAG"
        ),
        "back_edges_removed": [
            {
                "source_instance": source,
                "source_node_type": str(by_instance[source].get("id", "")),
                "source_action_technique": actions.get(source, ""),
                "target_instance": target,
                "target_node_type": str(by_instance[target].get("id", "")),
                "target_action_technique": actions.get(target, ""),
            }
            for source, target in sorted(
                back_edges,
                key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
            )
        ],
        "cyclic_scc_count": len(cyclic_components),
        "cyclic_sccs": [
            {
                "size": len(component),
                "action_techniques": [
                    actions[node] for node in component if node in actions
                ],
                "selected_path_nodes": sum(node in selected_nodes for node in component),
                "selected_path_action_techniques": [
                    actions[node]
                    for node in component
                    if node in selected_nodes and node in actions
                ],
            }
            for component in cyclic_components
        ],
        "longest_graph_path_nodes": len(best_path),
        "longest_path_actions": len(raw_sequence),
        "rows": max(0, len(raw_sequence) - 1),
    }
    if len(raw_sequence) < 2:
        return None, audit
    return (
        SequenceRecord(
            source="attack_flow",
            campaign_id=path.stem,
            raw_sequence=raw_sequence,
            metadata={"flow_file": path.name},
        ),
        audit,
    )


def build_attack_flow_records(corpus: Path) -> tuple[list[SequenceRecord], dict[str, Any]]:
    files = sorted(corpus.glob("*.afb"))
    if not files:
        raise FileNotFoundError(f"Attack Flow corpus missing: {corpus}")

    selected = [path for path in files if path.name not in ATTACK_FLOW_OVERLAPS]
    records: list[SequenceRecord] = []
    per_file: dict[str, Any] = {}
    zero_action_path_files: list[str] = []
    for path in selected:
        record, audit = parse_attack_flow(path)
        per_file[path.name] = audit
        if record is None:
            zero_action_path_files.append(path.name)
        else:
            records.append(record)

    return records, {
        "corpus_files": len(files),
        "selected_files": len(selected),
        "excluded_overlap_files": sorted(ATTACK_FLOW_OVERLAPS),
        "flows_with_rows": len(records),
        "zero_action_path_files": zero_action_path_files,
        "per_file": per_file,
    }


def split_top_level_yaml_records(text: str) -> list[list[str]]:
    records: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            if current:
                records.append(current)
                current = []
            continue
        if re.match(r"^-\s+", line) and current:
            records.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        records.append(current)
    return records


def parse_ability_map(abilities: Path) -> tuple[dict[str, str], dict[str, Any]]:
    mapping: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    files = sorted(abilities.rglob("*.yml"))
    for path in files:
        records = split_top_level_yaml_records(path.read_text(encoding="utf-8-sig"))
        for lines in records:
            ability_id: str | None = None
            technique_id: str | None = None
            nested_attack_id: str | None = None
            for line in lines:
                id_match = re.match(
                    r"^(\s*)(?:-\s*)?id:\s*[\"']?([^\s#\"']+)", line
                )
                if id_match and len(id_match.group(1)) <= 2:
                    ability_id = id_match.group(2).lower()
                technique_match = re.match(
                    r"^\s*technique_id:\s*[\"']?(T\d{4}(?:\.\d{3})?)", line
                )
                if technique_match:
                    technique_id = technique_match.group(1)
                attack_match = re.match(
                    r"^\s*attack_id:\s*[\"']?(T\d{4}(?:\.\d{3})?)", line
                )
                if attack_match and nested_attack_id is None:
                    nested_attack_id = attack_match.group(1)

            mapped = technique_id or nested_attack_id
            if ability_id is None and len(records) == 1 and mapped is not None:
                ability_id = path.stem.lower()
            if ability_id is None or mapped is None:
                continue
            existing = mapping.get(ability_id)
            if existing is not None and existing != mapped:
                conflicts.append(
                    {"ability_id": ability_id, "first": existing, "second": mapped}
                )
                continue
            mapping[ability_id] = mapped

    if conflicts:
        raise ValueError(f"conflicting Stockpile ability mappings: {conflicts[:3]}")
    return mapping, {
        "ability_yaml_files": len(files),
        "mapped_ability_ids": len(mapping),
        "mapping_conflicts": conflicts,
    }


def parse_profile(path: Path) -> tuple[str, str, tuple[str, ...]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    profile_id = ""
    profile_name = ""
    ordering: list[str] = []
    in_ordering = False

    for index, line in enumerate(lines):
        id_match = re.match(r"^id:\s*[\"']?([^\s#\"']+)", line)
        if id_match:
            profile_id = id_match.group(1).lower()
        name_match = re.match(r"^name:\s*(.*)$", line)
        if name_match:
            value = name_match.group(1).strip().strip("\"'")
            if value in {"|", ">", ""}:
                continuation = ""
                for candidate in lines[index + 1 :]:
                    if candidate.startswith(" ") and candidate.strip():
                        continuation = candidate.strip().strip("\"'")
                        break
                    if candidate and not candidate.startswith(" "):
                        break
                profile_name = continuation
            else:
                profile_name = value
        if re.match(r"^\s*atomic_ordering:\s*$", line):
            in_ordering = True
            continue
        if not in_ordering:
            continue
        ordering_match = re.match(r"^\s*-\s+([^\s#]+)", line)
        if ordering_match:
            ordering.append(ordering_match.group(1).lower())
        elif not line.strip() or line.lstrip().startswith("#"):
            continue
        elif line and not line.startswith(" "):
            in_ordering = False

    return profile_id or path.stem.lower(), profile_name or path.stem, tuple(ordering)


def build_stockpile_records(root: Path) -> tuple[list[SequenceRecord], dict[str, Any]]:
    ability_map, ability_audit = parse_ability_map(root / "data" / "abilities")
    profile_files = sorted((root / "data" / "adversaries").rglob("*.yml"))
    if not profile_files:
        raise FileNotFoundError(f"Stockpile profiles missing: {root}")

    records: list[SequenceRecord] = []
    unmapped_instances: list[dict[str, str]] = []
    per_profile: dict[str, Any] = {}
    ordering_steps = 0
    mapped_steps = 0
    for path in profile_files:
        profile_id, profile_name, ordering = parse_profile(path)
        ordering_steps += len(ordering)
        raw_sequence: list[str] = []
        profile_unmapped: list[str] = []
        for ability_id in ordering:
            technique = ability_map.get(ability_id)
            if technique is None:
                profile_unmapped.append(ability_id)
                unmapped_instances.append(
                    {
                        "profile_id": profile_id,
                        "profile_file": path.relative_to(root).as_posix(),
                        "ability_id": ability_id,
                    }
                )
            else:
                raw_sequence.append(technique)
        mapped_steps += len(raw_sequence)
        if raw_sequence:
            records.append(
                SequenceRecord(
                    source="stockpile",
                    campaign_id=profile_id,
                    raw_sequence=tuple(raw_sequence),
                    metadata={
                        "profile_name": profile_name,
                        "profile_file": path.relative_to(root).as_posix(),
                    },
                )
            )
        per_profile[profile_id] = {
            "profile_name": profile_name,
            "profile_file": path.relative_to(root).as_posix(),
            "ordering_steps": len(ordering),
            "mapped_steps": len(raw_sequence),
            "unmapped_steps": len(profile_unmapped),
            "unmapped_ability_ids": profile_unmapped,
            "rows": max(0, len(raw_sequence) - 1),
        }

    return records, {
        **ability_audit,
        "profile_files": len(profile_files),
        "profiles_with_mapped_steps": len(records),
        "ordering_steps": ordering_steps,
        "mapped_steps": mapped_steps,
        "unmapped_step_instances": len(unmapped_instances),
        "unique_unmapped_ability_ids": len(
            {item["ability_id"] for item in unmapped_instances}
        ),
        "unmapped_instances": unmapped_instances,
        "per_profile": per_profile,
    }


def rows_from_records(records: Sequence[SequenceRecord]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    source_counters: Counter[str] = Counter()
    for record in records:
        parents = tuple(parent_id(value) for value in record.raw_sequence)
        for target_index in range(1, len(record.raw_sequence)):
            source_counters[record.source] += 1
            row: dict[str, str | int] = {
                "sample_id": (
                    f"{record.source}::{record.campaign_id}::"
                    f"{target_index:04d}"
                ),
                "source": record.source,
                "campaign_id": record.campaign_id,
                "prefix_len": target_index,
                "prefix": json_array(parents[:target_index]),
                "raw_prefix": json_array(record.raw_sequence[:target_index]),
                "true_label": parents[target_index],
                "target_raw_id": record.raw_sequence[target_index],
            }
            row.update(record.metadata)
            rows.append(row)
    return rows


def summarize_rows(
    rows: Sequence[dict[str, str | int]],
    records: Sequence[SequenceRecord],
    vocab: set[str],
    tactic_map: dict[str, int],
    quality_threshold: float | None,
) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values are not unique")
    prefix_lengths = [int(row["prefix_len"]) for row in rows]
    out_of_vocab = [str(row["true_label"]) for row in rows if row["true_label"] not in vocab]
    monotonic = monotonicity(records, tactic_map)
    ratio = float(monotonic["ratio"])
    quality_passed = None if quality_threshold is None else ratio < quality_threshold
    return {
        "rows": len(rows),
        "campaigns": len({str(row["campaign_id"]) for row in rows}),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_target_labels": len({str(row["true_label"]) for row in rows}),
        "prefix_len_min": min(prefix_lengths) if prefix_lengths else 0,
        "prefix_len_max": max(prefix_lengths) if prefix_lengths else 0,
        "prefix_len_counts": {
            str(length): count
            for length, count in sorted(Counter(prefix_lengths).items())
        },
        "tactic_monotonicity": monotonic,
        "quality_threshold_strictly_below": quality_threshold,
        "quality_gate_passed": quality_passed,
        "out_of_vocab_target_rows": len(out_of_vocab),
        "out_of_vocab_target_counts": dict(sorted(Counter(out_of_vocab).items())),
    }


def write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row if key not in REQUIRED_COLUMNS})
    fieldnames = list(REQUIRED_COLUMNS) + extras
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def input_hashes(paths: Iterable[Path], project_root: Path) -> dict[str, str]:
    return {
        path.relative_to(project_root).as_posix(): sha256(path)
        for path in sorted(paths)
    }


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Rebuild cumulative Attack Flow and Stockpile prefixes"
    )
    parser.add_argument("--project-root", type=Path, default=default_project)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    external_root = project_root / "data_v2" / "repro_external"
    attack_corpus = external_root / "attack_flow" / "corpus"
    stockpile_root = external_root / "stockpile"
    output_root = external_root / "cumulative"
    vocab_path = project_root / "data_v2" / "core" / "rl_label_vocab.csv"
    tactic_path = project_root / "data" / "attack_lookup_dedup.csv"

    vocab_tuple = read_vocab(vocab_path)
    vocab = set(vocab_tuple)
    tactic_map = read_tactic_map(tactic_path)

    attack_records, attack_audit = build_attack_flow_records(attack_corpus)
    stockpile_records, stockpile_audit = build_stockpile_records(stockpile_root)
    attack_rows = rows_from_records(attack_records)
    stockpile_rows = rows_from_records(stockpile_records)

    attack_summary = summarize_rows(
        attack_rows, attack_records, vocab, tactic_map, quality_threshold=0.85
    )
    stockpile_summary = summarize_rows(
        stockpile_rows, stockpile_records, vocab, tactic_map, quality_threshold=None
    )
    stockpile_summary["quality_gate_note"] = (
        "pre-registered semi-synthetic exception; monotonicity is reported but not gated"
    )

    if attack_summary["campaigns"] != 35:
        raise ValueError(f"unexpected Attack Flow shape: {attack_summary}")
    if stockpile_summary["rows"] != 122 or stockpile_summary["campaigns"] != 27:
        raise ValueError(f"unexpected Stockpile shape: {stockpile_summary}")
    if stockpile_audit["ordering_steps"] != 183:
        raise ValueError(f"unexpected Stockpile ordering count: {stockpile_audit}")
    if stockpile_audit["unmapped_step_instances"] != 34:
        raise ValueError(f"unexpected Stockpile unmapped count: {stockpile_audit}")
    if attack_summary["quality_gate_passed"] is not True:
        raise ValueError(f"Attack Flow monotonicity quality gate failed: {attack_summary}")

    attack_path = output_root / "attack_flow_cumulative.csv"
    stockpile_path = output_root / "stockpile_cumulative.csv"
    report_path = output_root / "rebuild_report.json"
    write_csv(attack_path, attack_rows)
    write_csv(stockpile_path, stockpile_rows)

    raw_inputs = list(attack_corpus.glob("*.afb"))
    raw_inputs.extend((stockpile_root / "data" / "adversaries").rglob("*.yml"))
    raw_inputs.extend((stockpile_root / "data" / "abilities").rglob("*.yml"))
    raw_inputs.extend((vocab_path, tactic_path))

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "attack_flow_linearization": (
                "source-order DFS removes gray-ancestor back edges; on the resulting "
                "DAG maximize action-node count; tie by graph-node count; final tie "
                "by source-file node order; non-action nodes may be traversed"
            ),
            "attack_flow_overlap_exclusions": sorted(ATTACK_FLOW_OVERLAPS),
            "stockpile_sequence": (
                "drop unmapped atomic_ordering steps, retain remaining order, then emit "
                "all cumulative prefixes"
            ),
            "parent_mapping": "split ATT&CK ID at first period",
            "tactic_unknown_policy": "count pair in denominator as non-monotonic",
            "closed_set_resolution": (
                "preserve raw cumulative rows; primary experiment filters only rows "
                "whose true_label is outside the frozen 184-class vocabulary; prefix "
                "tokens are never filtered"
            ),
        },
        "vocab": {
            "path": vocab_path.relative_to(project_root).as_posix(),
            "sha256": sha256(vocab_path),
            "labels": len(vocab_tuple),
        },
        "attack_flow": {**attack_summary, "extraction": attack_audit},
        "stockpile": {**stockpile_summary, "extraction": stockpile_audit},
        "outputs": {
            attack_path.name: {"rows": len(attack_rows), "sha256": sha256(attack_path)},
            stockpile_path.name: {
                "rows": len(stockpile_rows),
                "sha256": sha256(stockpile_path),
            },
        },
        "input_sha256": input_hashes(raw_inputs, project_root),
        "blocking_findings": [],
        "resolved_findings": [],
    }
    total_out_of_vocab = int(attack_summary["out_of_vocab_target_rows"]) + int(
        stockpile_summary["out_of_vocab_target_rows"]
    )
    if total_out_of_vocab:
        report["resolved_findings"].append(
            {
                "code": "true_label_outside_fixed_vocab",
                "rows": total_out_of_vocab,
                "message": (
                    "Raw cumulative data contain targets outside the frozen 184-class "
                    "vocabulary. The pre-registered resolution is implemented by the "
                    "separate closed-set view builder; raw rows remain unchanged."
                ),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "attack_flow": attack_summary,
        "stockpile": stockpile_summary,
        "blocking_findings": report["blocking_findings"],
        "resolved_findings": report["resolved_findings"],
        "report": report_path.as_posix(),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
