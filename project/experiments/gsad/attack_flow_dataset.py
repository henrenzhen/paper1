from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


OVERLAPPING_CTID_FLOWS = {
    "Turla - Carbon Emulation Plan.afb",
    "Turla - Snake Emulation Plan.afb",
}


def _properties(obj: dict[str, Any]) -> dict[str, Any]:
    values = obj.get("properties", [])
    return {str(key): value for key, value in values} if isinstance(values, list) else {}


def _parse_one(path: Path, vocab: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    by_instance = {
        str(obj["instance"]): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("instance")
    }
    actions: dict[str, tuple[str, str]] = {}
    for instance, obj in by_instance.items():
        if obj.get("id") != "action":
            continue
        raw = str(_properties(obj).get("technique_id", "")).strip()
        if not raw.startswith("T"):
            continue
        actions[instance] = (raw, raw.split(".")[0])

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

    adjacency: dict[str, set[str]] = defaultdict(set)
    dynamic_lines = 0
    unresolved_lines = 0
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("id") != "dynamic_line":
            continue
        dynamic_lines += 1
        source = latch_to_node.get(str(obj.get("source", "")))
        target = latch_to_node.get(str(obj.get("target", "")))
        if source is None or target is None:
            unresolved_lines += 1
            continue
        adjacency[source].add(target)

    transitions: set[tuple[str, str]] = set()
    for source in actions:
        queue = deque(adjacency.get(source, ()))
        visited: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in actions:
                if node != source:
                    transitions.add((source, node))
                continue
            queue.extend(adjacency.get(node, ()))

    rows: list[dict[str, Any]] = []
    dropped_target = 0
    for edge_index, (source, target) in enumerate(sorted(transitions)):
        source_raw, source_parent = actions[source]
        target_raw, target_parent = actions[target]
        if target_parent not in vocab:
            dropped_target += 1
            continue
        rows.append(
            {
                "sample_id": f"{path.stem}::{edge_index}",
                "flow": path.stem,
                "source_action": source,
                "target_action": target,
                "prefix_ids": (source_parent,),
                "raw_prefix_ids": (source_raw,),
                "target": target_parent,
                "evaluation_next_raw_id": target_raw,
            }
        )
    audit = {
        "actions": len(actions),
        "dynamic_lines": dynamic_lines,
        "unresolved_lines": unresolved_lines,
        "collapsed_action_edges": len(transitions),
        "dropped_target_edges": dropped_target,
        "kept_edges": len(rows),
    }
    return rows, audit


def load_attack_flow_transitions(
    project_root: Path,
    vocab: Sequence[str],
    exclude_overlapping_ctid: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    corpus = (
        Path(project_root)
        / "data_v2"
        / "external_attack_flow"
        / "raw"
        / "attack-flow"
        / "corpus"
    )
    files = sorted(corpus.glob("*.afb"))
    if not files:
        raise FileNotFoundError(f"Attack Flow corpus is missing: {corpus}")
    excluded = [
        path for path in files if exclude_overlapping_ctid and path.name in OVERLAPPING_CTID_FLOWS
    ]
    selected = [path for path in files if path not in excluded]
    rows: list[dict[str, Any]] = []
    per_file: dict[str, dict[str, int]] = {}
    label_set = {str(label) for label in vocab}
    for path in selected:
        file_rows, file_audit = _parse_one(path, label_set)
        rows.extend(file_rows)
        per_file[path.name] = file_audit
    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        raise ValueError("Attack Flow corpus yielded no in-vocabulary transitions")
    if frame.duplicated("sample_id").any():
        raise ValueError("Attack Flow sample IDs are not unique")
    audit: dict[str, Any] = {
        "corpus_files": len(files),
        "selected_files": len(selected),
        "excluded_overlap_files": len(excluded),
        "rows": len(frame),
        "flows_with_rows": int(frame["flow"].nunique()),
        "raw_subtechnique_source_rate": float(
            frame["raw_prefix_ids"].map(lambda prefix: "." in prefix[-1]).mean()
        ),
        "per_file": per_file,
    }
    return frame.reset_index(drop=True), audit
