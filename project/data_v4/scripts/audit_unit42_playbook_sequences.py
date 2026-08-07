#!/usr/bin/env python3
"""Audit Unit42 Playbook Viewer bundles for explicit within-campaign order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_CARD = PROJECT_ROOT / "data_v4/protocols/unit42_playbook_sequence_audit_v1.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_v4/audits/unit42_playbook_sequence_v1"
SEQUENCE_TOKENS = (
    "order",
    "sequence",
    "step",
    "next",
    "preced",
    "follow",
    "before",
    "after",
    "execution_time",
    "observed_time",
)
TEMPORAL_RELATION_TYPES = {"precedes", "preceded-by", "follows", "followed-by", "next"}
TIME_FIELDS = (
    "created",
    "modified",
    "published",
    "first_seen",
    "last_seen",
    "start_time",
    "stop_time",
    "valid_from",
    "valid_until",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def git_value(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def external_ids(attack_pattern: dict[str, Any]) -> list[str]:
    return sorted(
        {
            reference["external_id"]
            for reference in attack_pattern.get("external_references", [])
            if reference.get("source_name") == "mitre-attack"
            and reference.get("external_id")
        }
    )


def sequence_like_fields(objects: Sequence[dict[str, Any]]) -> list[str]:
    fields = set()
    for item in objects:
        for key in item:
            lowered = key.lower()
            if any(token in lowered for token in SEQUENCE_TOKENS):
                fields.add(key)
    return sorted(fields)


def load_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("objects"), list):
        raise ValueError("not a STIX-like bundle with an objects list")
    objects = [item for item in value["objects"] if isinstance(item, dict)]
    if len(objects) != len(value["objects"]):
        raise ValueError("bundle objects contains a non-object item")
    return value, objects


def audit(source: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    index_path = source / "playbooks.json"
    bundle_dir = source / "playbook_json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if not isinstance(index, list):
        raise ValueError("playbooks.json is not a list")
    index_by_file = {row["pb_file"]: row for row in index}
    if len(index_by_file) != len(index):
        raise AssertionError("duplicate pb_file in playbooks.json")
    paths = sorted(bundle_dir.glob("*.json"))
    inventory: list[dict[str, Any]] = []
    campaigns: list[dict[str, Any]] = []
    relationship_counts: Counter[tuple[str, str, str]] = Counter()
    global_types: Counter[str] = Counter()
    global_report_labels: Counter[str] = Counter()
    global_time_fields: Counter[str] = Counter()
    global_sequence_fields: Counter[str] = Counter()
    invalid: list[tuple[str, str]] = []

    for path in paths:
        try:
            _, objects = load_bundle(path)
        except Exception as error:  # audit must retain invalid external files
            invalid.append((path.name, str(error)))
            inventory.append(
                {
                    "bundle_file": path.name,
                    "indexed": int(path.name in index_by_file),
                    "title": index_by_file.get(path.name, {}).get("title", ""),
                    "sha256": sha256(path),
                    "valid_json_bundle": 0,
                    "error": str(error),
                }
            )
            continue

        by_id = {item.get("id"): item for item in objects if item.get("id")}
        type_counts = Counter(item.get("type", "<missing>") for item in objects)
        global_types.update(type_counts)
        for item in objects:
            for field in TIME_FIELDS:
                if field in item:
                    global_time_fields[field] += 1
            if item.get("type") == "report":
                global_report_labels.update(item.get("labels", []))
        sequence_fields = sequence_like_fields(objects)
        global_sequence_fields.update(sequence_fields)

        relationships = [item for item in objects if item.get("type") == "relationship"]
        relationship_rows: list[tuple[dict[str, Any], str, str]] = []
        attack_pattern_edges: list[dict[str, Any]] = []
        for relationship in relationships:
            source_type = by_id.get(relationship.get("source_ref"), {}).get("type", "<missing>")
            target_type = by_id.get(relationship.get("target_ref"), {}).get("type", "<missing>")
            relation = relationship.get("relationship_type", "<missing>")
            relationship_counts[(relation, source_type, target_type)] += 1
            relationship_rows.append((relationship, source_type, target_type))
            if source_type == "attack-pattern" and target_type == "attack-pattern":
                attack_pattern_edges.append(relationship)

        campaign_objects = [item for item in objects if item.get("type") == "campaign"]
        campaign_reports = [
            item
            for item in objects
            if item.get("type") == "report" and "campaign" in item.get("labels", [])
        ]
        bundle_eligible = 0
        campaign_uses_total = 0
        for campaign in campaign_objects:
            campaign_id = campaign["id"]
            uses_edges = [
                relationship
                for relationship, source_type, target_type in relationship_rows
                if relationship.get("relationship_type") == "uses"
                and relationship.get("source_ref") == campaign_id
                and source_type == "campaign"
                and target_type == "attack-pattern"
            ]
            campaign_uses_total += len(uses_edges)
            technique_ids = sorted(
                {
                    identifier
                    for edge in uses_edges
                    for identifier in external_ids(by_id.get(edge.get("target_ref"), {}))
                }
            )
            directed_edges = [
                edge
                for edge in attack_pattern_edges
                if edge.get("source_ref") in {item.get("target_ref") for item in uses_edges}
                and edge.get("target_ref") in {item.get("target_ref") for item in uses_edges}
            ]
            temporal_edges = [
                edge
                for edge in directed_edges
                if edge.get("relationship_type") in TEMPORAL_RELATION_TYPES
            ]
            occurrence_sequence_fields = sorted(
                {
                    key
                    for edge in uses_edges
                    for key in edge
                    if any(token in key.lower() for token in SEQUENCE_TOKENS)
                }
            )
            reports = [report for report in campaign_reports if campaign_id in report.get("object_refs", [])]
            report_attack_pattern_refs = sorted(
                {
                    reference
                    for report in reports
                    for reference in report.get("object_refs", [])
                    if by_id.get(reference, {}).get("type") == "attack-pattern"
                }
            )
            eligible = int(bool(temporal_edges or occurrence_sequence_fields))
            bundle_eligible += eligible
            if eligible:
                reason = "explicit temporal attack-pattern edge or sequence-like occurrence field"
            elif len(technique_ids) < 2:
                reason = "fewer than two campaign-linked ATT&CK techniques"
            else:
                reason = "unordered campaign-to-technique membership only"
            campaigns.append(
                {
                    "bundle_file": path.name,
                    "campaign_id": campaign_id,
                    "campaign_name": campaign.get("name", ""),
                    "first_seen": campaign.get("first_seen", ""),
                    "last_seen": campaign.get("last_seen", ""),
                    "campaign_reports": len(reports),
                    "campaign_uses_attack_pattern_edges": len(uses_edges),
                    "unique_mitre_attack_ids": len(technique_ids),
                    "mitre_attack_ids": compact_json(technique_ids),
                    "uses_created_unique": len({edge.get("created") for edge in uses_edges if edge.get("created")}),
                    "uses_modified_unique": len({edge.get("modified") for edge in uses_edges if edge.get("modified")}),
                    "report_attack_pattern_refs": len(report_attack_pattern_refs),
                    "attack_pattern_to_attack_pattern_edges": len(directed_edges),
                    "temporal_attack_pattern_edges": len(temporal_edges),
                    "occurrence_sequence_fields": compact_json(occurrence_sequence_fields),
                    "sequence_eligible": eligible,
                    "reason": reason,
                }
            )

        inventory.append(
            {
                "bundle_file": path.name,
                "indexed": int(path.name in index_by_file),
                "title": index_by_file.get(path.name, {}).get("title", ""),
                "sha256": sha256(path),
                "valid_json_bundle": 1,
                "objects": len(objects),
                "campaigns": len(campaign_objects),
                "campaign_reports": len(campaign_reports),
                "attack_patterns": type_counts["attack-pattern"],
                "relationships": type_counts["relationship"],
                "campaign_uses_attack_pattern_edges": campaign_uses_total,
                "attack_pattern_to_attack_pattern_edges": len(attack_pattern_edges),
                "sequence_like_fields": compact_json(sequence_fields),
                "sequence_eligible_campaigns": bundle_eligible,
                "error": "",
            }
        )

    relationship_matrix = [
        {
            "relationship_type": relation,
            "source_object_type": source_type,
            "target_object_type": target_type,
            "count": count,
        }
        for (relation, source_type, target_type), count in sorted(relationship_counts.items())
    ]
    indexed = set(index_by_file)
    actual = {path.name for path in paths}
    summary = {
        "index_rows": len(index),
        "index_unique_files": len(indexed),
        "bundle_json_files": len(paths),
        "indexed_missing_files": sorted(indexed - actual),
        "unindexed_bundle_files": sorted(actual - indexed),
        "invalid_bundles": invalid,
        "object_type_counts": dict(sorted(global_types.items())),
        "report_label_counts": dict(sorted(global_report_labels.items())),
        "time_field_counts": {field: global_time_fields[field] for field in TIME_FIELDS},
        "sequence_like_field_bundle_counts": dict(sorted(global_sequence_fields.items())),
        "campaigns": len(campaigns),
        "campaigns_with_at_least_two_techniques": sum(row["unique_mitre_attack_ids"] >= 2 for row in campaigns),
        "campaigns_with_first_seen": sum(bool(row["first_seen"]) for row in campaigns),
        "campaigns_with_last_seen": sum(bool(row["last_seen"]) for row in campaigns),
        "campaigns_with_multiple_uses_created_values": sum(row["uses_created_unique"] > 1 for row in campaigns),
        "campaigns_with_attack_pattern_edges": sum(row["attack_pattern_to_attack_pattern_edges"] > 0 for row in campaigns),
        "campaigns_with_temporal_attack_pattern_edges": sum(row["temporal_attack_pattern_edges"] > 0 for row in campaigns),
        "sequence_eligible_campaigns": sum(row["sequence_eligible"] for row in campaigns),
    }
    return inventory, campaigns, relationship_matrix, summary


def report_markdown(summary: dict[str, Any], relationship_matrix: Sequence[dict[str, Any]]) -> str:
    relation_lookup = {
        (row["relationship_type"], row["source_object_type"], row["target_object_type"]): row["count"]
        for row in relationship_matrix
    }
    eligible = summary["sequence_eligible_campaigns"]
    decision = (
        "PASS: at least one campaign contains an explicit qualifying order carrier"
        if eligible
        else "REJECT as a direct sequential source: no campaign contains an explicit qualifying order carrier"
    )
    lines = [
        "# Unit42 Playbook Viewer sequence-sufficiency audit",
        "",
        "## Inventory",
        "",
        f"- `playbooks.json` rows / unique files: {summary['index_rows']} / {summary['index_unique_files']}",
        f"- JSON bundles: {summary['bundle_json_files']}",
        f"- Unindexed bundles: {compact_json(summary['unindexed_bundle_files'])}",
        f"- Invalid bundles: {len(summary['invalid_bundles'])}",
        f"- Campaign objects: {summary['campaigns']}",
        f"- Campaigns with at least two linked ATT&CK techniques: {summary['campaigns_with_at_least_two_techniques']}",
        "",
        "## Order carriers",
        "",
        f"- Campaigns with `first_seen`: {summary['campaigns_with_first_seen']}",
        f"- Campaigns with `last_seen`: {summary['campaigns_with_last_seen']}",
        f"- Campaigns whose campaign→technique `uses` edges have multiple `created` values: {summary['campaigns_with_multiple_uses_created_values']}",
        f"- Campaigns with any attack-pattern→attack-pattern edge: {summary['campaigns_with_attack_pattern_edges']}",
        f"- Campaigns with a qualifying temporal attack-pattern edge: {summary['campaigns_with_temporal_attack_pattern_edges']}",
        f"- Sequence-eligible campaigns: **{eligible}**",
        "",
        "Selected relationship counts:",
        "",
        "| Relationship | Source | Target | Count | Temporal meaning |",
        "|---|---|---|---:|---|",
        f"| uses | campaign | attack-pattern | {relation_lookup.get(('uses', 'campaign', 'attack-pattern'), 0)} | membership only |",
        f"| uses | intrusion-set | attack-pattern | {relation_lookup.get(('uses', 'intrusion-set', 'attack-pattern'), 0)} | membership only |",
        f"| uses | malware | attack-pattern | {relation_lookup.get(('uses', 'malware', 'attack-pattern'), 0)} | membership only |",
        "",
        "## Decision",
        "",
        f"**{decision}.**",
        "",
        "Campaign interval timestamps, STIX object lifecycle timestamps, `object_refs` position, kill-chain phase, and membership edges were not treated as technique execution order under the frozen protocol.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    required = (source / "playbooks.json", source / "playbook_json", source / "LICENSE")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"not a Unit42 Playbook Viewer checkout: {source}")
    managed = (
        "bundle_inventory.csv",
        "campaign_sequence_audit.csv",
        "relationship_type_matrix.csv",
        "report.md",
        "stdout.log",
        "audit_manifest.json",
    )
    if any((output / name).exists() for name in managed):
        raise FileExistsError("refusing to overwrite Unit42 sequence audit")
    inventory, campaigns, relationship_matrix, summary = audit(source)
    output.mkdir(parents=True, exist_ok=False)
    inventory_columns = (
        "bundle_file", "indexed", "title", "sha256", "valid_json_bundle",
        "objects", "campaigns", "campaign_reports", "attack_patterns",
        "relationships", "campaign_uses_attack_pattern_edges",
        "attack_pattern_to_attack_pattern_edges", "sequence_like_fields",
        "sequence_eligible_campaigns", "error",
    )
    campaign_columns = (
        "bundle_file", "campaign_id", "campaign_name", "first_seen", "last_seen",
        "campaign_reports", "campaign_uses_attack_pattern_edges",
        "unique_mitre_attack_ids", "mitre_attack_ids", "uses_created_unique",
        "uses_modified_unique", "report_attack_pattern_refs",
        "attack_pattern_to_attack_pattern_edges", "temporal_attack_pattern_edges",
        "occurrence_sequence_fields", "sequence_eligible", "reason",
    )
    write_csv(output / "bundle_inventory.csv", inventory, inventory_columns)
    write_csv(output / "campaign_sequence_audit.csv", campaigns, campaign_columns)
    write_csv(output / "relationship_type_matrix.csv", relationship_matrix, ("relationship_type", "source_object_type", "target_object_type", "count"))
    markdown = report_markdown(summary, relationship_matrix)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "stdout.log").write_text(markdown + "\n", encoding="utf-8")
    bundle_paths = sorted((source / "playbook_json").glob("*.json"))
    external_paths = [source / "playbooks.json", source / "LICENSE", *bundle_paths]
    output_hashes = {
        name: sha256(output / name)
        for name in managed
        if name != "audit_manifest.json" and (output / name).exists()
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "external_source": {
            "origin": git_value(source, "remote", "get-url", "origin"),
            "commit": git_value(source, "rev-parse", "HEAD"),
            "commit_date": git_value(source, "show", "-s", "--format=%cI", "HEAD"),
            "playbooks_json_sha256": sha256(source / "playbooks.json"),
            "license_sha256": sha256(source / "LICENSE"),
            "bundle_tree_sha256": tree_sha256(bundle_paths, source),
            "all_external_inputs_tree_sha256": tree_sha256(external_paths, source),
            "bundle_file_sha256": {path.name: sha256(path) for path in bundle_paths},
        },
        "script": {"path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "method_card": {"path": METHOD_CARD.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(METHOD_CARD)},
        "eligibility_rule": "explicit campaign-specific occurrence order/timestamp or temporal attack-pattern edge; membership/tactic/object order excluded",
        "summary": summary,
        "outputs_sha256": output_hashes,
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
