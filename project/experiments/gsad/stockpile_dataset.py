from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


STOCKPILE_COMMIT = "996ec41cd1c5d1c7cc09e620fc55dabe5aefd9cc"
SELECTED_PROFILE_NAMES = (
    "Advanced Thief via DropBox",
    "Nosy Neighbor",
    "Alice 2.0",
    "Super Spy",
    "Worm",
    "Signed Binary Proxy Execution",
    "Enumerator",
    "Service Creation Lateral Movement",
    "Ransack",
    "Printer Queue",
)


def _repository_root(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "data_v2"
        / "external_stockpile"
        / "raw"
        / f"stockpile-{STOCKPILE_COMMIT}"
    )


def _parse_ability_map(repository: Path) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    files = sorted((repository / "data" / "abilities").rglob("*.yml"))
    id_pattern = re.compile(r"^ {0,2}(?:-\s+)?id:\s*([^\s#]+)\s*$")
    attack_pattern = re.compile(r"^\s+attack_id:\s*(T\d{4}(?:\.\d{3})?)\s*$")
    for path in files:
        current_id: str | None = None
        current_attack: str | None = None

        def finish() -> None:
            nonlocal current_id, current_attack
            if current_id is not None and current_attack is not None:
                existing = mapping.get(current_id)
                if existing is not None and existing != current_attack:
                    raise ValueError(f"ability {current_id} has conflicting ATT&CK IDs")
                mapping[current_id] = current_attack

        for line in path.read_text(encoding="utf-8").splitlines():
            id_match = id_pattern.match(line)
            if id_match:
                finish()
                current_id = id_match.group(1).lower()
                current_attack = None
                continue
            attack_match = attack_pattern.match(line)
            if attack_match and current_id is not None and current_attack is None:
                current_attack = attack_match.group(1)
        finish()
    return mapping, len(files)


def _parse_profiles(repository: Path) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for path in sorted((repository / "data" / "adversaries").glob("*.yml")):
        profile_id = ""
        name = ""
        description = ""
        ordering: list[str] = []
        in_ordering = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("id:"):
                profile_id = line.split(":", 1)[1].strip().lower()
            elif line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"\'')
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"\'')
            elif line.startswith("atomic_ordering:"):
                in_ordering = True
            elif in_ordering:
                match = re.match(r"^\s*-\s+([^\s#]+)(?:\s+#.*)?$", line)
                if match:
                    ordering.append(match.group(1).lower())
                elif not line.strip() or line.lstrip().startswith("#"):
                    continue
                elif line and not line.startswith(" "):
                    in_ordering = False
        profiles.append(
            {
                "path": path,
                "id": profile_id,
                "name": name,
                "description": description,
                "ordering": tuple(ordering),
            }
        )
    return profiles


def load_stockpile_transitions(
    project_root: Path, vocab: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repository = _repository_root(Path(project_root))
    if not repository.exists():
        raise FileNotFoundError(f"Stockpile snapshot is missing: {repository}")
    ability_map, ability_files = _parse_ability_map(repository)
    all_profiles = _parse_profiles(repository)
    by_name = {str(profile["name"]): profile for profile in all_profiles}
    missing = sorted(set(SELECTED_PROFILE_NAMES) - set(by_name))
    if missing:
        raise ValueError(f"pre-registered Stockpile profiles are missing: {missing}")
    selected = [by_name[name] for name in SELECTED_PROFILE_NAMES]
    vocab_set = {str(label) for label in vocab}
    rows: list[dict[str, object]] = []
    mapped_steps = 0
    transition_events = 0
    dropped_target = 0
    per_profile: dict[str, dict[str, int | float]] = {}
    for profile in selected:
        ordering = tuple(str(item) for item in profile["ordering"])
        mapped = [ability_map.get(ability_id) for ability_id in ordering]
        mapped_steps += sum(technique is not None for technique in mapped)
        kept_for_profile = 0
        for position in range(len(mapped) - 1):
            source_raw = mapped[position]
            target_raw = mapped[position + 1]
            if source_raw is None or target_raw is None:
                continue
            transition_events += 1
            target_parent = target_raw.split(".")[0]
            if target_parent not in vocab_set:
                dropped_target += 1
                continue
            source_parent = source_raw.split(".")[0]
            rows.append(
                {
                    "sample_id": f"{profile['id']}::{position}",
                    "profile_id": str(profile["id"]),
                    "profile": str(profile["name"]),
                    "position": position,
                    "prefix_ids": (source_parent,),
                    "raw_prefix_ids": (source_raw,),
                    "target": target_parent,
                    "evaluation_next_raw_id": target_raw,
                }
            )
            kept_for_profile += 1
        per_profile[str(profile["name"])] = {
            "ordered_steps": len(ordering),
            "mapped_steps": sum(technique is not None for technique in mapped),
            "mapping_coverage": (
                sum(technique is not None for technique in mapped) / len(ordering)
                if ordering
                else 0.0
            ),
            "kept_transitions": kept_for_profile,
        }
    frame = pd.DataFrame(rows)
    if len(frame) == 0 or frame.duplicated("sample_id").any():
        raise ValueError("Stockpile transition extraction failed")
    audit: dict[str, Any] = {
        "commit": STOCKPILE_COMMIT,
        "root_profiles": len(all_profiles),
        "selected_profiles": len(selected),
        "ability_yaml_files": ability_files,
        "mapped_ability_ids": len(ability_map),
        "mapped_steps": mapped_steps,
        "transition_events_before_vocab_filter": transition_events,
        "dropped_target_events": dropped_target,
        "kept_rows": len(frame),
        "subtechnique_step_instances": int(
            sum("." in prefix[-1] for prefix in frame["raw_prefix_ids"])
            + sum("." in raw for raw in frame["evaluation_next_raw_id"])
        ),
        "per_profile": per_profile,
    }
    return frame.reset_index(drop=True), audit
