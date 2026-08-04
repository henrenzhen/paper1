from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


VALIDATION_ROOTS = frozenset(
    {
        "SIM_001",
        "SIM_004",
        "SIM_007",
        "SIM_019",
        "SIM_021",
        "SIM_054",
        "SIM_057",
        "SIM_062",
        "SIM_063",
        "SIM_064",
        "SIM_093",
        "SIM_107",
        "SIM_108",
        "SIM_113",
        "SIM_126",
        "SIM_130",
        "SIM_134",
        "SIM_153",
        "SIM_158",
        "SIM_160",
    }
)

CALIBRATION_ROOTS = frozenset(
    {
        "SIM_003",
        "SIM_013",
        "SIM_027",
        "SIM_038",
        "SIM_048",
        "SIM_067",
        "SIM_082",
        "SIM_085",
        "SIM_092",
        "SIM_103",
        "SIM_105",
        "SIM_106",
        "SIM_109",
        "SIM_111",
        "SIM_122",
        "SIM_137",
        "SIM_156",
        "SIM_161",
        "SIM_165",
        "SIM_171",
    }
)

TEST_ROOTS = frozenset(
    {
        "SIM_016",
        "SIM_024",
        "SIM_035",
        "SIM_037",
        "SIM_039",
        "SIM_046",
        "SIM_047",
        "SIM_053",
        "SIM_072",
        "SIM_073",
        "SIM_081",
        "SIM_083",
        "SIM_084",
        "SIM_095",
        "SIM_124",
        "SIM_128",
        "SIM_133",
        "SIM_141",
        "SIM_146",
        "SIM_169",
    }
)

EXTERNAL_SIM_ROOTS = frozenset(
    {
        "SIM_008",
        "SIM_010",
        "SIM_014",
        "SIM_030",
        "SIM_033",
        "SIM_040",
        "SIM_041",
        "SIM_044",
        "SIM_090",
    }
)

_REQUIRED_SOURCE_COLUMNS = (
    "sequence_id",
    "prefix_len",
    "prefix_technique_ids_parent",
    "next_technique_id_parent",
)

_ALLOWED_FEATURE_COLUMNS = frozenset(
    {
        "sequence_id",
        "root",
        "prefix_len",
        "prefix_ids",
        "entropy",
        "margin",
        "transition_surprise",
        "backoff_signal",
        "fit_distance",
        "set_size",
    }
)

_BLOCKED_EXACT_COLUMNS = frozenset(
    {
        "true_label",
        "matched_technique_name",
        "matched_description",
        "matched_command_summary",
    }
)


@dataclass(frozen=True)
class FrozenSplit:
    fit: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    excluded_roots: frozenset[str]
    audit: dict[str, Any]


def sim_root(sequence_id: str) -> str:
    """Return the actor/layer root shared by all SIM parts."""

    return re.sub(r"_part\d+$", "", str(sequence_id))


def _parse_prefix_ids(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split("||") if part.strip())


def _prepare_source(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_SOURCE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"SIM source is missing required columns: {missing}")
    prepared = frame.loc[:, _REQUIRED_SOURCE_COLUMNS].copy()
    prepared = prepared.rename(columns={"next_technique_id_parent": "target"})
    prepared["sequence_id"] = prepared["sequence_id"].astype(str)
    prepared["prefix_len"] = prepared["prefix_len"].astype(int)
    prepared["root"] = prepared["sequence_id"].map(sim_root)
    prepared["prefix_ids"] = prepared["prefix_technique_ids_parent"].map(
        _parse_prefix_ids
    )
    prepared["target"] = prepared["target"].astype(str).str.strip()
    if (prepared["prefix_ids"].map(len) != prepared["prefix_len"]).any():
        bad = prepared.loc[
            prepared["prefix_ids"].map(len) != prepared["prefix_len"],
            ["sequence_id", "prefix_len", "prefix_ids"],
        ].head(5)
        raise ValueError(f"prefix_len mismatch detected: {bad.to_dict('records')}")
    return prepared


def build_frozen_split(frames: Sequence[pd.DataFrame]) -> FrozenSplit:
    """Build the pre-registered 93/20/20/20 root-disjoint split."""

    if not frames:
        raise ValueError("at least one SIM source frame is required")
    combined = pd.concat([_prepare_source(frame) for frame in frames], ignore_index=True)
    if combined.duplicated(["sequence_id", "prefix_len"]).any():
        raise ValueError("duplicate (sequence_id, prefix_len) keys detected")

    source_rows = len(combined)
    excluded_mask = combined["root"].isin(EXTERNAL_SIM_ROOTS)
    excluded_rows = int(excluded_mask.sum())
    eligible = combined.loc[~excluded_mask].copy()
    eligible_roots = frozenset(eligible["root"].unique())

    named_roots = VALIDATION_ROOTS | CALIBRATION_ROOTS | TEST_ROOTS
    missing_named = sorted(named_roots - eligible_roots)
    if missing_named:
        raise ValueError(f"frozen roots missing from source: {missing_named}")
    fit_roots = eligible_roots - named_roots

    def select(roots: frozenset[str]) -> pd.DataFrame:
        return eligible.loc[eligible["root"].isin(roots)].reset_index(drop=True)

    fit = select(frozenset(fit_roots))
    validation = select(VALIDATION_ROOTS)
    calibration = select(CALIBRATION_ROOTS)
    test = select(TEST_ROOTS)

    partitions = (fit, validation, calibration, test)
    actual_root_counts = tuple(int(part["root"].nunique()) for part in partitions)
    actual_row_counts = tuple(len(part) for part in partitions)
    if actual_root_counts != (93, 20, 20, 20):
        raise ValueError(f"unexpected frozen root counts: {actual_root_counts}")
    if actual_row_counts != (7371, 1592, 1592, 1592):
        raise ValueError(f"unexpected frozen row counts: {actual_row_counts}")

    root_sets = [frozenset(part["root"].unique()) for part in partitions]
    overlaps = {
        f"{left_index}:{right_index}": sorted(root_sets[left_index] & root_sets[right_index])
        for left_index in range(len(root_sets))
        for right_index in range(left_index + 1, len(root_sets))
    }
    if any(overlaps.values()):
        raise ValueError(f"root leakage across frozen partitions: {overlaps}")

    audit: dict[str, Any] = {
        "source_rows": int(source_rows),
        "retained_rows": int(len(eligible)),
        "excluded_rows": excluded_rows,
        "eligible_roots": int(len(eligible_roots)),
        "excluded_roots": sorted(EXTERNAL_SIM_ROOTS),
        "partition_rows": {
            "fit": len(fit),
            "validation": len(validation),
            "calibration": len(calibration),
            "test": len(test),
        },
        "partition_roots": {
            "fit": int(fit["root"].nunique()),
            "validation": int(validation["root"].nunique()),
            "calibration": int(calibration["root"].nunique()),
            "test": int(test["root"].nunique()),
        },
        "root_overlaps": overlaps,
    }
    return FrozenSplit(
        fit=fit,
        validation=validation,
        calibration=calibration,
        test=test,
        excluded_roots=EXTERNAL_SIM_ROOTS,
        audit=audit,
    )


def audit_feature_columns(
    columns: Sequence[str], target_columns: Sequence[str] = ("target",)
) -> None:
    """Hard-fail when a feature could expose the answer or future context."""

    targets = {str(column).strip().lower() for column in target_columns}
    problems: list[str] = []
    for raw_column in columns:
        column = str(raw_column).strip().lower()
        if column in targets:
            problems.append(f"target column used as feature: {raw_column}")
        elif column.startswith("next_technique"):
            problems.append(f"future label field used as feature: {raw_column}")
        elif column in _BLOCKED_EXACT_COLUMNS:
            problems.append(f"future context field used as feature: {raw_column}")
        elif column not in _ALLOWED_FEATURE_COLUMNS:
            problems.append(f"feature is not allowlisted: {raw_column}")
    if problems:
        raise ValueError("; ".join(problems))


def normalize_ctid_actor(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if normalized.startswith("turla_") or normalized == "turla":
        return "turla"
    return normalized
