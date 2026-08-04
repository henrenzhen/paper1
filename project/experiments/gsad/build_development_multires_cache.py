from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import sha256_file, write_canonical_json
from .data_protocol import EXTERNAL_SIM_ROOTS, TEST_ROOTS, sim_root


SOURCE_COLUMNS = (
    "sequence_id",
    "prefix_len",
    "prefix_technique_ids",
    "prefix_technique_ids_parent",
    "next_technique_id_parent",
    "next_technique_id",
)


def build_cache(project_root: Path) -> tuple[Path, Path]:
    root = Path(project_root)
    core = root / "data_v2" / "core"
    sources = [
        core / "sim_train_parent_min3.csv",
        core / "sim_val_parent_min3.csv",
        core / "sim_test_parent_min3.csv",
    ]
    frames = [pd.read_csv(path, usecols=list(SOURCE_COLUMNS)) for path in sources]
    combined = pd.concat(frames, ignore_index=True)
    roots = combined["sequence_id"].astype(str).map(sim_root)
    excluded = set(EXTERNAL_SIM_ROOTS) | set(TEST_ROOTS)
    development = combined.loc[~roots.isin(excluded)].copy()
    development["_root"] = development["sequence_id"].astype(str).map(sim_root)
    development = development.sort_values(
        ["_root", "sequence_id", "prefix_len"]
    ).drop(columns="_root")
    if len(development) != 10555:
        raise AssertionError(f"unexpected development row count: {len(development)}")
    retained_roots = set(development["sequence_id"].astype(str).map(sim_root))
    if len(retained_roots) != 133 or retained_roots & excluded:
        raise AssertionError("development cache root boundary failed")
    cache = core / "sim_development_multires_min3.csv"
    development.to_csv(cache, index=False, encoding="utf-8", lineterminator="\n")
    manifest = core / "sim_development_multires_min3.manifest.json"
    write_canonical_json(
        manifest,
        {
            "cache": cache.name,
            "cache_sha256": sha256_file(cache),
            "excluded_external_roots": sorted(EXTERNAL_SIM_ROOTS),
            "excluded_locked_roots": sorted(TEST_ROOTS),
            "retained_rows": len(development),
            "retained_roots": len(retained_roots),
            "source_hashes": {path.name: sha256_file(path) for path in sources},
        },
    )
    return cache, manifest


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    cache_path, manifest_path = build_cache(project)
    print(cache_path)
    print(manifest_path)
