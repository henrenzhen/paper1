from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "data_v2" / "repro_external"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.experiments.gsad.attack_flow_dataset import (  # noqa: E402
    load_attack_flow_transitions,
)
from project.experiments.gsad.run_mrct_development import (  # noqa: E402
    load_multires_development,
)
from project.experiments.gsad.run_qmrct_external import (  # noqa: E402
    load_ctid_external,
)
from project.experiments.gsad.stockpile_dataset import (  # noqa: E402
    load_stockpile_transitions,
)


SOURCE_REVISIONS = {
    "attack_flow": {
        "url": "https://github.com/center-for-threat-informed-defense/attack-flow",
        "commit": "295d20d27cefce0a2d309b6c24781545e45f547d",
    },
    "ctid": {
        "url": "https://github.com/center-for-threat-informed-defense/adversary_emulation_library",
        "commit": "4467a6eed6e67d25009704130e1d27d1a8007f57",
    },
    "stockpile": {
        "url": "https://github.com/mitre/stockpile",
        "commit": "996ec41cd1c5d1c7cc09e620fc55dabe5aefd9cc",
    },
}


def _json_array(value: object) -> str:
    return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))


def _write_snapshot(
    frame: pd.DataFrame,
    source: str,
    campaign_id: pd.Series,
    metadata_columns: list[str],
) -> dict[str, object]:
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "source": source,
            "campaign_id": campaign_id.astype(str),
            "prefix_len": frame["prefix_ids"].map(len).astype(int),
            "prefix": frame["prefix_ids"].map(_json_array),
            "raw_prefix": frame["raw_prefix_ids"].map(_json_array),
            "true_label": frame["target"].astype(str),
            "target_raw_id": frame["evaluation_next_raw_id"].astype(str),
        }
    )
    for column in metadata_columns:
        output[column] = frame[column]
    path = OUTPUT_ROOT / "loader_snapshots" / f"{source}_current_loader.csv"
    output.to_csv(path, index=False, encoding="utf-8")
    return {
        "rows": int(len(output)),
        "campaigns": int(output["campaign_id"].nunique()),
        "unique_sample_ids": int(output["sample_id"].nunique()),
        "prefix_length_counts": {
            str(key): int(value)
            for key, value in output["prefix_len"].value_counts().sort_index().items()
        },
        "target_classes": int(output["true_label"].nunique()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "loader_snapshots").mkdir(parents=True, exist_ok=True)

    _, vocab, _ = load_multires_development(PROJECT_ROOT)
    attack_flow, attack_audit = load_attack_flow_transitions(PROJECT_ROOT, vocab)
    ctid = load_ctid_external(PROJECT_ROOT)
    stockpile, stockpile_audit = load_stockpile_transitions(PROJECT_ROOT, vocab)

    datasets = {
        "attack_flow": _write_snapshot(
            attack_flow,
            "attack_flow",
            attack_flow["flow"],
            ["flow", "source_action", "target_action"],
        ),
        "ctid": _write_snapshot(
            ctid,
            "ctid",
            ctid["sample_id"].astype(str).str.rsplit("::", n=1).str[0],
            ["org_name", "actor", "plan_id", "scenario_id", "mapping_quality"],
        ),
        "stockpile": _write_snapshot(
            stockpile,
            "stockpile",
            stockpile["profile_id"],
            ["profile_id", "profile", "position"],
        ),
    }

    audit = {
        "attack_flow": attack_audit,
        "stockpile": stockpile_audit,
    }
    (OUTPUT_ROOT / "loader_snapshots" / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = {}
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            relative = path.relative_to(OUTPUT_ROOT).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": 1,
        "source_revisions": SOURCE_REVISIONS,
        "datasets": datasets,
        "files": files,
    }
    (OUTPUT_ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(datasets, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
