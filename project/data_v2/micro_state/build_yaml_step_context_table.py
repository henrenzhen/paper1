import re
from pathlib import Path

import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parent

INPUT_YAML = Path("/root/project/data_v2/external_ctid/raw/APT29.yaml")
OUTPUT_CSV = BASE_DIR / "APT29_step_context_table.csv"


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def collapse_ws(x: str) -> str:
    return re.sub(r"\s+", " ", clean_text(x)).strip()


def summarize_command(cmd: str, max_len: int = 220) -> str:
    cmd = collapse_ws(cmd)
    if len(cmd) <= max_len:
        return cmd
    return cmd[: max_len - 3] + "..."


def extract_command(step: dict) -> str:
    executors = step.get("executors", [])
    if isinstance(executors, list) and executors:
        for ex in executors:
            if isinstance(ex, dict) and clean_text(ex.get("command")):
                return clean_text(ex.get("command"))

    platforms = step.get("platforms", {})
    if isinstance(platforms, dict):
        for _, os_block in platforms.items():
            if not isinstance(os_block, dict):
                continue
            for _, exec_block in os_block.items():
                if isinstance(exec_block, dict) and clean_text(exec_block.get("command")):
                    return clean_text(exec_block.get("command"))

    return ""


def is_plan_details_block(obj):
    return isinstance(obj, dict) and "emulation_plan_details" in obj


def is_step_block(obj):
    return (
        isinstance(obj, dict)
        and "id" in obj
        and "technique" in obj
        and isinstance(obj.get("technique"), dict)
    )


def infer_scenario_id(proc_step: str) -> str:
    proc_step = clean_text(proc_step)
    m = re.match(r"^(\d+)", proc_step)
    if m:
        return m.group(1)
    return "unknown"


def main():
    if not INPUT_YAML.exists():
        raise FileNotFoundError(f"Missing YAML: {INPUT_YAML}")

    with open(INPUT_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level YAML list.")

    source_org = INPUT_YAML.stem.lower()
    plan_id = source_org
    adversary_name = source_org

    rows = []
    step_idx = 0

    for item in data:
        if is_plan_details_block(item):
            meta = item["emulation_plan_details"]
            adversary_name = clean_text(meta.get("adversary_name")) or adversary_name
            continue

        if not is_step_block(item):
            continue

        step_idx += 1

        technique = item.get("technique", {}) or {}
        proc_step = clean_text(item.get("procedure_step"))
        scenario_id = infer_scenario_id(proc_step)

        cmd_full = extract_command(item)

        rows.append(
            {
                "source_org": source_org,
                "plan_id": plan_id,
                "adversary_name": adversary_name,
                "scenario_id": scenario_id,
                "step_idx": step_idx,
                "yaml_step_id": clean_text(item.get("id")),
                "procedure_step": proc_step,
                "tactic": clean_text(item.get("tactic")),
                "technique_attack_id": clean_text(technique.get("attack_id")),
                "technique_name": clean_text(technique.get("name")),
                "name": clean_text(item.get("name")),
                "description": collapse_ws(item.get("description")),
                "command_summary": summarize_command(cmd_full),
                "command_full": cmd_full,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No step blocks parsed from YAML.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Parsed YAML: {INPUT_YAML}")
    print(f"[INFO] Output CSV : {OUTPUT_CSV}")
    print(f"[INFO] Rows       : {len(df)}")
    print("\n[PREVIEW]")
    preview_cols = [
        "step_idx",
        "procedure_step",
        "technique_attack_id",
        "technique_name",
        "description",
        "command_summary",
    ]
    print(df[preview_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()