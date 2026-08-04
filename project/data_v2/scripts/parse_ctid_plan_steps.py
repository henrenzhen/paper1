from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml


def extract_attack_ids(text: str) -> list[str]:
    if text is None:
        return []
    return sorted(set(re.findall(r"T\d{4}(?:\.\d{3})?", str(text))))


def parent_of(tech_id: str) -> str:
    return tech_id.split(".")[0] if tech_id else tech_id


def flatten_obj(obj):
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {flatten_obj(v)}")
        return " | ".join(parts)
    if isinstance(obj, list):
        return " | ".join(flatten_obj(x) for x in obj)
    return str(obj)


def parse_yaml_plan(yaml_path: Path, org_name: str) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []

    def walk(node, path="root", scenario_hint=None):
        if isinstance(node, dict):
            local_scenario = scenario_hint

            for k in ["scenario", "Scenario", "name", "title", "id"]:
                if k in node and isinstance(node[k], str):
                    if "scenario" in node[k].lower():
                        local_scenario = node[k]

            blob = json.dumps(node, ensure_ascii=False)
            attack_ids = extract_attack_ids(blob)

            has_step_signal = (
                "step" in path.lower()
                or "procedure" in path.lower()
                or "command" in blob.lower()
                or "description" in node
                or "commands" in node
                or "attack" in blob.lower()
            )

            if has_step_signal and attack_ids:
                step_title = (
                    node.get("title")
                    or node.get("name")
                    or node.get("step")
                    or node.get("description")
                    or path.split("/")[-1]
                )
                step_text = flatten_obj(node)

                rows.append(
                    {
                        "org_name": org_name,
                        "plan_id": org_name,
                        "scenario_id": local_scenario or "unknown",
                        "step_id": f"{org_name}::{len(rows)+1}",
                        "step_order": len(rows) + 1,
                        "step_title": str(step_title)[:500],
                        "step_text": step_text[:20000],
                        "attack_technique_ids_raw": attack_ids,
                        "attack_technique_ids_parent": sorted(set(parent_of(x) for x in attack_ids)),
                        "source_file": str(yaml_path),
                        "mapping_method": "yaml_regex_extract",
                        "mapping_quality": "high",
                    }
                )

            for k, v in node.items():
                walk(v, f"{path}/{k}", local_scenario)

        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", scenario_hint)

    walk(data)
    return rows


def build_eval_from_steps(df_steps: pd.DataFrame) -> pd.DataFrame:
    eval_rows = []

    for (org_name, plan_id, scenario_id), g in df_steps.groupby(
        ["org_name", "plan_id", "scenario_id"], dropna=False
    ):
        g = g.sort_values(["step_order", "step_id"]).reset_index(drop=True)

        seq = []
        step_ids = []

        for _, row in g.iterrows():
            parents = row["attack_technique_ids_parent"]
            if isinstance(parents, str):
                parents = json.loads(parents)

            if not parents:
                continue

            # 当前先用“每步第一个 parent”形成粗序列
            parent = parents[0]
            if len(seq) == 0 or seq[-1] != parent:
                seq.append(parent)
                step_ids.append(row["step_id"])

        if len(seq) < 2:
            continue

        for i in range(1, len(seq)):
            eval_rows.append(
                {
                    "sample_id": f"{org_name}::{scenario_id}::{i}",
                    "org_name": org_name,
                    "plan_id": plan_id,
                    "scenario_id": scenario_id,
                    "prefix_len": i,
                    "prefix_technique_ids_parent": json.dumps(seq[:i], ensure_ascii=False),
                    "next_technique_id_parent": seq[i],
                    "prefix_source_step_ids": json.dumps(step_ids[:i], ensure_ascii=False),
                    "target_source_step_id": step_ids[i],
                    "mapping_quality": "high",
                }
            )

    return pd.DataFrame(eval_rows)


def main():
    import sys

    project_root = Path(__file__).resolve().parents[2]
    raw_root = project_root / "data_v2" / "external_ctid" / "raw" / "adversary_emulation_library-master"
    out_root = project_root / "data_v2" / "external_ctid" / "parsed"
    out_root.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python parse_ctid_plan_steps.py <org_name> <yaml_filename>")

    org_name = sys.argv[1]
    yaml_filename = sys.argv[2]

    yaml_path = raw_root / org_name / "Emulation_Plan" / "yaml" / yaml_filename
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    rows = parse_yaml_plan(yaml_path, org_name)
    df_steps = pd.DataFrame(rows)

    if df_steps.empty:
        raise RuntimeError(f"No steps parsed from {yaml_path}")

    df_steps = df_steps.sort_values(["scenario_id", "step_order", "step_id"]).reset_index(drop=True)

    long_path = out_root / f"ctid_steps_long_{org_name}.csv"
    df_steps_to_save = df_steps.copy()
    for col in ["attack_technique_ids_raw", "attack_technique_ids_parent"]:
        df_steps_to_save[col] = df_steps_to_save[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df_steps_to_save.to_csv(long_path, index=False, encoding="utf-8-sig")

    df_eval = build_eval_from_steps(df_steps.copy())
    eval_path = out_root / f"ctid_eval_parent_{org_name}.csv"
    df_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {long_path}")
    print(f"[OK] wrote: {eval_path}")
    print(f"[INFO] org={org_name}, steps={len(df_steps)}, eval_samples={len(df_eval)}")
    if len(df_eval) > 0:
        print(df_eval.head(10).to_string(index=False))

if __name__ == "__main__":
    main()