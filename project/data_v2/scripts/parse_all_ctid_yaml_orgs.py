from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data_v2" / "external_ctid" / "raw" / "adversary_emulation_library-master"
PARSED_ROOT = PROJECT_ROOT / "data_v2" / "external_ctid" / "parsed"
PARSED_ROOT.mkdir(parents=True, exist_ok=True)


TARGET_YAMLS = [
    ("apt29", "APT29.yaml", "apt29"),
    ("carbanak", "Carbanak.yaml", "carbanak"),
    ("fin6", "FIN6.yaml", "fin6"),
    ("fin7", "Fin7.yaml", "fin7"),
    ("menu_pass", "menupass.yaml", "menu_pass"),
    ("oilrig", "oilrig.yaml", "oilrig"),
    ("sandworm", "sandworm.yaml", "sandworm"),
    ("wizard_spider", "wizard_spider.yaml", "wizard_spider"),
    ("turla", "turla_carbon.yaml", "turla_carbon"),
    ("turla", "turla_snake.yaml", "turla_snake"),
]


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


def parse_yaml_plan(yaml_path: Path, plan_name: str) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []

    def walk(node, path="root", scenario_hint=None):
        if isinstance(node, dict):
            local_scenario = scenario_hint

            for k in ["scenario", "Scenario", "name", "title", "id"]:
                if k in node and isinstance(node[k], str):
                    v = node[k].strip()
                    if "scenario" in v.lower():
                        local_scenario = v

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
                        "org_name": plan_name,
                        "plan_id": plan_name,
                        "scenario_id": local_scenario or "unknown",
                        "step_id": f"{plan_name}::{len(rows)+1}",
                        "step_order": len(rows) + 1,
                        "step_title": str(step_title)[:500],
                        "step_text": step_text[:30000],
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

            # 当前仍采用“每步取第一个 parent-technique”的简化规则
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


def run_one(org_dir: str, yaml_filename: str, plan_name: str) -> dict:
    yaml_path = RAW_ROOT / org_dir / "Emulation_Plan" / "yaml" / yaml_filename
    if not yaml_path.exists():
        return {
            "plan_name": plan_name,
            "status": "missing_yaml",
            "yaml_path": str(yaml_path),
            "steps": 0,
            "eval_samples": 0,
        }

    rows = parse_yaml_plan(yaml_path, plan_name)
    if not rows:
        return {
            "plan_name": plan_name,
            "status": "parsed_empty",
            "yaml_path": str(yaml_path),
            "steps": 0,
            "eval_samples": 0,
        }

    df_steps = pd.DataFrame(rows).sort_values(
        ["scenario_id", "step_order", "step_id"]
    ).reset_index(drop=True)

    steps_out = PARSED_ROOT / f"ctid_steps_long_{plan_name}.csv"
    eval_out = PARSED_ROOT / f"ctid_eval_parent_{plan_name}.csv"

    save_steps = df_steps.copy()
    for col in ["attack_technique_ids_raw", "attack_technique_ids_parent"]:
        save_steps[col] = save_steps[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
    save_steps.to_csv(steps_out, index=False, encoding="utf-8-sig")

    eval_df = build_eval_from_steps(df_steps.copy())
    eval_df.to_csv(eval_out, index=False, encoding="utf-8-sig")

    return {
        "plan_name": plan_name,
        "status": "ok",
        "yaml_path": str(yaml_path),
        "steps": len(df_steps),
        "eval_samples": len(eval_df),
        "steps_out": str(steps_out),
        "eval_out": str(eval_out),
    }


def main():
    results = []
    for org_dir, yaml_filename, plan_name in TARGET_YAMLS:
        print(f"\n[RUN] {plan_name} <- {org_dir}/Emulation_Plan/yaml/{yaml_filename}")
        res = run_one(org_dir, yaml_filename, plan_name)
        results.append(res)
        print(res)

    summary_df = pd.DataFrame(results)
    summary_out = PARSED_ROOT / "ctid_yaml_batch_parse_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved -> {summary_out}")


if __name__ == "__main__":
    main()