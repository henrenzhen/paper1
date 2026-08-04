import json
import re
from pathlib import Path
import pandas as pd


BUNDLE_JSON = Path(r"E:\desktop\project_only\project\data\enterprise-attack-18.1.json")
OUTPUT_CSV = Path(r"E:\desktop\project_only\project\data\attack_kg_snippets.csv")


def clean_text(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def get_attack_id(obj):
    for ref in obj.get("external_references", []):
        ext_id = ref.get("external_id")
        if ext_id and str(ext_id).startswith("T"):
            return str(ext_id).strip()
    return ""


def main():
    if not BUNDLE_JSON.exists():
        raise FileNotFoundError(f"Bundle not found: {BUNDLE_JSON}")

    with open(BUNDLE_JSON, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Invalid STIX bundle: missing objects list")

    id2obj = {}
    for obj in objects:
        if isinstance(obj, dict) and "id" in obj:
            id2obj[obj["id"]] = obj

    rows = []

    # 1) attack-pattern snippets
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "attack-pattern":
            continue

        attack_id = get_attack_id(obj)
        name = clean_text(obj.get("name"))
        desc = clean_text(obj.get("description"))

        phases = []
        for kc in obj.get("kill_chain_phases", []):
            phase = clean_text(kc.get("phase_name"))
            if phase:
                phases.append(phase)

        text_parts = []
        if attack_id:
            text_parts.append(f"Technique {attack_id}")
        if name:
            text_parts.append(name)
        if phases:
            text_parts.append("Phases: " + ", ".join(phases))
        if desc:
            text_parts.append(desc)

        text = " | ".join(text_parts)
        if text:
            rows.append({
                "snippet_id": f"attack-pattern::{obj['id']}",
                "snippet_type": "attack-pattern",
                "attack_id": attack_id,
                "source_name": "",
                "target_name": name,
                "text": text,
            })

    # 2) relationship snippets
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "relationship":
            continue

        rel_type = clean_text(obj.get("relationship_type"))
        src_id = obj.get("source_ref", "")
        tgt_id = obj.get("target_ref", "")
        desc = clean_text(obj.get("description"))

        src_obj = id2obj.get(src_id, {})
        tgt_obj = id2obj.get(tgt_id, {})

        src_name = clean_text(src_obj.get("name"))
        tgt_name = clean_text(tgt_obj.get("name"))
        tgt_attack_id = get_attack_id(tgt_obj) if isinstance(tgt_obj, dict) else ""

        text_parts = []
        if src_name and rel_type and tgt_name:
            text_parts.append(f"{src_name} {rel_type} {tgt_name}")
        if tgt_attack_id:
            text_parts.append(f"Target ATT&CK ID: {tgt_attack_id}")
        if desc:
            text_parts.append(desc)

        text = " | ".join(text_parts)
        if text:
            rows.append({
                "snippet_id": f"relationship::{obj['id']}",
                "snippet_type": f"relationship::{rel_type}" if rel_type else "relationship",
                "attack_id": tgt_attack_id,
                "source_name": src_name,
                "target_name": tgt_name,
                "text": text,
            })

    # 3) entity description snippets
    valid_entity_types = {"intrusion-set", "campaign", "malware", "tool"}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("type")
        if obj_type not in valid_entity_types:
            continue

        name = clean_text(obj.get("name"))
        desc = clean_text(obj.get("description"))
        aliases = obj.get("aliases", [])
        aliases = [clean_text(a) for a in aliases if clean_text(a)]

        text_parts = [obj_type]
        if name:
            text_parts.append(name)
        if aliases:
            text_parts.append("Aliases: " + ", ".join(aliases[:10]))
        if desc:
            text_parts.append(desc)

        text = " | ".join(text_parts)
        if text:
            rows.append({
                "snippet_id": f"{obj_type}::{obj['id']}",
                "snippet_type": obj_type,
                "attack_id": "",
                "source_name": "",
                "target_name": name,
                "text": text,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["snippet_id"]).copy()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Bundle: {BUNDLE_JSON}")
    print(f"[INFO] Snippets: {len(df)}")
    print(f"[INFO] Output: {OUTPUT_CSV}")
    print("\n[BY TYPE]")
    print(df["snippet_type"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()