import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


SIM_TEST_CSV = Path(r"E:\desktop\project_only\project\data\sim_val_parent_min3.csv")
KG_SNIPPETS_CSV = Path(r"E:\desktop\project_only\project\data\attack_kg_snippets.csv")
OUTPUT_CSV = Path(r"E:\desktop\project_only\project\data\sim_val_parent_min3_kg_context.csv")

RECENT_K = 3
TOP_K = 8
MAX_CHARS_PER_SNIPPET = 260
MAX_TOTAL_CONTEXT_CHARS = 1400

TYPE_PRIORITY = {
    "relationship::uses": 4,
    "attack-pattern": 3,
    "intrusion-set": 2,
    "campaign": 2,
    "tool": 2,
    "malware": 2,
}


def clean_text(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def to_parent_tid(tid: str) -> str:
    tid = clean_text(tid).upper()
    m = re.match(r"^(T\d{4})", tid)
    return m.group(1) if m else tid


def split_prefix_ids(x: str):
    x = clean_text(x)
    if not x:
        return []

    if "||" in x:
        parts = x.split("||")
    elif "," in x:
        parts = x.split(",")
    else:
        parts = x.split()

    return [to_parent_tid(p) for p in parts if clean_text(p)]


def truncate_text(text: str, max_chars: int):
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip() + " ..."


def build_attack_index(kg_df: pd.DataFrame):
    attack_to_rows = defaultdict(list)

    for idx, row in kg_df.iterrows():
        attack_id = to_parent_tid(row.get("attack_id", ""))
        if attack_id:
            attack_to_rows[attack_id].append(idx)

    return attack_to_rows


def score_snippet(row, query_attack_ids, matched_attack_id):
    score = 0.0

    snippet_type = clean_text(row.get("snippet_type", ""))
    score += TYPE_PRIORITY.get(snippet_type, 1)

    if matched_attack_id:
        score += 5.0

    text = clean_text(row.get("text", ""))
    source_name = clean_text(row.get("source_name", ""))
    target_name = clean_text(row.get("target_name", ""))

    # 稍微偏好有 source/target name 的关系片段
    if source_name:
        score += 0.5
    if target_name:
        score += 0.5

    # 偏好中等长度文本，过短信息量小，过长噪声大
    text_len = len(text)
    if 80 <= text_len <= 350:
        score += 1.0
    elif text_len < 40:
        score -= 0.5

    # 如果 query 里多个 technique 都命中片段文本，略加分
    hit_count = 0
    text_upper = text.upper()
    for tid in query_attack_ids:
        if tid in text_upper:
            hit_count += 1
    score += 0.25 * hit_count

    return score


def retrieve_for_sample(recent_ids, kg_df, attack_to_rows, top_k=8):
    candidate_indices = set()

    # 先用 attack_id 精确召回
    for tid in recent_ids:
        for idx in attack_to_rows.get(tid, []):
            candidate_indices.add(idx)

    # 如果精确召回太少，退一步扫 text 包含 technique id 的片段
    if len(candidate_indices) < top_k:
        recent_set = set(recent_ids)
        for idx, row in kg_df.iterrows():
            text_upper = clean_text(row.get("text", "")).upper()
            attack_id = to_parent_tid(row.get("attack_id", ""))

            if attack_id in recent_set:
                candidate_indices.add(idx)
                continue

            for tid in recent_ids:
                if tid and tid in text_upper:
                    candidate_indices.add(idx)
                    break

    scored = []
    for idx in candidate_indices:
        row = kg_df.iloc[idx]
        matched_attack_id = to_parent_tid(row.get("attack_id", ""))
        if matched_attack_id not in recent_ids:
            matched_attack_id = ""
        s = score_snippet(row, recent_ids, matched_attack_id)
        scored.append((idx, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected_rows = []
    used_ids = set()

    for idx, s in scored:
        row = kg_df.iloc[idx]
        snippet_id = clean_text(row.get("snippet_id", ""))
        if not snippet_id or snippet_id in used_ids:
            continue
        used_ids.add(snippet_id)
        selected_rows.append(row)
        if len(selected_rows) >= top_k:
            break

    return selected_rows


def main():
    if not SIM_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing sim test csv: {SIM_TEST_CSV}")
    if not KG_SNIPPETS_CSV.exists():
        raise FileNotFoundError(f"Missing KG snippets csv: {KG_SNIPPETS_CSV}")

    sim_df = pd.read_csv(SIM_TEST_CSV, encoding="utf-8-sig")
    kg_df = pd.read_csv(KG_SNIPPETS_CSV, encoding="utf-8-sig")

    if "prefix_technique_ids_parent" not in sim_df.columns:
        raise ValueError("Missing column: prefix_technique_ids_parent")

    for col in ["snippet_id", "snippet_type", "attack_id", "source_name", "target_name", "text"]:
        if col not in kg_df.columns:
            raise ValueError(f"Missing KG column: {col}")

    kg_df = kg_df.fillna("")
    attack_to_rows = build_attack_index(kg_df)

    out_rows = []
    covered = 0

    total_n = len(sim_df)

    for i, (_, row) in enumerate(sim_df.iterrows(), start=1):
        prefix_ids = split_prefix_ids(row["prefix_technique_ids_parent"])
        recent_ids = prefix_ids[-RECENT_K:]

        retrieved = retrieve_for_sample(recent_ids, kg_df, attack_to_rows, top_k=TOP_K)

        snippet_ids = []
        snippet_types = []
        snippet_texts = []

        total_chars = 0
        for r in retrieved:
            sid = clean_text(r.get("snippet_id", ""))
            stype = clean_text(r.get("snippet_type", ""))
            text = truncate_text(clean_text(r.get("text", "")), MAX_CHARS_PER_SNIPPET)

            if not text:
                continue

            if total_chars + len(text) > MAX_TOTAL_CONTEXT_CHARS:
                break

            snippet_ids.append(sid)
            snippet_types.append(stype)
            snippet_texts.append(text)
            total_chars += len(text)

        kg_context_text = " ; ".join(snippet_texts).strip()

        if kg_context_text:
            covered += 1

        out = row.to_dict()
        out["recent_prefix_ids"] = " || ".join(recent_ids)
        out["retrieved_snippet_ids"] = " || ".join(snippet_ids)
        out["retrieved_snippet_types"] = " || ".join(snippet_types)
        out["kg_context_text"] = kg_context_text
        out_rows.append(out)

        if i % 200 == 0 or i == total_n:
            print(f"[INFO] Processed {i}/{total_n}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    coverage_rate = covered / len(out_df) if len(out_df) else 0.0

    print(f"\n[INFO] Input rows         : {len(sim_df)}")
    print(f"[INFO] Output rows        : {len(out_df)}")
    print(f"[INFO] KG coverage rows   : {covered}")
    print(f"[INFO] Coverage rate      : {coverage_rate:.4f}")
    print(f"[INFO] Saved output       : {OUTPUT_CSV}")

    preview_cols = [
        c for c in [
            "sequence_id",
            "prefix_technique_ids_parent",
            "recent_prefix_ids",
            "retrieved_snippet_ids",
            "retrieved_snippet_types",
            "kg_context_text",
            "next_technique_id_parent",
        ] if c in out_df.columns
    ]

    print("\n[PREVIEW]")
    print(out_df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()