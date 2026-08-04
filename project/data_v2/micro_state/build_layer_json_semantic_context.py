import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 你按实际路径改这里
LAYER_JSON_DIR = Path("/root/project/data_v2/layer_source")
SIM_TEST_CSV = Path("/root/project/data/sim_test_parent_min3.csv")

OUTPUT_CSV = BASE_DIR / "sim_test_parent_min3_semantic_context.csv"
OUTPUT_MAP_CSV = BASE_DIR / "layer_parent_tid_comment_map.csv"


RECENT_K = 3


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def collapse_ws(x: str) -> str:
    return re.sub(r"\s+", " ", clean_text(x)).strip()


def to_parent_tid(tid: str) -> str:
    tid = clean_text(tid).upper()
    m = re.match(r"^(T\d{4})", tid)
    return m.group(1) if m else tid


def split_prefix_ids(x: str):
    x = clean_text(x)
    if not x:
        return []
    # 兼容几种常见分隔格式
    if "||" in x:
        parts = x.split("||")
    elif "," in x:
        parts = x.split(",")
    else:
        parts = x.split()
    return [to_parent_tid(p) for p in parts if clean_text(p)]


def load_layer_comment_map(layer_dir: Path):
    if not layer_dir.exists():
        raise FileNotFoundError(f"Missing layer json dir: {layer_dir}")

    parent_to_comments = defaultdict(list)
    file_count = 0
    tech_rows = 0

    for p in sorted(layer_dir.glob("*.json")):
        file_count += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to parse {p.name}: {e}")
            continue

        techniques = data.get("techniques", [])
        if not isinstance(techniques, list):
            continue

        for t in techniques:
            if not isinstance(t, dict):
                continue

            tid = clean_text(t.get("techniqueID"))
            comment = collapse_ws(t.get("comment"))

            if not tid or not comment:
                continue

            parent_tid = to_parent_tid(tid)
            parent_to_comments[parent_tid].append(comment)
            tech_rows += 1

    print(f"[INFO] Parsed layer files : {file_count}")
    print(f"[INFO] Technique comments : {tech_rows}")
    print(f"[INFO] Parent-tech IDs    : {len(parent_to_comments)}")
    return parent_to_comments


def choose_comment(comments):
    """
    先取最长的 comment 作为默认摘要
    """
    comments = [collapse_ws(c) for c in comments if collapse_ws(c)]
    if not comments:
        return ""
    comments = sorted(set(comments), key=lambda x: (-len(x), x))
    return comments[0]


def build_parent_comment_df(parent_to_comments):
    rows = []
    for parent_tid, comments in sorted(parent_to_comments.items()):
        uniq = sorted(set(collapse_ws(c) for c in comments if collapse_ws(c)))
        chosen = choose_comment(uniq)
        rows.append({
            "parent_tid": parent_tid,
            "num_comments": len(uniq),
            "chosen_comment": chosen,
            "all_comments_joined": " || ".join(uniq),
        })
    return pd.DataFrame(rows)


def main():
    parent_to_comments = load_layer_comment_map(LAYER_JSON_DIR)
    comment_df = build_parent_comment_df(parent_to_comments)
    comment_df.to_csv(OUTPUT_MAP_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved map -> {OUTPUT_MAP_CSV}")

    if not SIM_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing sim test csv: {SIM_TEST_CSV}")

    df = pd.read_csv(SIM_TEST_CSV, encoding="utf-8-sig")

    # 自动识别 prefix 技术列
    candidate_cols = [
        "prefix_technique_ids_parent",
        "state",
        "prefix_ids",
    ]
    prefix_col = None
    for c in candidate_cols:
        if c in df.columns:
            prefix_col = c
            break
    if prefix_col is None:
        raise ValueError(f"Cannot find prefix technique column. Checked: {candidate_cols}")

    rows = []
    covered = 0
    total_recent = 0
    total_hits = 0

    for _, row in df.iterrows():
        prefix_ids = split_prefix_ids(row[prefix_col])
        recent_ids = prefix_ids[-RECENT_K:]
        recent_comments = []

        for tid in recent_ids:
            total_recent += 1
            comment = choose_comment(parent_to_comments.get(tid, []))
            if comment:
                total_hits += 1
            recent_comments.append(comment)

        semantic_parts = []
        for tid, comment in zip(recent_ids, recent_comments):
            if comment:
                semantic_parts.append(f"{tid}: {comment}")
            else:
                semantic_parts.append(f"{tid}:")
        semantic_context_text = " ; ".join(semantic_parts).strip()

        if any(clean_text(c) for c in recent_comments):
            covered += 1

        out = row.to_dict()
        out["recent_prefix_ids"] = " || ".join(recent_ids)
        out["recent_prefix_comments"] = " || ".join(recent_comments)
        out["semantic_context_text"] = semantic_context_text
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    coverage_rate = covered / len(out_df) if len(out_df) else 0.0
    hit_rate = total_hits / total_recent if total_recent else 0.0

    print(f"[INFO] Input rows              : {len(df)}")
    print(f"[INFO] Prefix column used     : {prefix_col}")
    print(f"[INFO] Semantic coverage rows : {covered}")
    print(f"[INFO] Coverage rate          : {coverage_rate:.4f}")
    print(f"[INFO] Recent-id hit rate     : {hit_rate:.4f}")
    print(f"[INFO] Saved output -> {OUTPUT_CSV}")

    preview_cols = [c for c in [
        "sequence_id",
        prefix_col,
        "recent_prefix_ids",
        "recent_prefix_comments",
        "semantic_context_text",
        "next_technique_id_parent",
    ] if c in out_df.columns]

    print("\n[PREVIEW]")
    print(out_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()