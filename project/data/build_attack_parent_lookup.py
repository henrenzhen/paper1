import re
import pandas as pd

IN_CSV = "/root/project/data/attack_lookup_dedup.csv"
OUT_CSV = "/root/project/data/attack_parent_lookup_for_llm.csv"

PARENT_RE = re.compile(r"^(T\d{4})")

def to_parent_tid(tid: str) -> str:
    tid = str(tid).strip()
    m = PARENT_RE.match(tid)
    return m.group(1) if m else tid

def uniq_join(values):
    vals = []
    seen = set()
    for v in values:
        if pd.isna(v):
            continue
        v = str(v).strip()
        if not v:
            continue
        if v not in seen:
            vals.append(v)
            seen.add(v)
    return " || ".join(vals)

def main():
    df = pd.read_csv(IN_CSV)

    required_cols = ["technique_id", "technique_name"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "tactic" not in df.columns:
        df["tactic"] = ""
    if "tactic_title" not in df.columns:
        df["tactic_title"] = ""

    df["technique_id"] = df["technique_id"].astype(str).str.strip()
    df["technique_name"] = df["technique_name"].astype(str).str.strip()
    df["technique_id_parent"] = df["technique_id"].apply(to_parent_tid)

    # 标记哪些行本身就是父技术行（非子技术）
    df["is_parent_row"] = df["technique_id"] == df["technique_id_parent"]

    rows = []
    grouped = df.groupby("technique_id_parent", sort=True)

    for parent_tid, g in grouped:
        g = g.copy()

        # 1) 优先取父技术本体行名称
        parent_rows = g[g["is_parent_row"]]
        if len(parent_rows) > 0:
            name_parent = parent_rows.iloc[0]["technique_name"]
        else:
            # 2) 回退：取出现频次最高/第一个非空名称
            non_empty_names = [x for x in g["technique_name"].tolist() if str(x).strip()]
            name_parent = non_empty_names[0] if non_empty_names else parent_tid

        tactic_join = uniq_join(g["tactic"].tolist())
        tactic_title_join = uniq_join(g["tactic_title"].tolist())

        # 子技术汇总，便于后续排查
        sub_ids = [
            x for x in g["technique_id"].astype(str).tolist()
            if x != parent_tid
        ]
        sub_names = [
            row["technique_name"]
            for _, row in g.iterrows()
            if str(row["technique_id"]) != parent_tid
        ]

        row = {
            "technique_id_parent": parent_tid,
            "technique_name_parent": name_parent,
            "tactic_ids": uniq_join(sub_ids if not tactic_join else g["tactic"].tolist()),
            "tactic_titles": tactic_title_join,
            "n_rows_grouped": len(g),
            "has_parent_row": int(len(parent_rows) > 0),
            "member_technique_ids": uniq_join(g["technique_id"].tolist()),
            "member_technique_names": uniq_join(g["technique_name"].tolist()),
            "subtechnique_ids": uniq_join(sub_ids),
            "subtechnique_names": uniq_join(sub_names),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # 列顺序整理
    preferred_cols = [
        "technique_id_parent",
        "technique_name_parent",
        "tactic_ids",
        "tactic_titles",
        "n_rows_grouped",
        "has_parent_row",
        "member_technique_ids",
        "member_technique_names",
        "subtechnique_ids",
        "subtechnique_names",
    ]
    out_df = out_df[preferred_cols]

    out_df.to_csv(OUT_CSV, index=False)

    print(f"saved: {OUT_CSV}")
    print(f"rows: {len(out_df)}")
    print(out_df.head(10).to_string())

    # 检查是否都是父技术格式
    bad = out_df[~out_df["technique_id_parent"].astype(str).str.fullmatch(r"T\d{4}")]
    print(f"\nnon-parent-format rows: {len(bad)}")
    if len(bad) > 0:
        print(bad.head(10).to_string())

if __name__ == "__main__":
    main()