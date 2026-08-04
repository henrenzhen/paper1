from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = PROJECT_ROOT / "rl" / "all_ctid_eval"
OUT_DIR = PROJECT_ROOT / "data_v2" / "micro_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("oilrig", RL_DIR / "rl_oilrig_predictions_top5.csv"),
    ("sandworm", RL_DIR / "rl_sandworm_predictions_top5.csv"),
    ("wizard_spider", RL_DIR / "rl_wizard_spider_predictions_top5.csv"),
    ("turla_carbon", RL_DIR / "rl_turla_carbon_predictions_top5.csv"),
]

OUT_PATH = OUT_DIR / "micro_state_gold_v3_seed.csv"


def sample_plan(df: pd.DataFrame, source_org: str, n_total: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["source_org"] = source_org

    # 高难样本
    hard = (
        df.sort_values(["true_rank", "prefix_len"], ascending=[False, False])
          .drop_duplicates("true_label")
          .head(4)
    )

    used = set(hard["sample_id"].tolist())

    # 中等难度
    mid_pool = df[
        (~df["sample_id"].isin(used)) &
        (df["true_rank"] > 5) &
        (df["true_rank"] <= 50)
    ].copy()
    mid = (
        mid_pool.sort_values(["true_rank", "prefix_len"], ascending=[False, False])
               .drop_duplicates("true_label")
               .head(3)
    )
    used.update(mid["sample_id"].tolist())

    # 相对容易样本
    easy_pool = df[
        (~df["sample_id"].isin(used)) &
        (df["true_rank"] <= 5)
    ].copy()
    easy = (
        easy_pool.sort_values(["true_rank", "prefix_len"], ascending=[True, False])
                .drop_duplicates("true_label")
                .head(3)
    )
    used.update(easy["sample_id"].tolist())

    out = pd.concat([hard, mid, easy], ignore_index=True).drop_duplicates("sample_id")

    # 不足时补齐
    if len(out) < n_total:
        remain = df[~df["sample_id"].isin(set(out["sample_id"]))].copy()
        fill = (
            remain.sort_values(["true_rank", "prefix_len"], ascending=[False, False])
                  .drop_duplicates("true_label")
                  .head(n_total - len(out))
        )
        out = pd.concat([out, fill], ignore_index=True).drop_duplicates("sample_id")

    return out.head(n_total).copy()


def make_schema(df: pd.DataFrame, start_idx: int = 41) -> pd.DataFrame:
    return pd.DataFrame({
        "annotation_id": [f"ms_{i:03d}" for i in range(start_idx, start_idx + len(df))],
        "sample_id": df["sample_id"].astype(str),
        "source_org": df["source_org"].astype(str),
        "plan_id": df.get("plan_id", "").astype(str) if "plan_id" in df.columns else "",
        "scenario_id": df.get("scenario_id", "").astype(str) if "scenario_id" in df.columns else "",
        "prefix_len": df["prefix_len"],
        "state": df["state"].astype(str),
        "true_label": df["true_label"].astype(str),
        "top1_pred": df["top1_pred"].astype(str),
        "true_rank": df["true_rank"],
        "current_access_context": "",
        "exhausted_action_constraint": "",
        "next_target_artifact": "",
        "next_micro_verb": "",
        "label_confidence": "",
        "annotation_notes": "",
    })


def main():
    picked = []
    for org, path in TARGETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        part = sample_plan(df, org, n_total=10)
        picked.append(part)

    merged = pd.concat(picked, ignore_index=True)

    # 保持按组织与难度大致有序
    merged["__org_order"] = merged["source_org"].map({
        "oilrig": 0,
        "sandworm": 1,
        "wizard_spider": 2,
        "turla_carbon": 3,
    })
    merged = merged.sort_values(["__org_order", "true_rank"], ascending=[True, False]).drop(columns="__org_order")

    out = make_schema(merged, start_idx=41)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {OUT_PATH}")
    print(f"[INFO] total={len(out)}")
    print(out[["annotation_id", "source_org", "sample_id", "true_label", "true_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()