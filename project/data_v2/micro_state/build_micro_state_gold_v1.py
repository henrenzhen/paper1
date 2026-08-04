from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_V2_ROOT = PROJECT_ROOT / "data_v2"

APT29_HARDEST = PROJECT_ROOT / "rl" / "apt29_analysis" / "hardest_cases_top50.csv"
FIN6_PRED = PROJECT_ROOT / "rl" / "rl_fin6_predictions_top5.csv"

OUT_DIR = DATA_V2_ROOT / "micro_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "micro_state_gold_v1_seed.csv"


def pick_apt29(df: pd.DataFrame, n=20) -> pd.DataFrame:
    df = df.copy()

    g1 = df[df["prefix_len"].between(1, 10)].sort_values("true_rank", ascending=False).head(5)
    g2 = df[df["prefix_len"].between(11, 30)].sort_values("true_rank", ascending=False).head(5)
    g3 = df[df["prefix_len"] >= 31].sort_values("true_rank", ascending=False).head(5)

    used = set(pd.concat([g1, g2, g3])["sample_id"].tolist())
    remain = df[~df["sample_id"].isin(used)].copy()
    g4 = remain.sort_values(["true_label", "true_rank"], ascending=[True, False]).drop_duplicates("true_label").head(5)

    out = pd.concat([g1, g2, g3, g4], ignore_index=True).drop_duplicates("sample_id").head(n)
    out["source_org"] = "apt29"
    return out


def pick_fin6(df: pd.DataFrame, n=10) -> pd.DataFrame:
    df = df.copy()
    df["is_top1_correct"] = (df["top1_pred"].astype(str) == df["true_label"].astype(str)).astype(int)
    df["is_top5_correct"] = (df["true_rank"] <= 5).astype(int)

    hard = df.sort_values("true_rank", ascending=False).head(4)
    mid = df[(df["true_rank"] > 5) & (df["true_rank"] <= 50)].sort_values("true_rank", ascending=False).head(3)
    easy = df[df["is_top5_correct"] == 1].sort_values("true_rank", ascending=True).head(3)

    out = pd.concat([hard, mid, easy], ignore_index=True).drop_duplicates("sample_id").head(n)
    out["source_org"] = "fin6"
    return out


def make_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "annotation_id": [f"ms_{i+1:03d}" for i in range(len(df))],
        "sample_id": df["sample_id"].astype(str),
        "source_org": df["source_org"].astype(str),
        "plan_id": df.get("plan_id", "").astype(str) if "plan_id" in df.columns else "",
        "scenario_id": df.get("scenario_id", "").astype(str) if "scenario_id" in df.columns else "",
        "prefix_len": df["prefix_len"],
        "state": df["state"].astype(str),
        "true_label": df["true_label"].astype(str),
        "top1_pred": df["top1_pred"].astype(str),
        "true_rank": df["true_rank"],

        # 待人工标注字段
        "current_access_context": "",
        "exhausted_action_constraint": "",
        "next_target_artifact": "",
        "next_micro_verb": "",

        # 辅助说明
        "label_confidence": "",
        "annotation_notes": "",
    })
    return out


def main():
    apt29_df = pd.read_csv(APT29_HARDEST, encoding="utf-8-sig")
    fin6_df = pd.read_csv(FIN6_PRED, encoding="utf-8-sig")

    apt29_pick = pick_apt29(apt29_df, n=20)
    fin6_pick = pick_fin6(fin6_df, n=10)

    merged = pd.concat([apt29_pick, fin6_pick], axis=0, ignore_index=True)
    out = make_schema(merged)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {OUT_PATH}")
    print(f"[INFO] total={len(out)}, apt29={len(apt29_pick)}, fin6={len(fin6_pick)}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()