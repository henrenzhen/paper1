from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = PROJECT_ROOT / "rl"
OUT_DIR = PROJECT_ROOT / "data_v2" / "micro_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APT29_PATH = RL_DIR / "rl_apt29_predictions_top5.csv"
FIN6_PATH = RL_DIR / "rl_fin6_predictions_top5.csv"
OUT_PATH = OUT_DIR / "micro_state_gold_v2_seed.csv"

APT29_IDS = [
    "apt29::unknown::4",
    "apt29::unknown::5",
    "apt29::unknown::7",
    "apt29::unknown::10",
    "apt29::unknown::14",
    "apt29::unknown::15",
    "apt29::unknown::20",
]

FIN6_IDS = [
    "fin6::unknown::3",
    "fin6::unknown::4",
    "fin6::unknown::8",
    "fin6::unknown::12",
]


def load_and_select(path: Path, ids: list[str], source_org: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["sample_id"].isin(ids)].copy()
    df["source_org"] = source_org
    return df


def make_schema(df: pd.DataFrame, start_idx: int = 30) -> pd.DataFrame:
    out = pd.DataFrame({
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
    return out


def main():
    apt29_df = load_and_select(APT29_PATH, APT29_IDS, "apt29")
    fin6_df = load_and_select(FIN6_PATH, FIN6_IDS, "fin6")

    merged = pd.concat([apt29_df, fin6_df], ignore_index=True)

    # 保持顺序
    desired_order = APT29_IDS + FIN6_IDS
    merged["__ord"] = merged["sample_id"].apply(lambda x: desired_order.index(x))
    merged = merged.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    out = make_schema(merged, start_idx=30)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {OUT_PATH}")
    print(f"[INFO] total={len(out)}, apt29={len(apt29_df)}, fin6={len(fin6_df)}")
    print(out[["annotation_id", "sample_id", "true_label", "true_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()