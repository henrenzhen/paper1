from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MICRO_DIR = PROJECT_ROOT / "data_v2" / "micro_state"

RL_CSV = MICRO_DIR / "rl_on_aligned_56_predictions_top5.csv"
LLM_CSV = MICRO_DIR / "llm_predicted_micro_state_to_nextttp_details.csv"

OUTPUT_SUMMARY_CSV = MICRO_DIR / "dynamic_rl_llm_fusion_56_summary.csv"
OUTPUT_BEST_DETAIL_CSV = MICRO_DIR / "dynamic_rl_llm_fusion_56_best_details.csv"
OUTPUT_TXT = MICRO_DIR / "dynamic_rl_llm_fusion_56_metrics.txt"

# alpha = clip(a * p1 + b * margin + c, 0, 1)
A_GRID = [0.5, 1.0, 1.5]
B_GRID = [0.0, 0.5, 1.0, 1.5]
C_GRID = [-0.2, 0.0, 0.2]

# LLM rank -> score 的缩放
GAMMA_GRID = [0.5, 1.0, 2.0]


def split_labels(x: str):
    x = str(x).strip()
    if not x:
        return []
    return [s.strip() for s in x.split("||") if s.strip()]


def split_probs(x: str):
    x = str(x).strip()
    if not x:
        return []
    return [float(s.strip()) for s in x.split("||") if s.strip()]


def compute_metrics_from_ranks(ranks):
    ranks = list(ranks)
    n = len(ranks)
    top1 = sum(int(r == 1) for r in ranks) / n
    top5 = sum(int(1 <= r <= 5) for r in ranks) / n
    mrr = sum((1.0 / r) if r > 0 else 0.0 for r in ranks) / n
    return top1, top5, mrr


def clip01(x):
    return max(0.0, min(1.0, x))


def alpha_from_rl(p1, p2, a, b, c):
    margin = p1 - p2
    return clip01(a * p1 + b * margin + c)


def rl_candidate_scores(top5_labels, top5_probs):
    return {lab: prob for lab, prob in zip(top5_labels, top5_probs)}


def llm_candidate_scores(top5_labels, gamma):
    # rank-based score
    scores = {}
    for i, lab in enumerate(top5_labels):
        scores[lab] = gamma * (1.0 / (i + 1))
    return scores


def build_rl_only(df):
    rows = []
    for _, row in df.iterrows():
        true_label = row["true_label"]
        top5_labels = split_labels(row["rl_top5_labels"])

        if true_label in top5_labels:
            rank = top5_labels.index(true_label) + 1
        else:
            rank = int(row["rl_true_rank"]) if pd.notna(row["rl_true_rank"]) else -1

        rows.append({
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": true_label,
            "rank": rank,
            "pred_top1": row["rl_top1_pred"],
            "top5_labels": row["rl_top5_labels"],
            "mode": "rl_only",
        })

    out = pd.DataFrame(rows)
    top1, top5, mrr = compute_metrics_from_ranks(out["rank"])
    return out, {"method": "rl_only", "top1": top1, "top5": top5, "mrr": mrr}


def build_llm_only(df):
    rows = []
    for _, row in df.iterrows():
        rank = int(row["llm_true_rank"]) if pd.notna(row["llm_true_rank"]) else -1
        rows.append({
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": row["true_label"],
            "rank": rank,
            "pred_top1": row["llm_pred_top1"],
            "top5_labels": row["llm_top5_labels"],
            "mode": "llm_only",
        })

    out = pd.DataFrame(rows)
    top1, top5, mrr = compute_metrics_from_ranks(out["rank"])
    return out, {"method": "llm_only", "top1": top1, "top5": top5, "mrr": mrr}


def fuse_one(row, a, b, c, gamma):
    true_label = row["true_label"]

    rl_labels = split_labels(row["rl_top5_labels"])
    rl_probs = split_probs(row["rl_top5_probs"])
    llm_labels = split_labels(row["llm_top5_labels"])

    p1 = float(row["rl_top1_prob"])
    p2 = rl_probs[1] if len(rl_probs) > 1 else 0.0
    alpha = alpha_from_rl(p1, p2, a, b, c)

    rl_scores = rl_candidate_scores(rl_labels, rl_probs)
    llm_scores = llm_candidate_scores(llm_labels, gamma)

    candidate_set = list(dict.fromkeys(rl_labels + llm_labels))

    fused_items = []
    for lab in candidate_set:
        s_rl = rl_scores.get(lab, 0.0)
        s_llm = llm_scores.get(lab, 0.0)
        s = alpha * s_rl + (1.0 - alpha) * s_llm
        fused_items.append((lab, s, s_rl, s_llm))

    fused_items.sort(key=lambda x: x[1], reverse=True)
    fused_labels = [x[0] for x in fused_items[:5]]

    if true_label in fused_labels:
        rank = fused_labels.index(true_label) + 1
    else:
        # 如果真值不在前5融合结果里，退回看更长排序
        full_labels = [x[0] for x in fused_items]
        rank = full_labels.index(true_label) + 1 if true_label in full_labels else -1

    return {
        "annotation_id": row["annotation_id"],
        "source_org": row["source_org"],
        "true_label": true_label,
        "rank": rank,
        "pred_top1": fused_labels[0] if fused_labels else "",
        "top5_labels": " || ".join(fused_labels),
        "alpha": alpha,
        "rl_top1_prob": p1,
        "rl_top2_prob": p2,
    }


def main():
    if not RL_CSV.exists():
        raise FileNotFoundError(f"Missing RL CSV: {RL_CSV}")
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"Missing LLM CSV: {LLM_CSV}")

    rl = pd.read_csv(RL_CSV, encoding="utf-8-sig")
    llm = pd.read_csv(LLM_CSV, encoding="utf-8-sig")

    rl = rl[[
        "annotation_id", "source_org", "true_label",
        "true_rank", "top1_pred", "top1_prob", "top5_labels", "top5_probs"
    ]].copy()

    llm = llm[[
        "annotation_id", "source_org", "true_label",
        "true_rank", "pred_top1", "top5_labels"
    ]].copy()

    for c in ["annotation_id", "source_org", "true_label"]:
        rl[c] = rl[c].astype(str).str.strip()
        llm[c] = llm[c].astype(str).str.strip()

    df = rl.merge(
        llm,
        on=["annotation_id", "source_org", "true_label"],
        suffixes=("_rl", "_llm"),
        how="inner",
    )

    if df.empty:
        raise ValueError("No overlapping rows between RL and LLM files.")

    df = df.rename(columns={
        "true_rank_rl": "rl_true_rank",
        "top1_pred": "rl_top1_pred",
        "top1_prob": "rl_top1_prob",
        "top5_labels_rl": "rl_top5_labels",
        "top5_probs": "rl_top5_probs",
        "true_rank_llm": "llm_true_rank",
        "pred_top1": "llm_pred_top1",
        "top5_labels_llm": "llm_top5_labels",
    })

    rl_eval_df, rl_metrics = build_rl_only(df)
    llm_eval_df, llm_metrics = build_llm_only(df)

    summary_rows = [
        {"method": "rl_only", "a": "", "b": "", "c": "", "gamma": "", **rl_metrics},
        {"method": "llm_only", "a": "", "b": "", "c": "", "gamma": "", **llm_metrics},
    ]

    best_tuple = None
    best_params = None
    best_detail_df = None

    for a in A_GRID:
        for b in B_GRID:
            for c in C_GRID:
                for gamma in GAMMA_GRID:
                    rows = []
                    for _, row in df.iterrows():
                        rows.append(fuse_one(row, a, b, c, gamma))

                    detail_df = pd.DataFrame(rows)
                    top1, top5, mrr = compute_metrics_from_ranks(detail_df["rank"])

                    summary_rows.append({
                        "method": "dynamic_fusion",
                        "a": a,
                        "b": b,
                        "c": c,
                        "gamma": gamma,
                        "top1": top1,
                        "top5": top5,
                        "mrr": mrr,
                        "mean_alpha": detail_df["alpha"].mean(),
                    })

                    key = (top1, mrr, top5)
                    if best_tuple is None or key > best_tuple:
                        best_tuple = key
                        best_params = (a, b, c, gamma)
                        best_detail_df = detail_df.copy()

    summary_df = pd.DataFrame(summary_rows)
    best_row = summary_df[
        (summary_df["method"] == "dynamic_fusion") &
        (summary_df["a"] == best_params[0]) &
        (summary_df["b"] == best_params[1]) &
        (summary_df["c"] == best_params[2]) &
        (summary_df["gamma"] == best_params[3])
    ].iloc[0]

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    best_detail_df.to_csv(OUTPUT_BEST_DETAIL_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=== DYNAMIC RL + LLM FUSION ON 56 ===")

    rl_row = summary_df[summary_df["method"] == "rl_only"].iloc[0]
    lines.append(
        f"RL-only: top1={rl_row['top1']:.4f}, top5={rl_row['top5']:.4f}, mrr={rl_row['mrr']:.4f}"
    )

    llm_row = summary_df[summary_df["method"] == "llm_only"].iloc[0]
    lines.append(
        f"LLM-only: top1={llm_row['top1']:.4f}, top5={llm_row['top5']:.4f}, mrr={llm_row['mrr']:.4f}"
    )

    lines.append(
        f"Best dynamic fusion: a={best_params[0]:.2f}, b={best_params[1]:.2f}, "
        f"c={best_params[2]:.2f}, gamma={best_params[3]:.2f}, "
        f"top1={best_row['top1']:.4f}, top5={best_row['top5']:.4f}, "
        f"mrr={best_row['mrr']:.4f}, mean_alpha={best_row['mean_alpha']:.4f}"
    )

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved -> {OUTPUT_SUMMARY_CSV}")
    print(f"Saved -> {OUTPUT_BEST_DETAIL_CSV}")
    print(f"Saved -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()