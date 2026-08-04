from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MICRO_DIR = PROJECT_ROOT / "data_v2" / "micro_state"

RL_CSV = MICRO_DIR / "rl_on_aligned_56_predictions_top5.csv"
LLM_CSV = MICRO_DIR / "llm_predicted_micro_state_to_nextttp_details.csv"

OUTPUT_SUMMARY_CSV = MICRO_DIR / "gated_rl_llm_rerank_56_summary.csv"
OUTPUT_BEST_DETAIL_CSV = MICRO_DIR / "gated_rl_llm_rerank_56_best_details.csv"
OUTPUT_TXT = MICRO_DIR / "gated_rl_llm_rerank_56_metrics.txt"


TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BETA_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]


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


def build_rl_only(df):
    rows = []
    for _, row in df.iterrows():
        true_label = row["true_label"]
        top5_labels = split_labels(row["top5_labels"])
        top5_probs = split_probs(row["top5_probs"])

        if true_label in top5_labels:
            rank = top5_labels.index(true_label) + 1
        else:
            # 用 rl true_rank；若 >5 也没问题，指标函数会处理
            rank = int(row["true_rank"]) if pd.notna(row["true_rank"]) else -1

        rows.append({
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": true_label,
            "rank": rank,
            "pred_top1": row["top1_pred"],
            "top5_labels": " || ".join(top5_labels),
            "mode": "rl_only",
        })
    out = pd.DataFrame(rows)
    top1, top5, mrr = compute_metrics_from_ranks(out["rank"])
    return out, {"method": "rl_only", "top1": top1, "top5": top5, "mrr": mrr}


def build_llm_only(df):
    rows = []
    for _, row in df.iterrows():
        rank = int(row["true_rank"]) if pd.notna(row["true_rank"]) else -1
        rows.append({
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": row["true_label"],
            "rank": rank,
            "pred_top1": row["pred_top1"],
            "top5_labels": row["top5_labels"],
            "mode": "llm_only",
        })
    out = pd.DataFrame(rows)
    top1, top5, mrr = compute_metrics_from_ranks(out["rank"])
    return out, {"method": "llm_only", "top1": top1, "top5": top5, "mrr": mrr}


def rerank_one(rl_row, llm_row, tau, beta):
    true_label = rl_row["true_label"]

    rl_labels = split_labels(rl_row["top5_labels"])
    rl_probs = split_probs(rl_row["top5_probs"])
    rl_top1_prob = float(rl_row["top1_prob"])

    llm_labels = split_labels(llm_row["top5_labels"])

    gated = rl_top1_prob < tau

    # 默认直接用 RL
    fused_labels = list(rl_labels)
    fused_scores = list(rl_probs)

    if gated:
        llm_bonus = {}
        for i, lab in enumerate(llm_labels):
            llm_bonus[lab] = 1.0 / (i + 1)

        fused_items = []
        for lab, base_score in zip(rl_labels, rl_probs):
            score = base_score + beta * llm_bonus.get(lab, 0.0)
            fused_items.append((lab, score))

        fused_items.sort(key=lambda x: x[1], reverse=True)
        fused_labels = [x[0] for x in fused_items]
        fused_scores = [x[1] for x in fused_items]

    if true_label in fused_labels:
        rank = fused_labels.index(true_label) + 1
    else:
        # 若真值不在 RL top5 内，重排也不可能救回来
        rank = int(rl_row["true_rank"]) if pd.notna(rl_row["true_rank"]) else -1

    return {
        "annotation_id": rl_row["annotation_id"],
        "source_org": rl_row["source_org"],
        "true_label": true_label,
        "rank": rank,
        "pred_top1": fused_labels[0] if fused_labels else "",
        "top5_labels": " || ".join(fused_labels),
        "gated": int(gated),
        "rl_top1_prob": rl_top1_prob,
    }


def main():
    if not RL_CSV.exists():
        raise FileNotFoundError(f"Missing RL CSV: {RL_CSV}")
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"Missing LLM CSV: {LLM_CSV}")

    rl = pd.read_csv(RL_CSV, encoding="utf-8-sig")
    llm = pd.read_csv(LLM_CSV, encoding="utf-8-sig")

    # 只保留必要列
    rl_cols = [
        "annotation_id", "source_org", "true_label",
        "true_rank", "top1_pred", "top1_prob", "top5_labels", "top5_probs"
    ]
    llm_cols = [
        "annotation_id", "source_org", "true_label",
        "true_rank", "pred_top1", "top5_labels"
    ]
    rl = rl[rl_cols].copy()
    llm = llm[llm_cols].copy()

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

    # 统一成便于访问的列名
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

    # 先算 RL only / LLM only
    rl_eval_df, rl_metrics = build_rl_only(pd.DataFrame({
        "annotation_id": df["annotation_id"],
        "source_org": df["source_org"],
        "true_label": df["true_label"],
        "true_rank": df["rl_true_rank"],
        "top1_pred": df["rl_top1_pred"],
        "top1_prob": df["rl_top1_prob"],
        "top5_labels": df["rl_top5_labels"],
        "top5_probs": df["rl_top5_probs"],
    }))

    llm_eval_df, llm_metrics = build_llm_only(pd.DataFrame({
        "annotation_id": df["annotation_id"],
        "source_org": df["source_org"],
        "true_label": df["true_label"],
        "true_rank": df["llm_true_rank"],
        "pred_top1": df["llm_pred_top1"],
        "top5_labels": df["llm_top5_labels"],
    }))

    summary_rows = [
        {"method": "rl_only", "tau": "", "beta": "", **rl_metrics},
        {"method": "llm_only", "tau": "", "beta": "", **llm_metrics},
    ]

    best_key = None
    best_detail_df = None
    best_metric = -1.0

    for tau in TAU_GRID:
        for beta in BETA_GRID:
            rows = []
            for _, row in df.iterrows():
                rl_row = {
                    "annotation_id": row["annotation_id"],
                    "source_org": row["source_org"],
                    "true_label": row["true_label"],
                    "true_rank": row["rl_true_rank"],
                    "top1_pred": row["rl_top1_pred"],
                    "top1_prob": row["rl_top1_prob"],
                    "top5_labels": row["rl_top5_labels"],
                    "top5_probs": row["rl_top5_probs"],
                }
                llm_row = {
                    "annotation_id": row["annotation_id"],
                    "source_org": row["source_org"],
                    "true_label": row["true_label"],
                    "true_rank": row["llm_true_rank"],
                    "pred_top1": row["llm_pred_top1"],
                    "top5_labels": row["llm_top5_labels"],
                }
                rows.append(rerank_one(rl_row, llm_row, tau, beta))

            detail_df = pd.DataFrame(rows)
            top1, top5, mrr = compute_metrics_from_ranks(detail_df["rank"])

            summary_rows.append({
                "method": "gated_rl_llm_rerank",
                "tau": tau,
                "beta": beta,
                "top1": top1,
                "top5": top5,
                "mrr": mrr,
                "gated_fraction": detail_df["gated"].mean(),
            })

            # 以 Top1 优先，其次 MRR
            key = (top1, mrr)
            if key > best_metric if isinstance(best_metric, tuple) else False:
                pass

            if best_key is None or key > best_key:
                best_key = key
                best_detail_df = detail_df.copy()

    summary_df = pd.DataFrame(summary_rows)

    # 重新找 best row
    gated_df = summary_df[summary_df["method"] == "gated_rl_llm_rerank"].copy()
    best_row = gated_df.sort_values(["top1", "mrr"], ascending=[False, False]).iloc[0]

    best_tau = float(best_row["tau"])
    best_beta = float(best_row["beta"])

    # 生成 best detail
    best_rows = []
    for _, row in df.iterrows():
        rl_row = {
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": row["true_label"],
            "true_rank": row["rl_true_rank"],
            "top1_pred": row["rl_top1_pred"],
            "top1_prob": row["rl_top1_prob"],
            "top5_labels": row["rl_top5_labels"],
            "top5_probs": row["rl_top5_probs"],
        }
        llm_row = {
            "annotation_id": row["annotation_id"],
            "source_org": row["source_org"],
            "true_label": row["true_label"],
            "true_rank": row["llm_true_rank"],
            "pred_top1": row["llm_pred_top1"],
            "top5_labels": row["llm_top5_labels"],
        }
        best_rows.append(rerank_one(rl_row, llm_row, best_tau, best_beta))

    best_detail_df = pd.DataFrame(best_rows)

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    best_detail_df.to_csv(OUTPUT_BEST_DETAIL_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=== GATED RL + LLM RERANK ON 56 ===")

    rl_row = summary_df[summary_df["method"] == "rl_only"].iloc[0]
    lines.append(
        f"RL-only: top1={rl_row['top1']:.4f}, top5={rl_row['top5']:.4f}, mrr={rl_row['mrr']:.4f}"
    )

    llm_row = summary_df[summary_df["method"] == "llm_only"].iloc[0]
    lines.append(
        f"LLM-only: top1={llm_row['top1']:.4f}, top5={llm_row['top5']:.4f}, mrr={llm_row['mrr']:.4f}"
    )

    lines.append(
        f"Best gated fusion: tau={best_tau:.2f}, beta={best_beta:.2f}, "
        f"top1={best_row['top1']:.4f}, top5={best_row['top5']:.4f}, "
        f"mrr={best_row['mrr']:.4f}, gated_fraction={best_row['gated_fraction']:.4f}"
    )

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved -> {OUTPUT_SUMMARY_CSV}")
    print(f"Saved -> {OUTPUT_BEST_DETAIL_CSV}")
    print(f"Saved -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()