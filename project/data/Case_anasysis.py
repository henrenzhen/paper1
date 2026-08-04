import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. 案例挖掘核心逻辑
# ==========================================
def get_rank(sorted_idx_row, true_lbl):
    return int(np.where(sorted_idx_row == true_lbl)[0][0]) + 1


def format_topk(pred_row, logit_row, id2l, k=3):
    return [f"{id2l[p]} ({logit_row[p]:.2f})" for p in pred_row[:k]]


def mine_golden_cases(df_test, y_te, Z_rl_te, Z_llm_te, Z_feat_fused, Z_global_fused, id2l, out_dir=".",
                      max_thinking_chars=500):
    print("\n" + "=" * 80)
    print("🎯 开启终极黄金案例挖掘 (含 Feature Fusion 对比)")
    print("=" * 80)

    pred_rl = np.argsort(Z_rl_te, axis=1)[:, ::-1]
    pred_llm = np.argsort(Z_llm_te, axis=1)[:, ::-1]
    pred_feat = np.argsort(Z_feat_fused, axis=1)[:, ::-1]
    pred_global = np.argsort(Z_global_fused, axis=1)[:, ::-1]

    records_cat1, records_cat2, records_cat3 = [], [], []

    for i in range(len(y_te)):
        true_lbl = y_te[i]

        rl_top1, llm_top1, feat_top1, global_top1 = pred_rl[i][0], pred_llm[i][0], pred_feat[i][0], pred_global[i][0]

        true_rank_rl = get_rank(pred_rl[i], true_lbl)
        true_rank_llm = get_rank(pred_llm[i], true_lbl)
        true_rank_feat = get_rank(pred_feat[i], true_lbl)
        true_rank_global = get_rank(pred_global[i], true_lbl)

        record = {
            "sample_idx": i,
            "true_label": id2l[true_lbl],
            "prefix_state": df_test["state"].iloc[i],
            "rank_RL": true_rank_rl,
            "rank_LLM": true_rank_llm,
            "rank_FeatFusion": true_rank_feat,
            "rank_GlobalFusion": true_rank_global,
            "logit_RL": float(Z_rl_te[i][true_lbl]),
            "logit_LLM": float(Z_llm_te[i][true_lbl]),
            "logit_FeatFusion": float(Z_feat_fused[i][true_lbl]),
            "logit_GlobalFusion": float(Z_global_fused[i][true_lbl]),
            "RL_top3": " | ".join(format_topk(pred_rl[i], Z_rl_te[i], id2l, k=3)),
            "LLM_top3": " | ".join(format_topk(pred_llm[i], Z_llm_te[i], id2l, k=3)),
            "FeatFusion_top3": " | ".join(format_topk(pred_feat[i], Z_feat_fused[i], id2l, k=3)),
            "GlobalFusion_top3": " | ".join(format_topk(pred_global[i], Z_global_fused[i], id2l, k=3)),
            "llm_thinking_process": str(df_test["llm_thinking_process"].iloc[i])[:max_thinking_chars]
        }

        # 类别 1: 绝杀救场 (RL错，Global对)
        if rl_top1 != true_lbl and global_top1 == true_lbl:
            rec = record.copy()
            rec["category"] = "cat1_rl_wrong_global_correct"
            rec["case_score"] = true_rank_rl - true_rank_global
            records_cat1.append(rec)

        # 类别 2: 鲁棒性证明 (RL对，LLM错，Global保持对)
        if rl_top1 == true_lbl and global_top1 == true_lbl and llm_top1 != true_lbl:
            rec = record.copy()
            rec["category"] = "cat2_rl_correct_global_stays_correct"
            rec["case_score"] = float(Z_llm_te[i][llm_top1] - Z_llm_te[i][true_lbl])
            records_cat2.append(rec)

        # 类别 3: 真正的接口鸿沟 (LLM对，Feat败，Global成)
        if (true_rank_llm <= 3) and (feat_top1 != true_lbl) and (global_top1 == true_lbl):
            rec = record.copy()
            rec["category"] = "cat3_interface_gap_proof"
            rec["case_score"] = true_rank_feat - true_rank_global
            records_cat3.append(rec)

    df_cat1 = pd.DataFrame(records_cat1).sort_values("case_score", ascending=False)
    df_cat2 = pd.DataFrame(records_cat2).sort_values("case_score", ascending=False)
    df_cat3 = pd.DataFrame(records_cat3).sort_values("case_score", ascending=False)

    print(f"\n[Cat 1] RL错, GlobalFusion拉正: {len(df_cat1)} 条")
    print(f"[Cat 2] LLM带偏, GlobalFusion扛住: {len(df_cat2)} 条")
    print(f"[Cat 3] 证实接口鸿沟 (Feat败, Global成): {len(df_cat3)} 条")

    out_path = Path(out_dir)
    df_cat1.to_csv(out_path / "golden_cases_cat1_rescue.csv", index=False, encoding="utf-8-sig")
    df_cat2.to_csv(out_path / "golden_cases_cat2_robust.csv", index=False, encoding="utf-8-sig")
    df_cat3.to_csv(out_path / "golden_cases_cat3_interface_gap.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ 案例已导出至 {out_dir} 目录下。快去打开 CSV 挑选天选样本吧！")


# ==========================================
# 2. 模型定义 (为了提取 Logits)
# ==========================================
class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.classifier(h.squeeze(0))


class LLMOnlyNet(nn.Module):
    def __init__(self, llm_dim=768, num_labels=184):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim), nn.Linear(llm_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def forward(self, x): return self.net(x)


# ==========================================
# 3. 主程序执行入口
# ==========================================
if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 正在加载测试集与映射表...")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    ckpt = torch.load(RL_DIR / "rl_baseline_v2_seed42.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]
    id2l = {v: k for k, v in l2id.items()}
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # --- 获取 RL Logits ---
    print("[INFO] 计算 GRU Logits...")
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()
    seqs = []
    for s in test_df["state"]:
        toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
        seqs.append(toks + [0] * (ckpt["max_len"] - len(toks)))
    with torch.no_grad():
        Z_rl_te = gru(torch.tensor(seqs, device=device)).cpu().numpy()

    # --- 获取 BGE & LLM Logits ---
    print("[INFO] 提取 BGE 并计算 LLM Probe Logits...")
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()
    feats = []
    txts = test_df["llm_thinking_process"].fillna("None").tolist()
    for i in range(0, len(txts), 64):
        batch = tk(txts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = md(**batch)
            feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
    H_te = np.vstack(feats)

    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / "llm_probe_seed42.pt", map_location=device))
    llm_p.eval()
    with torch.no_grad():
        Z_llm_te = llm_p(torch.tensor(H_te, device=device)).cpu().numpy()

    # --- 获取 Global Fusion Logits ---
    Z_global_fused = (1 - 0.18) * Z_rl_te + 0.18 * Z_llm_te

    # --- 获取 Feature Fusion Logits ---
    # 【⚠️ 极其重要】你需要把之前做 Feature Fusion 实验的预测结果加进来。
    # 这里我用一个占位符模拟。如果你之前把 feature fusion 的 logits 存成了 npy，请替换这一行：
    # Z_feat_fused = np.load("E:/.../feature_fusion_logits.npy")
    print("[WRN] 当前 Z_feat_fused 使用的是占位逻辑！请确保传入真实的特征融合 Logits 以分析第三类案例。")
    Z_feat_fused = Z_rl_te.copy()  # 占位

    # 开始执行挖掘
    mine_golden_cases(test_df, y_te, Z_rl_te, Z_llm_te, Z_feat_fused, Z_global_fused, id2l, out_dir=str(DATA_DIR))