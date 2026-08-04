import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# 模型定义
# =========================================================
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
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# RL 错误样本排名分布分析
# =========================================================
def analyze_rl_error_rank_distribution(y_te, Z_rl_te, Z_global_fused):
    print("\n" + "=" * 90)
    print("📉 开启 RL 错误样本排名分布深度分析 (Rank Distribution of RL-Wrong Samples)")
    print("=" * 90)

    y_te = np.asarray(y_te)

    pred_rl = np.argmax(Z_rl_te, axis=1)
    rl_wrong_mask = (pred_rl != y_te)

    y_wrong = y_te[rl_wrong_mask]
    Z_rl_wrong = Z_rl_te[rl_wrong_mask]
    Z_fused_wrong = Z_global_fused[rl_wrong_mask]

    total_wrong = len(y_wrong)
    print(f"总计 RL 预测错误样本数: {total_wrong} / {len(y_te)}\n")

    if total_wrong == 0:
        print("RL 没有预测错误的样本，分析结束。")
        return None, None, None

    def get_ranks(Z, y):
        ranks = []
        for i in range(len(y)):
            order = np.argsort(Z[i])[::-1]
            rank = np.where(order == y[i])[0][0] + 1
            ranks.append(rank)
        return np.array(ranks)

    rank_rl = get_ranks(Z_rl_wrong, y_wrong)
    rank_fused = get_ranks(Z_fused_wrong, y_wrong)
    rank_gain = rank_rl - rank_fused

    labels = ['Rank 1 (Rescued)', 'Rank 2-3', 'Rank 4-5', 'Rank 6-10', 'Rank 11-50', 'Rank > 50']

    def bucketize(ranks):
        return [
            np.sum(ranks == 1),
            np.sum((ranks >= 2) & (ranks <= 3)),
            np.sum((ranks >= 4) & (ranks <= 5)),
            np.sum((ranks >= 6) & (ranks <= 10)),
            np.sum((ranks >= 11) & (ranks <= 50)),
            np.sum(ranks > 50)
        ]

    counts_rl = bucketize(rank_rl)
    counts_fused = bucketize(rank_fused)

    df_dist = pd.DataFrame({
        "Rank Bucket": labels,
        "RL Count": counts_rl,
        "RL (%)": [f"{c / total_wrong * 100:.1f}%" for c in counts_rl],
        "Fusion Count": counts_fused,
        "Fusion (%)": [f"{c / total_wrong * 100:.1f}%" for c in counts_fused],
        "Shift (Δ Count)": [counts_fused[i] - counts_rl[i] for i in range(len(labels))]
    })

    print("[表 1：RL 错误样本的真实标签排名分布对比]")
    print(df_dist.to_string(index=False))

    rank_improved = np.sum(rank_gain > 0)
    rank_worsened = np.sum(rank_gain < 0)
    rank_same = np.sum(rank_gain == 0)

    avg_rank_rl = np.mean(rank_rl)
    avg_rank_fused = np.mean(rank_fused)

    df_shift = pd.DataFrame([{
        "Samples": total_wrong,
        "Improved (Count)": int(rank_improved),
        "Improved (%)": f"{rank_improved / total_wrong * 100:.1f}%",
        "Unchanged (Count)": int(rank_same),
        "Unchanged (%)": f"{rank_same / total_wrong * 100:.1f}%",
        "Worsened (Count)": int(rank_worsened),
        "Worsened (%)": f"{rank_worsened / total_wrong * 100:.1f}%",
        "Mean Rank Gain": f"{np.mean(rank_gain):.2f}",
        "Median Rank Gain": f"{np.median(rank_gain):.2f}",
        "Improved >= 5 Ranks": f"{np.mean(rank_gain >= 5) * 100:.1f}%",
        "Improved >= 10 Ranks": f"{np.mean(rank_gain >= 10) * 100:.1f}%",
        "Avg Rank (RL)": f"{avg_rank_rl:.2f}",
        "Avg Rank (Fusion)": f"{avg_rank_fused:.2f}",
    }])

    print("\n[表 2：个体样本排名升降统计 (Rank Shift Summary)]")
    print(df_shift.to_string(index=False))

    def group_rank(r):
        if r == 2:
            return "Rank 2"
        elif 3 <= r <= 5:
            return "Rank 3-5"
        elif 6 <= r <= 10:
            return "Rank 6-10"
        elif 11 <= r <= 20:
            return "Rank 11-20"
        else:
            return "Rank > 20"

    df_case = pd.DataFrame({
        "rank_rl": rank_rl,
        "rank_fused": rank_fused,
        "rank_gain": rank_gain
    })
    df_case["RL Error Severity"] = df_case["rank_rl"].apply(group_rank)

    group_order = ["Rank 2", "Rank 3-5", "Rank 6-10", "Rank 11-20", "Rank > 20"]
    rows = []

    for g in group_order:
        sub = df_case[df_case["RL Error Severity"] == g]
        if len(sub) == 0:
            continue

        rows.append({
            "RL Error Severity": g,
            "Samples": len(sub),
            "Fusion Top1 (%)": f"{np.mean(sub['rank_fused'] == 1) * 100:.1f}%",
            "Fusion Top5 (%)": f"{np.mean(sub['rank_fused'] <= 5) * 100:.1f}%",
            "Avg Rank Gain": f"{np.mean(sub['rank_gain']):.2f}",
            "Median Rank Gain": f"{np.median(sub['rank_gain']):.2f}",
            "Improved (%)": f"{np.mean(sub['rank_gain'] > 0) * 100:.1f}%"
        })

    df_group = pd.DataFrame(rows)

    print("\n[表 3：按 RL 原始错误严重程度分层的修复效果]")
    print(df_group.to_string(index=False))

    return df_dist, df_shift, df_group


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 加载数据与模型...")

    # 1. 读取测试集
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    # 2. 加载 RL checkpoint
    ckpt = torch.load(RL_DIR / "rl_baseline_v2_seed42.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]

    # 3. 构造 y_te
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # 4. 计算 RL logits
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()

    seqs = []
    for s in test_df["state"]:
        toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
        toks = toks + [0] * (ckpt["max_len"] - len(toks))
        seqs.append(toks)

    with torch.no_grad():
        Z_rl_te = gru(torch.tensor(seqs, dtype=torch.long, device=device)).cpu().numpy()

    # 5. 计算 LLM features
    print("[INFO] 提取 BGE 特征...")
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()

    txts = test_df["llm_thinking_process"].fillna("None").tolist()
    feats = []

    for i in range(0, len(txts), 64):
        batch = tk(
            txts[i:i+64],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            cls = md(**batch)[0][:, 0]
            cls = F.normalize(cls, p=2, dim=1)
            feats.append(cls.cpu().numpy())

    H_te = np.vstack(feats)

    # 6. 计算 LLM logits
    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / "llm_probe_seed42.pt", map_location=device))
    llm_p.eval()

    with torch.no_grad():
        Z_llm_te = llm_p(torch.tensor(H_te, dtype=torch.float32, device=device)).cpu().numpy()

    # 7. Global Fusion
    alpha = 0.18
    Z_global_fused = (1 - alpha) * Z_rl_te + alpha * Z_llm_te

    # 8. 分析
    df_dist, df_shift, df_group = analyze_rl_error_rank_distribution(
        y_te, Z_rl_te, Z_global_fused
    )

    # 9. 导出
    if df_dist is not None:
        df_dist.to_csv(DATA_DIR / "rl_wrong_rank_distribution.csv", index=False, encoding="utf-8-sig")
        df_shift.to_csv(DATA_DIR / "rl_wrong_rank_shift_summary.csv", index=False, encoding="utf-8-sig")
        df_group.to_csv(DATA_DIR / "rl_wrong_rank_grouped_recovery.csv", index=False, encoding="utf-8-sig")
        print("\n[INFO] 分析完成，CSV 已导出。")