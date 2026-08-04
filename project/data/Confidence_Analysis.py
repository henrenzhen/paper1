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
# 工具函数：计算指标
# ==========================================
def calc_metrics_for_subset(Z_pred, y_true):
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    t1, t5, mrr = [], [], []
    for i in range(len(y_true)):
        order = Z_pred[i].argsort()[::-1]
        t1.append(1 if order[0] == y_true[i] else 0)
        t5.append(1 if y_true[i] in order[:5] else 0)
        mrr.append(1.0 / (np.where(order == y_true[i])[0][0] + 1))
    return np.mean(t1), np.mean(t5), np.mean(mrr)


# ==========================================
# 核心分析逻辑
# ==========================================
def analyze_margin_stratification(y_te, Z_rl_te, Z_global_fused):
    print("\n" + "=" * 90)
    print("🔬 开启低置信度样本分层分析 (Top1-Top2 Margin Stratification)")
    print("=" * 90)

    # 1. 计算 RL Baseline 的 Top1-Top2 Probability Margin
    probs_rl = F.softmax(torch.tensor(Z_rl_te), dim=-1).numpy()
    sorted_probs = np.sort(probs_rl, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # Top1 概率 - Top2 概率

    # 2. 获取预测 Top-1 标签
    pred_rl = np.argmax(Z_rl_te, axis=1)
    pred_fused = np.argmax(Z_global_fused, axis=1)

    # 3. 按照 Margin 将测试集 5 等分 (Q1 最不确定 -> Q5 最确定)
    try:
        df_analysis = pd.DataFrame({'margin': margin, 'y_true': y_te, 'pred_rl': pred_rl, 'pred_fused': pred_fused})
        df_analysis['bucket'], bins = pd.qcut(df_analysis['margin'], q=5, retbins=True,
                                              labels=['Q1 (Lowest 20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)',
                                                      'Q5 (Highest 20%)'])
    except ValueError:
        # 如果由于 margin 重复值太多导致无法 qcut，则改用等距划分或 rank
        df_analysis['bucket'] = pd.qcut(df_analysis['margin'].rank(method='first'), q=5,
                                        labels=['Q1 (Lowest 20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)',
                                                'Q5 (Highest 20%)'])

    # 4. 逐桶统计指标
    results_perf = []
    results_behavior = []

    for bucket in ['Q1 (Lowest 20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)', 'Q5 (Highest 20%)']:
        mask = df_analysis['bucket'] == bucket
        idx = df_analysis.index[mask].tolist()
        num_samples = len(idx)

        if num_samples == 0: continue

        # 提取当前桶的数据
        z_rl_sub = Z_rl_te[idx]
        z_fused_sub = Z_global_fused[idx]
        y_sub = y_te[idx]

        # 计算基础指标
        rl_t1, rl_t5, rl_mrr = calc_metrics_for_subset(z_rl_sub, y_sub)
        fus_t1, fus_t5, fus_mrr = calc_metrics_for_subset(z_fused_sub, y_sub)

        results_perf.append({
            "RL Margin Bucket": bucket,
            "Samples": num_samples,
            "RL Top1": f"{rl_t1:.4f}",
            "Fusion Top1": f"{fus_t1:.4f}",
            "ΔTop1": f"{fus_t1 - rl_t1:+.4f}",
            "RL MRR": f"{rl_mrr:.4f}",
            "Fusion MRR": f"{fus_mrr:.4f}",
            "ΔMRR": f"{fus_mrr - rl_mrr:+.4f}"
        })

        # 计算行为指标 (Rescue, Preserve, Harm)
        rl_correct = (pred_rl[idx] == y_sub)
        rl_wrong = (pred_rl[idx] != y_sub)
        fus_correct = (pred_fused[idx] == y_sub)
        fus_wrong = (pred_fused[idx] != y_sub)

        rescue_cnt = np.sum(rl_wrong & fus_correct)
        harm_cnt = np.sum(rl_correct & fus_wrong)
        preserve_cnt = np.sum(rl_correct & fus_correct)

        # 计算基于特定基数的 Rate
        rescue_rate = rescue_cnt / max(1, np.sum(rl_wrong)) * 100  # 在 RL 错误样本中的纠错率
        harm_rate = harm_cnt / max(1, np.sum(rl_correct)) * 100  # 在 RL 正确样本中的带偏率

        results_behavior.append({
            "RL Margin Bucket": bucket,
            "RL Wrongs": np.sum(rl_wrong),
            "Rescued (Count)": rescue_cnt,
            "Rescue Rate": f"{rescue_rate:.1f}%",
            "RL Corrects": np.sum(rl_correct),
            "Harmed (Count)": harm_cnt,
            "Harm Rate": f"{harm_rate:.1f}%",
        })

    print("\n[表 1：各置信度区间的宏观性能提升]")
    df_perf = pd.DataFrame(results_perf)
    print(df_perf.to_string(index=False))

    print("\n[表 2：行为纠正率与伤害率深度解析]")
    print("* 注: Rescue Rate = Rescued / RL Wrongs. Harm Rate = Harmed / RL Corrects.")
    df_beh = pd.DataFrame(results_behavior)
    print(df_beh.to_string(index=False))


# ==========================================
# 模型与执行入口 (复用之前环境)
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


if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 加载模型与数据...")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    ckpt = torch.load(RL_DIR / "rl_baseline_v2_seed42.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # RL Logits
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()
    seqs = []
    for s in test_df["state"]:
        toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
        seqs.append(toks + [0] * (ckpt["max_len"] - len(toks)))
    with torch.no_grad():
        Z_rl_te = gru(torch.tensor(seqs, device=device)).cpu().numpy()

    # LLM Logits
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()
    feats = []
    txts = test_df["llm_thinking_process"].fillna("None").tolist()
    for i in range(0, len(txts), 64):
        batch = tk(txts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            feats.append(F.normalize(md(**batch)[0][:, 0], p=2, dim=1).cpu().numpy())
    H_te = np.vstack(feats)

    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / "llm_probe_seed42.pt", map_location=device))
    llm_p.eval()
    with torch.no_grad():
        Z_llm_te = llm_p(torch.tensor(H_te, device=device)).cpu().numpy()

    # Global Fusion
    Z_global_fused = (1 - 0.18) * Z_rl_te + 0.18 * Z_llm_te

    # 运行分层分析
    analyze_margin_stratification(y_te, Z_rl_te, Z_global_fused)