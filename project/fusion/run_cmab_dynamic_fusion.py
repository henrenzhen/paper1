import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from scipy.stats import entropy
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


# =========================
# 1. Hybrid Bandit 核心类
# =========================
class HybridExpertBandit:
    def __init__(self, n_arms, d=770):
        self.n_arms = n_arms
        self.d = d
        self.A = [np.eye(d) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)]

    def get_context(self, bge_feat, gru_logits):
        probs = F.softmax(torch.tensor(gru_logits), dim=-1).numpy()
        conf = np.max(probs)
        ent = entropy(probs)
        return np.hstack([bge_feat, [conf, ent]])

    def select_arm(self, x_t, explore_alpha=0.1):
        x_t = x_t.reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            try:
                A_inv = np.linalg.inv(self.A[a])
                theta_a = A_inv @ self.b[a]
                mean = float(theta_a.T @ x_t)
                cb = explore_alpha * np.sqrt(float(x_t.T @ A_inv @ x_t))
                p[a] = mean + cb
            except:
                p[a] = 0.0
        return np.argmax(p)

    def update(self, arm, x_t, reward):
        x_t = x_t.reshape(-1, 1)
        self.A[arm] += x_t @ x_t.T
        self.b[arm] += reward * x_t


# =========================
# 2. 模型结构定义
# =========================
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

    def forward(self, x): return self.net(x)


def calc_metrics(Z_pred, y_te):
    t1, mrr, y_p = [], [], []
    for i in range(len(y_te)):
        order = Z_pred[i].argsort()[::-1]
        t1.append(1 if order[0] == y_te[i] else 0)
        mrr.append(1.0 / (np.where(order == y_te[i])[0][0] + 1))
        y_p.append(order[0])
    return np.mean(t1), np.mean(mrr), f1_score(y_te, y_p, average='weighted', zero_division=0)


# =========================
# 3. 完整执行逻辑
# =========================
def main(seed=42):
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[🚀] 启动 Hybrid Expert Bandit 最终修正版 (Seed {seed})")

    # 1. 数据加载
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    ckpt = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]
    y_val = np.array([l2id[str(l).strip()] for l in val_df["true_label"]])
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # 2. GRU 推断
    print("[INFO] 正在计算 GRU Logits...")
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()

    def get_gru_logits(df):
        seqs = []
        for s in df["state"]:
            toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
            seqs.append(toks + [0] * (ckpt["max_len"] - len(toks)))
        with torch.no_grad():
            return gru(torch.tensor(seqs, device=device)).cpu().numpy()

    Z_rl_val = get_gru_logits(val_df)
    Z_rl_te = get_gru_logits(test_df)

    # 3. BGE 特征提取
    print("[INFO] 正在提取 BGE 语义特征...")
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()

    def get_bge_feats(df):
        feats = []
        txts = df["llm_thinking_process"].fillna("None").tolist()
        for i in range(0, len(txts), 64):
            batch = tk(txts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = md(**batch)
                emb = F.normalize(out[0][:, 0], p=2, dim=1)
                feats.append(emb.cpu().numpy())
        return np.vstack(feats)

    H_val = get_bge_feats(val_df)
    H_te = get_bge_feats(test_df)

    # 4. LLM Probe 推断
    print("[INFO] 正在加载 LLM Probe...")
    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt", map_location=device))
    llm_p.eval()
    with torch.no_grad():
        Z_llm_val = llm_p(torch.tensor(H_val, device=device)).cpu().numpy()
        Z_llm_te = llm_p(torch.tensor(H_te, device=device)).cpu().numpy()

    # 5. Hybrid Bandit 训练
    policy_alphas = [0.05, 0.18, 0.45]
    bandit = HybridExpertBandit(n_arms=len(policy_alphas))

    print("[INFO] 🎰 Hybrid Bandit 正在进行专家路由学习...")
    for epoch in range(5):
        indices = np.random.permutation(len(y_val))
        for i in indices:
            # 核心修正点：确保使用 Z_rl_val 和 Z_llm_val (带m)
            x_t = bandit.get_context(H_val[i], Z_rl_val[i])
            arm = bandit.select_arm(x_t, explore_alpha=0.2)
            alpha = policy_alphas[arm]
            z_f = (1 - alpha) * Z_rl_val[i] + alpha * Z_llm_val[i]
            reward = 1.0 if np.argmax(z_f) == y_val[i] else 0.0
            bandit.update(arm, x_t, reward)

    # 6. 测试
    print("[INFO] 🚀 执行测试集推理...")
    Z_final, choices = [], []
    for i in range(len(y_te)):
        x_t = bandit.get_context(H_te[i], Z_rl_te[i])
        arm = bandit.select_arm(x_t, explore_alpha=0.0)
        a = policy_alphas[arm]
        choices.append(a)
        Z_final.append((1 - a) * Z_rl_te[i] + a * Z_llm_te[i])

    # 7. 最终战报
    m_gb = calc_metrics((1 - 0.18) * Z_rl_te + 0.18 * Z_llm_te, y_te)
    m_hb = calc_metrics(np.array(Z_final), y_te)

    print("\n" + "=" * 85)
    print(f"🔥 Hybrid Expert Bandit 最终战报 (Seed {seed})")
    print("-" * 85)
    print(f"{'Method':<35} | {'Top-1':<10} | {'MRR':<10} | {'Weighted-F1':<10}")
    print("-" * 85)
    print(f"Global Fusion (alpha=0.18)          | {m_gb[0]:.4f}     | {m_gb[1]:.4f}     | {m_gb[2]:.4f}")
    print(f"👑 Hybrid Expert Bandit             | {m_hb[0]:.4f}     | {m_hb[1]:.4f}     | {m_hb[2]:.4f}")
    print("=" * 85)
    print(f"专家路由分布: {Counter(choices)}")


if __name__ == "__main__":
    main()