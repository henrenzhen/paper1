import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RL_DIR = PROJECT_ROOT / "rl"
CKPT_DIR = BASE_DIR / "checkpoints"

# --- 核心评估函数 ---
def eval_probs(probs_np, y_true_np):
    top1_hits, top5_hits, mrr_sum = 0, 0, 0.0
    n = len(y_true_np)
    for i in range(n):
        order = probs_np[i].argsort()[::-1]
        if order[0] == y_true_np[i]: top1_hits += 1
        if y_true_np[i] in order[:5]: top5_hits += 1
        rank = np.where(order == y_true_np[i])[0][0] + 1
        mrr_sum += 1.0 / rank
    return top1_hits / n, top5_hits / n, mrr_sum / n

# --- 网络结构定义 (用于加载权重) ---
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
    def forward(self, h_llm): return self.net(h_llm)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 引擎启动 | 计算设备: {device.type.upper()}")

    # 1. 加载数据
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df   = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df  = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    
    ckpt = torch.load(RL_DIR / "rl_baseline_v2.pt", map_location=device)
    t2id, max_len, num_labels = ckpt["token2id"], ckpt["max_len"], ckpt["num_labels"]
    l2id = ckpt["label2id"]
    y_val = np.array([l2id[l] for l in val_df["true_label"]])
    y_te  = np.array([l2id[l] for l in test_df["true_label"]])

    # 2. 提取 RL 的概率分布 (Softmax)
    print("[INFO] 提取 RL 原生概率...")
    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(ckpt["model_state_dict"])
    rl_model.eval()

    def get_rl_probs(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            seq = seq[-max_len:] if len(seq) > max_len else seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)
        with torch.no_grad():
            return F.softmax(rl_model(torch.tensor(seqs, dtype=torch.long).to(device)), dim=1).cpu().numpy()

    P_rl_val = get_rl_probs(val_df)
    P_rl_te  = get_rl_probs(test_df)
    rl_t1, rl_t5, rl_mrr = eval_probs(P_rl_te, y_te)

    # 3. 提取 LLM 的概率分布 (加载之前跑出来的 LLM Baseline 权重)
    print("[INFO] 提取 LLM 独立判断概率...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()
    
    def get_llm_probs(df):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i+256], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        h_llm = torch.FloatTensor(np.vstack(feats)).to(device)
        
        # 必须确保你之前跑出过这个文件！如果没有，这里会报错
        llm_net = LLMOnlyNet(768, num_labels).to(device)
        llm_net.load_state_dict(torch.load(CKPT_DIR / "2._LLM_Only_(BGE_Probing)_best.pt", map_location=device))
        llm_net.eval()
        with torch.no_grad():
            return F.softmax(llm_net(h_llm), dim=1).cpu().numpy()

    P_llm_val = get_llm_probs(val_df)
    P_llm_te  = get_llm_probs(test_df)

    # ==========================================
    # 🔥 核心决策层融合搜索算法 (纯数学操作，瞬间完成)
    # ==========================================
    print("\n[INFO] ================= 开启纯数学决策层 Late Fusion =================\n")
    results = {"1. RL Only (Baseline)": (rl_t1, rl_t5, rl_mrr)}

    # --- Scheme A: 全局 Softmax 软融合 ---
    best_a, best_val_mrr = 0.0, 0.0
    for alpha in np.linspace(0, 1.0, 101):
        P_val = (1 - alpha) * P_rl_val + alpha * P_llm_val
        _, _, v_mrr = eval_probs(P_val, y_val)
        if v_mrr > best_val_mrr:
            best_val_mrr, best_a = v_mrr, alpha
    
    P_te_A = (1 - best_a) * P_rl_te + best_a * P_llm_te
    results[f"2. Global Soft Fusion (a={best_a:.2f})"] = eval_probs(P_te_A, y_te)

    # --- Scheme D (User's 2): 双路召回联合重排 (Dual-Recall) ---
    def dual_recall_rerank(P_rl, P_llm, k_rl, m_llm, alpha):
        P_fused = np.zeros_like(P_rl)
        for i in range(len(P_rl)):
            pool = set(P_rl[i].argsort()[::-1][:k_rl].tolist()) | set(P_llm[i].argsort()[::-1][:m_llm].tolist())
            for c in pool:
                P_fused[i, c] = (1 - alpha) * P_rl[i, c] + alpha * P_llm[i, c]
        return P_fused

    # 简单网格搜索双路召回的最优 alpha
    best_a_dual, best_val_mrr = 0.0, 0.0
    k_rl, m_llm = 10, 3 # 设定 RL 召回 10 个，LLM 提名 3 个
    for alpha in np.linspace(0, 1.0, 51):
        P_val = dual_recall_rerank(P_rl_val, P_llm_val, k_rl, m_llm, alpha)
        _, _, v_mrr = eval_probs(P_val, y_val)
        if v_mrr > best_val_mrr:
            best_val_mrr, best_a_dual = v_mrr, alpha
            
    P_te_D = dual_recall_rerank(P_rl_te, P_llm_te, k_rl, m_llm, best_a_dual)
    results[f"3. Dual-Recall Union (k={k_rl}, m={m_llm}, a={best_a_dual:.2f})"] = eval_probs(P_te_D, y_te)

    # --- Scheme E (User's 3): 自适应双路召回 (Adaptive Dual-Recall) ---
    def adaptive_dual_recall(P_rl, P_llm, theta, k_low, k_high, m_llm, alpha):
        P_fused = np.zeros_like(P_rl)
        for i in range(len(P_rl)):
            sorted_rl = P_rl[i].argsort()[::-1]
            margin = P_rl[i, sorted_rl[0]] - P_rl[i, sorted_rl[1]]
            
            if margin > theta:
                # RL 极其自信，不让 LLM 捣乱，极小候选池
                pool = set(sorted_rl[:k_low].tolist())
                current_alpha = 0.0 # 纯信 RL
            else:
                # RL 摇摆，扩大候选池，接纳 LLM 提名
                pool = set(sorted_rl[:k_high].tolist()) | set(P_llm[i].argsort()[::-1][:m_llm].tolist())
                current_alpha = alpha
                
            for c in pool:
                P_fused[i, c] = (1 - current_alpha) * P_rl[i, c] + current_alpha * P_llm[i, c]
        return P_fused

    # 搜索最佳阈值 theta 和 alpha
    best_theta, best_a_adapt, best_val_mrr = 0.0, 0.0, 0.0
    for theta in [0.1, 0.2, 0.3, 0.5]:
        for alpha in np.linspace(0.1, 1.0, 10):
            P_val = adaptive_dual_recall(P_rl_val, P_llm_val, theta, 3, 15, 5, alpha)
            _, _, v_mrr = eval_probs(P_val, y_val)
            if v_mrr > best_val_mrr:
                best_val_mrr, best_theta, best_a_adapt = v_mrr, theta, alpha

    P_te_E = adaptive_dual_recall(P_rl_te, P_llm_te, best_theta, 3, 15, 5, best_a_adapt)
    results[f"4. Adaptive Dual-Recall (theta={best_theta:.2f}, a={best_a_adapt:.2f})"] = eval_probs(P_te_E, y_te)

    # --- 终极打印 ---
    print("\n" + "="*85)
    print(f"【MITRE 184微观预测 —— 决策层晚期融合 (Late Fusion) 方案终极裁决】")
    print("="*85)
    print(f"{'融合策略 (Fusion Strategy)':<48} | {'Top-1':<8} | {'Top-5':<8} | {'MRR':<8}")
    print("-" * 85)
    for name, (t1, t5, mrr) in results.items():
        print(f"{name:<48} | {t1:<8.4f} | {t5:<8.4f} | {mrr:<8.4f}")
    print("="*85)

if __name__ == "__main__":
    main()