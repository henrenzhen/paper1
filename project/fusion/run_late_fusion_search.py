import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import random
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RL_DIR = PROJECT_ROOT / "rl"
CKPT_DIR = BASE_DIR / "checkpoints"

# --- 严谨的工具函数 ---
def encode_labels(labels, label2id, split_name):
    missing = sorted(set([l for l in labels if l not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[l] for l in labels], dtype=np.int64)

def eval_metrics(scores_np, y_true_np):
    top1_hits, top5_hits, mrr_sum = 0, 0, 0.0
    n = len(y_true_np)
    for i in range(n):
        order = scores_np[i].argsort()[::-1]
        if order[0] == y_true_np[i]: top1_hits += 1
        if y_true_np[i] in order[:5]: top5_hits += 1
        rank = np.where(order == y_true_np[i])[0][0] + 1
        mrr_sum += 1.0 / rank
    return top1_hits / n, top5_hits / n, mrr_sum / n

def print_coverage_matrix(Z_rl, Z_llm, y_true):
    print("\n" + "="*65)
    print("【论文硬核证据】多维度候选集召回率理论上限 (Coverage)")
    print("="*65)
    n = len(y_true)
    rl_ranks = np.array([np.where(Z_rl[i].argsort()[::-1] == y_true[i])[0][0] + 1 for i in range(n)])
    llm_ranks = np.array([np.where(Z_llm[i].argsort()[::-1] == y_true[i])[0][0] + 1 for i in range(n)])
    
    for k in [1, 3, 5, 10]:
        print(f"真值在 RL  Top-{k:<2}: {(rl_ranks <= k).mean()*100:5.2f}%  |  LLM Top-{k:<2}: {(llm_ranks <= k).mean()*100:5.2f}%")
    print("-" * 65)
    
    # 打印多组联合召回上限
    combinations = [(5, 3), (10, 3), (10, 5), (20, 5)]
    for r_k, l_m in combinations:
        hits = sum([1 for i in range(n) if y_true[i] in set(Z_rl[i].argsort()[::-1][:r_k]) | set(Z_llm[i].argsort()[::-1][:l_m])])
        print(f"🔥 联合提名 (RL Top-{r_k:<2} ∪ LLM Top-{l_m:<2}) 召回率: {hits/n*100:5.2f}%")
    print("="*65)

# --- 网络结构定义 ---
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

# --- 核心网格搜索逻辑 ---
def run_single_seed(seed, device, test_df, val_df, H_llm_val, H_llm_te, ckpt_rl_path, ckpt_llm_path):
    # --- 1. 数据对齐与加载 ---
    ckpt = torch.load(ckpt_rl_path, map_location=device)
    t2id, max_len, num_labels, l2id = ckpt["token2id"], ckpt["max_len"], ckpt["num_labels"], ckpt["label2id"]
    y_val = encode_labels(val_df["true_label"], l2id, "Val")
    y_te  = encode_labels(test_df["true_label"], l2id, "Test")

    # --- 2. 提取 RL 分布 ---
    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(ckpt["model_state_dict"])
    rl_model.eval()

    def get_rl_preds(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            seq = seq[-max_len:] if len(seq) > max_len else seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)
        with torch.no_grad():
            logits = rl_model(torch.tensor(seqs, dtype=torch.long).to(device))
            return logits.cpu().numpy(), F.softmax(logits, dim=1).cpu().numpy()

    Z_rl_val, P_rl_val = get_rl_preds(val_df)
    Z_rl_te,  P_rl_te  = get_rl_preds(test_df)

    # --- 3. 提取 LLM 分布 ---
    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(ckpt_llm_path, map_location=device))
    llm_net.eval()
    with torch.no_grad():
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        P_llm_val = F.softmax(torch.tensor(Z_llm_val), dim=1).numpy()
        Z_llm_te  = llm_net(H_llm_te).cpu().numpy()
        P_llm_te  = F.softmax(torch.tensor(Z_llm_te), dim=1).numpy()

    # 打印 Coverage (仅在第一个 Seed 打印)
    if seed == 42: print_coverage_matrix(Z_rl_te, Z_llm_te, y_te)

    results = {"1. RL Only (Baseline)": eval_metrics(P_rl_te, y_te)}

    # --- A. Global Soft Fusion (Probability) ---
    best_a, best_v_mrr = 0.0, 0.0
    for alpha in np.linspace(0, 1.0, 51):
        P_val = (1 - alpha) * P_rl_val + alpha * P_llm_val
        _, _, v_mrr = eval_metrics(P_val, y_val)
        if v_mrr > best_v_mrr: best_v_mrr, best_a = v_mrr, alpha
    P_te_A = (1 - best_a) * P_rl_te + best_a * P_llm_te
    results[f"2. Global Prob Fusion (a={best_a:.2f})"] = eval_metrics(P_te_A, y_te)

    # --- B. Global Logit Fusion ---
    best_a_z, best_v_mrr_z = 0.0, 0.0
    for alpha in np.linspace(0, 1.0, 51):
        Z_val = (1 - alpha) * Z_rl_val + alpha * Z_llm_val
        _, _, v_mrr = eval_metrics(Z_val, y_val)
        if v_mrr > best_v_mrr_z: best_v_mrr_z, best_a_z = v_mrr, alpha
    Z_te_B = (1 - best_a_z) * Z_rl_te + best_a_z * Z_llm_te
    results[f"3. Global Logit Fusion (a={best_a_z:.2f})"] = eval_metrics(Z_te_B, y_te)

    # --- C. Adaptive Global Fusion (New: 动态路由，不截断候选) ---
    def adaptive_global_fusion(P_rl, P_llm, theta, alpha):
        P_fused = np.copy(P_rl)
        for i in range(len(P_rl)):
            sorted_rl = P_rl[i].argsort()[::-1]
            margin = P_rl[i, sorted_rl[0]] - P_rl[i, sorted_rl[1]]
            if margin <= theta: # 仅在摇摆时启用全局融合
                P_fused[i] = (1 - alpha) * P_rl[i] + alpha * P_llm[i]
        return P_fused

    best_theta_g, best_a_g, best_v_mrr_g = 0.0, 0.0, 0.0
    for theta in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for alpha in np.linspace(0.1, 1.0, 10):
            P_val = adaptive_global_fusion(P_rl_val, P_llm_val, theta, alpha)
            _, _, v_mrr = eval_metrics(P_val, y_val)
            if v_mrr > best_v_mrr_g: best_v_mrr_g, best_theta_g, best_a_g = v_mrr, theta, alpha
    P_te_C = adaptive_global_fusion(P_rl_te, P_llm_te, best_theta_g, best_a_g)
    results[f"4. Adaptive Global Routing (theta={best_theta_g:.2f}, a={best_a_g:.2f})"] = eval_metrics(P_te_C, y_te)

    # --- D. Adaptive Dual-Recall Reranking (Score Reranking) ---
    def adaptive_dual_recall(P_rl, P_llm, theta, k_high, m_llm, alpha):
        S_fused = np.copy(P_rl) # 默认继承 RL 概率分布 (Do no harm)
        for i in range(len(P_rl)):
            sorted_rl = P_rl[i].argsort()[::-1]
            margin = P_rl[i, sorted_rl[0]] - P_rl[i, sorted_rl[1]]
            if margin <= theta:
                # 只有低置信度时，才进入 Score Reranking 通道
                S_fused[i] = 0.0 # 候选外归零 (分数截断)
                pool = set(sorted_rl[:k_high].tolist()) | set(P_llm[i].argsort()[::-1][:m_llm].tolist())
                for c in pool:
                    S_fused[i, c] = (1 - alpha) * P_rl[i, c] + alpha * P_llm[i, c]
        return S_fused

    best_theta_d, best_a_d, best_v_mrr_d = 0.0, 0.0, 0.0
    for theta in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for alpha in np.linspace(0.1, 1.0, 10):
            S_val = adaptive_dual_recall(P_rl_val, P_llm_val, theta, 10, 3, alpha)
            _, _, v_mrr = eval_metrics(S_val, y_val)
            if v_mrr > best_v_mrr_d: best_v_mrr_d, best_theta_d, best_a_d = v_mrr, theta, alpha
    S_te_D = adaptive_dual_recall(P_rl_te, P_llm_te, best_theta_d, 10, 3, best_a_d)
    results[f"5. Adaptive Dual-Recall (theta={best_theta_d:.2f}, a={best_a_d:.2f})"] = eval_metrics(S_te_D, y_te)

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 引擎启动 | 计算设备: {device.type.upper()}")

    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    val_df  = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    
    # === 将 BGE 提取彻底前置，脱离 Seed 循环 ===
    print("[INFO] 一次性预计算 BGE 稠密特征 (脱离多 Seed 循环)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()
    
    def extract_bge(df):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i+256], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.FloatTensor(np.vstack(feats)).to(device)
        
    H_llm_val = extract_bge(val_df)
    H_llm_te  = extract_bge(test_df)
    del embedder # 释放 4090 显存

    # === 兼容多 Seed 的执行入口 ===
    # 假设你当前只有 Seed 42 的权重。未来只需添加 [42, 43, 44, 45, 46] 即可一键跑满所有表格
    SEEDS = [42] 
    all_results = {}

    for seed in SEEDS:
        print(f"\n[INFO] ================= 正在处理 Seed {seed} =================")
        # 这里的命名逻辑假设了你以后多 seed 会按照 _seed42.pt 命名
        # 目前先写死读取你现在的权重文件，如果改名了请自行调整
        ckpt_rl = RL_DIR / "rl_baseline_v2.pt"
        ckpt_llm = CKPT_DIR / "2._LLM_Only_(BGE_Probing)_best.pt" 
        
        res = run_single_seed(seed, device, test_df, val_df, H_llm_val, H_llm_te, ckpt_rl, ckpt_llm)
        for k, v in res.items():
            if k not in all_results: all_results[k] = []
            all_results[k].append(v)

    # === 终极 LaTeX 友好表格打印 ===
    print("\n" + "="*95)
    print(f"【MITRE 184微观预测 —— 严格 Late Fusion 与消融证据对标】")
    print("="*95)
    print(f"{'融合策略 (Fusion Strategy)':<50} | {'Top-1 (Mean±Std)':<14} | {'Top-5 (Mean±Std)':<14} | {'MRR (Mean±Std)':<14}")
    print("-" * 95)
    
    for name, metrics_list in all_results.items():
        t1s, t5s, mrrs = [m[0] for m in metrics_list], [m[1] for m in metrics_list], [m[2] for m in metrics_list]
        t1_str = f"{np.mean(t1s):.4f} ± {np.std(t1s):.4f}" if len(SEEDS)>1 else f"{t1s[0]:.4f}"
        t5_str = f"{np.mean(t5s):.4f} ± {np.std(t5s):.4f}" if len(SEEDS)>1 else f"{t5s[0]:.4f}"
        mrr_str = f"{np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}" if len(SEEDS)>1 else f"{mrrs[0]:.4f}"
        
        print(f"{name:<50} | {t1_str:<14} | {t5_str:<14} | {mrr_str:<14}")
    print("="*95)

if __name__ == "__main__":
    main()