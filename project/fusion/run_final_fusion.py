import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# =========================
# 0. 路径自动寻址 (请根据你的实际文件夹调整)
# =========================
BASE_DIR = Path(__file__).resolve().parent
# 假设你在 project/fusion 下运行，这里自动推导 project/ 根目录
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "fusion" else BASE_DIR

DATA_DIR = PROJECT_ROOT / "data"
RL_DIR = PROJECT_ROOT / "rl"
LLM_CKPT_DIR = PROJECT_ROOT / "llm" / "checkpoints"


# =========================
# 1. 网络结构定义 (用于加载权重)
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
            nn.LayerNorm(llm_dim), nn.Linear(llm_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def forward(self, x): return self.net(x)


# =========================
# 2. 严谨评估体系 (加入 F1)
# =========================
def encode_labels(labels, label2id):
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def eval_metrics(scores_np, y_true_np):
    top1, top5, mrr_sum = 0, 0, 0.0
    n = len(y_true_np)
    y_pred = []

    for i in range(n):
        order = scores_np[i].argsort()[::-1]
        y_pred.append(order[0])
        if order[0] == y_true_np[i]: top1 += 1
        if y_true_np[i] in order[:5]: top5 += 1
        mrr_sum += 1.0 / (np.where(order == y_true_np[i])[0][0] + 1)

    mac_f1 = f1_score(y_true_np, y_pred, average='macro', zero_division=0)
    wei_f1 = f1_score(y_true_np, y_pred, average='weighted', zero_division=0)
    return top1 / n, top5 / n, mrr_sum / n, mac_f1, wei_f1


def print_coverage_matrix(Z_rl, Z_llm, y_true):
    print("\n" + "=" * 75)
    print("【论文硬核证据】多维度候选集召回率与 LLM 独立增量分析 (基于 Seed 42)")
    print("=" * 75)
    n = len(y_true)
    rl_ranks = np.array([np.where(Z_rl[i].argsort()[::-1] == y_true[i])[0][0] + 1 for i in range(n)])
    llm_ranks = np.array([np.where(Z_llm[i].argsort()[::-1] == y_true[i])[0][0] + 1 for i in range(n)])

    for k in [1, 3, 5, 10, 20]:
        print(
            f"真值在 RL  Top-{k:<2}: {(rl_ranks <= k).mean() * 100:5.2f}%  |  LLM Top-{k:<2}: {(llm_ranks <= k).mean() * 100:5.2f}%")
    print("-" * 75)

    for r_k, l_m in [(5, 3), (10, 3), (10, 5), (20, 5)]:
        base_cov = (rl_ranks <= r_k).mean() * 100
        hits = sum([1 for i in range(n) if
                    y_true[i] in set(Z_rl[i].argsort()[::-1][:r_k]) | set(Z_llm[i].argsort()[::-1][:l_m])])
        union_cov = hits / n * 100
        print(
            f"🔥 联合提名 (RL Top-{r_k:<2} ∪ LLM Top-{l_m:<2}) 召回率: {union_cov:5.2f}% (增量: +{union_cov - base_cov:4.2f} pts)")
    print("=" * 75)


# =========================
# 3. 核心融合执行流
# =========================
def run_single_seed(seed, device, test_df, val_df, H_llm_val, H_llm_te, ckpt_rl_path, ckpt_llm_path):
    ckpt = torch.load(ckpt_rl_path, map_location=device)
    t2id, max_len, num_labels, l2id = ckpt["token2id"], ckpt["max_len"], ckpt["num_labels"], ckpt["label2id"]
    y_val = encode_labels(val_df["true_label"], l2id)
    y_te = encode_labels(test_df["true_label"], l2id)

    # 提取 RL
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
    Z_rl_te, P_rl_te = get_rl_preds(test_df)

    # 提取 LLM
    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(ckpt_llm_path, map_location=device))
    llm_net.eval()
    with torch.no_grad():
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        P_llm_val = F.softmax(torch.tensor(Z_llm_val), dim=1).numpy()
        Z_llm_te = llm_net(H_llm_te).cpu().numpy()
        P_llm_te = F.softmax(torch.tensor(Z_llm_te), dim=1).numpy()

    if seed == 42: print_coverage_matrix(Z_rl_te, Z_llm_te, y_te)

    results = {"1. RL Only (Baseline)": eval_metrics(Z_rl_te, y_te)}

    # A. Global Prob Fusion
    best_a, best_v_mrr = 0.0, 0.0
    for alpha in np.linspace(0, 1.0, 51):
        P_val = (1 - alpha) * P_rl_val + alpha * P_llm_val
        if eval_metrics(P_val, y_val)[2] > best_v_mrr: best_v_mrr, best_a = eval_metrics(P_val, y_val)[2], alpha
    results[f"2. Global Prob Fusion (a={best_a:.2f})"] = eval_metrics((1 - best_a) * P_rl_te + best_a * P_llm_te, y_te)

    # B. Global Logit Fusion
    best_a_z, best_v_mrr_z = 0.0, 0.0
    for alpha in np.linspace(0, 1.0, 51):
        Z_val = (1 - alpha) * Z_rl_val + alpha * Z_llm_val
        if eval_metrics(Z_val, y_val)[2] > best_v_mrr_z: best_v_mrr_z, best_a_z = eval_metrics(Z_val, y_val)[2], alpha
    results[f"3. Global Logit Fusion (a={best_a_z:.2f})"] = eval_metrics((1 - best_a_z) * Z_rl_te + best_a_z * Z_llm_te,
                                                                         y_te)

    # C. Adaptive Dual-Recall (Score Reranking)
    def adaptive_dual_recall(P_rl, P_llm, theta, k_high, m_llm, alpha):
        S_fused = np.copy(P_rl)
        for i in range(len(P_rl)):
            sorted_rl = P_rl[i].argsort()[::-1]
            if (P_rl[i, sorted_rl[0]] - P_rl[i, sorted_rl[1]]) <= theta:
                S_fused[i] = 0.0
                for c in (set(sorted_rl[:k_high].tolist()) | set(P_llm[i].argsort()[::-1][:m_llm].tolist())):
                    S_fused[i, c] = (1 - alpha) * P_rl[i, c] + alpha * P_llm[i, c]
        return S_fused

    best_theta_d, best_a_d, best_v_mrr_d = 0.0, 0.0, 0.0
    for theta in [0.01, 0.05, 0.10, 0.15, 0.20]:
        for alpha in np.linspace(0.1, 1.0, 10):
            if eval_metrics(adaptive_dual_recall(P_rl_val, P_llm_val, theta, 10, 3, alpha), y_val)[2] > best_v_mrr_d:
                best_v_mrr_d, best_theta_d, best_a_d = \
                eval_metrics(adaptive_dual_recall(P_rl_val, P_llm_val, theta, 10, 3, alpha), y_val)[2], theta, alpha
    results[f"4. Adaptive Dual-Recall (theta={best_theta_d:.2f}, a={best_a_d:.2f})"] = eval_metrics(
        adaptive_dual_recall(P_rl_te, P_llm_te, best_theta_d, 10, 3, best_a_d), y_te)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 引擎启动 | 计算设备: {device.type.upper()}")

    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")

    print("[INFO] 一次性提取 BGE 特征...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def extract_bge(df):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                device)
            with torch.no_grad(): feats.append(F.normalize(embedder(**inputs)[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.FloatTensor(np.vstack(feats)).to(device)

    H_llm_val = extract_bge(val_df)
    H_llm_te = extract_bge(test_df)
    del embedder
    torch.cuda.empty_cache()

    SEEDS = [42, 43, 44, 45, 46]
    all_results = {}

    for seed in SEEDS:
        print(f"\n[INFO] 正在联合推理 Seed {seed} ...")
        # 确保路径与你刚才保存的对应
        ckpt_rl = RL_DIR / f"rl_baseline_v2_seed{seed}.pt"
        ckpt_llm = LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt"

        res = run_single_seed(seed, device, test_df, val_df, H_llm_val, H_llm_te, ckpt_rl, ckpt_llm)
        for k, v in res.items():
            # 统一提取方法名前缀（去掉括号里的参数），防止合并时出错
            base_name = k.split("(")[0].strip()
            if base_name not in all_results: all_results[base_name] = []
            all_results[base_name].append(v)

    # === 终极 LaTeX 友好表格打印 ===
    print("\n" + "=" * 125)
    print(f"【MITRE 184微观预测 —— 多种子 (N=5) 终极评估报告】")
    print("=" * 125)
    print(
        f"{'融合策略 (Fusion Strategy)':<32} | {'Top-1 (M±S)':<15} | {'Top-5 (M±S)':<15} | {'MRR (M±S)':<15} | {'Mac-F1 (M±S)':<15} | {'Wei-F1 (M±S)':<15}")
    print("-" * 125)

    for name, metrics_list in all_results.items():
        t1s, t5s, mrrs, macs, weis = zip(*metrics_list)

        def fmt(lst): return f"{np.mean(lst):.4f}±{np.std(lst):.4f}"

        print(f"{name:<32} | {fmt(t1s):<15} | {fmt(t5s):<15} | {fmt(mrrs):<15} | {fmt(macs):<15} | {fmt(weis):<15}")
    print("=" * 125)


if __name__ == "__main__":
    main()