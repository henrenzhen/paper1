import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

# =========================
# 0. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
RL_DIR = Path(r"E:\desktop\project_only\project\rl")
LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")


# =========================
# 1. 模型结构定义
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
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 2. 辅助函数
# =========================
def get_mrr(scores_np, y_true):
    mrr_sum = 0.0
    for i in range(len(y_true)):
        order = scores_np[i].argsort()[::-1]
        mrr_sum += 1.0 / (np.where(order == y_true[i])[0][0] + 1)
    return mrr_sum / len(y_true)


def encode_labels(labels, label2id, split_name):
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def calc_metrics(Z_pred, y_te):
    # 【优化】显式指定 dtype，规范内存与计算
    t1 = np.zeros(len(y_te), dtype=np.int32)
    t5 = np.zeros(len(y_te), dtype=np.int32)
    mrr = np.zeros(len(y_te), dtype=np.float32)

    for i in range(len(y_te)):
        tl = y_te[i]
        order = Z_pred[i].argsort()[::-1]
        if order[0] == tl: t1[i] = 1
        if tl in order[:5]: t5[i] = 1
        mrr[i] = 1.0 / (np.where(order == tl)[0][0] + 1)
    return t1.mean(), t5.mean(), mrr.mean()


# =========================
# 3. 核心评估流程
# =========================
def run_eval_on_seed(seed, device):
    print(f"\n[INFO] 正在初始化 Seed {seed} 的 CoT vs Empty 消融评估...")

    # --- A. 加载数据 ---
    val_cot = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_cot = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    val_empty = pd.read_csv(DATA_DIR / "sim_val_llm_empty.csv")
    test_empty = pd.read_csv(DATA_DIR / "sim_test_llm_empty.csv")

    # 【优化】安全检查：确保标签完全一致
    assert (val_cot["true_label"].astype(str).str.strip().values == val_empty["true_label"].astype(
        str).str.strip().values).all(), \
        "CoT 与 Empty 的 val 标签不一致！"
    assert (test_cot["true_label"].astype(str).str.strip().values == test_empty["true_label"].astype(
        str).str.strip().values).all(), \
        "CoT 与 Empty 的 test 标签不一致！"

    # --- B. 加载标签体系与 Baseline ---
    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location=device)
    t2id, max_len, num_labels, l2id = ckpt_rl["token2id"], ckpt_rl["max_len"], ckpt_rl["num_labels"], ckpt_rl[
        "label2id"]

    y_val = encode_labels(val_cot["true_label"], l2id, "Val")
    y_te = encode_labels(test_cot["true_label"], l2id, "Test")
    n = len(y_te)

    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(ckpt_rl["model_state_dict"])
    rl_model.eval()

    def get_rl_preds(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            seq = seq[-max_len:] if len(seq) > max_len else seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)
        with torch.no_grad():
            return rl_model(torch.tensor(seqs, dtype=torch.long, device=device)).cpu().numpy()

    Z_rl_val = get_rl_preds(val_cot)
    Z_rl_te = get_rl_preds(test_cot)

    # --- C. 提取 BGE 特征 ---
    print(f"[INFO] 正在提取 BGE 特征...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def extract_bge(df):
        feats = []
        texts = [str(t) if pd.notna(t) and str(t).strip() != "" else "No reasoning." for t in
                 df["llm_thinking_process"].tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.tensor(np.vstack(feats), dtype=torch.float32, device=device)

    H_cot_val = extract_bge(val_cot)
    H_cot_te = extract_bge(test_cot)
    H_empty_val = extract_bge(val_empty)
    H_empty_te = extract_bge(test_empty)

    del embedder
    torch.cuda.empty_cache()

    # --- D. 加载两个探针并推断 ---
    def load_and_infer_probe(model_suffix, H_val, H_te):
        net = LLMOnlyNet(768, num_labels).to(device)
        model_path = LLM_CKPT_DIR / f"llm_probe_{model_suffix}_seed{seed}.pt" if model_suffix else LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt"
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        with torch.no_grad():
            return net(H_val).cpu().numpy(), net(H_te).cpu().numpy()

    print(f"[INFO] 正在推断 LLM-Only Logits...")
    Z_cot_val, Z_cot_te = load_and_infer_probe("", H_cot_val, H_cot_te)
    Z_empty_val, Z_empty_te = load_and_infer_probe("empty", H_empty_val, H_empty_te)

    # --- E. 独立搜索最优 Alpha 并执行 Fusion ---
    def optimize_and_fuse(Z_llm_val, Z_llm_te):
        best_alpha, best_mrr = 0.0, -1.0
        for alpha in np.linspace(0, 1.0, 51):
            Z_tmp = (1.0 - alpha) * Z_rl_val + alpha * Z_llm_val
            cmrr = get_mrr(Z_tmp, y_val)
            if cmrr > best_mrr:
                best_mrr, best_alpha = cmrr, alpha
        Z_fused_te = (1.0 - best_alpha) * Z_rl_te + best_alpha * Z_llm_te
        return best_alpha, best_mrr, Z_fused_te  # 【优化】一并返回 best_mrr

    alpha_cot, val_mrr_cot, Z_fused_cot = optimize_and_fuse(Z_cot_val, Z_cot_te)
    alpha_empty, val_mrr_empty, Z_fused_empty = optimize_and_fuse(Z_empty_val, Z_empty_te)

    # --- F. 计算指标 ---
    metrics_llm_cot = calc_metrics(Z_cot_te, y_te)
    metrics_llm_empty = calc_metrics(Z_empty_te, y_te)

    metrics_fus_cot = calc_metrics(Z_fused_cot, y_te)
    metrics_fus_empty = calc_metrics(Z_fused_empty, y_te)

    metrics_base = calc_metrics(Z_rl_te, y_te)

    # --- G. 打印二维战报大表 ---
    print(f"\n{'=' * 110}")
    print(f"🔥 LLM Reasoning (CoT vs Empty) 阶段性战报 (Seed {seed}) | 测试样本数: {n}")
    print(
        f"   -> [Baseline GRU] Top-1: {metrics_base[0]:.4f} | Top-5: {metrics_base[1]:.4f} | MRR: {metrics_base[2]:.4f}")
    # 【优化】打印 Val MRR
    print(f"   -> [CoT]   最优 Alpha: {alpha_cot:.2f} (Val MRR: {val_mrr_cot:.4f})")
    print(f"   -> [Empty] 最优 Alpha: {alpha_empty:.2f} (Val MRR: {val_mrr_empty:.4f})")
    print(f"{'=' * 110}")

    header = f"{'Variant':<10} | {'LLM-only Top-1':<15} | {'LLM-only Top-5':<15} | {'LLM-only MRR':<15} || {'Fusion Top-1':<15} | {'Fusion Top-5':<15} | {'Fusion MRR':<15}"
    print(header)
    print("-" * 110)

    def print_row(name, m_llm, m_fus):
        print(
            f"{name:<10} | {m_llm[0]:<15.4f} | {m_llm[1]:<15.4f} | {m_llm[2]:<15.4f} || {m_fus[0]:<15.4f} | {m_fus[1]:<15.4f} | {m_fus[2]:<15.4f}")

    print_row("CoT", metrics_llm_cot, metrics_fus_cot)
    print_row("Empty", metrics_llm_empty, metrics_fus_empty)
    print(f"{'=' * 110}")

    # 【优化】打印极致直观的差值对比
    print("\n[CoT 相对 Empty 的绝对增益 (Δ)]")
    print(f"  -> LLM-only  Top-1 Δ: {metrics_llm_cot[0] - metrics_llm_empty[0]:+.4f}")
    print(f"  -> LLM-only  Top-5 Δ: {metrics_llm_cot[1] - metrics_llm_empty[1]:+.4f}")
    print(f"  -> LLM-only  MRR   Δ: {metrics_llm_cot[2] - metrics_llm_empty[2]:+.4f}")
    print(f"  -> Fusion    Top-1 Δ: {metrics_fus_cot[0] - metrics_fus_empty[0]:+.4f}")
    print(f"  -> Fusion    Top-5 Δ: {metrics_fus_cot[1] - metrics_fus_empty[1]:+.4f}")
    print(f"  -> Fusion    MRR   Δ: {metrics_fus_cot[2] - metrics_fus_empty[2]:+.4f}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_eval_on_seed(42, device)