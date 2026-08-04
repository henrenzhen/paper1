import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
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
# 2. 辅助函数 (恢复严格安全检查)
# =========================
def encode_labels(labels, label2id, split_name):
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def get_mrr(scores_np, y_true):
    mrr_sum = 0.0
    for i in range(len(y_true)):
        order = scores_np[i].argsort()[::-1]
        mrr_sum += 1.0 / (np.where(order == y_true[i])[0][0] + 1)
    return mrr_sum / len(y_true)


def calc_metrics(Z_pred, y_te):
    t1 = np.zeros(len(y_te), dtype=np.int32)
    t5 = np.zeros(len(y_te), dtype=np.int32)
    mrr = np.zeros(len(y_te), dtype=np.float32)
    y_pred_top1 = []

    for i in range(len(y_te)):
        tl = y_te[i]
        order = Z_pred[i].argsort()[::-1]
        top1 = order[0]
        y_pred_top1.append(top1)

        if top1 == tl: t1[i] = 1
        if tl in order[:5]: t5[i] = 1
        mrr[i] = 1.0 / (np.where(order == tl)[0][0] + 1)

    macro_f1 = f1_score(y_te, y_pred_top1, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred_top1, average='weighted', zero_division=0)

    return t1.mean(), t5.mean(), mrr.mean(), macro_f1, weighted_f1


# =========================
# 3. 核心评估流程
# =========================
def run_eval(seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在补齐 GRU 3-Way Ablation 的 F1 指标 (Seed {seed}) ...")

    val_cot = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_cot = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    val_nocot = pd.read_csv(DATA_DIR / "sim_val_llm_no_cot.csv")
    test_nocot = pd.read_csv(DATA_DIR / "sim_test_llm_no_cot.csv")
    val_empty = pd.read_csv(DATA_DIR / "sim_val_llm_empty.csv")
    test_empty = pd.read_csv(DATA_DIR / "sim_test_llm_empty.csv")

    # --- 核心安全检查区 ---
    # 1. 检查 Label 对齐
    assert (val_cot["true_label"].astype(str).str.strip().values == val_nocot["true_label"].astype(
        str).str.strip().values).all(), "Val 标签不对齐 (CoT vs No-CoT)"
    assert (val_cot["true_label"].astype(str).str.strip().values == val_empty["true_label"].astype(
        str).str.strip().values).all(), "Val 标签不对齐 (CoT vs Empty)"
    assert (test_cot["true_label"].astype(str).str.strip().values == test_nocot["true_label"].astype(
        str).str.strip().values).all(), "Test 标签不对齐 (CoT vs No-CoT)"
    assert (test_cot["true_label"].astype(str).str.strip().values == test_empty["true_label"].astype(
        str).str.strip().values).all(), "Test 标签不对齐 (CoT vs Empty)"

    # 2. 检查 Sequence ID 对齐
    if "sequence_id" in val_cot.columns:
        assert (val_cot["sequence_id"].astype(str).values == val_nocot["sequence_id"].astype(
            str).values).all(), "Val Sequence ID 不对齐 (CoT vs No-CoT)"
        assert (val_cot["sequence_id"].astype(str).values == val_empty["sequence_id"].astype(
            str).values).all(), "Val Sequence ID 不对齐 (CoT vs Empty)"
        assert (test_cot["sequence_id"].astype(str).values == test_nocot["sequence_id"].astype(
            str).values).all(), "Test Sequence ID 不对齐 (CoT vs No-CoT)"
        assert (test_cot["sequence_id"].astype(str).values == test_empty["sequence_id"].astype(
            str).values).all(), "Test Sequence ID 不对齐 (CoT vs Empty)"
        print("[INFO] Sequence ID & Label 严格对齐检查全部通过！")

    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location="cpu")
    t2id, max_len, num_labels, l2id = ckpt_rl["token2id"], ckpt_rl["max_len"], ckpt_rl["num_labels"], ckpt_rl[
        "label2id"]

    y_val = encode_labels(val_cot["true_label"], l2id, "Val")
    y_te = encode_labels(test_cot["true_label"], l2id, "Test")

    # --- 推断 GRU ---
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

    # --- 提取 BGE ---
    print(f"[INFO] 正在提取 BGE 特征 (约需1分钟)...")
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

    H_cot_val, H_cot_te = extract_bge(val_cot), extract_bge(test_cot)
    H_nocot_val, H_nocot_te = extract_bge(val_nocot), extract_bge(test_nocot)
    H_empty_val, H_empty_te = extract_bge(val_empty), extract_bge(test_empty)
    del embedder;
    torch.cuda.empty_cache()

    # --- 推断 LLM Probe ---
    def load_and_infer_probe(model_suffix, H_val, H_te):
        net = LLMOnlyNet(768, num_labels).to(device)
        model_path = LLM_CKPT_DIR / f"llm_probe_{model_suffix}_seed{seed}.pt" if model_suffix else LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt"
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        with torch.no_grad():
            return net(H_val).cpu().numpy(), net(H_te).cpu().numpy()

    Z_cot_val, Z_cot_te = load_and_infer_probe("", H_cot_val, H_cot_te)
    Z_nocot_val, Z_nocot_te = load_and_infer_probe("no_cot", H_nocot_val, H_nocot_te)
    Z_empty_val, Z_empty_te = load_and_infer_probe("empty", H_empty_val, H_empty_te)

    # --- 搜索 Alpha & 融合 ---
    def optimize_and_fuse(Z_llm_val, Z_llm_te):
        best_alpha, best_mrr = 0.0, -1.0
        for alpha in np.linspace(0, 1.0, 51):
            Z_tmp = (1.0 - alpha) * Z_rl_val + alpha * Z_llm_val
            cmrr = get_mrr(Z_tmp, y_val)
            if cmrr > best_mrr:
                best_mrr, best_alpha = cmrr, alpha
        Z_fused_te = (1.0 - best_alpha) * Z_rl_te + best_alpha * Z_llm_te
        return best_alpha, Z_fused_te

    alpha_cot, Z_fused_cot = optimize_and_fuse(Z_cot_val, Z_cot_te)
    alpha_nocot, Z_fused_nocot = optimize_and_fuse(Z_nocot_val, Z_nocot_te)
    alpha_empty, Z_fused_empty = optimize_and_fuse(Z_empty_val, Z_empty_te)

    # --- 计算指标 (包含 F1) ---
    m_base = calc_metrics(Z_rl_te, y_te)

    m_llm_cot = calc_metrics(Z_cot_te, y_te)
    m_llm_nocot = calc_metrics(Z_nocot_te, y_te)
    m_llm_empty = calc_metrics(Z_empty_te, y_te)

    m_fus_cot = calc_metrics(Z_fused_cot, y_te)
    m_fus_nocot = calc_metrics(Z_fused_nocot, y_te)
    m_fus_empty = calc_metrics(Z_fused_empty, y_te)

    # --- 打印战报 ---
    print(f"\n{'=' * 110}")
    print(f"🔥 填表专用全指标战报 (Seed {seed})")
    print(f"{'=' * 110}")
    header = f"{'Method':<30} | {'Top-1':<10} | {'Top-5':<10} | {'MRR':<10} | {'Macro-F1':<10} | {'Weight-F1':<10}"
    print(header)
    print("-" * 110)

    def print_row(name, m):
        print(f"{name:<30} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f} | {m[4]:<10.4f}")

    print("[单体基线表填空区]")
    print_row("GRU Baseline", m_base)
    print_row("LLM (CoT) Only", m_llm_cot)
    print_row("LLM (No-CoT) Only", m_llm_nocot)
    print_row("LLM (Empty) Only", m_llm_empty)
    print("-" * 110)

    print("[融合结果表填空区]")
    print_row(f"CoT Fusion (\u03B1={alpha_cot:.2f})", m_fus_cot)
    print_row(f"No-CoT Fusion (\u03B1={alpha_nocot:.2f})", m_fus_nocot)
    print_row(f"Empty Fusion (\u03B1={alpha_empty:.2f})", m_fus_empty)
    print(f"{'=' * 110}")


if __name__ == "__main__":
    run_eval(seed=42)