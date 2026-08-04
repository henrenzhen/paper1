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
class PolicyTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_heads=4, hidden_dim=256, num_layers=2, num_labels=184, pad_idx=0,
                 max_len=20):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_labels)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        src_key_padding_mask = (x == self.pad_idx)

        emb = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)

        # Masked Mean Pooling
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        masked_out = out * mask
        sum_out = masked_out.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_out / denom

        return self.classifier(pooled)


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
def encode_labels(labels, label2id, split_name):
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def encode_dataset(df, vocab, label2id, max_len, pad_idx):
    X, Y = [], []
    unk_idx = vocab.get("<UNK>", 1)
    for _, row in df.iterrows():
        seq = [vocab.get(tok.strip(), unk_idx) for tok in str(row["state"]).split("||") if tok.strip()]
        seq = seq[-max_len:] if len(seq) > max_len else seq + [pad_idx] * (max_len - len(seq))
        X.append(seq)
        Y.append(label2id[str(row["true_label"]).strip()])
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


# 【优化 2】专用 MRR 函数，极速搜索 Alpha
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
def run_transformer_fusion(seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在初始化 Transformer + LLM (CoT) 融合实验 (Seed {seed}) ...")

    # --- A. 加载数据与字典 ---
    train_cot = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_cot = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_cot = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    ckpt_tf = torch.load(RL_DIR / f"rl_transformer_v2_seed{seed}.pt", map_location="cpu")
    vocab = ckpt_tf["token2id"]
    label2id = ckpt_tf["label2id"]
    max_len = ckpt_tf["max_len"]
    num_labels = ckpt_tf["num_labels"]
    pad_idx = vocab.get("<PAD>", 0)

    y_val = encode_labels(val_cot["true_label"], label2id, "Val")
    y_te = encode_labels(test_cot["true_label"], label2id, "Test")

    # --- B. 推断 Transformer ---
    print("[INFO] 正在计算 Transformer Logits...")
    X_val, _ = encode_dataset(val_cot, vocab, label2id, max_len, pad_idx)
    X_te, _ = encode_dataset(test_cot, vocab, label2id, max_len, pad_idx)

    # 【预留防坑】如果你后续在 checkpoint 里存了 config，可以直接在这里替换动态读取
    tf_model = PolicyTransformer(len(vocab), num_labels=num_labels, pad_idx=pad_idx, max_len=max_len).to(device)
    tf_model.load_state_dict(ckpt_tf["model_state_dict"])
    tf_model.eval()

    with torch.no_grad():
        Z_tf_val = tf_model(X_val.to(device)).cpu().numpy()
        Z_tf_te = tf_model(X_te.to(device)).cpu().numpy()

    # --- C. 提取 LLM BGE 特征 ---
    print(f"[INFO] 正在提取 LLM CoT BGE 特征...")
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

    H_llm_val = extract_bge(val_cot)
    H_llm_te = extract_bge(test_cot)
    del embedder;
    torch.cuda.empty_cache()

    # --- D. 推断 LLM Logits ---
    print("[INFO] 正在推断 LLM Probe Logits...")
    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt", map_location=device))
    llm_net.eval()

    with torch.no_grad():
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        Z_llm_te = llm_net(H_llm_te).cpu().numpy()

    # --- E. 在 Val 上搜索最优 Alpha ---
    print("\n[INFO] 正在 Val 集上搜索最优融合比例 Alpha ...")

    # 【优化 6】打印独立模型的 Val MRR，为 Alpha 提供解释背景
    val_mrr_tf = get_mrr(Z_tf_val, y_val)
    val_mrr_llm = get_mrr(Z_llm_val, y_val)
    print(f"  -> [参考] Transformer 单体 Val MRR: {val_mrr_tf:.4f}")
    print(f"  -> [参考] LLM (CoT) 单体 Val MRR: {val_mrr_llm:.4f}")

    best_alpha, best_mrr = 0.0, -1.0
    for alpha in np.linspace(0, 1.0, 101):
        Z_tmp = (1.0 - alpha) * Z_tf_val + alpha * Z_llm_val
        mrr_tmp = get_mrr(Z_tmp, y_val)  # 【优化 2】极速搜索
        if mrr_tmp > best_mrr:
            best_mrr, best_alpha = mrr_tmp, alpha

    print(f"  -> 最优 Alpha 锁定为: {best_alpha:.2f} (融合 Val MRR: {best_mrr:.4f})\n")

    # --- F. 在 Test 集上执行最终评估 ---
    Z_fused_te = (1.0 - best_alpha) * Z_tf_te + best_alpha * Z_llm_te

    metrics_tf = calc_metrics(Z_tf_te, y_te)
    metrics_llm = calc_metrics(Z_llm_te, y_te)
    metrics_fused = calc_metrics(Z_fused_te, y_te)

    # --- G. 打印战报 ---
    print(f"{'=' * 110}")
    print(f"🔥 Transformer + LLM (CoT) Logit Fusion 战报 | 测试样本数: {len(y_te)}")
    print(f"{'=' * 110}")
    header = f"{'Method':<25} | {'Top-1':<10} | {'Top-5':<10} | {'MRR':<10} | {'Macro-F1':<10} | {'Weight-F1':<10}"
    print(header)
    print("-" * 110)

    def print_row(name, m):
        print(f"{name:<25} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f} | {m[4]:<10.4f}")

    print_row("Transformer Only", metrics_tf)
    print_row("LLM (CoT) Only", metrics_llm)
    print("-" * 110)
    print_row(f"👑 Fusion (\u03B1={best_alpha:.2f})", metrics_fused)
    print(f"{'=' * 110}")

    print("\n[🎯 核心结论：相对纯 Transformer 的绝对增益]")
    print(f"  -> Top-1 Δ:     {metrics_fused[0] - metrics_tf[0]:+.4f}")
    print(f"  -> Top-5 Δ:     {metrics_fused[1] - metrics_tf[1]:+.4f}")
    print(f"  -> MRR Δ:       {metrics_fused[2] - metrics_tf[2]:+.4f}")
    print(f"  -> Macro-F1 Δ:  {metrics_fused[3] - metrics_tf[3]:+.4f}")
    print(f"  -> Weight-F1 Δ: {metrics_fused[4] - metrics_tf[4]:+.4f}")


if __name__ == "__main__":
    run_transformer_fusion(seed=42)