import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
from pathlib import Path
import random
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# =========================
# 0. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
RL_DIR = Path(r"E:\desktop\project_only\project\rl")
RL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. 辅助函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_dataset(df, vocab, label2id, max_len, pad_idx):
    X, Y = [], []
    unk_idx = vocab.get("<UNK>", 1)
    for _, row in df.iterrows():
        seq = [vocab.get(tok.strip(), unk_idx) for tok in str(row["state"]).split("||") if tok.strip()]
        seq = seq[-max_len:] if len(seq) > max_len else seq + [pad_idx] * (max_len - len(seq))
        X.append(seq)
        Y.append(label2id[str(row["true_label"]).strip()])
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


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
# 2. Transformer Baseline 模型定义 (Masked Mean Pooling)
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

        # 【核心修复】：Masked Mean Pooling，彻底隔离 PAD 干扰
        mask = (x != self.pad_idx).unsqueeze(-1).float()  # shape: (batch_size, seq_len, 1)
        masked_out = out * mask
        sum_out = masked_out.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)  # 防止全零除法报错
        pooled = sum_out / denom

        return self.classifier(pooled)


# =========================
# 3. 训练流程
# =========================
def train_transformer_baseline(seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在初始化严谨版 Transformer Baseline (Seed {seed}) ...")

    # 1. 加载数据
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    # 2. 【核心修复】：绝对无泄漏，直接复用 GRU 的 Checkpoint 字典
    rl_ckpt_path = RL_DIR / f"rl_baseline_v2_seed{seed}.pt"
    print(f"[INFO] 正在从 {rl_ckpt_path.name} 载入严谨对齐的 Vocab 和 Label 空间...")
    ckpt_rl = torch.load(rl_ckpt_path, map_location="cpu")

    vocab = ckpt_rl["token2id"]
    label2id = ckpt_rl["label2id"]
    max_len = ckpt_rl["max_len"]
    num_labels = ckpt_rl["num_labels"]
    pad_idx = vocab.get("<PAD>", 0)
    print(f"  -> Vocab Size: {len(vocab)} | Num Labels: {num_labels} | Max Len: {max_len}")

    # 3. 编码数据
    X_tr, y_tr = encode_dataset(train_df, vocab, label2id, max_len, pad_idx)
    X_val, y_val = encode_dataset(val_df, vocab, label2id, max_len, pad_idx)
    X_te, y_te = encode_dataset(test_df, vocab, label2id, max_len, pad_idx)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    X_val, y_val = X_val.to(device), y_val.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    # 4. 初始化模型
    model = PolicyTransformer(len(vocab), num_labels=num_labels, pad_idx=pad_idx, max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_mrr = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    patience = 15

    # 5. 训练循环
    for epoch in range(100):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Val 评估
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()

            # 使用提取出的指标计算函数
            metrics_val = calc_metrics(logits_val, y_val_np)
            current_mrr = metrics_val[2]

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

        if epoch % 5 == 0:
            print(
                f"  -> Epoch {epoch:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val MRR: {current_mrr:.4f}")

        if patience_counter >= patience:
            print(
                f"  [!] Epoch {epoch:02d}: Early stopping 触发 (最优 Val MRR: {best_mrr:.4f} 出现于 Epoch {best_epoch:02d})")
            break

    # 6. 【核心修复】：保存丰富的元数据
    assert best_state is not None
    save_path = RL_DIR / f"rl_transformer_v2_seed{seed}.pt"

    ckpt = {
        "model_state_dict": best_state,
        "token2id": vocab,
        "label2id": label2id,
        "max_len": max_len,
        "num_labels": num_labels,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mrr": best_mrr
    }
    torch.save(ckpt, save_path)
    print(f"\n[SUCCESS] Transformer Baseline 已保存至: {save_path}")

    # 7. Test 集最终战报
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(X_te).cpu().numpy()
        y_te_np = y_te.cpu().numpy()

        metrics = calc_metrics(logits_te, y_te_np)

        print(f"\n{'=' * 100}")
        print(f"🔥 Transformer 基线 (Seed {seed}) 终极战报 | 测试样本数: {len(y_te_np)}")
        print(f"{'=' * 100}")
        header = f"{'Method':<25} | {'Top-1':<10} | {'Top-5':<10} | {'MRR':<10} | {'Macro-F1':<10} | {'Weight-F1':<10}"
        print(header)
        print("-" * 100)
        print(
            f"{'Transformer Encoder':<25} | {metrics[0]:<10.4f} | {metrics[1]:<10.4f} | {metrics[2]:<10.4f} | {metrics[3]:<10.4f} | {metrics[4]:<10.4f}")
        print(f"{'=' * 100}")


if __name__ == "__main__":
    train_transformer_baseline(seed=42)