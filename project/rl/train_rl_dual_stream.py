import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from sklearn.metrics import f1_score # [新增] F1 评估库

# =========================
# 0. [新增] 强力锁死随机种子 (保证实验 100% 可复现)
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from pathlib import Path

# =========================
# 1. 绝对路径寻址 & 读取数据
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
# 假设你的 csv 数据都在 project/data 目录下。
# 如果就在 project/rl 目录下，把 DATA_DIR 改成 BASE_DIR 即可！
DATA_DIR = PROJECT_ROOT / "data" 

train_df = pd.read_csv(DATA_DIR / "sim_train_parent_min3.csv")
val_df = pd.read_csv(DATA_DIR / "sim_val_parent_min3.csv")
test_df = pd.read_csv(DATA_DIR / "sim_test_parent_min3.csv")
label_vocab = pd.read_csv(DATA_DIR / "rl_label_vocab.csv")

NUM_LABELS = len(label_vocab)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

label2id = dict(zip(label_vocab["technique_id_parent"], label_vocab["label_id"]))
id2label = dict(zip(label_vocab["label_id"], label_vocab["technique_id_parent"]))

# =========================
# 2. 构建 state token 词表
# =========================
def tokenize_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]

token_set = set()
for s in train_df["prefix_technique_ids_parent"].astype(str):
    token_set.update(tokenize_state(s.replace("||", " ")))

token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in sorted(token_set):
    token2id[tok] = len(token2id)

vocab_size = len(token2id)

print("state vocab size:", vocab_size)
print("num labels:", NUM_LABELS)

# =========================
# 3. 超参数
# =========================
MAX_LEN = 20
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 奖励加权强度
LAMBDA_REWARD = 1.0

# =========================
# 4. 数据处理 (保持原样)
# =========================
def normalize_prefix(prefix_str: str) -> str:
    items = [x.strip() for x in str(prefix_str).split("||") if x.strip()]
    return " ".join(items)

def encode_tokens(tokens):
    ids = [token2id.get(t, token2id[UNK_TOKEN]) for t in tokens[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [token2id[PAD_TOKEN]] * (MAX_LEN - len(ids))
    return ids

class RLDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.states = []
        self.labels = []
        self.meta = []

        for _, row in df.iterrows():
            state_str = normalize_prefix(row["prefix_technique_ids_parent"])
            tokens = tokenize_state(state_str)
            label_name = str(row["next_technique_id_parent"]).strip()
            label_id = label2id[label_name]

            self.states.append(torch.tensor(encode_tokens(tokens), dtype=torch.long))
            self.labels.append(torch.tensor(label_id, dtype=torch.long))
            self.meta.append({
                "sequence_id": row["sequence_id"],
                "prefix_len": int(row["prefix_len"]),
                "state": state_str,
                "true_label": label_name
            })

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx], self.meta[idx]

train_ds = RLDataset(train_df)
val_ds = RLDataset(val_df)
test_ds = RLDataset(test_df)

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# =========================
# 5. [重构] 核心模型：多尺度因果双流网络
# =========================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out

class MultiScaleCausalEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_labels=184, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        # 全局流
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 局部流 (多尺度因果 CNN)
        conv_out_dim = 64
        self.conv_k2 = CausalConv1d(emb_dim, conv_out_dim, kernel_size=2)
        self.conv_k3 = CausalConv1d(emb_dim, conv_out_dim, kernel_size=3)
        self.conv_k5 = CausalConv1d(emb_dim, conv_out_dim, kernel_size=5)
        
        self.local_proj = nn.Sequential(
            nn.Linear(conv_out_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x, return_hidden=False):
        emb = self.embedding(x)
        
        # 全局特征
        _, h_n = self.gru(emb)
        h_global = h_n.squeeze(0)
        
        # 局部特征
        emb_t = emb.transpose(1, 2)
        c2 = F.gelu(self.conv_k2(emb_t))[:, :, -1]
        c3 = F.gelu(self.conv_k3(emb_t))[:, :, -1]
        c5 = F.gelu(self.conv_k5(emb_t))[:, :, -1]
        
        h_local = self.local_proj(torch.cat([c2, c3, c5], dim=-1))
        
        # 门控融合
        g = self.gate(torch.cat([h_global, h_local], dim=-1))
        h_fused = g * h_local + (1 - g) * h_global
        
        logits = self.classifier(h_fused)
        if return_hidden:
            return logits, h_fused
        return logits

# [修改] 实例化新模型
model = MultiScaleCausalEncoder(
    vocab_size=vocab_size,
    emb_dim=128,
    hidden_dim=128,
    num_labels=NUM_LABELS,
    pad_idx=token2id[PAD_TOKEN]
).to(DEVICE)

criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# 6. 奖励函数 (保持原样)
# =========================
def compute_rank_rewards(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(probs, dim=1, descending=True)
    rewards = []
    for i in range(probs.size(0)):
        true_label = y[i].item()
        ranking = sorted_idx[i].tolist()
        rank = ranking.index(true_label) + 1
        if rank == 1: r = 1.0
        elif rank <= 5: r = 0.5
        else: r = 0.0
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32, device=probs.device)

# =========================
# 7. [重构] 评估 (加入严谨的 F1 统计)
# =========================
@torch.no_grad()
def evaluate(loader, save_predictions=False, pred_outfile=None):
    model.eval()

    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    mrr_sum = 0.0
    
    all_preds = []
    all_labels = []
    prediction_rows = []

    for x, y, metas in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        ce_losses = criterion(logits, y)
        loss = ce_losses.mean()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        pred1 = probs.argmax(dim=1)
        top1 += (pred1 == y).sum().item()
        
        # 收集用于计算 F1 的数据
        all_preds.extend(pred1.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        k = min(5, probs.size(1))
        top5_vals, top5_idx = torch.topk(probs, k=k, dim=1)
        top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()

        sorted_idx = torch.argsort(probs, dim=1, descending=True)
        for i in range(x.size(0)):
            true_label = y[i].item()
            ranking = sorted_idx[i].tolist()
            rank = ranking.index(true_label) + 1
            mrr_sum += 1.0 / rank

            if save_predictions:
                pred_ids = top5_idx[i].tolist()
                pred_probs = top5_vals[i].tolist()
                pred_names = [id2label[idx] for idx in pred_ids]

                prediction_rows.append({
                    "sequence_id": metas[i]["sequence_id"],
                    "prefix_len": metas[i]["prefix_len"],
                    "state": metas[i]["state"],
                    "true_label": id2label[true_label],
                    "true_rank": rank,
                    "top1_pred": pred_names[0] if pred_names else "",
                    "top1_prob": pred_probs[0] if pred_probs else "",
                    "top5_labels": " || ".join(pred_names),
                    "top5_probs": " || ".join([f"{p:.6f}" for p in pred_probs])
                })

    # 计算全局 F1
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    wei_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        "loss": total_loss / total,
        "top1": top1 / total,
        "top5": top5 / total,
        "mrr": mrr_sum / total,
        "mac_f1": mac_f1,
        "wei_f1": wei_f1
    }

    if save_predictions and pred_outfile is not None:
        pd.DataFrame(prediction_rows).to_csv(pred_outfile, index=False, encoding="utf-8-sig")

    return metrics

# =========================
# 8. 训练
# =========================
best_val_top5 = -1.0
best_val_mrr = -1.0
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y, _ in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        ce_losses = criterion(logits, y)
        rewards = compute_rank_rewards(probs.detach(), y)

        weights = 1.0 + LAMBDA_REWARD * rewards
        loss = (ce_losses * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    train_loss = total_loss / total
    val_metrics = evaluate(val_loader, save_predictions=False)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} -> {val_metrics['loss']:.4f} | "
        f"Top1: {val_metrics['top1']:.4f} | "
        f"Top5: {val_metrics['top5']:.4f} | "
        f"MRR: {val_metrics['mrr']:.4f} | "
        f"Mac-F1: {val_metrics['mac_f1']:.4f} | "
        f"Wei-F1: {val_metrics['wei_f1']:.4f}"
    )

    improved = False
    if val_metrics["top5"] > best_val_top5:
        improved = True
    elif val_metrics["top5"] == best_val_top5 and val_metrics["mrr"] > best_val_mrr:
        improved = True

    if improved:
        best_val_top5 = val_metrics["top5"]
        best_val_mrr = val_metrics["mrr"]
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "token2id": token2id,
                "label2id": label2id,
                "id2label": id2label,
                "num_labels": NUM_LABELS,
                "max_len": MAX_LEN
            },
            "rl_baseline_v2.pt" # 这里直接覆盖你原来的文件，方便后续跑 Late Fusion
        )
        print(" [✨ SOTA] Saved best model -> rl_baseline_v2.pt")
    else:
        patience_counter += 1
        print(f" [!] Early stop counter = {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# =========================
# 9. 终极盲盒测试
# =========================
ckpt = torch.load("rl_baseline_v2.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

test_metrics = evaluate(
    test_loader,
    save_predictions=True,
    pred_outfile="rl_v2_test_predictions_top5.csv"
)

print("\n" + "="*50)
print("=== 🚀 TEST SET 终极表现 (Dual-Stream CNN-GRU) ===")
print("="*50)
print(f"Test Loss:  {test_metrics['loss']:.4f}")
print(f"Test Top-1: {test_metrics['top1']:.4f}  <-- 盯紧这个，看有没有突破 0.5496！")
print(f"Test Top-5: {test_metrics['top5']:.4f}  <-- 以及这个，看有没有突破 0.8771！")
print(f"Test MRR:   {test_metrics['mrr']:.4f}")
print("-" * 50)
print(f"Test Mac-F1: {test_metrics['mac_f1']:.4f}  <-- 长尾识别率")
print(f"Test Wei-F1: {test_metrics['wei_f1']:.4f}  <-- 综合体量识别率")
print("="*50)