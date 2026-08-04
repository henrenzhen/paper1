import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

# =========================
# 0. 绝对路径与 Seed 锁
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# 1. 读取数据
# =========================
train_df = pd.read_csv(DATA_DIR / "sim_train_parent_min3.csv")
val_df = pd.read_csv(DATA_DIR / "sim_val_parent_min3.csv")
test_df = pd.read_csv(DATA_DIR / "sim_test_parent_min3.csv")
label_vocab = pd.read_csv(DATA_DIR / "rl_label_vocab.csv")

NUM_LABELS = len(label_vocab)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

label2id = dict(zip(label_vocab["technique_id_parent"], label_vocab["label_id"]))
id2label = dict(zip(label_vocab["label_id"], label_vocab["technique_id_parent"]))

def tokenize_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]

token_set = set()
for s in train_df["prefix_technique_ids_parent"].astype(str):
    token_set.update(tokenize_state(s.replace("||", " ")))

token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tok in sorted(token_set): token2id[tok] = len(token2id)
vocab_size = len(token2id)

MAX_LEN = 20
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_REWARD = 1.0

# =========================
# 2. 数据处理
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
        self.states, self.labels, self.meta = [], [], []
        for _, row in df.iterrows():
            state_str = normalize_prefix(row["prefix_technique_ids_parent"])
            tokens = tokenize_state(state_str)
            label_name = str(row["next_technique_id_parent"]).strip()
            self.states.append(torch.tensor(encode_tokens(tokens), dtype=torch.long))
            self.labels.append(torch.tensor(label2id[label_name], dtype=torch.long))
            self.meta.append({"sequence_id": row["sequence_id"], "prefix_len": int(row["prefix_len"]), "state": state_str, "true_label": label_name})
    def __len__(self): return len(self.states)
    def __getitem__(self, idx): return self.states[idx], self.labels[idx], self.meta[idx]

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas

train_loader = DataLoader(RLDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(RLDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(RLDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# =========================
# 3. 原汁原味的 PolicyGRU
# =========================
class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)
    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return self.classifier(h.squeeze(0))

def compute_rank_rewards(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(probs, dim=1, descending=True)
    rewards = []
    for i in range(probs.size(0)):
        rank = sorted_idx[i].tolist().index(y[i].item()) + 1
        if rank == 1: r = 1.0
        elif rank <= 5: r = 0.5
        else: r = 0.0
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32, device=probs.device)

# =========================
# 4. 严谨评估 (加回了 F1)
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total, top1, top5, mrr_sum = 0.0, 0, 0, 0, 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss(reduction="none")

    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, y).mean()
        
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        pred1 = probs.argmax(dim=1)
        top1 += (pred1 == y).sum().item()
        all_preds.extend(pred1.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        top5_idx = torch.topk(probs, k=min(5, probs.size(1)), dim=1)[1]
        top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()

        sorted_idx = torch.argsort(probs, dim=1, descending=True)
        for i in range(x.size(0)):
            rank = sorted_idx[i].tolist().index(y[i].item()) + 1
            mrr_sum += 1.0 / rank

    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    wei_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {"loss": total_loss/total, "top1": top1/total, "top5": top5/total, "mrr": mrr_sum/total, "mac_f1": mac_f1, "wei_f1": wei_f1}

# =========================
# 5. 主循环：跑满 5 个 Seed
# =========================
def run_seed(seed):
    print(f"\n{'='*50}")
    print(f"🚀 开始训练 RL Baseline | Seed: {seed}")
    print(f"{'='*50}")
    set_seed(seed)
    
    model = PolicyGRU(vocab_size=vocab_size, emb_dim=128, hidden_dim=128, num_labels=NUM_LABELS, pad_idx=token2id[PAD_TOKEN]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_top5, best_val_mrr = -1.0, -1.0
    patience_counter = 0
    save_path = BASE_DIR / f"rl_baseline_v2_seed{seed}.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            ce_losses = criterion(logits, y)
            rewards = compute_rank_rewards(probs.detach(), y)
            loss = (ce_losses * (1.0 + LAMBDA_REWARD * rewards)).mean()
            
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader)
        
        improved = False
        if val_metrics["top5"] > best_val_top5: improved = True
        elif val_metrics["top5"] == best_val_top5 and val_metrics["mrr"] > best_val_mrr: improved = True

        if improved:
            best_val_top5, best_val_mrr = val_metrics["top5"], val_metrics["mrr"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(), "token2id": token2id,
                "label2id": label2id, "id2label": id2label, "num_labels": NUM_LABELS, "max_len": MAX_LEN
            }, save_path)
            print(f"Epoch {epoch:02d} | Val T5: {val_metrics['top5']:.4f} | MRR: {val_metrics['mrr']:.4f} [✨ Saved]")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

    # 盲盒测试
    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    tm = evaluate(model, test_loader)
    print(f"\n🎯 Seed {seed} 终极战绩 | T1: {tm['top1']:.4f} | T5: {tm['top5']:.4f} | MRR: {tm['mrr']:.4f} | Mac-F1: {tm['mac_f1']:.4f}")

if __name__ == "__main__":
    for s in [42, 43, 44, 45, 46]:
        run_seed(s)
    print("\n[SUCCESS] 5 个 Seed 的 RL 权重已全部就绪！")