import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 读取数据
# =========================
train_df = pd.read_csv("sim_train_parent_min3.csv")
val_df = pd.read_csv("sim_val_parent_min3.csv")
test_df = pd.read_csv("sim_test_parent_min3.csv")
label_vocab = pd.read_csv("rl_label_vocab.csv")

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
# 4. 数据处理
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
# 5. 模型
# =========================
class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = h.squeeze(0)
        logits = self.classifier(h)
        return logits

model = PolicyGRU(
    vocab_size=vocab_size,
    emb_dim=128,
    hidden_dim=128,
    num_labels=NUM_LABELS,
    pad_idx=token2id[PAD_TOKEN]
).to(DEVICE)

# 单样本 CE，方便做 reward weighting
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# 6. 奖励函数
# =========================
def compute_rank_rewards(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    probs: [B, C]
    y: [B]
    reward:
      rank=1 -> 1.0
      rank<=5 -> 0.5
      else -> 0.0
    """
    sorted_idx = torch.argsort(probs, dim=1, descending=True)
    rewards = []

    for i in range(probs.size(0)):
        true_label = y[i].item()
        ranking = sorted_idx[i].tolist()
        rank = ranking.index(true_label) + 1

        if rank == 1:
            r = 1.0
        elif rank <= 5:
            r = 0.5
        else:
            r = 0.0
        rewards.append(r)

    return torch.tensor(rewards, dtype=torch.float32, device=probs.device)

# =========================
# 7. 评估
# =========================
@torch.no_grad()
def evaluate(loader, save_predictions=False, pred_outfile=None):
    model.eval()

    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    mrr_sum = 0.0
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

    metrics = {
        "loss": total_loss / total,
        "top1": top1 / total,
        "top5": top5 / total,
        "mrr": mrr_sum / total
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

        ce_losses = criterion(logits, y)  # [B]
        rewards = compute_rank_rewards(probs.detach(), y)  # [B]

        # reward-weighted CE
        # rank越好，权重越大；错样本仍有基础CE
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
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_top1={val_metrics['top1']:.4f} | "
        f"val_top5={val_metrics['top5']:.4f} | "
        f"val_mrr={val_metrics['mrr']:.4f}"
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
            "rl_baseline_v2.pt"
        )
        print("saved best model -> rl_baseline_v2.pt")
    else:
        patience_counter += 1
        print(f"early_stop_counter = {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# =========================
# 9. 测试
# =========================
ckpt = torch.load("rl_baseline_v2.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

test_metrics = evaluate(
    test_loader,
    save_predictions=True,
    pred_outfile="rl_v2_test_predictions_top5.csv"
)

print("\n=== TEST ===")
print(f"test_loss={test_metrics['loss']:.4f}")
print(f"test_top1={test_metrics['top1']:.4f}")
print(f"test_top5={test_metrics['top5']:.4f}")
print(f"test_mrr={test_metrics['mrr']:.4f}")
print("saved predictions -> rl_v2_test_predictions_top5.csv")