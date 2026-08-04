import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import copy
import warnings
import random

warnings.filterwarnings("ignore")

# =========================
# 0. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
RL_DIR = Path(r"E:\desktop\project_only\project\rl")
LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
LLM_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. 制造 Empty 数据集
# =========================
print("[INFO] 正在生成 Empty 数据集...")


def make_empty(src_name, dst_name):
    src_path = DATA_DIR / src_name
    if not src_path.exists():
        print(f"  [!] 找不到源文件 {src_name}，跳过...")
        return
    df = pd.read_csv(src_path)
    df["llm_thinking_process"] = "No reasoning."
    df.to_csv(DATA_DIR / dst_name, index=False)
    print(f"  -> saved: {dst_name}")


make_empty("sim_train_llm_cot.csv", "sim_train_llm_empty.csv")
make_empty("sim_val_llm_cot.csv", "sim_val_llm_empty.csv")
make_empty("sim_test_llm_cot.csv", "sim_test_llm_empty.csv")


# =========================
# 2. 辅助函数与模型定义
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_labels(labels, label2id, split_name):
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


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
# 3. 训练 Empty 探针
# =========================
def train_empty_probe(seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在提取 BGE 特征并训练 Empty Probe (Seed {seed}) ...")

    # 加载标签字典
    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location=device)
    l2id = ckpt_rl["label2id"]
    num_labels = ckpt_rl["num_labels"]

    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_empty.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_empty.csv")

    y_tr = torch.tensor(encode_labels(train_df["true_label"], l2id, "Train"), dtype=torch.long, device=device)
    y_val = torch.tensor(encode_labels(val_df["true_label"], l2id, "Val"), dtype=torch.long, device=device)

    # 提取 BGE (内容全是 "No reasoning.")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def extract_bge(df):
        feats = []
        texts = df["llm_thinking_process"].tolist()
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.tensor(np.vstack(feats), dtype=torch.float32, device=device)

    H_tr = extract_bge(train_df)
    H_val = extract_bge(val_df)

    del embedder
    torch.cuda.empty_cache()

    # 训练 Probe
    model = LLMOnlyNet(768, num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_mrr = -1.0
    best_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits_tr = model(H_tr)
        loss = F.cross_entropy(logits_tr, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(H_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()

            mrr_sum = 0.0
            for i in range(len(y_val_np)):
                order = logits_val[i].argsort()[::-1]
                mrr_sum += 1.0 / (np.where(order == y_val_np[i])[0][0] + 1)
            current_mrr = mrr_sum / len(y_val_np)

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

        if epoch % 5 == 0:
            print(f"  -> Epoch {epoch:02d} | Train CE Loss: {loss.item():.4f} | Val MRR: {current_mrr:.4f}")

        if patience_counter >= patience:
            print(f"  [!] Epoch {epoch:02d}: Early stopping 触发 (最优 Val MRR: {best_mrr:.4f})")
            break

    assert best_state is not None, "best_state is None, Empty probe 训练失败！"

    save_path = LLM_CKPT_DIR / f"llm_probe_empty_seed{seed}.pt"
    torch.save(best_state, save_path)
    print(f"[SUCCESS] Empty 探针已保存至: {save_path}")


if __name__ == "__main__":
    train_empty_probe(seed=42)