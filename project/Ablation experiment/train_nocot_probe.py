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
# 1. 辅助函数与模型定义
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


# 【优化 1】抽出 get_mrr，保持代码整洁不重复
def get_mrr(scores_np, y_true):
    mrr_sum = 0.0
    for i in range(len(y_true)):
        order = scores_np[i].argsort()[::-1]
        mrr_sum += 1.0 / (np.where(order == y_true[i])[0][0] + 1)
    return mrr_sum / len(y_true)


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
# 2. 训练 No-CoT 探针
# =========================
def train_nocot_probe(seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在提取 BGE 特征并训练 No-CoT Probe (Seed {seed}) ...")

    # 加载标签字典
    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location=device)
    l2id = ckpt_rl["label2id"]
    num_labels = ckpt_rl["num_labels"]

    # 读取刚才新鲜出炉的 no_cot 数据
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_no_cot.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_no_cot.csv")

    # 【优化 4】训练前检查标签分布和数据量是否对齐
    print(f"[INFO] 数据分布对齐检查 - Train size: {len(train_df)} | Val size: {len(val_df)} | Num labels: {num_labels}")

    # 【优化 3】快速 Sanity Check，防止生成了一堆 Error
    print("\n[INFO] No-CoT 示例文本 (Sanity Check):")
    print(f"  Train[0]: {str(train_df['llm_thinking_process'].iloc[0])[:200]}")
    print(f"  Val[0]:   {str(val_df['llm_thinking_process'].iloc[0])[:200]}\n")

    y_tr = torch.tensor(encode_labels(train_df["true_label"], l2id, "Train"), dtype=torch.long, device=device)
    y_val = torch.tensor(encode_labels(val_df["true_label"], l2id, "Val"), dtype=torch.long, device=device)

    # 提取 BGE 文本特征
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

    H_tr = extract_bge(train_df)
    H_val = extract_bge(val_df)

    del embedder
    torch.cuda.empty_cache()

    # 训练 Probe
    model = LLMOnlyNet(768, num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_mrr = -1.0
    best_state = None
    best_epoch = -1  # 【优化 2】记录最优 epoch
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

            # 使用提取出的 get_mrr 函数
            current_mrr = get_mrr(logits_val, y_val_np)

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch  # 更新最优 epoch
                patience_counter = 0
            else:
                patience_counter += 1

        if epoch % 5 == 0:
            print(f"  -> Epoch {epoch:02d} | Train CE Loss: {loss.item():.4f} | Val MRR: {current_mrr:.4f}")

        if patience_counter >= patience:
            print(
                f"  [!] Epoch {epoch:02d}: Early stopping 触发 (最优 Val MRR: {best_mrr:.4f} 出现于 Epoch {best_epoch:02d})")
            break

    assert best_state is not None, "best_state is None, No-CoT probe 训练失败！"

    save_path = LLM_CKPT_DIR / f"llm_probe_no_cot_seed{seed}.pt"
    torch.save(best_state, save_path)

    # 【优化 2】打印最终详细信息
    print(f"\n[SUCCESS] No-CoT 探针已保存至: {save_path} | Best Epoch: {best_epoch:02d} | Best Val MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    train_nocot_probe(seed=42)