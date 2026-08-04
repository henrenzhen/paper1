import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import random
import os
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# =========================
# 0. 绝对路径与环境准备
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"  # 确保和你的 RL 脚本路径一致
CKPT_DIR = BASE_DIR / "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_labels(labels, label2id):
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)

# =========================
# 1. 评估函数 (加入 F1)
# =========================
def eval_metrics(logits_np, y_true_np):
    top1_hits, top5_hits, mrr_sum = 0, 0, 0.0
    n = len(y_true_np)
    
    y_pred = []
    for i in range(n):
        order = logits_np[i].argsort()[::-1]
        y_pred.append(order[0])
        if order[0] == y_true_np[i]: top1_hits += 1
        if y_true_np[i] in order[:5]: top5_hits += 1
        rank = np.where(order == y_true_np[i])[0][0] + 1
        mrr_sum += 1.0 / rank
        
    y_pred_np = np.array(y_pred)
    mac_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    wei_f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
    
    return top1_hits / n, top5_hits / n, mrr_sum / n, mac_f1, wei_f1

# =========================
# 2. LLM 专属探针网络
# =========================
class LLMOnlyNet(nn.Module):
    def __init__(self, llm_dim=768, num_labels=184):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    def forward(self, x): return self.net(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 引擎启动 | 计算设备: {device.type.upper()}")

    # =========================
    # 3. 加载数据与对齐字典
    # =========================
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df   = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df  = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    label_vocab = pd.read_csv(DATA_DIR / "rl_label_vocab.csv")
    
    num_labels = len(label_vocab)
    label2id = dict(zip(label_vocab["technique_id_parent"], label_vocab["label_id"]))

    y_tr  = encode_labels(train_df["true_label"], label2id)
    y_val = encode_labels(val_df["true_label"], label2id)
    y_te  = encode_labels(test_df["true_label"], label2id)

    # =========================
    # 4. 全局提取 BGE 稠密特征
    # =========================
    print("\n[INFO] 正在全局提取 BGE 稠密特征 (极其省时，仅执行一次)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()
    
    def extract_bge(df):
        feats = []
        # 确保列名与你数据集中包含大模型推演文本的列名一致
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i+256], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.FloatTensor(np.vstack(feats)).to(device)

    H_llm_tr  = extract_bge(train_df)
    H_llm_val = extract_bge(val_df)
    H_llm_te  = extract_bge(test_df)
    
    del embedder
    torch.cuda.empty_cache()

    train_loader = DataLoader(TensorDataset(H_llm_tr, torch.LongTensor(y_tr)), batch_size=256, shuffle=True)

    # =========================
    # 5. 循环 5 个 Seed 暴躁刷怪
    # =========================
    SEEDS = [42, 43, 44, 45, 46]
    
    for seed in SEEDS:
        print(f"\n" + "="*50)
        print(f"🚀 开始训练 LLM Probe | Seed: {seed}")
        print("="*50)
        
        set_seed(seed)
        
        model = LLMOnlyNet(768, num_labels).to(device)
        opt = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_v_mrr = 0.0
        ckpt_path = CKPT_DIR / f"llm_probe_seed{seed}.pt"

        # 训练 150 轮，Val MRR 早停
        for epoch in range(1, 151):
            model.train()
            for h_l, y_b in train_loader:
                opt.zero_grad()
                loss = criterion(model(h_l.to(device)), y_b.to(device))
                loss.backward()
                opt.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(H_llm_val.to(device)).cpu().numpy()
                    _, _, v_mrr, _, _ = eval_metrics(val_logits, y_val)
                
                if v_mrr > best_v_mrr:
                    best_v_mrr = v_mrr
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  -> Epoch {epoch:03d} | 新的 Val MRR 巅峰: {v_mrr:.4f} (权重已保存)")

        # 盲盒测试报告
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            test_logits = model(H_llm_te.to(device)).cpu().numpy()
            t1, t5, mrr, mac_f1, wei_f1 = eval_metrics(test_logits, y_te)
            print(f"🎯 Seed {seed} 终极战绩 | T1: {t1:.4f} | T5: {t5:.4f} | MRR: {mrr:.4f} | Mac-F1: {mac_f1:.4f}")

    print("\n[SUCCESS] 所有 5 个 Seed 的 LLM Probe 权重已全部保存至 checkpoints 目录！")

if __name__ == "__main__":
    main()