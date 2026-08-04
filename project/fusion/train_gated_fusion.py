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
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RL_DIR = PROJECT_ROOT / "rl"
CKPT_DIR = BASE_DIR / "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ==========================================
# 0. 绝对严谨的实验环境锁定
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 彻底关闭底层优化器的非确定性

def encode_labels(labels, label2id, split_name):
    missing = sorted(set([l for l in labels if l not in label2id]))
    assert not missing, f"[{split_name} 数据集报错] 发现字典中不存在的未知标签: {missing[:10]}"
    return np.array([label2id[l] for l in labels], dtype=np.int64)

def eval_metrics(logits_np, y_true_np):
    top1_hits, top5_hits, mrr_sum = 0, 0, 0.0
    n = len(y_true_np)
    for i in range(n):
        full_order = logits_np[i].argsort()[::-1]
        if full_order[0] == y_true_np[i]: top1_hits += 1
        if y_true_np[i] in full_order[:5]: top5_hits += 1
        rank = np.where(full_order == y_true_np[i])[0][0] + 1
        mrr_sum += 1.0 / rank
    return top1_hits / n, top5_hits / n, mrr_sum / n

# ==========================================
# 1. 网络架构定义区
# ==========================================
class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, x, return_hidden=False):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h_sq = h.squeeze(0)
        logits = self.classifier(h_sq)
        if return_hidden: return logits, h_sq
        return logits

# Baseline 2: LLM Only
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
    def forward(self, h_rl, h_llm): return self.net(h_llm)

# Baseline 3: 直接拼接 (Direct Concat)
class DirectConcatNet(nn.Module):
    def __init__(self, rl_dim=128, llm_dim=768, num_labels=184):
        super().__init__()
        self.rl_ln = nn.LayerNorm(rl_dim)
        self.llm_ln = nn.LayerNorm(llm_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rl_dim + llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    def forward(self, h_rl, h_llm):
        fused = torch.cat([self.rl_ln(h_rl), self.llm_ln(h_llm)], dim=-1)
        return self.classifier(fused)

# Baseline 4: 粗粒度标量门控 (Gated Fusion 768)
class GatedFusion768Net(nn.Module):
    def __init__(self, rl_dim=128, llm_dim=768, num_labels=184):
        super().__init__()
        self.rl_ln = nn.LayerNorm(rl_dim)
        self.llm_ln = nn.LayerNorm(llm_dim)
        self.gate = nn.Sequential(
            nn.Linear(rl_dim + llm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # 标量门控
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(rl_dim + llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    def forward(self, h_rl, h_llm):
        r, l = self.rl_ln(h_rl), self.llm_ln(h_llm)
        g = self.gate(torch.cat([r, l], dim=-1))
        fused = torch.cat([r, l * g], dim=-1)
        return self.classifier(fused)

# Ours 5: 维度预算平衡 + 细粒度向量门控 (Budget-Balanced Vector Gating)
class BalancedVectorGatedNet(nn.Module):
    def __init__(self, rl_dim=128, llm_dim=768, proj_dim=128, num_labels=184):
        super().__init__()
        self.rl_ln = nn.LayerNorm(rl_dim)
        self.llm_ln = nn.LayerNorm(llm_dim)
        
        # 降维投影：实现 Budget Balancing
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
        
        # 向量门控：逐维度审查语义特征
        self.vector_gate = nn.Sequential(
            nn.Linear(rl_dim + proj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim), # 输出与投影维度对齐的 Mask
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(rl_dim + proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, h_rl, h_llm):
        r = self.rl_ln(h_rl)
        l_proj = self.llm_proj(self.llm_ln(h_llm))
        
        g_vec = self.vector_gate(torch.cat([r, l_proj], dim=-1))
        fused = torch.cat([r, l_proj * g_vec], dim=-1)
        return self.classifier(fused)

# ==========================================
# 2. 统一训练与评估引擎 (公平的 MRR 裁判)
# ==========================================
def train_and_eval_model(model, name, train_loader, val_loader, test_loader, device, y_val, y_test, epochs=60):
    print(f"\n[{name}] 开始训练...")
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_v_mrr = 0.0
    ckpt_path = CKPT_DIR / f"{name.replace(' ', '_')}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for h_r, h_l, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(h_r.to(device), h_l.to(device)), y_b.to(device))
            loss.backward()
            opt.step()

        if epoch % 5 == 0:
            model.eval()
            val_logits = []
            with torch.no_grad():
                for h_r, h_l, _ in val_loader:
                    val_logits.append(model(h_r.to(device), h_l.to(device)).cpu().numpy())
            _, _, v_mrr = eval_metrics(np.vstack(val_logits), y_val)
            
            # 以 MRR 为唯一真理进行 Early Stopping
            if v_mrr > best_v_mrr:
                best_v_mrr = v_mrr
                torch.save(model.state_dict(), ckpt_path)

    # 加载最佳盲盒测试
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_logits = []
    with torch.no_grad():
        for h_r, h_l, _ in test_loader:
            test_logits.append(model(h_r.to(device), h_l.to(device)).cpu().numpy())
            
    t1, t5, mrr = eval_metrics(np.vstack(test_logits), y_test)
    print(f"[{name}] 完毕 | Test Top-1: {t1:.4f} | Top-5: {t5:.4f} | MRR: {mrr:.4f}")
    return t1, t5, mrr

# ==========================================
# 3. 主干流程
# ==========================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 引擎启动 | 设备: {device.type.upper()} | 随机种子: 42 (强锁定)")

    # --- 数据准备 ---
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df   = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df  = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    checkpoint = torch.load(RL_DIR / "rl_baseline_v2.pt", map_location=device)
    t2id, max_len, num_labels = checkpoint["token2id"], checkpoint["max_len"], checkpoint["num_labels"]
    l2id = checkpoint["label2id"]

    y_train = encode_labels(train_df["true_label"], l2id, "Train")
    y_val   = encode_labels(val_df["true_label"], l2id, "Val")
    y_test  = encode_labels(test_df["true_label"], l2id, "Test")

    # --- 提取 RL 特征 ---
    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(checkpoint["model_state_dict"])
    rl_model.eval()

    def get_rl_feats(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            seq = seq[-max_len:] if len(seq) > max_len else seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)
        with torch.no_grad():
            logits, h = rl_model(torch.tensor(seqs, dtype=torch.long).to(device), return_hidden=True)
            return logits.cpu().numpy(), h.cpu().numpy()

    L_rl_te, H_rl_te = get_rl_feats(test_df)
    _, H_rl_tr = get_rl_feats(train_df)
    _, H_rl_val = get_rl_feats(val_df)
    rl_t1, rl_t5, rl_mrr = eval_metrics(L_rl_te, y_test)

    # --- 提取 LLM 特征 ---
    print("\n[INFO] 启动 BGE 提取器...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def get_llm_feats(df, batch_size=256):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return np.vstack(feats)

    H_llm_tr = get_llm_feats(train_df)
    H_llm_val = get_llm_feats(val_df)
    H_llm_te = get_llm_feats(test_df)

    # --- 构造 Dataloader ---
    def make_loader(hr, hl, y, shuffle=False):
        return DataLoader(TensorDataset(torch.FloatTensor(hr), torch.FloatTensor(hl), torch.LongTensor(y)), batch_size=256, shuffle=shuffle)

    tr_loader = make_loader(H_rl_tr, H_llm_tr, y_train, shuffle=True)
    va_loader = make_loader(H_rl_val, H_llm_val, y_val)
    te_loader = make_loader(H_rl_te, H_llm_te, y_test)

    # --- 核心消融矩阵执行 ---
    results = {"1. RL Only (Baseline)": (rl_t1, rl_t5, rl_mrr)}
    
    models = {
        "2. LLM Only (BGE Probing)": LLMOnlyNet(768, num_labels),
        "3. Direct Concat (No Gate)": DirectConcatNet(128, 768, num_labels),
        "4. Gated Fusion (Scalar 768)": GatedFusion768Net(128, 768, num_labels),
        "5. Budget Balanced Vector Gating (Ours)": BalancedVectorGatedNet(128, 768, 128, num_labels)
    }

    for name, model in models.items():
        results[name] = train_and_eval_model(model.to(device), name, tr_loader, va_loader, te_loader, device, y_val, y_test)

    # --- 终极 LaTeX 级 Markdown 表格 ---
    print("\n\n" + "="*85)
    print(f"【MITRE 184微观预测 —— 核心消融实验矩阵 (基于 Val MRR 严格早停)】")
    print("="*85)
    print(f"{'模型架构 (Architecture)':<40} | {'Top-1':<8} | {'Top-5':<8} | {'MRR':<8}")
    print("-" * 85)
    for name, (t1, t5, mrr) in results.items():
        print(f"{name:<40} | {t1:<8.4f} | {t5:<8.4f} | {mrr:<8.4f}")
    print("="*85)
    print("注：公式 $$ h_{fused} = [h_{rl} \\oplus (g_{vec} \\otimes W_{proj}h_{llm})] $$ 已在模型 5 中实现。")

if __name__ == "__main__":
    main()