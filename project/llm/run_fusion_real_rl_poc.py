import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RL_DIR = PROJECT_ROOT / "rl"

# ==========================================
# 1. 定义网络结构
# ==========================================
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
        return h

# ==========================================
# 2. 加载数据与 RL 模型 (🔥全面启用 GPU)
# ==========================================
CSV_PATH = DATA_DIR / "sim_test_llm_cot_100.csv" 
RL_MODEL_PATH = RL_DIR / "rl_baseline_v2.pt"

df = pd.read_csv(CSV_PATH)
texts = df["llm_thinking_process"].fillna("").tolist()
states = df["state"].tolist()
labels = df["true_label"].tolist()

# 🔥 自动调用你的 4090
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[1] 正在将 RL 模型加载至 {device.type.upper()} 节点...")

checkpoint = torch.load(RL_MODEL_PATH, map_location=device)
token2id = checkpoint["token2id"]
max_len = checkpoint["max_len"]
vocab_size = len(token2id)

model = PolicyGRU(
    vocab_size=vocab_size,
    emb_dim=128,
    hidden_dim=128,
    num_labels=checkpoint["num_labels"],
    pad_idx=token2id.get("<PAD>", 0)
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def encode_prefix(prefix_str):
    items = [x.strip() for x in str(prefix_str).split("||") if x.strip()]
    seq = [token2id.get(tok, token2id.get("<UNK>", 1)) for tok in items]
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = seq + [token2id.get("<PAD>", 0)] * (max_len - len(seq))
    return seq

# 🔥 矩阵化批处理 (Batching)：不要写 for 循环，直接把 100 条打成一个大矩阵塞进显存！
print(f" -> 正在 4090 上进行矩阵并发特征提取...")
seqs = [encode_prefix(s) for s in states]
tensor_seqs = torch.tensor(seqs, dtype=torch.long).to(device)

with torch.no_grad():
    h_vecs = model(tensor_seqs) # 你的 4090 在这一步只需要零点零几秒
    X_rl = h_vecs.cpu().numpy()
print(f" -> 成功提取 RL 特征: {X_rl.shape}")

# ==========================================
# 3. 提取 LLM 文本语义特征 (降维至 128 对等制衡)
# ==========================================
print("\n[2] 正在提取 LLM 推演文本的语义特征...")
vec = TfidfVectorizer(max_features=128)
X_llm = vec.fit_transform(texts).toarray()
print(f" -> 成功提取 LLM 特征: {X_llm.shape}")

# ==========================================
# 4. 特征级融合与归一化 (防量级碾压)
# ==========================================
print("\n[3] 正在进行特征归一化与融合...")
scaler = StandardScaler()
X_rl_scaled = scaler.fit_transform(X_rl)
X_fused = np.hstack([X_rl_scaled, X_llm])
print(f" -> 融合后总特征维度: {X_fused.shape}")

# ==========================================
# 5. 快速验证 (加入实时进度条，防止假死焦虑)
# ==========================================
print("\n[4] 启动留一法交叉验证 (LOOCV) 评估性能 (Top-1, Top-5, MRR)...")
loo = LeaveOneOut()

def eval_metrics(clf, X_train, y_train, X_test, y_true):
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[0]
    classes_ = clf.classes_
    
    order = probs.argsort()[::-1]
    ranked_labels = [classes_[j] for j in order]
    
    top1 = int(ranked_labels[0] == y_true)
    top5 = int(y_true in ranked_labels[:5])
    rr = 1.0 / (ranked_labels.index(y_true) + 1) if y_true in ranked_labels else 0.0
    return top1, top5, rr

metrics = {"rl": [0,0,0], "llm": [0,0,0], "fused": [0,0,0]}
n = len(labels)

# 加入进度打印
for idx, (train_idx, test_idx) in enumerate(loo.split(X_fused)):
    # \r 会让进度条在同一行刷新，不会刷屏
    print(f"\r -> 正在训练第 {idx+1}/{n} 个折叠模型...", end="", flush=True)
    
    y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
    y_true = y_test[0]
    
    clf_rl = LogisticRegression(max_iter=1000, solver='lbfgs')
    metrics["rl"] = [sum(x) for x in zip(metrics["rl"], eval_metrics(clf_rl, X_rl_scaled[train_idx], y_train, X_rl_scaled[test_idx], y_true))]
        
    clf_llm = LogisticRegression(max_iter=1000, solver='lbfgs')
    metrics["llm"] = [sum(x) for x in zip(metrics["llm"], eval_metrics(clf_llm, X_llm[train_idx], y_train, X_llm[test_idx], y_true))]

    clf_fused = LogisticRegression(max_iter=1000, solver='lbfgs')
    metrics["fused"] = [sum(x) for x in zip(metrics["fused"], eval_metrics(clf_fused, X_fused[train_idx], y_train, X_fused[test_idx], y_true))]

print("\n\n=== 样本(100条) LOOCV 完整指标评估 ===")
print(f"{'模型':<20} | {'Top-1':<8} | {'Top-5':<8} | {'MRR':<8}")
print("-" * 50)
print(f"{'纯 RL 序列记忆提取':<16} | {metrics['rl'][0]/n:<8.4f} | {metrics['rl'][1]/n:<8.4f} | {metrics['rl'][2]/n:<8.4f}")
print(f"{'纯 LLM 语义推演':<17} | {metrics['llm'][0]/n:<8.4f} | {metrics['llm'][1]/n:<8.4f} | {metrics['llm'][2]/n:<8.4f}")
print(f"{'融合特征 (RL + LLM)':<15} | {metrics['fused'][0]/n:<8.4f} | {metrics['fused'][1]/n:<8.4f} | {metrics['fused'][2]/n:<8.4f}")
print("=========================================")