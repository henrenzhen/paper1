import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
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
# 1. LLM 探针模型结构
# =========================
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


def calc_metrics(scores_np, y_te):
    t1 = np.zeros(len(y_te), dtype=np.int32)
    t5 = np.zeros(len(y_te), dtype=np.int32)
    mrr = np.zeros(len(y_te), dtype=np.float32)
    y_pred_top1 = []

    for i in range(len(y_te)):
        tl = y_te[i]
        order = scores_np[i].argsort()[::-1]
        top1 = order[0]
        y_pred_top1.append(top1)

        if top1 == tl: t1[i] = 1
        if tl in order[:5]: t5[i] = 1
        mrr[i] = 1.0 / (np.where(order == tl)[0][0] + 1)

    macro_f1 = f1_score(y_te, y_pred_top1, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred_top1, average='weighted', zero_division=0)

    return t1.mean(), t5.mean(), mrr.mean(), macro_f1, weighted_f1


# =========================
# 3. 核心马尔可夫模型 (带 Laplace 平滑)
# =========================
class MarkovProbBaseline:
    def __init__(self, label2id, num_labels, alpha_smooth=1e-3):
        self.label2id = label2id
        self.num_labels = num_labels
        self.alpha_smooth = alpha_smooth
        self.global_prior = Counter()
        self.markov_1st = defaultdict(Counter)
        self.markov_2nd = defaultdict(Counter)
        self.global_prob = np.full(num_labels, alpha_smooth)  # 【改进】平滑初始化

    def fit(self, df):
        for _, row in df.iterrows():
            seq_str = str(row["state"])
            target = str(row["true_label"]).strip()
            self.global_prior[target] += 1

            items = [x.strip() for x in seq_str.split("||") if x.strip()]
            if not items: continue

            state_1 = items[-1]
            self.markov_1st[state_1][target] += 1

            if len(items) >= 2:
                state_2 = (items[-2], items[-1])
                self.markov_2nd[state_2][target] += 1

        # 计算全局平滑概率分布
        for k, v in self.global_prior.items():
            if k in self.label2id:
                self.global_prob[self.label2id[k]] += v
        self.global_prob /= self.global_prob.sum()

    def predict_proba(self, df, order=2):
        probs = np.full((len(df), self.num_labels), self.alpha_smooth)
        hit_2nd, hit_1st, hit_global = 0, 0, 0

        for i, row in enumerate(df.iterrows()):
            _, r = row
            seq_str = str(r["state"])
            items = [x.strip() for x in seq_str.split("||") if x.strip()]

            hit = False

            # --- 严格分支控制 ---
            if order == 2:
                # 先尝试二阶
                if len(items) >= 2:
                    state_2 = (items[-2], items[-1])
                    if state_2 in self.markov_2nd:
                        counter = self.markov_2nd[state_2]
                        for k, v in counter.items():
                            if k in self.label2id: probs[i, self.label2id[k]] += v
                        hit_2nd += 1
                        hit = True
                # 二阶未命中，降级尝试一阶
                if not hit and len(items) >= 1:
                    state_1 = items[-1]
                    if state_1 in self.markov_1st:
                        counter = self.markov_1st[state_1]
                        for k, v in counter.items():
                            if k in self.label2id: probs[i, self.label2id[k]] += v
                        hit_1st += 1
                        hit = True

            elif order == 1:
                # 只尝试一阶
                if len(items) >= 1:
                    state_1 = items[-1]
                    if state_1 in self.markov_1st:
                        counter = self.markov_1st[state_1]
                        for k, v in counter.items():
                            if k in self.label2id: probs[i, self.label2id[k]] += v
                        hit_1st += 1
                        hit = True

            # 如果 order == 0，上方逻辑全部跳过，hit 保持 False，直接进入下方全局先验

            # 3. 终极回退：全局先验
            if not hit:
                probs[i] = self.global_prob.copy()
                hit_global += 1
                continue

            probs[i] /= probs[i].sum()

        stats = {"2nd": hit_2nd, "1st": hit_1st, "global": hit_global, "total": len(df)}
        return probs, stats


# =========================
# 4. 主流程：概率级特征融合
# =========================
def run_markov_fusion(seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 正在初始化 2nd-Order Markov + LLM (CoT) 融合实验 (Seed {seed}) ...")

    # --- A. 加载数据与字典 ---
    train_cot = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_cot = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_cot = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location="cpu")
    label2id = ckpt_rl["label2id"]
    num_labels = ckpt_rl["num_labels"]

    y_val = encode_labels(val_cot["true_label"], label2id, "Val")
    y_te = encode_labels(test_cot["true_label"], label2id, "Test")

    # --- B. 训练并获取 Markov 概率 ---
    print("[INFO] 正在计算 2nd-Order Markov 概率矩阵 (带 Laplace 平滑)...")
    mk_model = MarkovProbBaseline(label2id, num_labels, alpha_smooth=1e-3)
    mk_model.fit(train_cot)

    P_mk_val, _ = mk_model.predict_proba(val_cot, order=2)
    P_mk_te, stats_te = mk_model.predict_proba(test_cot, order=2)

    P_global_te, _ = mk_model.predict_proba(test_cot, order=0)  # 提取 Global Prior 作为对比基线

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

    # --- D. 推断 LLM Logits 和 概率 ---
    print("[INFO] 正在推断 LLM Probe ...")
    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt", map_location=device))
    llm_net.eval()

    with torch.no_grad():
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        Z_llm_te = llm_net(H_llm_te).cpu().numpy()

        P_llm_val = F.softmax(torch.tensor(Z_llm_val), dim=-1).numpy()
        P_llm_te = F.softmax(torch.tensor(Z_llm_te), dim=-1).numpy()

    # --- 核心：将 Markov 概率转为 Logits 以备 Logit Fusion ---
    Z_mk_val = np.log(P_mk_val + 1e-9)
    Z_mk_te = np.log(P_mk_te + 1e-9)

    # --- E. 独立搜索最优 Alpha ---
    print("[INFO] 正在 Val 集上搜索最优融合比例 Alpha ...")

    # 1. Probability Fusion 搜索
    best_alpha_prob, best_mrr_prob = 0.0, -1.0
    for alpha in np.linspace(0, 1.0, 51):
        P_tmp = (1.0 - alpha) * P_mk_val + alpha * P_llm_val
        mrr_tmp = calc_metrics(P_tmp, y_val)[2]
        if mrr_tmp > best_mrr_prob: best_mrr_prob, best_alpha_prob = mrr_tmp, alpha

    # 2. Logit Fusion 搜索
    best_alpha_logit, best_mrr_logit = 0.0, -1.0
    for alpha in np.linspace(0, 1.0, 51):
        Z_tmp = (1.0 - alpha) * Z_mk_val + alpha * Z_llm_val
        mrr_tmp = calc_metrics(Z_tmp, y_val)[2]
        if mrr_tmp > best_mrr_logit: best_mrr_logit, best_alpha_logit = mrr_tmp, alpha

    # --- F. 在 Test 集上执行最终评估 ---
    P_fused_prob_te = (1.0 - best_alpha_prob) * P_mk_te + best_alpha_prob * P_llm_te
    Z_fused_logit_te = (1.0 - best_alpha_logit) * Z_mk_te + best_alpha_logit * Z_llm_te

    metrics_global = calc_metrics(P_global_te, y_te)
    metrics_mk = calc_metrics(P_mk_te, y_te)
    metrics_llm = calc_metrics(Z_llm_te, y_te)
    metrics_f_pr = calc_metrics(P_fused_prob_te, y_te)
    metrics_f_lg = calc_metrics(Z_fused_logit_te, y_te)

    # --- G. 打印战报 ---
    print(f"\n{'=' * 115}")
    print(f"🔥 Markov (2nd-Order) + LLM (CoT) 深度融合战报 | 测试样本数: {len(y_te)}")
    print(
        f"   [命中分解] 2nd-Order: {stats_te['2nd']} | 1st-Order Fallback: {stats_te['1st']} | Global Fallback: {stats_te['global']}")
    print(f"{'=' * 115}")
    header = f"{'Method':<30} | {'Top-1':<10} | {'Top-5':<10} | {'MRR':<10} | {'Macro-F1':<10} | {'Weight-F1':<10}"
    print(header)
    print("-" * 115)

    def print_row(name, m):
        print(f"{name:<30} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f} | {m[4]:<10.4f}")

    print_row("Global Prior (0-Order)", metrics_global)
    print_row("Markov Only (2nd-Order)", metrics_mk)
    print_row("LLM (CoT) Only", metrics_llm)
    print("-" * 115)
    print_row(f"Prob Fusion  (\u03B1={best_alpha_prob:.2f})", metrics_f_pr)
    print_row(f"👑 Logit Fusion (\u03B1={best_alpha_logit:.2f})", metrics_f_lg)
    print(f"{'=' * 115}")


if __name__ == "__main__":
    run_markov_fusion(seed=42)