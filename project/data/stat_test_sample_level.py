import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar  # [新增] 用于精确 McNemar 检验
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

# =========================
# 0. 路径配置 (请确保与本地一致)
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "fusion" else BASE_DIR

DATA_DIR = Path(r"E:\desktop\project_only\project\data")
RL_DIR = Path(r"E:\desktop\project_only\project\rl")
LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")


# =========================
# 1. 模型结构定义
# =========================
class PolicyGRU(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = torch.nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.classifier(h.squeeze(0))


class LLMOnlyNet(torch.nn.Module):
    def __init__(self, llm_dim=768, num_labels=184):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(llm_dim), torch.nn.Linear(llm_dim, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(256, num_labels)
        )

    def forward(self, x): return self.net(x)


# =========================
# 2. 统计检验与辅助函数
# =========================
def mcnemar_test(base_hits, fused_hits):
    """ 升级版：使用 statsmodels 的 Exact McNemar 检验 """
    a = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 1 and base_hits[i] == 1)  # 都对
    b = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 1 and base_hits[i] == 0)  # 救场 (融合对, 基线错)
    c = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 0 and base_hits[i] == 1)  # 带偏 (融合错, 基线对)
    d = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 0 and base_hits[i] == 0)  # 都错

    if b + c == 0:
        return 1.0, b, c

    # 构建 2x2 混淆矩阵
    table = [[a, c],
             [b, d]]

    # exact=True 执行精确二项检验，更适合严谨的论文报告
    mc_result = mcnemar(table, exact=True)
    return mc_result.pvalue, b, c


def get_mrr(scores_np, y_true):
    mrr_sum = 0.0
    for i in range(len(y_true)):
        order = scores_np[i].argsort()[::-1]
        mrr_sum += 1.0 / (np.where(order == y_true[i])[0][0] + 1)
    return mrr_sum / len(y_true)


def encode_labels(labels, label2id, split_name):
    """ [新增] 安全的标签映射函数 """
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def run_significance_on_seed(seed, device, val_df, test_df, H_llm_val, H_llm_te):
    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location=device)
    t2id, max_len, num_labels, l2id = ckpt_rl["token2id"], ckpt_rl["max_len"], ckpt_rl["num_labels"], ckpt_rl[
        "label2id"]

    # [修改] 使用统一的安全映射函数
    y_val = encode_labels(val_df["true_label"], l2id, "Val")
    y_te = encode_labels(test_df["true_label"], l2id, "Test")
    n = len(y_te)

    # 1. 加载并推断 GRU 基线
    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(ckpt_rl["model_state_dict"])
    rl_model.eval()

    def get_rl_preds(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            seq = seq[-max_len:] if len(seq) > max_len else seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)
        with torch.no_grad():
            return rl_model(torch.tensor(seqs, dtype=torch.long).to(device)).cpu().numpy()

    Z_rl_val = get_rl_preds(val_df)
    Z_rl_te = get_rl_preds(test_df)

    # 2. 加载并推断 LLM
    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt", map_location=device))
    llm_net.eval()
    with torch.no_grad():
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        Z_llm_te = llm_net(H_llm_te).cpu().numpy()

    # 3. 极其严谨的 Val 集 Alpha 搜索
    # [修改] 初值设为 -1.0 防止极端情况
    best_alpha, best_val_mrr = 0.0, -1.0
    for alpha in np.linspace(0, 1.0, 51):
        Z_val_tmp = (1 - alpha) * Z_rl_val + alpha * Z_llm_val
        current_mrr = get_mrr(Z_val_tmp, y_val)
        if current_mrr > best_val_mrr:
            best_val_mrr = current_mrr
            best_alpha = alpha

    print(f"\n[INFO] Seed {seed} | 验证集搜索完毕最优 Alpha-Z = {best_alpha:.2f}")

    # 4. 生成最终融合 Test 预测
    Z_fused_te = (1 - best_alpha) * Z_rl_te + best_alpha * Z_llm_te

    # 5. 统计样本级命中情况
    base_t1, fused_t1 = np.zeros(n), np.zeros(n)
    base_t5, fused_t5 = np.zeros(n), np.zeros(n)
    base_mrr, fused_mrr = np.zeros(n), np.zeros(n)

    for i in range(n):
        true_lbl = y_te[i]

        # 基线表现
        rl_order = Z_rl_te[i].argsort()[::-1]
        if rl_order[0] == true_lbl: base_t1[i] = 1
        if true_lbl in rl_order[:5]: base_t5[i] = 1
        base_mrr[i] = 1.0 / (np.where(rl_order == true_lbl)[0][0] + 1)

        # 融合表现
        fused_order = Z_fused_te[i].argsort()[::-1]
        if fused_order[0] == true_lbl: fused_t1[i] = 1
        if true_lbl in fused_order[:5]: fused_t5[i] = 1
        fused_mrr[i] = 1.0 / (np.where(fused_order == true_lbl)[0][0] + 1)

    # =========================
    # 6. 打印论文级检验战报
    # =========================
    print(f"\n{'=' * 75}")
    # [修改] 把 best_alpha 放进大标题
    print(f"🔥 样本级统计显著性检验 (Seed {seed}) | 测试集样本数: {n} | 最优 Alpha-Z: {best_alpha:.2f}")
    print(f"{'=' * 75}")

    # 指标绝对值对比
    print(f"{'Metric':<10} | {'Baseline (RNN)':<15} | {'Global Fusion':<15} | {'Delta':<10}")
    print("-" * 75)
    print(
        f"{'Top-1':<10} | {base_t1.mean():.4f}          | {fused_t1.mean():.4f}          | {fused_t1.mean() - base_t1.mean():+.4f}")
    print(
        f"{'Top-5':<10} | {base_t5.mean():.4f}          | {fused_t5.mean():.4f}          | {fused_t5.mean() - base_t5.mean():+.4f}")
    print(
        f"{'MRR':<10} | {base_mrr.mean():.4f}          | {fused_mrr.mean():.4f}          | {fused_mrr.mean() - base_mrr.mean():+.4f}")
    print(f"{'=' * 75}")

    # MRR - Wilcoxon
    # [修改] 增加异常保护机制
    try:
        stat, p_mrr = wilcoxon(base_mrr, fused_mrr, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p_mrr = 0.0, 1.0

    print(f"[MRR] Wilcoxon Signed-Rank Test:")
    print(f"  -> p-value = {p_mrr:.6f} ({'显著提升! ✅' if p_mrr < 0.05 else '不显著 ❌'})")

    # Top-5 - McNemar
    p_t5, b_t5, c_t5 = mcnemar_test(base_t5, fused_t5)
    print(f"\n[Top-5] McNemar's Test:")
    print(f"  -> 救场样本 (基线错, 融合对): {b_t5} 个")
    print(f"  -> 带偏样本 (基线对, 融合错): {c_t5} 个")
    print(f"  -> 净收益样本数 (Net Gain)  : {b_t5 - c_t5} 个 🚀")
    print(f"  -> p-value = {p_t5:.6f} ({'显著提升! ✅' if p_t5 < 0.05 else '不显著 ❌'})")

    # Top-1 - McNemar
    p_t1, b_t1, c_t1 = mcnemar_test(base_t1, fused_t1)
    print(f"\n[Top-1] McNemar's Test:")
    print(f"  -> 救场样本 (基线错, 融合对): {b_t1} 个")
    print(f"  -> 带偏样本 (基线对, 融合错): {c_t1} 个")
    print(f"  -> 净收益样本数 (Net Gain)  : {b_t1 - c_t1} 个")
    print(f"  -> p-value = {p_t1:.6f} ({'显著提升! ✅' if p_t1 < 0.05 else '不显著 ❌'})")
    print(f"{'=' * 75}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 正在加载数据并一次性提取 BGE 特征 (Val + Test)...")

    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def extract_bge(df):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                device)
            with torch.no_grad(): feats.append(F.normalize(embedder(**inputs)[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.FloatTensor(np.vstack(feats)).to(device)

    H_llm_val = extract_bge(val_df)
    H_llm_te = extract_bge(test_df)

    del embedder
    torch.cuda.empty_cache()

    # 对 Seed 42 进行严格闭环检验
    run_significance_on_seed(42, device, val_df, test_df, H_llm_val, H_llm_te)


if __name__ == "__main__":
    main()