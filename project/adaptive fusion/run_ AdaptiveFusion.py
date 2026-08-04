import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# =========================
# 0. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
RL_DIR = Path(r"E:\desktop\project_only\project\rl")
LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")


# =========================
# 1. 模型结构定义
# =========================
class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.classifier(h.squeeze(0))


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


class LogisticGate(nn.Module):
    def __init__(self, input_dim=15):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, meta_features):
        return torch.sigmoid(self.fc(meta_features))  # 自由输出 [0, 1]


# =========================
# 2. 统计检验与辅助函数
# =========================
def mcnemar_test(base_hits, fused_hits):
    a = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 1 and base_hits[i] == 1)
    b = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 1 and base_hits[i] == 0)
    c = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 0 and base_hits[i] == 1)
    d = sum(1 for i in range(len(base_hits)) if fused_hits[i] == 0 and base_hits[i] == 0)
    if b + c == 0:
        return 1.0, b, c
    mc_result = mcnemar([[a, c], [b, d]], exact=True)
    return mc_result.pvalue, b, c


def get_mrr(scores_np, y_true):
    mrr_sum = 0.0
    for i in range(len(y_true)):
        order = scores_np[i].argsort()[::-1]
        mrr_sum += 1.0 / (np.where(order == y_true[i])[0][0] + 1)
    return mrr_sum / len(y_true)


def encode_labels(labels, label2id, split_name):
    missing = sorted(set([str(l).strip() for l in labels if str(l).strip() not in label2id]))
    assert not missing, f"[{split_name}] 发现未知标签: {missing[:10]}"
    return np.array([label2id[str(l).strip()] for l in labels], dtype=np.int64)


def compare_against_ref(ref_t1, new_t1, name_ref, name_new):
    win = sum(1 for i in range(len(ref_t1)) if new_t1[i] == 1 and ref_t1[i] == 0)
    lose = sum(1 for i in range(len(ref_t1)) if new_t1[i] == 0 and ref_t1[i] == 1)
    print(f"  -> [{name_new} vs {name_ref}] 优于对方: {win} | 劣于对方: {lose} | 净胜出: {win - lose:+d}")


def build_meta_features(Z_rl, Z_llm):
    """
    15维 Meta-features:
      RL基础 4维 | LLM基础 4维 | 排名分歧 3维 | 交叉信任 4维
    """
    P_rl = F.softmax(torch.tensor(Z_rl, dtype=torch.float32), dim=1).numpy()
    P_llm = F.softmax(torch.tensor(Z_llm, dtype=torch.float32), dim=1).numpy()

    feats = []
    for i in range(len(Z_rl)):
        rl_order = np.argsort(P_rl[i])[::-1]
        llm_order = np.argsort(P_llm[i])[::-1]

        rl_top1_idx, rl_top2_idx = rl_order[0], rl_order[1]
        llm_top1_idx, llm_top2_idx = llm_order[0], llm_order[1]

        rl_top1, rl_top2 = P_rl[i, rl_top1_idx], P_rl[i, rl_top2_idx]
        llm_top1, llm_top2 = P_llm[i, llm_top1_idx], P_llm[i, llm_top2_idx]

        rl_entropy = -np.sum(P_rl[i] * np.log(P_rl[i] + 1e-9))
        llm_entropy = -np.sum(P_llm[i] * np.log(P_llm[i] + 1e-9))

        is_agree = 1.0 if rl_top1_idx == llm_top1_idx else 0.0
        rl_top1_in_llm_rank = float(np.where(llm_order == rl_top1_idx)[0][0])
        llm_top1_in_rl_rank = float(np.where(rl_order == llm_top1_idx)[0][0])

        llm_prob_on_rl_top1 = P_llm[i, rl_top1_idx]
        rl_prob_on_llm_top1 = P_rl[i, llm_top1_idx]
        prob_diff_top1 = rl_top1 - llm_top1

        rl_top5 = rl_order[:5]
        llm_top5 = llm_order[:5]
        top5_overlap_ratio = len(np.intersect1d(rl_top5, llm_top5)) / 5.0

        feats.append(
            [
                rl_top1, rl_top2, rl_top1 - rl_top2, rl_entropy,
                llm_top1, llm_top2, llm_top1 - llm_top2, llm_entropy,
                is_agree, rl_top1_in_llm_rank, llm_top1_in_rl_rank,
                llm_prob_on_rl_top1, rl_prob_on_llm_top1, prob_diff_top1, top5_overlap_ratio,
            ]
        )

    return np.array(feats, dtype=np.float32)


# =========================
# 3. 核心流程
# =========================
def run_significance_on_seed(seed, device, train_df, val_df, test_df, H_llm_tr, H_llm_val, H_llm_te):
    ckpt_rl = torch.load(RL_DIR / f"rl_baseline_v2_seed{seed}.pt", map_location=device)
    t2id = ckpt_rl["token2id"]
    max_len = ckpt_rl["max_len"]
    num_labels = ckpt_rl["num_labels"]
    l2id = ckpt_rl["label2id"]

    y_tr = encode_labels(train_df["true_label"], l2id, "Train")
    y_val = encode_labels(val_df["true_label"], l2id, "Val")
    y_te = encode_labels(test_df["true_label"], l2id, "Test")
    n = len(y_te)

    # --- A. 加载 RL 与 LLM，并推断 Train/Val/Test ---
    rl_model = PolicyGRU(len(t2id), 128, 128, num_labels, t2id.get("<PAD>", 0)).to(device)
    rl_model.load_state_dict(ckpt_rl["model_state_dict"])
    rl_model.eval()

    def get_rl_preds(df):
        seqs = []
        for s in df["state"].tolist():
            items = [x.strip() for x in str(s).split("||") if x.strip()]
            seq = [t2id.get(tok, t2id.get("<UNK>", 1)) for tok in items]
            if len(seq) > max_len:
                seq = seq[-max_len:]
            else:
                seq = seq + [t2id.get("<PAD>", 0)] * (max_len - len(seq))
            seqs.append(seq)

        with torch.no_grad():
            x = torch.tensor(seqs, dtype=torch.long, device=device)
            return rl_model(x).cpu().numpy()

    Z_rl_tr = get_rl_preds(train_df)
    Z_rl_val = get_rl_preds(val_df)
    Z_rl_te = get_rl_preds(test_df)

    llm_net = LLMOnlyNet(768, num_labels).to(device)
    llm_net.load_state_dict(torch.load(LLM_CKPT_DIR / f"llm_probe_seed{seed}.pt", map_location=device))
    llm_net.eval()

    with torch.no_grad():
        Z_llm_tr = llm_net(H_llm_tr).cpu().numpy()
        Z_llm_val = llm_net(H_llm_val).cpu().numpy()
        Z_llm_te = llm_net(H_llm_te).cpu().numpy()

    # --- B. 构建并标准化 15维 Meta-features ---
    print("\n[INFO] 正在构建并标准化 15维 Meta-features...")
    phi_tr = build_meta_features(Z_rl_tr, Z_llm_tr)
    phi_val = build_meta_features(Z_rl_val, Z_llm_val)
    phi_te = build_meta_features(Z_rl_te, Z_llm_te)

    scaler = StandardScaler()
    phi_tr = scaler.fit_transform(phi_tr)
    phi_val = scaler.transform(phi_val)
    phi_te = scaler.transform(phi_te)

    T_Z_rl_tr = torch.tensor(Z_rl_tr, dtype=torch.float32, device=device)
    T_Z_llm_tr = torch.tensor(Z_llm_tr, dtype=torch.float32, device=device)
    T_y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
    T_phi_tr = torch.tensor(phi_tr, dtype=torch.float32, device=device)

    T_Z_rl_val = torch.tensor(Z_rl_val, dtype=torch.float32, device=device)
    T_Z_llm_val = torch.tensor(Z_llm_val, dtype=torch.float32, device=device)
    T_phi_val = torch.tensor(phi_val, dtype=torch.float32, device=device)

    # --- C. 训练 Regularized Logistic Gate ---
    print("\n[INFO] 开始训练 Regularized Logistic Gate (A' 正规协议, 非对称正则化)...")

    gate_model = LogisticGate(input_dim=15).to(device)

    # 让初始 alpha ≈ 0.30 -> sigmoid(-0.847) ≈ 0.30
    nn.init.constant_(gate_model.fc.bias, -0.847)
    nn.init.zeros_(gate_model.fc.weight)

    optimizer = optim.Adam(gate_model.parameters(), lr=1e-2, weight_decay=1e-4)
    patience = 15
    lambda_reg = 2.0  # 非对称正则化力度

    best_val_mrr = -1.0
    best_gate_state = None
    patience_counter = 0

    for epoch in range(100):
        gate_model.train()
        optimizer.zero_grad()

        # 1. 前向计算 alpha
        alpha_tr = gate_model(T_phi_tr) # [N, 1]
        Z_fused_tr = (1.0 - alpha_tr) * T_Z_rl_tr + alpha_tr * T_Z_llm_tr
        ce_loss = F.cross_entropy(Z_fused_tr, T_y_tr)

        # 2. 计算非对称正则化 (Asymmetric Penalty)
        rl_preds = T_Z_rl_tr.argmax(dim=1)
        # 如果 RL 本来就对，mask=1.0，否则为 0.0
        rl_correct_mask = (rl_preds == T_y_tr).float().unsqueeze(1)
        reg_loss = (rl_correct_mask * (alpha_tr ** 2)).mean()

        # 3. 总 Loss
        loss = ce_loss + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        gate_model.eval()
        with torch.no_grad():
            alpha_val = gate_model(T_phi_val)
            Z_fused_val = (1.0 - alpha_val) * T_Z_rl_val + alpha_val * T_Z_llm_val
            current_val_mrr = get_mrr(Z_fused_val.cpu().numpy(), y_val)

        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr
            best_gate_state = copy.deepcopy(gate_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= patience:
            with torch.no_grad():
                alpha_val_np = alpha_val.cpu().numpy().squeeze()
            print(
                f"  -> Epoch {epoch:02d} | Total Loss: {loss.item():.4f} "
                f"(CE: {ce_loss.item():.4f}, Reg: {reg_loss.item():.4f}) | "
                f"Val MRR: {current_val_mrr:.4f} | Val Alpha Mean: {alpha_val_np.mean():.4f}"
            )

        if patience_counter >= patience:
            print(f"  [!] Early stopping 触发 (最优 Val MRR: {best_val_mrr:.4f})")
            break

    assert best_gate_state is not None, "best_gate_state is None, Logistic Gate 训练失败！"

    # --- D. Test 集闭眼评估 ---
    gate_model.load_state_dict(best_gate_state)
    gate_model.eval()

    with torch.no_grad():
        T_phi_te = torch.tensor(phi_te, dtype=torch.float32, device=device)
        T_Z_rl_te = torch.tensor(Z_rl_te, dtype=torch.float32, device=device)
        T_Z_llm_te = torch.tensor(Z_llm_te, dtype=torch.float32, device=device)

        alpha_te_tensor = gate_model(T_phi_te)
        Z_logistic_te = (1.0 - alpha_te_tensor) * T_Z_rl_te + alpha_te_tensor * T_Z_llm_te

        alpha_te = alpha_te_tensor.cpu().numpy().squeeze()
        Z_logistic_te = Z_logistic_te.cpu().numpy()

    # --- E. 计算 Baseline / Global / Rule-based 对照 ---
    print("\n[INFO] 正在对齐计算 Baseline, Global Fusion 和 Rule-based Fusion...")

    best_g_alpha, best_g_val_mrr = 0.0, -1.0
    for alpha in np.linspace(0, 1.0, 51):
        Z_v_tmp = (1.0 - alpha) * Z_rl_val + alpha * Z_llm_val
        cmrr = get_mrr(Z_v_tmp, y_val)
        if cmrr > best_g_val_mrr:
            best_g_val_mrr = cmrr
            best_g_alpha = alpha
    Z_global_te = (1.0 - best_g_alpha) * Z_rl_te + best_g_alpha * Z_llm_te

    probs_rl_val = F.softmax(torch.tensor(Z_rl_val, dtype=torch.float32), dim=1).numpy()
    sorted_val = np.sort(probs_rl_val, axis=1)[:, ::-1]
    margin_rl_val = sorted_val[:, 0] - sorted_val[:, 1]

    best_r_mrr, best_theta, best_a_low = -1.0, 0.0, 0.0
    for theta in [0.05, 0.10, 0.15, 0.20]:
        for a_low in [0.10, 0.20, 0.30, 0.40]:
            a_val_arr = np.where(margin_rl_val > theta, 0.0, a_low)[:, None]
            Z_r_val = (1.0 - a_val_arr) * Z_rl_val + a_val_arr * Z_llm_val
            cmrr = get_mrr(Z_r_val, y_val)
            if cmrr > best_r_mrr:
                best_r_mrr = cmrr
                best_theta = theta
                best_a_low = a_low

    probs_rl_te = F.softmax(torch.tensor(Z_rl_te, dtype=torch.float32), dim=1).numpy()
    sorted_te = np.sort(probs_rl_te, axis=1)[:, ::-1]
    margin_rl_te = sorted_te[:, 0] - sorted_te[:, 1]
    alpha_rule_te = np.where(margin_rl_te > best_theta, 0.0, best_a_low)[:, None]
    Z_rule_te = (1.0 - alpha_rule_te) * Z_rl_te + alpha_rule_te * Z_llm_te

    print(f"[INFO] Global Fusion 最优 Alpha = {best_g_alpha:.2f} | Val MRR = {best_g_val_mrr:.4f}")
    print(
        f"[INFO] Rule-based 最优 Theta = {best_theta:.2f}, Alpha_low = {best_a_low:.2f} | "
        f"Val MRR = {best_r_mrr:.4f}"
    )
    print(f"[INFO] RegGate 最优 Val MRR = {best_val_mrr:.4f}")

    # --- F. 统计 Test 指标 ---
    base_t1 = np.zeros(n, dtype=np.int32)
    base_t5 = np.zeros(n, dtype=np.int32)
    base_mrr = np.zeros(n, dtype=np.float32)

    g_t1 = np.zeros(n, dtype=np.int32)
    g_t5 = np.zeros(n, dtype=np.int32)
    g_mrr = np.zeros(n, dtype=np.float32)

    r_t1 = np.zeros(n, dtype=np.int32)
    r_t5 = np.zeros(n, dtype=np.int32)
    r_mrr = np.zeros(n, dtype=np.float32)

    l_t1 = np.zeros(n, dtype=np.int32)
    l_t5 = np.zeros(n, dtype=np.int32)
    l_mrr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        tl = y_te[i]

        o_base = Z_rl_te[i].argsort()[::-1]
        if o_base[0] == tl:
            base_t1[i] = 1
        if tl in o_base[:5]:
            base_t5[i] = 1
        base_mrr[i] = 1.0 / (np.where(o_base == tl)[0][0] + 1)

        o_g = Z_global_te[i].argsort()[::-1]
        if o_g[0] == tl:
            g_t1[i] = 1
        if tl in o_g[:5]:
            g_t5[i] = 1
        g_mrr[i] = 1.0 / (np.where(o_g == tl)[0][0] + 1)

        o_r = Z_rule_te[i].argsort()[::-1]
        if o_r[0] == tl:
            r_t1[i] = 1
        if tl in o_r[:5]:
            r_t5[i] = 1
        r_mrr[i] = 1.0 / (np.where(o_r == tl)[0][0] + 1)

        o_l = Z_logistic_te[i].argsort()[::-1]
        if o_l[0] == tl:
            l_t1[i] = 1
        if tl in o_l[:5]:
            l_t5[i] = 1
        l_mrr[i] = 1.0 / (np.where(o_l == tl)[0][0] + 1)

    # --- G. 打印战报 ---
    print("\n" + "=" * 95)
    print(f"🔥 综合战报 (Seed {seed}) | 严格 A' 协议 | 测试样本数: {n}")
    print("=" * 95)

    print(
        f"{'Metric':<10} | {'Baseline':<12} | {'Global':<12} | {'Rule-based':<12} | {'RegGate':<16}"
    )
    print("-" * 95)
    print(
        f"{'Top-1':<10} | {base_t1.mean():.4f}       | {g_t1.mean():.4f}       | {r_t1.mean():.4f}       | {l_t1.mean():.4f}"
    )
    print(
        f"{'Top-5':<10} | {base_t5.mean():.4f}       | {g_t5.mean():.4f}       | {r_t5.mean():.4f}       | {l_t5.mean():.4f}"
    )
    print(
        f"{'MRR':<10} | {base_mrr.mean():.4f}       | {g_mrr.mean():.4f}       | {r_mrr.mean():.4f}       | {l_mrr.mean():.4f}"
    )
    print("-" * 95)

    def print_diagnostics(name, t1_arr):
        b = sum(1 for i in range(n) if t1_arr[i] == 1 and base_t1[i] == 0)
        c = sum(1 for i in range(n) if t1_arr[i] == 0 and base_t1[i] == 1)
        print(f"[{name:<16} Top-1] 救场: {b:<3} | 带偏: {c:<3} | 净收益: {b - c}")

    print_diagnostics("Global", g_t1)
    print_diagnostics("Rule-based", r_t1)
    print_diagnostics("RegGate", l_t1)

    print("\n[直面对决 (净优势)]")
    compare_against_ref(g_t1, r_t1, "Global Fusion", "Rule-based")
    compare_against_ref(g_t1, l_t1, "Global Fusion", "RegGate")
    compare_against_ref(r_t1, l_t1, "Rule-based", "RegGate")

    try:
        _, p_mrr = wilcoxon(base_mrr, l_mrr, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p_mrr = 1.0
    p_t1, _, _ = mcnemar_test(base_t1, l_t1)
    p_t5, _, _ = mcnemar_test(base_t5, l_t5)

    print("\n[RegGate 显著性检验 (vs Baseline)]")
    print(f"  -> MRR   p-value: {p_mrr:.6f} ({'✅' if p_mrr < 0.05 else '❌'})")
    print(f"  -> Top-1 p-value: {p_t1:.6f} ({'✅' if p_t1 < 0.05 else '❌'})")
    print(f"  -> Top-5 p-value: {p_t5:.6f} ({'✅' if p_t5 < 0.05 else '❌'})")

    print("\n[RegGate Alpha 动态性诊断]")
    print(f"  -> Mean: {alpha_te.mean():.4f} | Std: {alpha_te.std():.4f}")
    print(f"  -> Min : {alpha_te.min():.4f} | Max: {alpha_te.max():.4f}")
    print(
        f"  -> P10 : {np.percentile(alpha_te, 10):.4f} | "
        f"P50: {np.percentile(alpha_te, 50):.4f} | "
        f"P90: {np.percentile(alpha_te, 90):.4f}"
    )
    print(f"  -> Alpha > 0.30 比例: {(alpha_te > 0.30).mean() * 100:.2f}%")
    print(f"  -> Alpha > 0.40 比例: {(alpha_te > 0.40).mean() * 100:.2f}%")

    correlation = np.corrcoef(margin_rl_te, alpha_te)[0, 1]
    print(f"  -> [关键核验] Alpha 与 RL Margin 的皮尔逊相关系数: {correlation:.4f}")
    print("=" * 95)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 正在加载数据并一次性提取 BGE 特征 (Train + Val + Test)...")

    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    embedder = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device)
    embedder.eval()

    def extract_bge(df):
        feats = []
        texts = [str(t) for t in df["llm_thinking_process"].fillna("").tolist()]
        for i in range(0, len(texts), 256):
            inputs = tokenizer(
                texts[i:i + 256],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = embedder(**inputs)
                feats.append(F.normalize(out[0][:, 0], p=2, dim=1).cpu().numpy())
        return torch.tensor(np.vstack(feats), dtype=torch.float32, device=device)

    H_llm_tr = extract_bge(train_df)
    H_llm_val = extract_bge(val_df)
    H_llm_te = extract_bge(test_df)

    del embedder
    torch.cuda.empty_cache()

    run_significance_on_seed(42, device, train_df, val_df, test_df, H_llm_tr, H_llm_val, H_llm_te)


if __name__ == "__main__":
    main()