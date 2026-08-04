import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 核心分析逻辑：Coverage & Joint Recall
# ==========================================
def analyze_coverage_and_upper_bound(y_te, Z_gru_te, Z_llm_te):
    print("\n" + "=" * 90)
    print("🎯 开启联合召回上限与 Coverage 分析 (Coverage & Joint Recall Upper Bound)")
    print("=" * 90)

    y_te = np.asarray(y_te)
    total_samples = len(y_te)

    # 辅助函数：计算 Top-K 命中掩码 (Boolean Array)
    def get_topk_mask(Z, y, k):
        preds = np.argsort(Z, axis=1)[:, ::-1][:, :k]
        hits = np.any(preds == y[:, None], axis=1)
        return hits

    # ---------------------------------------------------------
    # 1. 单模型 Top-K Coverage
    # ---------------------------------------------------------
    k_values = [1, 3, 5, 10, 20]
    ind_results = []

    for k in k_values:
        gru_hits = get_topk_mask(Z_gru_te, y_te, k)
        llm_hits = get_topk_mask(Z_llm_te, y_te, k)

        ind_results.append({
            "K": k,
            "Metric": f"Top-{k} Coverage",
            "GRU_Coverage_Pct": np.mean(gru_hits) * 100,
            "LLM_Coverage_Pct": np.mean(llm_hits) * 100
        })

    df_ind = pd.DataFrame(ind_results)

    # 打印格式化版本
    df_ind_print = df_ind.copy()
    df_ind_print['GRU_Coverage_Pct'] = df_ind_print['GRU_Coverage_Pct'].map("{:.2f}%".format)
    df_ind_print['LLM_Coverage_Pct'] = df_ind_print['LLM_Coverage_Pct'].map("{:.2f}%".format)
    print("\n[表 1：单模型 Top-K Coverage (Recall Upper Bound)]")
    print(df_ind_print.drop(columns=['K']).to_string(index=False))

    # ---------------------------------------------------------
    # 2. 联合召回上限分析 (Joint Coverage)
    # ---------------------------------------------------------
    joint_pairs = [(5, 3), (10, 3), (10, 5), (20, 5)]  # (GRU_K, LLM_K)
    joint_results = []

    for gru_k, llm_k in joint_pairs:
        gru_hits = get_topk_mask(Z_gru_te, y_te, gru_k)
        llm_hits = get_topk_mask(Z_llm_te, y_te, llm_k)

        # 联合命中 (并集)
        joint_hits = gru_hits | llm_hits
        gru_cov = np.mean(gru_hits) * 100
        joint_cov = np.mean(joint_hits) * 100
        delta_cov = joint_cov - gru_cov

        # LLM 的独家贡献 (LLM 命中了，但 GRU 没命中)
        llm_unique_adds = np.sum(~gru_hits & llm_hits)
        llm_unique_adds_pct = (llm_unique_adds / total_samples) * 100

        # 定义该联合配置的学术定位 (Oracle Gain Type)
        if gru_k <= 5:
            oracle_type = "Compact Candidate Pool"
        elif gru_k <= 10:
            oracle_type = "Standard Joint Recall"
        else:
            oracle_type = "Coverage Ceiling Approach"

        joint_results.append({
            "GRU_K": gru_k,
            "LLM_K": llm_k,
            "Union_Condition": f"GRU Top-{gru_k} ∪ LLM Top-{llm_k}",
            "Oracle_Type": oracle_type,
            "Base_GRU_Cov": gru_cov,
            "Joint_Coverage": joint_cov,
            "Delta_over_GRU": delta_cov,
            "LLM_Unique_Adds_Pct": llm_unique_adds_pct,
            "LLM_Unique_Adds_Count": int(llm_unique_adds)
        })

    df_joint = pd.DataFrame(joint_results)

    # 打印格式化版本
    df_joint_print = df_joint.copy()
    df_joint_print['Base_GRU_Cov'] = df_joint_print['Base_GRU_Cov'].map("{:.2f}%".format)
    df_joint_print['Joint_Coverage'] = df_joint_print['Joint_Coverage'].map("{:.2f}%".format)
    df_joint_print['Delta_over_GRU'] = df_joint_print['Delta_over_GRU'].map("+{:.2f}%".format)
    df_joint_print['LLM_Unique_Adds'] = df_joint_print.apply(
        lambda row: f"+{row['LLM_Unique_Adds_Pct']:.2f}% ({row['LLM_Unique_Adds_Count']} samples)", axis=1
    )
    cols_to_print = ['Union_Condition', 'Base_GRU_Cov', 'Joint_Coverage', 'Delta_over_GRU', 'LLM_Unique_Adds',
                     'Oracle_Type']

    print("\n[表 2：GRU 与 LLM 联合召回上限 (Joint Coverage / Union)]")
    print("* 注: Delta over GRU 直接反映了融合带来的召回天花板提升。")
    print("* 注: LLM Unique Adds 表示 LLM 成功命中、且 GRU 完全漏掉的增量样本。")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_joint_print[cols_to_print].to_string(index=False))
    print("\n" + "=" * 90)

    return df_ind, df_joint


# ==========================================
# 模型与执行入口
# ==========================================
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
            nn.LayerNorm(llm_dim), nn.Linear(llm_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def forward(self, x): return self.net(x)


if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    # 为了避免混淆，虽然路径叫 rl，我们变量名统一叫 GRU_DIR
    GRU_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 正在加载模型与数据...")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")
    ckpt = torch.load(GRU_DIR / "rl_baseline_v2_seed42.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # --- 提取 GRU Logits ---
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()
    seqs = []
    for s in test_df["state"]:
        toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
        seqs.append(toks + [0] * (ckpt["max_len"] - len(toks)))
    with torch.no_grad():
        Z_gru_te = gru(torch.tensor(seqs, device=device)).cpu().numpy()

    # --- 提取 LLM Logits ---
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()
    feats = []
    txts = test_df["llm_thinking_process"].fillna("None").tolist()
    for i in range(0, len(txts), 64):
        batch = tk(txts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            feats.append(F.normalize(md(**batch)[0][:, 0], p=2, dim=1).cpu().numpy())
    H_te = np.vstack(feats)

    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / "llm_probe_seed42.pt", map_location=device))
    llm_p.eval()
    with torch.no_grad():
        Z_llm_te = llm_p(torch.tensor(H_te, device=device)).cpu().numpy()

    # 执行覆盖率分析并接收返回值
    df_ind, df_joint = analyze_coverage_and_upper_bound(y_te, Z_gru_te, Z_llm_te)

    # 保存纯净数值的 CSV 文件
    out_path1 = DATA_DIR / "coverage_individual_upper_bound.csv"
    out_path2 = DATA_DIR / "coverage_joint_upper_bound.csv"
    df_ind.to_csv(out_path1, index=False, encoding="utf-8-sig")
    df_joint.to_csv(out_path2, index=False, encoding="utf-8-sig")
    print(f"[INFO] 纯数值分析结果已导出至:\n  -> {out_path1}\n  -> {out_path2}")