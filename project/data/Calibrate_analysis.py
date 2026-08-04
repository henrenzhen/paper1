import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# =========================================================
# 模型定义
# =========================================================
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
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 校准分析函数
# =========================================================
def expected_calibration_error(y_true, probs, n_bins=10):
    """
    计算 top-label ECE (Expected Calibration Error)
    """
    y_true = np.asarray(y_true)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        if bin_lower == 0.0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            gap = abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += gap * prop_in_bin

            bin_stats.append({
                "Bin Range": f"({bin_lower:.1f}, {bin_upper:.1f}]",
                "Samples": int(np.sum(in_bin)),
                "Avg Confidence": float(avg_confidence_in_bin),
                "True Accuracy": float(accuracy_in_bin),
                "Gap": float(gap)
            })

    return ece, pd.DataFrame(bin_stats)


def multiclass_brier_score(y_true, probs):
    """
    多分类 Brier Score:
    对每个样本计算 predicted prob 与 one-hot label 的平方误差和，
    再对样本求平均。越低越好。
    """
    y_true = np.asarray(y_true)
    y_true_onehot = np.eye(probs.shape[1])[y_true]
    return np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))


def analyze_calibration(y_te, Z_gru, Z_llm, Z_fused, n_bins=10):
    print("\n" + "=" * 90)
    print("⚖️ 开启置信度校准分析 (Confidence Calibration: ECE & Brier Score)")
    print("=" * 90)

    y_te = np.asarray(y_te)

    # logits -> probabilities
    P_gru = F.softmax(torch.tensor(Z_gru, dtype=torch.float32), dim=-1).cpu().numpy()
    P_llm = F.softmax(torch.tensor(Z_llm, dtype=torch.float32), dim=-1).cpu().numpy()
    P_fused = F.softmax(torch.tensor(Z_fused, dtype=torch.float32), dim=-1).cpu().numpy()

    models = {
        "GRU Baseline": P_gru,
        "LLM Probe": P_llm,
        "Global Logit Fusion": P_fused
    }

    results = []
    reliability_tables = {}

    for name, probs in models.items():
        ece, df_bins = expected_calibration_error(y_te, probs, n_bins=n_bins)
        brier = multiclass_brier_score(y_te, probs)

        results.append({
            "Model": name,
            "ECE": ece,
            "ECE (%)": ece * 100,
            "Brier Score": brier
        })

        reliability_tables[name] = df_bins

    df_results = pd.DataFrame(results).sort_values("ECE")

    print("\n[表 1：模型校准度核心指标 (越低越好)]")
    print(df_results[["Model", "ECE (%)", "Brier Score"]].to_string(
        index=False,
        formatters={
            "ECE (%)": lambda x: f"{x:.2f}%",
            "Brier Score": lambda x: f"{x:.4f}"
        }
    ))

    print("\n[表 2：GRU Reliability Bin Stats]")
    print(reliability_tables["GRU Baseline"].to_string(
        index=False,
        formatters={
            "Avg Confidence": lambda x: f"{x:.4f}",
            "True Accuracy": lambda x: f"{x:.4f}",
            "Gap": lambda x: f"{x:.4f}"
        }
    ))

    print("\n[表 3：LLM Reliability Bin Stats]")
    print(reliability_tables["LLM Probe"].to_string(
        index=False,
        formatters={
            "Avg Confidence": lambda x: f"{x:.4f}",
            "True Accuracy": lambda x: f"{x:.4f}",
            "Gap": lambda x: f"{x:.4f}"
        }
    ))

    print("\n[表 4：Fusion Reliability Bin Stats]")
    print(reliability_tables["Global Logit Fusion"].to_string(
        index=False,
        formatters={
            "Avg Confidence": lambda x: f"{x:.4f}",
            "True Accuracy": lambda x: f"{x:.4f}",
            "Gap": lambda x: f"{x:.4f}"
        }
    ))

    print("\n[说明]")
    print("ECE 越低表示模型置信度与实际命中率越匹配。")
    print("Brier Score 越低表示整体概率分布与真实标签更一致。")

    return df_results, reliability_tables


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 加载数据与模型...")

    # 1. 测试集
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    # 2. 加载 GRU checkpoint
    ckpt = torch.load(RL_DIR / "rl_baseline_v2_seed42.pt", map_location="cpu")
    t2id, l2id = ckpt["token2id"], ckpt["label2id"]

    # 3. 构造 y_te
    y_te = np.array([l2id[str(l).strip()] for l in test_df["true_label"]])

    # 4. 计算 GRU logits
    gru = PolicyGRU(len(t2id), 128, 128, ckpt["num_labels"]).to(device)
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()

    seqs = []
    for s in test_df["state"]:
        toks = [t2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-ckpt["max_len"]:]
        toks = toks + [0] * (ckpt["max_len"] - len(toks))
        seqs.append(toks)

    with torch.no_grad():
        Z_gru_te = gru(torch.tensor(seqs, dtype=torch.long, device=device)).cpu().numpy()

    # 5. 提取 LLM features
    print("[INFO] 提取 BGE 特征...")
    tk = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5", use_safetensors=True).to(device).eval()

    txts = test_df["llm_thinking_process"].fillna("None").tolist()
    feats = []

    for i in range(0, len(txts), 64):
        batch = tk(
            txts[i:i+64],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            cls = md(**batch)[0][:, 0]
            cls = F.normalize(cls, p=2, dim=1)
            feats.append(cls.cpu().numpy())

    H_te = np.vstack(feats)

    # 6. 计算 LLM logits
    llm_p = LLMOnlyNet(768, ckpt["num_labels"]).to(device)
    llm_p.load_state_dict(torch.load(LLM_CKPT_DIR / "llm_probe_seed42.pt", map_location=device))
    llm_p.eval()

    with torch.no_grad():
        Z_llm_te = llm_p(torch.tensor(H_te, dtype=torch.float32, device=device)).cpu().numpy()

    # 7. Global Logit Fusion
    alpha = 0.18
    Z_global_fused = (1 - alpha) * Z_gru_te + alpha * Z_llm_te

    # 8. 校准分析
    df_calib, reliability_tables = analyze_calibration(
        y_te, Z_gru_te, Z_llm_te, Z_global_fused, n_bins=10
    )

    # 9. 导出 CSV
    df_calib.to_csv(DATA_DIR / "calibration_summary.csv", index=False, encoding="utf-8-sig")
    reliability_tables["GRU Baseline"].to_csv(
        DATA_DIR / "calibration_bins_gru.csv", index=False, encoding="utf-8-sig"
    )
    reliability_tables["LLM Probe"].to_csv(
        DATA_DIR / "calibration_bins_llm.csv", index=False, encoding="utf-8-sig"
    )
    reliability_tables["Global Logit Fusion"].to_csv(
        DATA_DIR / "calibration_bins_fusion.csv", index=False, encoding="utf-8-sig"
    )

    print("\n[INFO] 校准分析完成，CSV 已导出。")