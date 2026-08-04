import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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
# 工具函数
# =========================================================
def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timed_run(fn, n_warmup=20, n_iters=100, desc=""):
    """
    计时一个无参数闭包 fn()，返回:
    - total_time_ms
    - avg_time_per_iter_ms
    """
    # warmup
    for _ in range(n_warmup):
        fn()
    sync_if_needed()

    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    sync_if_needed()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    avg_ms = elapsed_ms / n_iters
    return elapsed_ms, avg_ms


def build_sequences(state_series, token2id, max_len):
    seqs = []
    for s in state_series:
        toks = [token2id.get(t.strip(), 1) for t in str(s).split("||") if t.strip()][-max_len:]
        toks = toks + [0] * (max_len - len(toks))
        seqs.append(toks)
    return np.array(seqs, dtype=np.int64)


def make_batches(n_items, batch_size):
    indices = np.arange(n_items)
    return [indices[i:i + batch_size] for i in range(0, n_items, batch_size)]


# =========================================================
# 核心分析
# =========================================================
def benchmark_real_inference_cost(
    data_dir,
    rl_dir,
    llm_ckpt_dir,
    sample_size=256,
    batch_size=32,
    seed=42,
    alpha=0.18
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("🚀 真实模型推理成本分析 (Real Inference Cost Analysis)")
    print("=" * 90)
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Sample size : {sample_size}")
    print(f"[INFO] Batch size  : {batch_size}")
    print(f"[INFO] Alpha       : {alpha}")
    print()

    # -----------------------------------------------------
    # 1. 加载数据
    # -----------------------------------------------------
    print("[INFO] 加载测试集...")
    test_df = pd.read_csv(data_dir / "sim_test_llm_cot.csv")
    if sample_size > len(test_df):
        sample_size = len(test_df)

    sampled_df = test_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # -----------------------------------------------------
    # 2. 加载 GRU checkpoint 与模型
    # -----------------------------------------------------
    print("[INFO] 加载 GRU checkpoint...")
    ckpt = torch.load(rl_dir / "rl_baseline_v2_seed42.pt", map_location="cpu")
    token2id = ckpt["token2id"]
    label2id = ckpt["label2id"]
    max_len = ckpt["max_len"]
    num_labels = ckpt["num_labels"]

    gru = PolicyGRU(
        vocab_size=len(token2id),
        emb_dim=128,
        hidden_dim=128,
        num_labels=num_labels
    ).to(device)

    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()

    # -----------------------------------------------------
    # 3. 构造真实输入序列
    # -----------------------------------------------------
    print("[INFO] 构造 GRU 输入序列...")
    seqs_np = build_sequences(sampled_df["state"], token2id, max_len)
    seqs_t = torch.tensor(seqs_np, dtype=torch.long, device=device)

    # -----------------------------------------------------
    # 4. 加载 BGE 与 LLM probe
    # -----------------------------------------------------
    print("[INFO] 加载 BGE encoder 与 LLM probe...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    bge_model = AutoModel.from_pretrained(
        "BAAI/bge-base-zh-v1.5",
        use_safetensors=True
    ).to(device).eval()

    llm_probe = LLMOnlyNet(llm_dim=768, num_labels=num_labels).to(device)
    llm_probe.load_state_dict(torch.load(llm_ckpt_dir / "llm_probe_seed42.pt", map_location=device))
    llm_probe.eval()

    texts = sampled_df["llm_thinking_process"].fillna("None").astype(str).tolist()

    # -----------------------------------------------------
    # 5. 预先切 batch
    # -----------------------------------------------------
    batches = make_batches(len(sampled_df), batch_size)

    # -----------------------------------------------------
    # 6. 定义各组件计时闭包
    # -----------------------------------------------------
    # 6.1 GRU forward
    def run_gru_fullset():
        with torch.no_grad():
            for idx in batches:
                _ = gru(seqs_t[idx])

    # 6.2 BGE encoding
    def run_bge_fullset():
        with torch.no_grad():
            for idx in batches:
                batch_texts = [texts[i] for i in idx]
                batch = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                _ = bge_model(**batch)[0][:, 0]

    # 6.3 BGE encoding + normalize + collect
    def compute_bge_features():
        feats = []
        with torch.no_grad():
            for idx in batches:
                batch_texts = [texts[i] for i in idx]
                batch = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                cls = bge_model(**batch)[0][:, 0]
                cls = F.normalize(cls, p=2, dim=1)
                feats.append(cls)
        return torch.cat(feats, dim=0)

    # 先真实算一遍特征，供 probe / fusion 计时
    print("[INFO] 预计算一份 BGE 特征供 probe/fusion 测试...")
    H_te_t = compute_bge_features()

    # 6.4 LLM probe
    def run_probe_fullset():
        with torch.no_grad():
            for idx in batches:
                _ = llm_probe(H_te_t[idx])

    # 6.5 先真实算一遍 logits，供 fusion 计时
    print("[INFO] 预计算一份 GRU / Probe logits 供 fusion 测试...")
    with torch.no_grad():
        Z_gru_t = []
        for idx in batches:
            Z_gru_t.append(gru(seqs_t[idx]))
        Z_gru_t = torch.cat(Z_gru_t, dim=0)

        Z_llm_t = []
        for idx in batches:
            Z_llm_t.append(llm_probe(H_te_t[idx]))
        Z_llm_t = torch.cat(Z_llm_t, dim=0)

    # 6.6 Fusion
    def run_fusion_fullset():
        with torch.no_grad():
            _ = (1 - alpha) * Z_gru_t + alpha * Z_llm_t

    # 6.7 Full online pipeline: GRU + BGE + Probe + Fusion
    def run_online_fullset():
        fused_outputs = []
        with torch.no_grad():
            for idx in batches:
                # GRU
                z_gru = gru(seqs_t[idx])

                # BGE
                batch_texts = [texts[i] for i in idx]
                batch = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                cls = bge_model(**batch)[0][:, 0]
                cls = F.normalize(cls, p=2, dim=1)

                # Probe
                z_llm = llm_probe(cls)

                # Fusion
                z_fused = (1 - alpha) * z_gru + alpha * z_llm
                fused_outputs.append(z_fused)

        return fused_outputs

    # 6.8 Precomputed reasoning pipeline: GRU + Probe + Fusion
    def run_precomputed_fullset():
        fused_outputs = []
        with torch.no_grad():
            for idx in batches:
                z_gru = gru(seqs_t[idx])
                z_llm = llm_probe(H_te_t[idx])
                z_fused = (1 - alpha) * z_gru + alpha * z_llm
                fused_outputs.append(z_fused)
        return fused_outputs

    # -----------------------------------------------------
    # 7. 正式计时
    # -----------------------------------------------------
    print("[INFO] 开始计时...")

    gru_total_ms, gru_avg_iter_ms = timed_run(
        run_gru_fullset, n_warmup=10, n_iters=50, desc="GRU"
    )
    bge_total_ms, bge_avg_iter_ms = timed_run(
        run_bge_fullset, n_warmup=5, n_iters=20, desc="BGE"
    )
    probe_total_ms, probe_avg_iter_ms = timed_run(
        run_probe_fullset, n_warmup=10, n_iters=50, desc="Probe"
    )
    fusion_total_ms, fusion_avg_iter_ms = timed_run(
        run_fusion_fullset, n_warmup=10, n_iters=200, desc="Fusion"
    )
    precomp_total_ms, precomp_avg_iter_ms = timed_run(
        run_precomputed_fullset, n_warmup=5, n_iters=20, desc="Precomputed Fusion"
    )
    online_total_ms, online_avg_iter_ms = timed_run(
        run_online_fullset, n_warmup=3, n_iters=10, desc="Online Fusion"
    )

    # -----------------------------------------------------
    # 8. 计算 per-sample
    # -----------------------------------------------------
    def per_sample(avg_fullset_ms):
        return avg_fullset_ms / sample_size

    # 组件级
    gru_ps = per_sample(gru_avg_iter_ms)
    bge_ps = per_sample(bge_avg_iter_ms)
    probe_ps = per_sample(probe_avg_iter_ms)
    fusion_ps = per_sample(fusion_avg_iter_ms)

    # 系统级
    precomp_ps = per_sample(precomp_avg_iter_ms)
    online_ps = per_sample(online_avg_iter_ms)

    # 相对成本
    rel_precomp = precomp_ps / max(gru_ps, 1e-12)
    rel_online = online_ps / max(gru_ps, 1e-12)
    rel_llm_probe_stack = (bge_ps + probe_ps + fusion_ps) / max(gru_ps, 1e-12)

    # -----------------------------------------------------
    # 9. 输出表 1：组件级成本
    # -----------------------------------------------------
    df_components = pd.DataFrame([
        {
            "Component": "GRU Forward",
            "Avg Latency / Sample (ms)": gru_ps,
            "Relative to GRU": 1.0
        },
        {
            "Component": "BGE Encoding",
            "Avg Latency / Sample (ms)": bge_ps,
            "Relative to GRU": bge_ps / max(gru_ps, 1e-12)
        },
        {
            "Component": "LLM Probe MLP",
            "Avg Latency / Sample (ms)": probe_ps,
            "Relative to GRU": probe_ps / max(gru_ps, 1e-12)
        },
        {
            "Component": "Global Logit Fusion",
            "Avg Latency / Sample (ms)": fusion_ps,
            "Relative to GRU": fusion_ps / max(gru_ps, 1e-12)
        }
    ])

    # -----------------------------------------------------
    # 10. 输出表 2：系统级场景
    # -----------------------------------------------------
    df_system = pd.DataFrame([
        {
            "Method": "GRU Baseline",
            "Setting": "Online",
            "Avg Latency / Sample (ms)": gru_ps,
            "Relative Cost vs GRU": 1.0
        },
        {
            "Method": "LLM Semantic Stack",
            "Setting": "Encoding + Probe + Fusion only",
            "Avg Latency / Sample (ms)": bge_ps + probe_ps + fusion_ps,
            "Relative Cost vs GRU": rel_llm_probe_stack
        },
        {
            "Method": "Global Fusion",
            "Setting": "Precomputed reasoning/embedding",
            "Avg Latency / Sample (ms)": precomp_ps,
            "Relative Cost vs GRU": rel_precomp
        },
        {
            "Method": "Global Fusion",
            "Setting": "Online encoding included",
            "Avg Latency / Sample (ms)": online_ps,
            "Relative Cost vs GRU": rel_online
        }
    ])

    # -----------------------------------------------------
    # 11. 打印
    # -----------------------------------------------------
    print("\n" + "=" * 90)
    print("📊 表 1：组件级推理成本拆解 (Component-wise Inference Cost)")
    print("=" * 90)
    print(df_components.to_string(
        index=False,
        formatters={
            "Avg Latency / Sample (ms)": lambda x: f"{x:.4f}",
            "Relative to GRU": lambda x: f"{x:.2f}×"
        }
    ))

    print("\n" + "=" * 90)
    print("📊 表 2：系统级部署场景成本估算 (System-level Deployment Cost)")
    print("=" * 90)
    print(df_system.to_string(
        index=False,
        formatters={
            "Avg Latency / Sample (ms)": lambda x: f"{x:.4f}",
            "Relative Cost vs GRU": lambda x: f"{x:.2f}×"
        }
    ))

    print("\n[说明]")
    print("1. 上述结果基于真实测试样本、真实 checkpoint 和真实 reasoning 文本测得。")
    print("2. 'Precomputed reasoning/embedding' 表示 reasoning 文本或语义向量已离线缓存。")
    print("3. 'Online encoding included' 包含 BGE 在线编码成本，但不包含外部 LLM 生成 reasoning 文本的成本。")
    print("4. 若 reasoning 文本需要在线生成，则额外成本应单独报告；该成本通常远高于融合器本身。")

    # -----------------------------------------------------
    # 12. 导出 CSV
    # -----------------------------------------------------
    out1 = data_dir / "cost_breakdown_components.csv"
    out2 = data_dir / "cost_breakdown_system.csv"
    df_components.to_csv(out1, index=False, encoding="utf-8-sig")
    df_system.to_csv(out2, index=False, encoding="utf-8-sig")

    print(f"\n[INFO] 结果已导出:")
    print(f"  -> {out1}")
    print(f"  -> {out2}")

    return df_components, df_system


# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    DATA_DIR = Path(r"E:\desktop\project_only\project\data")
    RL_DIR = Path(r"E:\desktop\project_only\project\rl")
    LLM_CKPT_DIR = Path(r"E:\desktop\project_only\project\llm\checkpoints")

    benchmark_real_inference_cost(
        data_dir=DATA_DIR,
        rl_dir=RL_DIR,
        llm_ckpt_dir=LLM_CKPT_DIR,
        sample_size=256,   # 可改 128 / 256 / 512
        batch_size=32,     # 按显存调
        seed=42,
        alpha=0.18
    )