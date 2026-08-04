import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 0. 配置路径
# =========================
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
files = {
    "Train": "sim_train_llm_cot.csv",
    "Val": "sim_val_llm_cot.csv",
    "Test": "sim_test_llm_cot.csv"
}


def load_and_preprocess(name, filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f" [!] 警告: 找不到文件 {path}")
        return None
    df = pd.read_csv(path)

    # 统一清洗数据 (处理空格和空值)
    if "sequence_id" in df.columns:
        df["sequence_id"] = df["sequence_id"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    df["true_label"] = df["true_label"].astype(str).str.strip()

    # 优化计算 prefix 长度 (避免 apply axis=1)
    df["prefix_len"] = df["state"].apply(lambda x: len([t for t in str(x).split("||") if t.strip()]))
    return df


# =========================
# 1. 基础预处理
# =========================
dfs = {name: load_and_preprocess(name, fname) for name, fname in files.items()}
dfs = {k: v for k, v in dfs.items() if v is not None}

# =========================
# 2. 基本统计表 (Dataset Statistics)
# =========================
print("\n" + "=" * 90)
print("📊 第一部分：数据集基本统计 (Dataset Statistics)")
print("=" * 90)

stats_rows = []
for name, df in dfs.items():
    stats = {
        "Split": name,
        "Samples": len(df),
        "Unique Seq_IDs": df["sequence_id"].nunique() if "sequence_id" in df.columns else "N/A",
        "Labels": df["true_label"].nunique(),
        "Unique States": df["state"].nunique(),
        "Unique (S,L) Pairs": df[["state", "true_label"]].drop_duplicates().shape[0],
        "Prefix Mean": f"{df['prefix_len'].mean():.2f}",
        "Prefix Median": int(df['prefix_len'].median()),
        "Prefix Max": df['prefix_len'].max()
    }
    stats_rows.append(stats)

print(pd.DataFrame(stats_rows).to_string(index=False))

# =========================
# 3. 防泄漏检查 (Data Leakage Audit)
# =========================
print("\n" + "=" * 90)
print("🛡️ 第二部分：防泄漏检查 (Data Leakage Audit)")
print("=" * 90)


def check_intersection(set_a, set_b, label_a, label_b, title):
    inter = set_a.intersection(set_b)
    denom = max(1, min(len(set_a), len(set_b)))  # 防止除以 0
    ratio = len(inter) / denom * 100
    print(f" [{title}] {label_a} ∩ {label_b} -> 重叠数量: {len(inter):<4} | 占较小集合比例: {ratio:.2f}%")
    return inter


keys = list(dfs.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        n1, n2 = keys[i], keys[j]
        d1, d2 = dfs[n1], dfs[n2]
        print(f"\n--- 交叉核对: {n1} vs {n2} ---")

        # 1. Sequence ID 检查 (最严格的物理隔离)
        if "sequence_id" in d1.columns and "sequence_id" in d2.columns:
            id1 = set(d1["sequence_id"])
            id2 = set(d2["sequence_id"])
            check_intersection(id1, id2, n1, n2, "Sequence ID Overlap")

        # 2. State 检查 (允许少量重叠，代表相似攻击上下文)
        s1 = set(d1["state"])
        s2 = set(d2["state"])
        check_intersection(s1, s2, n1, n2, "State Overlap      ")

        # 3. (State, True_Label) 检查 (绝对不能重叠的强泄漏信号)
        p1 = set(zip(d1["state"], d1["true_label"]))
        p2 = set(zip(d2["state"], d2["true_label"]))
        check_intersection(p1, p2, n1, n2, "(S,L) Pair Overlap ")

        # 4. Reasoning Text 检查 (过滤固定模板和极短文本)
        if "llm_thinking_process" in d1.columns and "llm_thinking_process" in d2.columns:
            def get_valid_reasoning(series):
                s = series.fillna("").astype(str).str.strip()
                return set(s[(s.str.len() >= 20) & (~s.isin(["No reasoning.", "[ERROR]"]))])


            r1 = get_valid_reasoning(d1["llm_thinking_process"])
            r2 = get_valid_reasoning(d2["llm_thinking_process"])
            check_intersection(r1, r2, n1, n2, "Filtered CoT Overlap")

# =========================
# 4. 标签 ID 泄露抽查
# =========================
print("\n" + "=" * 90)
print("🔍 第三部分：直接标签 ID 泄露检查 (Direct Label-ID Leakage Check in CoT)")
print("=" * 90)
print("* 注: 此项仅检查 true_label 字符串是否直接暴露于思维链中，不包含 technique 别名或描述映射。")

for name, df in dfs.items():
    if "llm_thinking_process" in df.columns:
        # 过滤出 true_label 长度 > 3 的有效样本进行匹配
        mask = df.apply(
            lambda x: len(str(x['true_label']).strip()) > 3 and
                      str(x['true_label']).lower().strip() in str(x['llm_thinking_process']).lower(),
            axis=1
        )
        leak_count = mask.sum()
        ratio = (leak_count / len(df)) * 100
        print(f" [{name}] CoT 直接暴露 Label ID 的样本数: {leak_count} / {len(df)} ({ratio:.2f}%)")
print("\n")