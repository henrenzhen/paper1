import pandas as pd
import re
from pathlib import Path

# =========================
# 0. 路径配置
# =========================
DATA_DIR = Path(r"E:\desktop\project_only\project\data")
files = {
    "Train": "sim_train_llm_cot.csv",
    "Val": "sim_val_llm_cot.csv",
    "Test": "sim_test_llm_cot.csv"
}

# 匹配 ATT&CK 技术的正则 (例如 T1059, T1059.001)
T_PATTERN = re.compile(r'T\d{4}(?:\.\d{3})?')


def audit_refined_contamination():
    print("\n" + "=" * 90)
    print("🔍 第四部分：精细化前瞻性暴露审计 (Refined Anticipatory Exposure Audit)")
    print("=" * 90)

    for name, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "llm_thinking_process" not in df.columns:
            continue

        cat1_restatement = 0  # 类别1：合法复述 (Label在State中，且在CoT中出现)
        cat2_anticipation = 0  # 类别2：可疑前瞻 (Label不在State中，但在CoT中出现)
        cat3_safe = 0  # 类别3：安全闭口 (Label不在State中，也不在CoT中)

        total_new_ids_mentioned = 0  # 统计大模型平均每条“凭空”脑补了几个新ID

        for _, row in df.iterrows():
            state = str(row['state'])
            cot = str(row['llm_thinking_process'])
            label = str(row['true_label']).strip()

            # 1. 提取 State 中的所有 Txxxx ID
            state_ids = set(T_PATTERN.findall(state))
            # 2. 提取 CoT 中的所有 Txxxx ID
            cot_ids = set(T_PATTERN.findall(cot))

            # 计算 CoT 中新增的 ID (不在 Prefix 中的)
            new_ids_in_cot = cot_ids - state_ids
            total_new_ids_mentioned += len(new_ids_in_cot)

            # 判定逻辑 (结合 ID 匹配和子串匹配以防万一)
            label_in_state = (label in state_ids) or (label in state)
            label_in_cot = (label in cot_ids) or (label.lower() in cot.lower())

            if label_in_state:
                if label_in_cot:
                    cat1_restatement += 1
            else:
                if label_in_cot:
                    cat2_anticipation += 1
                else:
                    cat3_safe += 1

        total_samples = len(df)
        print(f"\n[{name} 集] 深度审计结果 (总样本: {total_samples}):")
        print(
            f"  -> 类别 1 (合法复述): {cat1_restatement:<4} 样本 (占比 {(cat1_restatement / total_samples) * 100:.2f}%) - Label已在Prefix中，CoT提及属于总结历史。")
        print(
            f"  -> 类别 2 (前瞻暴露): {cat2_anticipation:<4} 样本 (占比 {(cat2_anticipation / total_samples) * 100:.2f}%) - Label是全新的，但被CoT提前点名。")
        print(
            f"  -> 类别 3 (安全闭口): {cat3_safe:<4} 样本 (占比 {(cat3_safe / total_samples) * 100:.2f}%) - Label是全新的，CoT未提前点名。")
        print(
            f"  * 附加指标: CoT 中平均每条样本引入的『全新 ATT&CK ID』数量为 {total_new_ids_mentioned / total_samples:.2f} 个。")


if __name__ == "__main__":
    audit_refined_contamination()