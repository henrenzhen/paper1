import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"E:\desktop\project_only\project\data")

print("[INFO] 开始进行数据行级对齐热修复...")

for split in ["train", "val", "test"]:
    cot_path = DATA_DIR / f"sim_{split}_llm_cot.csv"
    nocot_path = DATA_DIR / f"sim_{split}_llm_no_cot.csv"

    if not cot_path.exists() or not nocot_path.exists():
        continue

    df_cot = pd.read_csv(cot_path)
    df_nocot = pd.read_csv(nocot_path)

    # 提取 No-CoT 中真正有价值的 LLM 输出，做一个基于 sequence_id 的字典映射
    mapping_think = dict(zip(df_nocot['sequence_id'], df_nocot['llm_thinking_process']))
    mapping_pred = dict(zip(df_nocot['sequence_id'], df_nocot['predicted_next_ttps']))

    # 完全克隆 CoT 的母表结构（保证所有的 label 和顺序 100% 绝对一致）
    df_nocot_aligned = df_cot.copy()

    # 把 No-CoT 的文本按照正确的 sequence_id 映射进去
    df_nocot_aligned['llm_thinking_process'] = df_cot['sequence_id'].map(mapping_think)
    df_nocot_aligned['predicted_next_ttps'] = df_cot['sequence_id'].map(mapping_pred)

    # 检查是否有漏掉的映射
    missing = df_nocot_aligned['llm_thinking_process'].isna().sum()
    if missing > 0:
        print(f"[!] {split} 发现 {missing} 条未对齐样本，已安全兜底填充。")
        df_nocot_aligned['llm_thinking_process'].fillna("No reasoning.", inplace=True)
        df_nocot_aligned['predicted_next_ttps'].fillna("[]", inplace=True)

    # 覆盖保存为 No-CoT 修复版
    df_nocot_aligned.to_csv(nocot_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] {split} 表重新对齐完成！行数: {len(df_nocot_aligned)}")

print("\n[🎉 修复完毕] 请现在直接重新运行 run_3way_ablation.py！")