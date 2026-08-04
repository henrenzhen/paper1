import numpy as np
import pandas as pd

# =========================================================
# 配置区
# 说明：
# 1) 建议所有指标都统一用 0~1 小数表示，而不是百分比
# 2) 多 seed 汇总表统一保留 3 位小数，更适合论文展示
# 3) 标准差采用样本标准差 (ddof=1)
# =========================================================

def format_multiseed(results_list, decimals=3):
    """
    将多个 seed 的结果格式化为 Mean ± Std 字符串
    例如: [0.544, 0.548, 0.541] -> '0.544 ± 0.004'
    """
    arr = np.asarray(results_list, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_multiseed_summary(data_dict, metric_order=None, decimals_map=None):
    """
    根据多 seed 结果字典生成汇总表 DataFrame

    参数:
        data_dict: dict
            格式示例:
            {
                "GRU Baseline": {
                    "Top-1": [0.544, 0.541, 0.548, 0.543, 0.546],
                    "MRR":   [0.654, 0.651, 0.658, 0.653, 0.656]
                },
                ...
            }

        metric_order: list[str] or None
            指标列顺序，例如 ["Top-1", "Top-5", "MRR", "Macro-F1"]
            若为 None，则自动使用每个模型里出现过的所有指标，按首次出现顺序排列

        decimals_map: dict or None
            可为不同指标设置不同保留位数
            例如 {"Top-1": 3, "MRR": 3, "Macro-F1": 4}
            若为 None，则默认全部 3 位

    返回:
        df_summary: pandas.DataFrame
    """
    if decimals_map is None:
        decimals_map = {}

    # 自动收集指标顺序
    if metric_order is None:
        metric_order = []
        seen = set()
        for _, metrics in data_dict.items():
            for metric_name in metrics.keys():
                if metric_name not in seen:
                    metric_order.append(metric_name)
                    seen.add(metric_name)

    records = []
    for model_name, metrics in data_dict.items():
        record = {
            "Model / Variant": model_name,
            "N Seeds": len(next(iter(metrics.values())))
        }

        for metric_name in metric_order:
            if metric_name in metrics:
                dec = decimals_map.get(metric_name, 3)
                record[metric_name] = format_multiseed(metrics[metric_name], decimals=dec)
            else:
                record[metric_name] = "-"

        records.append(record)

    df_summary = pd.DataFrame(records)
    return df_summary


if __name__ == "__main__":
    # =========================================================
    # 在这里填入你的真实多 seed 数据
    # 注意：下面只是示例格式
    # 推荐统一使用 0~1 小数，而不是百分数
    # =========================================================
    data = {
        "GRU Baseline": {
            "Top-1":    [0.5444, 0.5412, 0.5480, 0.5435, 0.5460],
            "MRR":      [0.6540, 0.6510, 0.6580, 0.6530, 0.6560]
        },
        "Global Logit Fusion (CoT)": {
            "Top-1":    [0.5620, 0.5605, 0.5640, 0.5615, 0.5635],
            "MRR":      [0.6720, 0.6700, 0.6750, 0.6710, 0.6740]
        },
        "Global Logit Fusion (No-CoT)": {
            "Top-1":    [0.5510, 0.5495, 0.5530, 0.5485, 0.5505],
            "MRR":      [0.6600, 0.6580, 0.6630, 0.6570, 0.6610]
        },
        "Transformer Baseline": {
            "Top-1":    [0.5310, 0.5150, 0.5420, 0.5280, 0.5390],
            "MRR":      [0.6400, 0.6250, 0.6480, 0.6350, 0.6450]
        },
        "Dynamic Gating Fusion": {
            "Top-1":    [0.5420, 0.5380, 0.5450, 0.5410, 0.5395],
            "MRR":      [0.6500, 0.6480, 0.6530, 0.6490, 0.6470]
        }
    }

    # 你可以在这里调整指标显示顺序
    metric_order = ["Top-1", "MRR"]

    # 你可以为不同指标设置不同小数位
    decimals_map = {
        "Top-1": 3,
        "MRR": 3,
    }

    df_summary = build_multiseed_summary(
        data_dict=data,
        metric_order=metric_order,
        decimals_map=decimals_map
    )

    print("=" * 78)
    print("📊 论文标准多 Seed 汇总表 (Mean ± Std)")
    print("=" * 78)
    print(df_summary.to_markdown(index=False))

    # 导出 CSV
    output_csv = "multiseed_summary_table.csv"
    df_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 汇总表已导出到: {output_csv}")