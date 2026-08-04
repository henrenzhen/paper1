import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# =========================
# 0. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\desktop\project_only\project\data")


# =========================
# 1. 核心马尔可夫模型定义
# =========================
class MarkovBaseline:
    def __init__(self):
        self.global_prior = Counter()
        self.markov_1st = defaultdict(Counter)
        self.markov_2nd = defaultdict(Counter)

    def fit(self, df):
        for _, row in df.iterrows():
            seq_str = str(row["state"])
            target = str(row["true_label"]).strip()

            # 统计全局先验
            self.global_prior[target] += 1

            # 解析序列
            items = [x.strip() for x in seq_str.split("||") if x.strip()]
            if not items:
                continue

            # 一阶状态 (只看最后一个动作)
            state_1 = items[-1]
            self.markov_1st[state_1][target] += 1

            # 二阶状态 (看最后两个动作)
            if len(items) >= 2:
                state_2 = (items[-2], items[-1])
                self.markov_2nd[state_2][target] += 1

        # 预计算全局 Top-K 备用
        self.global_topk = [item for item, _ in self.global_prior.most_common()]

    def predict(self, df, order=1):
        y_true = []
        preds_list = []

        for _, row in df.iterrows():
            seq_str = str(row["state"])
            target = str(row["true_label"]).strip()
            y_true.append(target)

            items = [x.strip() for x in seq_str.split("||") if x.strip()]
            preds = []

            # 【修复】严格分支隔离
            if order == 0:
                preds = []  # 纯靠后面的 fallback 填补全局最高频

            elif order == 1:
                if len(items) >= 1:
                    state_1 = items[-1]
                    if state_1 in self.markov_1st:
                        preds = [k for k, _ in self.markov_1st[state_1].most_common()]

            elif order == 2:
                # 先尝试二阶
                if len(items) >= 2:
                    state_2 = (items[-2], items[-1])
                    if state_2 in self.markov_2nd:
                        preds = [k for k, _ in self.markov_2nd[state_2].most_common()]
                # 二阶未命中或长度不够，平滑回退到一阶
                if not preds and len(items) >= 1:
                    state_1 = items[-1]
                    if state_1 in self.markov_1st:
                        preds = [k for k, _ in self.markov_1st[state_1].most_common()]

            # --- Fallback: 补齐缺失的候选项 ---
            seen = set(preds)
            for fallback_item in self.global_topk:
                if fallback_item not in seen:
                    preds.append(fallback_item)
                    seen.add(fallback_item)

            preds_list.append(preds)

        return y_true, preds_list

    # 【新增】统计未命中率 (OOV Coverage)
    def coverage_stats(self, df):
        total = len(df)
        hit_1, hit_2 = 0, 0
        for _, row in df.iterrows():
            items = [x.strip() for x in str(row["state"]).split("||") if x.strip()]
            if len(items) >= 1 and items[-1] in self.markov_1st:
                hit_1 += 1
            if len(items) >= 2 and (items[-2], items[-1]) in self.markov_2nd:
                hit_2 += 1
        return hit_1 / total if total > 0 else 0, hit_2 / total if total > 0 else 0


# =========================
# 2. 评估函数 (新增 Weighted-F1)
# =========================
def evaluate_metrics(y_true, preds_list):
    t1, t5, mrr = 0.0, 0.0, 0.0
    n = len(y_true)
    y_pred_top1 = []

    for yt, preds in zip(y_true, preds_list):
        top1_pred = preds[0]
        y_pred_top1.append(top1_pred)

        # Top-1
        if top1_pred == yt:
            t1 += 1
        # Top-5
        if yt in preds[:5]:
            t5 += 1
        # MRR
        try:
            rank = preds.index(yt) + 1
            mrr += 1.0 / rank
        except ValueError:
            pass

    macro_f1 = f1_score(y_true, y_pred_top1, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_top1, average='weighted', zero_division=0)

    return t1 / n, t5 / n, mrr / n, macro_f1, weighted_f1


# =========================
# 3. 主流程
# =========================
def main():
    print("[INFO] 正在加载数据...")
    train_df = pd.read_csv(DATA_DIR / "sim_train_llm_cot.csv")
    val_df = pd.read_csv(DATA_DIR / "sim_val_llm_cot.csv")
    test_df = pd.read_csv(DATA_DIR / "sim_test_llm_cot.csv")

    print("[INFO] 正在拟合马尔可夫转移矩阵 (Train集)...")
    model = MarkovBaseline()
    model.fit(train_df)

    # 打印 Test 集 OOV 覆盖度
    cov_1st, cov_2nd = model.coverage_stats(test_df)
    print(f"\n[🔬 状态空间覆盖率分析 (Test集)]")
    print(f"  -> 1st-Order 命中率 (单状态在Train见过): {cov_1st * 100:.2f}%")
    print(f"  -> 2nd-Order 命中率 (双状态对在Train见过): {cov_2nd * 100:.2f}%")

    # --- 阶段 1: 在 Val 集上选定最优模型 (Best Markov) ---
    print("\n[INFO] 正在 Val 集上进行超参数(阶数)选择...")
    best_val_mrr = -1.0
    best_order = 1

    for order in [0, 1, 2]:
        y_val_true, preds_val = model.predict(val_df, order=order)
        metrics_val = evaluate_metrics(y_val_true, preds_val)
        val_mrr = metrics_val[2]
        print(f"  -> Order {order} | Val MRR: {val_mrr:.4f}")
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_order = order

    print(f"[✅ 模型选择] 最佳马尔可夫阶数判定为: Order {best_order}")

    # --- 阶段 2: 在 Test 集上跑全部结果 ---
    print("\n[INFO] 正在 Test 集上评估所有变体...")
    y_test_true, preds_test_0 = model.predict(test_df, order=0)
    y_test_true, preds_test_1 = model.predict(test_df, order=1)
    y_test_true, preds_test_2 = model.predict(test_df, order=2)

    metrics_0 = evaluate_metrics(y_test_true, preds_test_0)
    metrics_1 = evaluate_metrics(y_test_true, preds_test_1)
    metrics_2 = evaluate_metrics(y_test_true, preds_test_2)

    # 取出 Best Markov 对应的 Test 指标
    if best_order == 0:
        best_metrics = metrics_0
    elif best_order == 1:
        best_metrics = metrics_1
    else:
        best_metrics = metrics_2

    # 打印超级战报
    print(f"\n{'=' * 110}")
    print(f"🔥 传统统计学基线 (Markov Chain) 终极战报 | 测试样本数: {len(y_test_true)}")
    print(f"{'=' * 110}")
    header = f"{'Method':<25} | {'Top-1':<10} | {'Top-5':<10} | {'MRR':<10} | {'Macro-F1':<10} | {'Weight-F1':<10}"
    print(header)
    print("-" * 110)

    def print_row(name, m):
        print(f"{name:<25} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f} | {m[4]:<10.4f}")

    print_row("Global Prior (0-Order)", metrics_0)
    print_row("1st-Order Markov", metrics_1)
    print_row("2nd-Order Markov", metrics_2)
    print("-" * 110)
    print_row(f"👑 Best Markov (Order {best_order})", best_metrics)
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()