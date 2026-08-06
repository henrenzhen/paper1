# 30 条 LLM 语义试点可行性筛查 v1

## 定位

本筛查是**探索性、解盲、非确认性**分析，只回答现有 30 条生成结果是否包含值得继续全量实验的初步信号。它不能替代 v4.2 的 groundedness 人工盲审，也不能证明 B1 原论文方法成立。

一旦查看本筛查结果，这 30 条即被视为开发/筛查样本。若后续发布新协议并用本筛查决定方法、指标或是否继续，确认性结果必须排除这 30 条，并披露这一适应性决策。

## 固定输入

- `project/data_v4/external_reasoning/pilot/pilot_sample_30.csv`
- `project/data_v4/external_reasoning/pilot/runs/20260806T081038Z_da72df77/pilot_raw_results.csv`
- 三个固定 184 类闭集文件
- `project/data_v2/core/rl_label_vocab.csv`

唯一键固定为 `(source, campaign_id, prefix_len)`。使用 intention-to-treat：无效 LLM Top-5 计为 B0 失败；B2 对该行回退为 A。

## 方法

1. **A0：**只由外层两个训练来源的标签频率构造，固定 184 类与总平滑质量 `alpha_s=0.1`。
2. **A：**v4.2 三层插值主干，order-2 / order-1 / unigram 权重 `0.5/0.3/0.2`，只对可用层重新归一化。
3. **B0：**DeepSeek 返回的五个候选及原顺序；只计算 Top-1、Hit@5、MRR@5。
4. **B2：**v4.2 固定排名先验与 A 的 log-opinion pooling。`lambda=0,0.05,...,1`，只在两个外层训练来源之间做 inner-LODO，以 campaign-macro Top-1 选择；并列取最小 lambda。外层留出来源的 10 条不得参与 lambda 选择。

不运行 B1：每个外层折只有 20 条 pilot 训练 reasoning，却要拟合 184 类 probe，结果不具可解释性。

## 指标与解释

每个来源恰好 10 条、来自 10 个不同 campaign，因此在该 pilot 子集上 row-micro 与 campaign-macro 数值相同。报告 Top-1、Hit@5、MRR；B0 明确标记为 MRR@5。

本筛查不预设“成立”阈值，不执行显著性检验，不根据结果修改 prompt、模型或融合规则。结果只用于评估继续全量实验的信息价值。

## 输出

输出到最终生成 run 下的 `exploratory_feasibility_screen_v1/`，包括：

- `predictions.csv`
- `results.json`
- `inner_lambda_selection.csv`
- `manifest.json`
- `stdout.log`

该目录含 true label，严禁交给 groundedness 盲审者。
