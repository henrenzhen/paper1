# 实验产物契约
状态：已建立，尚未运行实验。
## 强制文件
每个 E1–E9 实验目录必须保留并在运行时完整写入：
- `REPORT.md`：实验目的、方法、预注册判据、结果和唯一判定（通过 / 判死 / 不确定）。
- `results.csv`：机器可读的汇总指标、置信区间和校正前后 p 值。
- `predictions.jsonl`：逐样本原始预测；每行至少包含 sample_id、campaign_id、split、seed、model、alpha、true_label，以及 logits 或 top-20 完整排名。
- `config.json`：配置快照；至少包含随机种子、超参数、每 seed 的 alpha 候选与验证集选择结果、alpha 选择准则、模型名和精确版本、数据文件及哈希。
- `run.log`：从命令启动到结束的完整 stdout/stderr，不得只保留摘要。
## 固定实验规则
- Phase 0 冻结前不得运行 P1/P2。
- alpha 仅在验证集选择，每个 seed 独立选择；主准则为 MRR，除非任务另有明确规定。
- 主实验不得修改 test split；只有 E7 可按规定重做划分，且必须独立报告。
- Top-1 使用 McNemar 配对检验。
- MRR、Top-5、Macro-F1 使用按 campaign_id 聚类的 paired bootstrap，固定 10,000 次。
- 多重比较使用 Holm–Bonferroni，并同时报告原始和校正后 p 值。
- 负结果直接报告；命中判死条件时结论必须写“判死”。
## 占位文件说明
当前 E1–E9 中的文件仅用于锁定结构，均显式标记 `NOT_RUN`，不构成实验结果。运行某条实验时必须原子化替换其占位内容，且运行结束后立即停止并汇报。
