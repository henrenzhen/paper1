# SECRYPT 2026 clean-room 划分审计

本目录只保存本项目独立实现的审计结果，不保存或分发 Raj et al. 的源码、ATT&CK v16 工作簿、生成链明细或模型权重。

## 已冻结运行

当前冻结结果：[`20260807_split_audit_v1/report.md`](20260807_split_audit_v1/report.md)。

关键边界：

- `P-pair` 和 `P-campaign` 都是公开构造数据上的 `H=1` 单标签 Accuracy；
- `P-source` 才是本项目三源 future-3 Top-5 主任务；
- 三者任务和指标不同，禁止直接比较绝对分数；
- 当前运行只含 `FREQ/PREFIX/M1/M2`，不含 LSTM、Hybrid 或 LLM。

## 复算命令

准备公开仓库固定 commit 的 ATT&CK v16 工作簿。若原文件为 `.xls`，先做无语义内容改动的 `.xlsx` 格式转换，并把原文件与转换文件同时传入以记录两个哈希。

```bash
PYTHONHASHSEED=0 python project/data_v4/scripts/audit_secrypt_split_protocols.py \
  --attack-workbook /absolute/path/enterprise-attack-v16.0.xlsx \
  --original-workbook /absolute/path/enterprise-attack-v16.0.xls \
  --output-dir project/data_v4/repro_secrypt/<new_run_id> \
  --external-commit e188cc6ec96df0288470380dbafccda1591e2c95
```

脚本拒绝复用已存在的输出目录，以防覆盖原始审计。必须在进程启动前设置 `PYTHONHASHSEED=0`；在脚本内部设置无效。

## 文件说明

| 文件 | 内容 |
|---|---|
| `manifest.json` | 输入/脚本哈希、环境、冻结配置、构造计数 |
| `summary.csv` | 协议 × 聚合方式 × 方法的 Accuracy |
| `split_diagnostics.csv` | 完全重复率、prefix 覆盖率、campaign 重合率 |
| `campaign_loco.csv` | 33 个留出 campaign 的逐折结果 |
| `report.md` | 面向人工阅读的中文摘要与解读边界 |
| `stdout.log` | 完整运行日志 |

## 引用边界

正式论文 DOI：<https://doi.org/10.5220/0015075400004103>。

`report.md` 中的重建 pair 数、重复率和 LOCO 基线均是本项目独立复算，不得写成 Raj et al. 原论文报告的数值。
