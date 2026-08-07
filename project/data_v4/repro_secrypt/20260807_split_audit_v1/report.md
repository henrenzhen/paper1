# SECRYPT 2026 划分协议审计

> 以下数值均为本仓库对公开材料的 clean-room 独立复算，不是 Raj et al. 原论文报告的结果。

## 冻结来源

- 公开仓库 commit：`e188cc6ec96df0288470380dbafccda1591e2c95`
- `PYTHONHASHSEED`: `0`
- 原始 `.xls` SHA-256：`d8784efcf9d264e9a526574235d23e52d727d1d87b5f86208b51409d60afd80a`
- 转换后 `.xlsx` SHA-256：`c2f9d6fa21ec808e3e34d65d219539baa07856afc4c23807e6c21b092330cfcc`
- 审计脚本 SHA-256：`cbc56e59971575ea4349c05691ee6a885cf4c95c82a98b497be509cddb63fb46`

## released-order 数据重建

| campaign | 生成链 | occurrence | pair | 唯一 `(prefix,target)` | 词表 |
|---:|---:|---:|---:|---:|---:|
| 33 | 4849 | 132621 | 127772 | 21722 | 239 |

重建 pair 数严格等于 `sum(chain_length-1)`。正式论文表格写 128,413，与本审计相差641；必须披露，不能静默修正。

## 主随机 80/20 划分诊断

| 训练行 | 测试行 | `(prefix,target)` 完全重复 | prefix 覆盖 | 测试行所属 campaign 在训练出现 |
|---:|---:|---:|---:|---:|
| 102217 | 25555 | 87.5954% | 91.4185% | 100.0000% |

## 确定性基线 Accuracy

| 方法 | pair 80/20 | campaign LOCO macro | campaign LOCO pooled |
|---|---:|---:|---:|
| FREQ | 0.0282 | 0.0404 | 0.0291 |
| PREFIX | 0.7501 | 0.0456 | 0.0323 |
| M1 | 0.3773 | 0.1006 | 0.1123 |
| M2 | 0.6915 | 0.1015 | 0.1128 |

`PREFIX` 是查表诊断，不是论文候选预测模型。随机 pair 与 campaign-LOCO 的差距量化了同 campaign 排列复用对任务难度的影响。

## 解读边界

- 不得把这些复算值写成 SECRYPT 原论文结果。
- 不得把这些 `H=1` Accuracy 与本项目 future-3 NDCG@5/Recall@5 直接比较。
- 本次运行不含 LSTM、Hybrid、LLM、API 请求、token 或付费推理。
- canonical-order 只作敏感性检查；固定 hash seed 的 released-order 才是公开代码路径主审计。
