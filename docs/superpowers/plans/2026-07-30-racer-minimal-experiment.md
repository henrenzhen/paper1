# RACER 最小实验实施计划

> 本计划是 GSAD 两轮失败后的预注册分支；继续执行 root-disjoint、测试先行、锁定测试一次性访问和失败轮简记规则。

## Task 1：SAR-CTW 概率模型

- 在 `project/tests/test_racer_context_tree.py` 先写失败测试：零支持回退、独立 root 支持决定收缩、复制单一长 root 不改变概率、概率归一、真实数据无标签泄漏 smoke test。
- 在 `project/experiments/gsad/context_tree.py` 实现训练、推理元数据和确定性配置。
- 运行单文件测试及全部 `test_gsad*.py` / `test_racer*.py` 回归。

## Task 2：MC-RCU 名次 conformal 集合

- 在 `project/tests/test_racer_rank_conformal.py` 先写失败测试：确定性名次、有限样本分位数、专家并集等价性、非空集合、calibration/prediction ID 隔离。
- 在 `project/experiments/gsad/rank_conformal.py` 实现多专家分数、beta 配置、校准器和审计。
- 对 global APS 做同一 calibration split 的单元级对照。

## Task 3：嵌套 root-OOF 集成

- 在 `project/tests/test_racer_development_integration.py` 先写失败测试：外层 root 覆盖完整、角色互斥、锁定根不可见、强基线与候选逐行配对、非 self 指标、负对照、不可覆盖输出目录。
- 在 `project/experiments/gsad/run_racer_development.py` 实现 5 折开发 OOF、validation 选参、calibration 校准、2,000 次 paired root bootstrap、门槛与审计文件。
- 候选开发失败则只更新迭代简表；通过且负对照失败才写冻结令牌。

## Task 4：开发验证与消融

- 正式运行 `racer`，检查排序门槛、集合门槛、非 self 切片及负对照。
- 运行 fixed-weight、no-root-balance、single-expert-rank、no-SAR-expert 四个核心消融。
- 未通过则根据失败部件进入下一候选，仍不访问锁定测试。

## Task 5：通过后的单次锁定与外部压力测试

- 只有冻结令牌有效时实现并运行一次 locked SIM test。
- 随后在 CTID 9 actor 弱标签集上做不调参的 exploratory actor-macro 压力测试。
- 最终详细交付包括代码、配置、哈希、命令、完整指标、bootstrap、消融、负对照、限制和文献差异；失败轮仅保留简表。

