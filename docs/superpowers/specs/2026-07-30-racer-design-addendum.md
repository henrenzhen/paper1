# RACER 设计补充：GSAD 失败后的排序与集合双阶段候选

日期：2026-07-30  
状态：开发集预注册；锁定测试继续封存

## 1. 失败证据与转向条件

GSAD-Core 与 GSAD-Shift 在同一 133-root、10,555-row、五折 root-disjoint OOF 上均失败。两者的基础 root-macro Top-1 为 46.764%，90% clustered APS 集合大小为 33–125、均值 72.967，没有单例；DAG 展示压缩把叶等价集合进一步扩大到 91.059。学习型安全分数不改变标签排序，因此五折 exact 阈值均禁用。

这满足原设计的停止条件：不再调整安全门控、DAG 压缩预算或 clustered APS 参数，转而直接改进基础排序和原始集合构造。

## 2. 新候选概述

新候选命名为 RACER（Root-Adaptive Context Ensemble with Rank-conformal decoding），包含两个可独立否决的算法部件：

1. SAR-CTW：支持度自适应、root-balanced 的上下文树概率模型；
2. MC-RCU：多上下文专家的名次 conformal 并集。

只有两个部件各自通过开发门槛，才允许组成最终候选；任一失败都不能接触锁定测试。

## 3. SAR-CTW

对长度为 l 的后缀上下文 c，先对每个支持该上下文的训练 root 求条件分布，再跨 root 等权平均：

\[
\bar p_l(y\mid c)=|G_c|^{-1}\sum_{g\in G_c}N_g(c,y)/N_g(c).
\]

从短上下文向长上下文递归收缩：

\[
p_l(y\mid c)=\omega_c\bar p_l(y\mid c)+(1-\omega_c)p_{l-1}(y\mid suffix(c)),
\quad \omega_c=|G_c|/(|G_c|+\kappa).
\]

零支持上下文完全回退；所有概率包含固定总质量 Dirichlet 平滑。最大上下文长度和 kappa 只在 validation roots 上选择，目标依次为 root-macro Top-1、MRR、NLL，并用确定性配置键打破平局。

## 4. MC-RCU

专家族由预注册的 root-balanced unigram、固定二阶/三阶回退、tactic-aware 与 SAR-CTW 组成。对标签 y 定义：

\[
A(x,y)=\min_m rank_m(x,y)/\beta_m.
\]

beta 只能在 validation roots 上从小型固定网格选择；独立 calibration roots 上用有限样本分位数得到 q。预测集合为：

\[
C(x)=\{y:A(x,y)\le q\}=\cup_m Top_{\lfloor q\beta_m\rfloor}(p_m).
\]

名次按概率降序、标签索引升序确定性打破平局。集合至少包含总体最优专家的 Top-1。ATT&CK DAG 只作为展示映射与层级误差分析，不再允许扩张统计叶集合。

## 4.1 RACER-OP 排序扩展

RACER-Core 开发轮仅有 MRR 门槛失败后，集合分支冻结不再调参。排序分支增加低容量的受约束意见池。令 SAR-CTW 为左专家、validation 已选的固定上下文模型之一为右专家，只允许两种单纯形池：

\[
p_{lin}=w p_{sar}+(1-w)p_m,
\]

\[
p_{log}(y)\propto p_{sar}(y)^w p_m(y)^{1-w}.
\]

其中右专家只能从 baseline、unigram、bigram、trigram、tactic-aware 中选择，w 只能取 `{0,0.1,...,1.0}`。专家、池类型与 w 在 validation roots 上按 root-macro MRR、Top-1、规范配置键依次选择；calibration 和 outer roots 不参与。该扩展不引入逐样本门控，也不能访问安全标签。

## 5. 开发门槛

排序门槛（全部满足）：

- 相对最强固定上下文基线，root-macro Top-1 提升至少 1.0 个百分点；
- root-macro MRR 提升至少 0.01；
- 两项 paired root-bootstrap 95% CI 下界均大于 0；
- Hit@5 不下降超过 0.5 个百分点；
- 非 self 难度切片的 Top-1 与 MRR 均不下降。

集合门槛（全部满足）：

- 行覆盖和 root-macro 覆盖均在 88%–92%；
- 相对 global APS，覆盖差绝对值不超过 0.5 个百分点；
- 平均集合大小至少缩小 5%，paired root-bootstrap 95% CI 下界大于 0；
- full-set rate 不增加，tail-label 覆盖不下降超过 2 个百分点；
- 至少两个核心消融失败或显著更差：固定权重替代支持收缩、去掉 root 均衡、单专家 rank-conformal、去掉 SAR-CTW 专家。

负对照：置换训练 target 后，排序与集合 PRIMARY 均不得通过。

## 6. 数据伪影审计

开发数据中 parent target 等于最后 parent 状态的比例约 38%，而 raw technique 真自重复不足 1%。主要原因是多个 sub-technique 折叠到同一 parent。所有正式结果必须同时报告：

- 全样本；
- target_parent != last_parent 的非 self 切片；
- self/non-self 比例及 raw-ID 对照。

禁止把 parent copy 增益解释为攻击推理能力。

## 7. 冻结与停止

开发期继续仅使用 133 个开发 roots。20 个锁定 SIM roots 与 9 个 CTID actor roots保持不可见。开发 PRIMARY 通过且置换负对照失败后，写入包含配置、数据摘要、ATT&CK 版本和代码清单摘要的冻结令牌；随后锁定测试只能执行一次。
