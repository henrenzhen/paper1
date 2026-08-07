# Procedure-Aware Cross-Source ATT&CK Recommendation：研究路线与后续工作

> 记录日期：2026-08-06；最近核验：2026-08-07
> 状态：三源 `P-source` 主实验、30条开发试点、784条正式DeepSeek生成、HM+S/HM+P/HM+T/HM+ST和全部预注册统计均已完成；核心融合主张已按预注册规则判定为缺乏证据。
> 最近修订：DeepSeek直接Top-5 `B0` 的来源等权NDCG@5为0.2028；事后探索`T+B0`为0.2053，相对B0仅+0.0025且95% CI [-0.0007,+0.0078]。Unit42官方85个bundle中的153个campaign已完成审计，0个具有合格的技术执行顺序，不能直接升级为第四折。当前证据支持“LLM直接排序含跨来源信号”，不支持现有摘要probe或统计主干提供稳定互补增益。
> 性质：本文档记录论文级路线、决策依据与执行顺序；具体数据哈希、超参数、API 参数和统计实现以冻结实验协议为准。

## 1. 一句话路线

> 以 SECRYPT 2026 的 Hybrid LSTM–Markov 作为最近邻**架构基线**，在严格跨来源 LODO 的真实攻击流上，检验 LLM 对 procedure-level 行为文本的语义建模，能否为大量未见技术转移提供可迁移的短窗口候选排序信号。

这里不预设 SECRYPT 方法是“强基线”或 SOTA。只有当其独立适配版本在本研究统一协议下超过频率、共现、Markov、LSTM 等简单对照时，才可在结果章节称为强基线。

论文不再围绕旧的“184 类唯一下一项 Top-1”任务修补，也不以在合成 SIM 数据上继续提高数字为目标。

## 2. 为什么改变路线

旧实验已经得到以下相互支持的证据：

1. SIM 数据的战术阶段单调不减比例为 100%，高度模板化。
2. SIM 同分布随机划分下，二阶 Markov 可取得约 0.5562 Top-1，任务主要奖励转移记忆。
3. 真实跨来源数据的技术转移大多未在训练来源出现，Markov Top-1 下降至约 0.02–0.04。
4. 域等权转移估计和简单战术知识没有稳定改善真实跨来源结果。
5. 唯一下一项 Top-1 对少数命中极其敏感，也不符合分析师实际使用方式。

因此，新的研究问题不是“怎样把原模型准确率再抬高一点”，而是：

> 当序列转移统计因跨来源稀疏而失效时，观察步骤中的真实行为语义是否包含更稳定的攻击状态和意图信号？

## 3. 冻结任务方向

### 3.1 输入

每个样本只使用当前时刻之前已经观察到的内容：

- 有序 ATT&CK 父技术 ID 前缀；
- 每一步清洗后的真实 procedure/事件描述；
- 每个父技术对应的全部合法战术，使用 multi-hot 表示；
- 显式文本缺失标记。

以下内容不进入主模型或 LLM prompt：

- `source`、campaign、actor、文件名；
- 未来步骤、目标标签或未来文本；
- 跨来源覆盖严重不一致的 asset、process、file、tool、privilege、platform、executor、command 等字段。

`source` 只允许用于 LODO 划分、campaign bootstrap、分层统计和结果汇报。

### 3.2 输出

给定冻结攻击路径的当前前缀：

1. 取未来最多三个动作；
2. 映射为唯一 ATT&CK 父技术集合；
3. 在固定 184 类标签空间内输出 Top-5 排名。

主任务是无序的未来三步候选集合，不声称恢复未来三步的精确顺序。

### 3.3 指标

- 主指标：campaign-macro NDCG@5；
- 辅助指标：campaign-macro Recall@5、Precision@5、Hit@5；
- 主窗口：`H=3`；
- `H=1` 和 `H=5` 只作为预注册敏感性分析，不得试完后选择表现最好的窗口。

旧唯一下一项任务的 Top-1、旧 `0.0873` 规则结果和 future-3 指标禁止直接比较。

### 3.4 准确的应用表述

推荐使用：

> Cross-source short-horizon ATT&CK technique recommendation over documented and emulated attack flows.

禁止夸大为：

> 对真实生产网络中攻击者下一次动作的实时准确预测。

现有数据代表文档化攻击流、真实报告衍生路径和 emulation/adversary plan，不是统一采集的生产遥测事件流。

## 4. 主数据与潜在扩展

### 4.1 三源主实验

| 来源 | 性质 | 角色 |
|---|---|---|
| CTID | 有序 adversary emulation procedure | 真实语义主力来源 |
| Attack Flow | 真实报告衍生图上的冻结最长路径 | 最大真实报告衍生来源 |
| Stockpile | CALDERA adversary profile 的 `atomic_ordering` | 半合成来源，必须单独标注 |

当前 v8.1 已冻结并实测的数据形状为：

- 对齐步骤：898步（CTID 283、Attack Flow 466、Stockpile 149）；
- 全量：814 个 future-3 样本、72 个 campaign；
- 已解盲开发集：30 条；
- 正式主评估：784 条（CTID 263、Attack Flow 412、Stockpile 109），保持10/35/27个campaign；
- 固定输出词表：184 个父技术。

原 v7 预期的816/786因CTID旧解析器把嵌套`technique`字典当步骤、并从任意文本抽取ID而失效。v8.1只读取顶层结构化`technique.attack_id`，排除26个无法映射的非ATT&CK标签，并完成确定性重建；上述814/784是正式分母。任何后续campaign消失、分母变化或路径不一致，仍必须先升级协议，不得在实验中临场修补。

### 4.2 Unit42 第四来源审计结论

SECRYPT 的公开仓库表明其 Markov 数据使用 Unit42 Playbook Viewer 与 MITRE Attack Flow。真正值得评估的是 **Unit42 原始 playbook**，而不是论文生成的全部路径或预训练转移矩阵。

仓库报告从 85 个 Unit42 STIX bundle 和 39 个 Attack Flow 文档经 DFS 得到 8,437 条路径、72 个起始状态和 1,621 个唯一转移。这些是数据构建产物，不是 8,437 个相互独立的真实事件或 campaign。独立性必须回溯到原始 bundle、报告或 campaign root。

Unit42 只有全部通过以下审计后才可加入：

1. 能恢复可解释、冻结且可复现的动作顺序；
2. 每一步能对齐到非标签别名式的 procedure 文本；
3. 与现有 CTID、Attack Flow、Stockpile 的报告和 campaign 去重；
4. 同一原始 playbook 生成的多条路径不得跨训练/测试集合；
5. 不通过排列组合制造大量伪独立样本；
6. 全部标签、文本清洗和唯一键规则与三源一致；
7. 许可允许数据处理和论文复现。

所有来自同一 bundle/报告的路径必须绑定同一个 group id，并在划分、bootstrap 和敏感性分析中作为同一独立单元处理。不得用 DFS 路径数充当 campaign 数、样本独立性或统计功效。

冻结审计 `unit42_playbook_sequence_v1` 已在官方归档 commit `4cdeeb3378c7f2da1b7f2d93d8bc1d6582ef1100` 上完成：

- `playbooks.json` 83个索引项，`playbook_json/` 85个有效bundle；
- 153个campaign，其中151个关联至少两个ATT&CK技术；
- 2,641条campaign→attack-pattern `uses`边，全部是成员关系；
- 0条attack-pattern→attack-pattern边；
- 0个显式step/order/sequence/next/precedes/follows/execution-time字段；
- 152/150个campaign有`first_seen/last_seen`，但只界定campaign整体区间；
- **合格有序campaign：0/153。**

因此Unit42被排除为可直接加入的第四序列来源。不得按`object_refs`位置、relationship `created`、战术阶段或membership star上的DFS构造顺序。若以后回到Unit42，只能从其对应原始叙事报告重新抽取明确事件顺序；这将是一套新数据，必须另写冻结协议，不能称为现有bundle的直接解析。

### 4.3 不进入主实验的数据

- SIM 合成模板数据：只用于说明旧 benchmark 的模板化效应；
- SECRYPT 的 4,849 条 campaign chain：仓库说明其由 33 个 campaign 按战术分桶后排列/采样生成，不视为 4,849 个独立真实 campaign；
- SECRYPT 的预训练 LSTM、Markov 转移矩阵或全量模型输出：可能包含本研究 Attack Flow 留出样本；
- 未经审计的全量 ATT&CK campaign/group 图：可能携带测试 campaign 信息。

## 5. SECRYPT 2026 的定位与使用方式

### 5.1 正式发表状态

论文：

> *MITRE ATT&CK-based Attack Chain Prediction Using Hybrid LSTM-Markov Models for Cyber Risk Assessment*

截至 2026-08-07，已经核实：

- 正式发表于 *Proceedings of the 23rd International Conference on Security and Cryptography - Volume 2: SECRYPT*；
- DOI：`10.5220/0015075400004103`；
- 页码：1039–1050；
- ISBN：`978-989-758-858-7`；ISSN：`2184-7711`；
- 官方技术日程编号为 `SECRYPT26-RP-129`，口头报告日期为 2026-07-17；
- 官方论文集前言记录 215 篇投稿，其中 19% 被接收并正式出版为 Full Paper。

公开仓库底部残留的 `under review` 和旧 BibTeX 已经过期，正式 SCITEPRESS/Crossref 元数据优先。该工作现在可以按正式会议论文引用，不得再写成 preprint、submission 或 under review。19% 录用率说明它经过正式同行评审，但不等于该方法已经成为统一 benchmark 上的 SOTA，也不改变 SECRYPT 不具有 JCR/中科院期刊分区这一事实。

### 5.2 为什么仍然需要对比

它与本研究共享以下核心结构：

- 观察 ATT&CK 技术前缀；
- LSTM 建模长程依赖；
- 一阶 Markov 提供短程经验转移；
- constrained beam search 生成多步候选路径。

因此，它是目前最接近的近期非语义序列架构。对比它的原因是任务和结构接近，而不是因为 SECRYPT 是顶级会议、其 86% 可迁移到本任务，或该方法已在统一 benchmark 上被公认为 SOTA。

### 5.3 正式论文与公开代码的协议审计

正式论文和公开仓库足以确认其 headline 数字的评估边界：

1. 4,849 条 campaign chain 由 33 个满足长度要求的 MITRE campaign 经过战术分桶、桶内排列/采样和跨桶组合构建，不能视为 4,849 个独立真实 campaign；
2. 每条生成链通过滑动前缀展开为 prefix–target pair，随后公开代码直接对 pair 索引运行 `np.random.permutation`，没有按 campaign 分组；
3. 论文明确说明 `100% / 50-50 / 80-20` 只衡量训练数据量敏感性，不构成 temporal-holdout 或 campaign-holdout 泛化研究，并把 leave-one-campaign-out 列为未来工作；
4. 公开数据重建得到 132,621 个 technique occurrence 和 4,849 条链，因此按 `sum(L-1)` 应产生 **127,772** 个 prefix–target pair，公开代码重建结果与此一致；
5. 论文表格同时报告 **128,413** 个训练 pair，与上一项相差 641，属于必须在复现报告中披露的内部计数矛盾；
6. 由于公开实现把 technique 集合经未排序 `set` 转回列表，仅设置 `random.seed(42)` 而未冻结 `PYTHONHASHSEED`，重复运行得到约 21,717–21,724 个唯一 pair；“21,720”可作为某次运行的审计值，不能在未冻结环境前写成普适常数；
7. 在已审计的公开实现上，默认 90/10 pair 划分中，测试 `(prefix, target)` 在训练集完全出现的比例约为 **88.82%–88.89%**；正式 80/20 场景约为 **87.56%–87.63%**。因此训练/测试非独立不是推测，而是可量化的完全重复污染；
8. prefix-majority 字典在随机 pair 划分下可取得约 74%–76%，但精确值受 tie-break、未见 prefix fallback 和未冻结 set 顺序影响；在冻结可复现脚本前，不使用单一的 `74.42%`；
9. 公开论文与仓库均未报告 campaign-holdout 下的 `5.9%`。若后续独立实验得到该数字，必须标成“本研究对 Raj et al. 公开方法/数据的重新评估”，不得写成原论文结果；
10. 42.3% 是仓库定义下的 **tactic-level alignment/coverage**，而本研究现有 13.3% 是跨来源 **technique-level bigram coverage**，二者粒度、分母和划分不同，禁止并列作高低结论；
11. 26,051 条 beam forecast 是模型生成路径，不是新增真实观测。

冻结运行 `20260807_split_audit_v1` 进一步得到以下独立复算结果：

| 方法 | 随机 pair 80/20 Accuracy | campaign-LOCO macro Accuracy | campaign-LOCO pooled Accuracy |
|---|---:|---:|---:|
| 全局频率 | 0.0282 | 0.0404 | 0.0291 |
| 完整 prefix 字典 | 0.7501 | 0.0456 | 0.0323 |
| 一阶 Markov 众数 | 0.3773 | 0.1006 | 0.1123 |
| 二阶 Markov 后退 | 0.6915 | 0.1015 | 0.1128 |

该运行固定 `PYTHONHASHSEED=0`，released-order 得到21,722个唯一 `(prefix,target)`，canonical-order 敏感性分析得到21,716个。两次独立运行的 `summary.csv`、`split_diagnostics.csv` 和 `campaign_loco.csv` 逐字一致。这些数值均是本研究复算，不得归因给 Raj et al.。

因此，SECRYPT 对本研究的主要价值是提供一个待严格验证的候选生成与重排架构，以及提示 Unit42 这一潜在新来源；它报告的 86% 只进入相关工作和协议差异讨论，不进入主结果数值比较。

在 v8 中必须用同一标签空间、同一训练/测试折和同一定义，分别重算 technique-level 与 tactic-level coverage，形成可比较的覆盖率矩阵。未经同口径重算，不得使用“42.3% 对 13.3%”支撑粒度结论。

### 5.4 三协议泛化审计

v8 必须把“随机划分高、严格留出低”从叙述变为可复现实验。对 SECRYPT-adapted 架构预注册三种协议：

| 协议 | 数据与任务 | 独立单位 | 用途 |
|---|---|---|---|
| `P-pair` | SECRYPT 公开 239 类、`H=1` next-step | 随机 prefix–target pair | 复核 86% 所代表的同分布插值，并测重复率/字典基线 |
| `P-campaign` | 与 `P-pair` 相同 | campaign/group holdout | 量化去除排列同源重复后的跨 campaign 泛化下降 |
| `P-source` | 本研究固定 184 类、future-3 Top-5 | CTID/Attack Flow/Stockpile LODO | 论文主实验，衡量跨来源迁移 |

`P-pair` 与 `P-campaign` 必须使用相同的 SECRYPT 数据构建、标签空间和 `H=1` 指标，才能把差值解释为划分协议效应。`P-source` 的任务、标签空间和指标不同，只能作为更严格的外部验证，绝不能把三个协议的绝对准确率直接相减后声称方法下降了多少。

三协议都必须报告频率、prefix-majority 字典、Markov-only、LSTM-only 和 Hybrid LSTM–Markov。`P-pair` 还必须报告：总 pair、唯一 pair、测试 pair 完全重复率、prefix 覆盖率以及按 campaign 聚类后的有效独立单位数。

只有在脚本、输入哈希、commit、`PYTHONHASHSEED`、tie-break 和 fallback 全部冻结后，才允许写入精确的字典准确率或 campaign-holdout 数字。现阶段不得把未复现的 `5.9%` 当作路线判据。

### 5.5 公平独立适配原则

公开仓库当前声明许可仍在机构审查中。主实验不得复制、改写或分发其未明确授权的代码，也不得继承作者的模型或统计量；只依据正式论文和公开方法说明做 clean-room 式独立适配。方法名写成 `SECRYPT-adapted`，不得声称 exact reproduction。

必须在每个外层折中：

1. 只使用训练来源拟合 LSTM；
2. 只使用训练来源估计 Markov 转移；
3. 只使用 inner-LODO 选择 epoch、融合权重和 beam 参数；
4. 外层测试来源不得进入任何预训练统计；
5. 所有方法使用相同的 184 类词表、future-3 目标和 Top-5 指标。

SECRYPT 的风险评分分支（EPSS、CAPEC、KEV、D3FEND、OCTAVE、NCISS）不属于本研究的 future-technique relevance 目标，不进入主基线。

### 5.6 future-3 适配

冻结 beam horizon 为 3。对所有保留路径 `p`，将未来三步技术的路径概率聚合为标签分数：

```text
score(c) = sum_p P(p) * 1[c appears in the first 3 generated steps of p]
```

按 `score(c)` 降序输出 Top-5。并列规则、beam width、分支数、概率下界和路径去重必须在 v8 中事前冻结。

不得把 SECRYPT 报告的 86% next-step accuracy 与本研究的 future-3 LODO 指标直接比较。v8 还必须加入 Markov-only constrained beam 消融，从而区分“候选约束本身”的收益与 LSTM 重排的收益；LSTM–Markov 的路径融合公式必须事前写死，不能根据外层结果在纯重排、概率乘积或其他规则之间切换。

## 6. 方法与对照矩阵

### 6.1 必做简单基线

| 代号 | 方法 | 目的 |
|---|---|---|
| A0 | 训练折全局目标频率 | 不看输入的下限 |
| CO | 训练折技术共现推荐 | 简单集合补全基线 |
| TIE-local | 只在训练来源重训的 TIE 式关联推荐 | 对比公开 ATT&CK 推荐范式 |
| KUW | Kuwano 式协同过滤/战术过滤 | 对比分析师候选推荐工作 |
| M | 一阶/二阶 Markov | 纯局部转移统计 |

`TIE-public` 如运行，只能作为可能包含外部重合报告的 open-world 诊断，不进入主判定。

### 6.2 必做神经序列基线

| 代号 | 方法 | 目的 |
|---|---|---|
| LSTM | ID-only LSTM | 长程序列对照 |
| TR | ID-only Transformer/DeepOP 式序列模型 | 注意力序列对照 |
| MB | Markov-only constrained horizon-3 beam | 隔离候选约束收益 |
| HM | 独立实现的 SECRYPT-adapted LSTM–Markov + horizon-3 beam | 最近邻架构基线 |

### 6.3 文本、战术和语义阶梯

保留 v7 的主要对照思想：

| 代号 | 方法 | 可回答的问题 |
|---|---|---|
| A | 多标签 ID 上下文 relevance 主干 | ID 历史是否有信息 |
| T | A + 多战术软先验 | 战术归纳偏置是否有用 |
| R | A + 原始 procedure text probe | 原始文本是否有增量 |
| S | A + LLM 状态摘要 | LLM 语义转换是否有增量 |
| ST | A + LLM 状态摘要 + 战术软先验 | 战术能否扩展语义方法 |
| P | A + 训练摘要置换 | 排除容量和普通集成效应 |
| B0 | LLM 直接 Top-5 | LLM 自身排名是否有信号 |
| K | 单调战术过滤 | 旧强规则在新任务下的诊断 |

### 6.4 加入 SECRYPT 后的核心比较

v8 应增加：

| 代号 | 方法 | 角色 |
|---|---|---|
| HM | LSTM–Markov | 新的核心无语义序列基线 |
| HM+R | HM + 原始文本 | 文本信号控制 |
| HM+S | HM + LLM 状态语义 | 候选主方法 |
| HM+P | HM + 置换状态语义 | 等容量内容控制 |
| HM+ST | HM + LLM 语义 + 战术软先验 | 预注册扩展 |

v8 必须在看到外层结果前明确：`HM+S` 是否替代 v7 的 `S` 成为论文主方法。推荐将 `HM+S` 固定为主方法，原 `A/S` 阶梯作为机制消融，因为这能直接回答“语义是否超越最近邻序列架构”。

### 6.5 已完成的 P-source 结果（2026-08-07）

全部方法使用同一784行正式集合、同一三折LODO、campaign-macro聚合和三来源等权总体。神经模型先在两个训练来源之间做inner-LODO，再以五个seed训练；没有方法使用外层测试来源选参。

| 方法 | CTID | Attack Flow | Stockpile | 来源等权 NDCG@5 |
|---|---:|---:|---:|---:|
| A0 | 0.1755 | 0.1897 | 0.0644 | 0.1432 |
| CO | 0.0357 | 0.0486 | 0.0080 | 0.0307 |
| A | 0.1559 | 0.1765 | 0.1241 | 0.1522 |
| K | 0.1450 | 0.1637 | 0.2244 | 0.1777 |
| T | 0.1559 | 0.1765 | 0.1434 | 0.1586 |
| R（A+原始文本） | 0.1597 | 0.1765 | 0.0634 | 0.1332 |
| LSTM | 0.1991 | 0.1255 | 0.0630 | 0.1292 |
| Transformer | 0.1097 | 0.1165 | 0.0622 | 0.0961 |
| MB | 0.1000 | 0.1191 | 0.1939 | 0.1377 |
| HM | 0.1202 | 0.1470 | 0.1063 | 0.1245 |
| HM+R | 0.1202 | 0.1470 | 0.0634 | 0.1102 |
| HM+S | 0.1202 | 0.1470 | 0.0645 | 0.1106 |
| HM+P | 0.1202 | 0.1470 | 0.0632 | 0.1101 |
| HM+T | 0.1202 | 0.1470 | 0.1063 | 0.1245 |
| HM+ST | 0.1414 | 0.1470 | 0.0645 | 0.1176 |
| B0（DeepSeek直接Top-5诊断） | 0.1943 | 0.2401 | 0.1741 | 0.2028 |
| T+B0（事后探索） | 0.1944 | 0.2401 | 0.1815 | 0.2053 |

当前结论必须按以下边界表述：

1. 没有一个已运行方法在三个来源上稳定占优；每个来源的最佳方法不同，域差异仍是主导因素。
2. K 的总体最高值主要来自半合成 Stockpile；它在 CTID 和 Attack Flow 上均低于 A0 与 A，因此不能称为真实跨来源的统一强基线。
3. LSTM 只在 CTID 上明显较强，但在 Attack Flow 和 Stockpile 上退化；Transformer 总体显著低于 A。增加 ID-only 模型容量没有形成稳定迁移收益。
4. MB 低于 A，`MB-K` 的95% CI为[-0.0805,-0.0022]；经验转移约束会排除未见但正确的候选。
5. HM 的inner beta分别为0.1/0.2/0.9，来源间差异很大；总体低于MB、LSTM、A和K，其中`HM-A`与`HM-K`的95% CI均完全小于0。因此SECRYPT-adapted架构在本协议下不是强基线。
6. R总体相对A为-0.0190，95% CI为[-0.0393,-0.0044]。all-unseen层的来源等权正差主要来自Stockpile；CTID约为0、Attack Flow因inner选择回退A而为0，不能据此声称原始文本在两个真实来源上补足未见转移。
7. HM+R在两个真实来源都机械回退到HM；Stockpile选择纯原始文本后变差。总体`HM+R-HM=-0.0143`，95% CI为[-0.0221,-0.0077]，排除了“R只因A主干不合适而失败”的解释。

8. 正式DeepSeek生成共784/784行有效；787次计费attempt，输入641,862 tokens、输出127,285 tokens，实测费用0.0741152384美元。所有摘要均通过无ATT&CK ID、无source/campaign字面量和无内部thinking通道的机械门。
9. 预注册`HM+S`在CTID和Attack Flow均由inner-LODO选择lambda=0，在Stockpile选择纯S后变差；来源等权`HM+S-HM=-0.0139`，95% CI为[-0.0214,-0.0073]，命中v8“总体不超过HM”的判死条件。
10. `HM+S`只比置换控制`HM+P`高0.0005，CI跨0；比`HM+R`高0.0004，CI跨0。当前summary→BGE-M3→MLP probe链路没有得到可归因的内容增益。
11. HM+T三个外层折均由inner-LODO选择战术权重0，精确退化为HM。HM+ST只在CTID高于HM/HM+S，另外两源分别退化为纯HM和纯S；只满足1/3来源，预注册互补性条件不成立。
12. B0在三个来源均高于HM，来源等权NDCG@5为0.2028；相对A0为+0.0596，95% CI [0.0157,0.1077]。这说明DeepSeek的直接候选排序存在跨来源信号，但它是诊断结果，不等于融合主张成立。
13. 事后冻结的探索性rank fusion得到`HM+B0=0.1965`，低于B0；`HM+B0-B0=-0.0064`，95% CI [-0.0216,0.0066]。准确结论是“当前HM融合未检出增量，也不能排除小幅增益或损害”，不得写成所有转移信息确定无互补价值。
14. 使用更合适的软战术主干后，`T+B0=0.2053`，相对B0为+0.0025，95% CI [-0.0007,+0.0078]；相对T为+0.0467，95% CI [+0.0094,+0.0891]。这封闭了“HM太弱导致融合假阴性”的漏洞，但仍未达到可声称互补的统计证据。

至此，已经排除了“简单加容量”“原始文本直接编码”“摘要embedding probe”“战术软先验”“Markov beam约束”和“LSTM-Markov混合”六类替代方案。核心融合方法在当前三源任务设定下缺乏证据；但B0表明问题不等于“LLM完全无信号”，更可能是监督probe压缩和转移融合路径丢失了零样本LLM的排序能力。

## 7. LLM 语义分支

LLM 不直接作为最终标签分类器，而先把已观察历史转换为三个状态字段：

```text
stage_assessment
observed_capabilities
likely_next_intents
```

三字段不得包含 ATT&CK ID，也不得复述未来候选列表。状态文本再由冻结多语编码器编码，并通过共享多标签 probe 产生 184 维 relevance。

必须同时保留：

- raw-text 分支，检验是否根本不需要 LLM；
- shuffled-summary 分支，检验是否只是容量或额外分支收益；
- direct-LLM Top-5，检验 LLM 排名本身；
- tactic-only 分支，检验是否只是战术阶段信息。

旧 30 条试点只证明旧 API/JSON 流程可运行，且已参与任务重设计。由于 future-3 prompt 和输入字段发生变化，它们只能作为开发集，不能进入正式训练、调参或测试。

任何新的 v8 API 调用都必须重新获得明确授权，说明发送字段、样本数量、模型和按 token 计费；旧授权不能自动扩展到新任务。

## 8. 评估设计

### 8.1 主评估：三源 LODO

| 外层折 | 训练来源 | 测试来源 |
|---|---|---|
| 1 | Attack Flow + Stockpile | CTID |
| 2 | CTID + Stockpile | Attack Flow |
| 3 | CTID + Attack Flow | Stockpile |

若 Unit42 审计通过，则升级为四源 LODO；不得在看到其结果后决定是否纳入主实验。

### 8.2 超参数选择

- 外层测试来源完全不可见；
- 在两个训练来源之间执行 inner-LODO；
- 所有 epoch、融合权重、beam 参数和阈值只由 inner-validation 的 campaign-macro NDCG@5 选择；
- 禁止逐样本选择融合权重；
- 禁止根据外层测试结果切换主方法。

### 8.3 统计单位

- 指标先在每个 campaign 内聚合，再对 campaign 宏平均；
- campaign 为 bootstrap 和配对检验单位；
- 运行 2,000 次 campaign cluster bootstrap，报告 95% CI；
- CTID 只有约 10 个 campaign，必须增加 leave-one-campaign-out influence 分析，避免一个 campaign 主导结论；
- 不使用“接近显著”等措辞美化跨 0 的 CI。

### 8.4 必报分层

1. **seen/unseen transition**：真实 `(last_parent, target_parent)` 是否在训练来源出现；
2. **文本长度**：`<40` 与 `>=40` 字符；
3. **来源**：CTID、Attack Flow、Stockpile 分别报告；
4. **目标基数**：`|Y|=1/2/3`；
5. **prefix 长度**：预注册长度分层；
6. **开发集影响**：主结果排除旧 30 条，全量只作敏感性分析。

核心机制假设为：语义增益应主要出现在 unseen-transition 层。若增益只出现在 seen 层，不能声称语义补足未见转移。

## 9. 执行顺序

### 阶段 0：发布并冻结 v8

**状态：已完成。** v8.1 addendum记录了CTID结构化解析修正、正式分母和确定性哈希。

在任何正式外层实验或新 API 调用前：

1. 把 SECRYPT-adapted HM、TIE-local、Kuwano、LSTM、Transformer 加入协议；
2. 冻结 HM 的数据输入、beam、概率聚合和超参数网格；
3. 冻结 LSTM–Markov 路径融合公式，并加入 Markov-only beam 消融；
4. 冻结 `P-pair/P-campaign/P-source` 三协议、独立单位与指标；
5. 冻结 SECRYPT 审计的 `PYTHONHASHSEED`、pair 唯一键、tie-break 和字典 fallback；
6. 冻结 technique/tactic coverage 的同口径计算方法；
7. 冻结 `HM+S` 是否为主方法；
8. 冻结 `H=1/3/5` 的角色；
9. 写死成立、判死和失败处理规则；
10. 重新生成配置与数据哈希。

### 阶段 1：完成数据、SECRYPT 与 Unit42 审计

**状态：三源数据、SECRYPT划分和Unit42顺序审计已完成；Unit42不具备第四折所需顺序，统一coverage矩阵仍待做。**

1. 已重建并核对898步、814个三源future-3样本；
2. 已验证784行主评估、30行开发集和10/35/27个campaign分母；
3. 已运行唯一键、时序、闭集、prompt字段和文本ID清洗泄漏断言；
4. 已固化SECRYPT公开仓库并复核4,849 chains、132,621 occurrences、127,772 pairs及论文128,413的矛盾；
5. 已完成`P-pair/P-campaign`简单基线、重复率与确定性审计；
6. 已完成Unit42官方playbook审计：许可清楚，但0/153 campaign具有可接受的技术顺序，直接四源升级终止；
7. 待用统一定义重算各源technique/tactic coverage；如需新来源，必须另行搜索显式有序数据或从原始报告重建。

### 阶段 2：先跑全部非 LLM 基线

**状态：P-source的A0/CO/A/K/T/R/LSTM/TR/MB/HM/HM+R已完成；TIE-local/KUW原式核验和P-pair/P-campaign神经模型仍待做。**

优先完成：

1. 在 `P-pair/P-campaign` 上运行频率、prefix-majority、Markov、LSTM 和 HM，量化协议差距；
2. 在 `P-source` future-3 主任务上运行 A0/CO/TIE-local/KUW；
3. Markov；
4. ID-only LSTM；
5. Transformer/DeepOP 式模型；
6. Markov-only constrained beam；
7. SECRYPT-adapted HM；
8. tactic-only 和单调过滤诊断。

这一阶段回答：

- future-3 任务是否存在超越频率/共现的可学习信息；
- 随机 pair 划分的高分中有多少可由完全重复和字典查表解释；
- 去除同 campaign 排列重复后，SECRYPT-adapted 方法的跨 campaign 性能是多少；
- SECRYPT 式方法在严格 LODO 下是否仍然有效；
- 任务困难是否主要来自跨来源还是来源内部稀疏；
- LLM 需要打赢的真实门槛是多少。

当前P-source主干、最近邻架构、原始文本、LLM摘要、置换、战术和直接LLM诊断均已完成。尚待完成的是TIE-local/KUW原式核验、P-pair/P-campaign神经模型和Unit42第四来源审计；这些工作不能改变已经封口的三源HM+S结论。

### 阶段 3：运行新的 30 条开发试点

**状态：已完成。** 30/30首次生成成功，所有JSON、summary长度、空值、Top-5和泄漏机械门通过；试点只用于流程验证，未进入正式评估。

使用已经排除于主评估的 30 条开发集：

1. 验证新 prompt、JSON schema、关闭内部思考和 token 预算；
2. 验证摘要不含 ATT&CK ID、source、campaign 或未来文本；
3. 检查状态摘要不是单纯重述技术名称；
4. 保存全部原始响应、token、费用和 manifest；
5. 试点完成后停止，等待人工确认。

试点只能用于机械质量门和 prompt 开发，不用于估计正式效果。

### 阶段 4：全量生成与正式训练

**状态：已完成。** 784/784正式样本生成成功；原始响应先提交，随后完成S/P编码和HM+S/HM+P/HM+T/HM+ST。API总费用约0.0741美元。

获得新的明确 API 授权后：

1. 全量生成未经后处理的原始 LLM 输出；
2. 立即保存、哈希并提交原始版本；
3. 编码 R/S/P 分支；
4. 运行 `HM+R/HM+S/HM+P/HM+ST` 的 inner-LODO 与外层 LODO；
5. 每个神经模型运行冻结的 5 seeds；
6. 落盘逐样本 Top-20、分数、唯一键、配置和 stdout。

### 阶段 5：统计、机制与决策

**状态：核心判定已完成。** campaign bootstrap、seen/unseen、文本长度和目标基数分层均已落盘；HM+S判死与HM+ST互补性不成立已经封口。T+B0事后探索没有检出相对B0的可靠增量；CTID leave-one-campaign influence与其他新来源上的前瞻确认仍待完成。

1. 计算 campaign-macro 指标和 cluster bootstrap CI；
2. 完成 seen/unseen、文本长度和目标基数分层；
3. 完成 source-level 与 leave-one-campaign-out influence 分析；
4. 按预注册条件判定核心主张；
5. 禁止基于结果修改主任务、主指标或挑选主方法。

## 10. 论文成立条件

最终数值阈值由 v8 在外层结果产生前写死。论文级最低证据要求为：

1. `HM+S` 相对 `HM` 的 campaign-macro NDCG@5 在至少两个真实来源上为正；
2. 聚合差值的 95% CI 最好完全大于 0；
3. `HM+S` 超过 `HM+P`，否则不能归因于摘要内容；
4. `HM+S` 超过或稳定不劣于 `HM+R`，否则只能结论为“原始文本有用”，不能声称 LLM 必要；
5. `HM+S` 超过 tactic-only/`HM+T`，否则不能排除只是战术阶段信息；
6. `HM+S` 至少与 TIE-local、Kuwano、LSTM、Transformer 和 HM 等候选基线正面对比，并明确哪些方法在统一协议下实际构成强基线；
7. 主要增益应出现在 unseen-transition 层，支持预设机制解释；
8. 结果不能只由半合成 Stockpile 单一来源支撑。

若只比简单 Markov 高一点、只在排列生成数据上有效，或提升主要来自未来窗口扩大，均不足以支撑 LLM 语义方法论文。

### 10.1 当前实际判定（2026-08-07）

- 条件1不满足：HM+S相对HM没有两个来源为正；CTID和Attack Flow均为0，Stockpile为负。
- 条件3不满足稳健归因：HM+S仅比HM+P高0.0005，95% CI跨0。
- 条件4不提供LLM必要性证据：HM+S与HM+R几乎相同，差值0.0004，95% CI跨0。
- 条件5不满足方法级优势：HM+T等于HM，HM+ST总体仍低于HM，战术与摘要互补性不成立。
- 条件7不满足：没有得到“摘要融合专门改善unseen-transition”的稳定证据。
- 条件8不满足正向支撑：唯一非零S外层权重出现在半合成Stockpile，且结果为负。

正式结论为：**LLM语义融合在当前任务设定下缺乏证据。** 这不否定B0直接LLM排序的独立信号，但禁止把B0替换成预注册主方法或用事后rank fusion改写主结论。

## 11. 三种结果分支

### 11.1 强正结果：语义方法路线

若上述条件大体满足，论文主线为：

> 现有序列方法在容易划分或排列扩增数据上表现良好，但在来源完全留出的真实攻击流上受未见转移限制；procedure-level 语义提供了更可迁移的状态信号。

候选贡献：

1. 跨来源短窗口 ATT&CK recommendation 任务；
2. 三源或四源 procedure-aligned benchmark；
3. LLM 状态语义与序列规律融合；
4. 严格的 LODO、campaign 统计和污染控制；
5. seen/unseen 机制证据；
6. 对同一相邻架构执行 pair-random、campaign-holdout、source-LODO 的协议敏感性审计；
7. 合成/排列扩增 benchmark 的真实性审计。

### 11.2 部分正结果：文本有用但 LLM 非必要

若 `R>A/HM+R>HM`，但 `S<=R/HM+S<=HM+R`：

- 允许结论：真实 procedure 文本包含 ID 序列之外的预测信号；
- 禁止结论：LLM 语义推理是必要组件；
- 后续应简化方法、强化 benchmark 和跨来源文本表示贡献，而不是包装 LLM。

### 11.3 负结果：现实检验/benchmark 路线

若语义、原始文本和战术均不能稳定超过强序列/共现基线：

- 如实判定当前输入不足以支持可靠短窗口预测；
- 论文转向“合成与排列扩增如何夸大 ATT&CK 预测性能，以及严格跨来源评估为何失效”；
- 要争取二区，应补充 Unit42 第四来源、真实生产遥测、分析师效用实验或更大规模人工标注；
- conformal candidate sets 可作为下一项目，但不在当前实验中事后加入救结果。

## 12. 论文定位

不推荐：

> 首次融合 LLM 和序列模型预测下一 ATT&CK 技术。

推荐：

> A rigorous cross-source evaluation of whether procedure-level semantics can complement sparse transition statistics for short-horizon ATT&CK technique recommendation.

候选标题：

> **Beyond Transition Memorization: Procedure-Aware Cross-Source ATT&CK Technique Recommendation**

二区潜力不来自“比 SECRYPT 高几个点”，而来自完整证据链：

1. 真实任务重新定义；
2. 可审计的跨来源数据；
3. 最近邻架构基线的公平独立适配；
4. 文本、LLM、战术和集成效应的严格拆分；
5. unseen-transition 机制解释；
6. 可复现代码、数据和统计产物。

## 13. 红线

1. 不使用外层测试来源选择任何参数。
2. 不逐样本选择融合权重。
3. 不把 source 或缺失模式输入模型。
4. 不把未来文本、未来战术或目标标签写入 prompt。
5. 不把 SECRYPT 的外部转移矩阵用于主 LODO。
6. 不把生成路径数量当作独立 campaign 数量。
7. 不把不同任务、标签空间或划分的绝对数字直接比较。
8. 不把 tactic-level 42.3% 与 technique-level 13.3% 直接比较。
9. 不复制或派生使用许可状态未明确的外部仓库代码。
10. 不把独立复现数字写成 Raj et al. 原论文报告结果。
11. 不在缺少冻结脚本与 manifest 时引用 `5.9%` 或精确 `74.42%`。
12. 不用旧 next-step `0.0873` 作为 future-3 方法的通过阈值。
13. 不在结果为负后新增子集、指标或 oracle 挽救结论。
14. 不静默丢弃 API 失败、空文本或 OOV 目标。
15. 不在未获得新授权前发送 v8 数据到外部 API。

## 14. 下一步交付物

已完成：v8/v8.1、三源future-3数据、SECRYPT划分审计、P-source主要基线、30条试点、784条全量生成、S/P/T/ST实验、逐样本结果和核心判定。

后续按证据价值排序：

1. 完成CTID leave-one-campaign influence，确认B0与各基线差值是否被少数campaign主导。
2. 搜索其他具有显式事件顺序和procedure文本的新来源；Unit42 bundle已判定不合格，不再把它当作既定第四折。
3. 若选择从Unit42对应叙事报告重新抽取顺序，先冻结报告选择、事件抽取、分支线性化和独立group规则，再构建全新数据。
4. 在真正未参与本轮选择的新来源上前瞻验证B0与一个**事先定义**的直接LLM reranker/校准器；不得复用三源T+B0权重后再称预注册。
5. 完成TIE-local/KUW原式核验和`P-pair/P-campaign`神经模型，补齐相关工作公平对比。
6. 生成统一technique/tactic coverage矩阵，量化粒度与划分协议的可解性变化。
7. 若B0前瞻复现且可校准，形成“跨来源LLM直接候选排序”新方法；若不能复现，转向benchmark/现实检验论文。
8. 最后再改论文方法、实验与讨论章节；当前不得继续润色原“HM+S有效”叙事。

## 15. 关键参考

- Raj et al. (SECRYPT 2026) DOI：<https://doi.org/10.5220/0015075400004103>
- Raj et al. 正式 SCITEPRESS 页面：<https://www.scitepress.org/Link.aspx?doi=10.5220/0015075400004103>
- SECRYPT 2026 官方论文集前言（215 submissions；19% Full Papers）：<https://www.scitepress.org/ProceedingsDetails.aspx?ID=GrF7dp3BzeY%3D&t=1>
- SECRYPT 2026 官方技术日程：<https://www.insticc.org/node/TechnicalProgram/secrypt/2026/presentationDetails/150754>
- SECRYPT Hybrid LSTM–Markov 仓库：<https://github.com/mayank02raj/MITRE-ATTACK-based-Attack-Chain-Prediction>
- MITRE Technique Inference Engine：<https://github.com/center-for-threat-informed-defense/technique-inference-engine>
- MITRE TIE 说明：<https://ctid.mitre.org/blog/2024/09/09/know-your-adversarys-next-move-with-tie/>
- Kuwano et al. (2023)：<https://doi.org/10.2197/ipsjjip.31.802>
- DeepOP (2025)：<https://doi.org/10.3390/electronics14020257>
- Attack Flow 官方说明：<https://center-for-threat-informed-defense.github.io/attack-flow/overview/>
- 当前 v7 协议：`project/data_v4/protocols/LLM_semantic_future3_lodo_validation_v7.md`
