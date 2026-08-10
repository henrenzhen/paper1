# LLM 语义与攻击序列预测：已完成尝试、证据状态与当前路线

> 更新时间：2026-08-10  
> 仓库：`henrenzhen/paper1`  
> 本文性质：项目状态总结。精确实现、哈希、统计口径和 API 审计以各冻结协议、manifest 与逐样本产物为准。

## 1. 当前结论

截至 2026-08-10，已经有证据支持：

1. **DeepSeek 直接读取已观察 procedure 文本并输出 Top-5（B0）含有跨来源排序信号。** 在 784 条正式三源 LODO 发现集上，B0 的来源等权 campaign-macro NDCG@5 为 `0.2028`，高于当前全部单一非语义基线的点估计。
2. **尚无证据证明“LLM 语义 + 跨来源 ATT&CK 技术转移统计”的融合有效。** 预注册融合、固定权重、战术软先验、混合 LSTM–Markov、候选级门控、局部重排、专家路由和让 LLM 直接读取转移证据等方案均未形成稳定正增益。
3. **失败集中在转移证据，而不是 LLM 本身。** 跨来源 technique bigram 覆盖率约 `13.3%`；在大量无支撑样本上，转移分支退化为平滑频率或噪声。最新 EDLR 开发试点中，正确转移证据甚至低于打乱证据。
4. **现阶段不能声称核心融合方法成立，也不应继续在同一 784 行上搜索更多权重并挑最好结果。** 这些数据已经参与多轮方法开发；后续任何正结果在没有新来源确认前都只能是事后探索。

因此，当前应当封闭的具体路线是：

> `LLM 语义排序 + 跨 campaign 的稀疏 technique bigram/order-2 计数`。

这不是数学上证明任何 LLM–序列方法都不可能。若继续保留核心思想，下一条实质不同的候选路线是：

> 建模连续前缀下 **LLM 语义状态/候选信念的变化轨迹**，把“序列规律”定义为语义状态随观测增长的动态，而不再依赖不同 campaign 重复出现相同 technique bigram。

该路线目前只是下一步方向，尚未得到实验结果。

## 2. 为什么原论文实验不能继续沿用

前置审计发现原稿的主要问题不是简单的“创新不足”，而是数据、实现和叙述之间存在不可接受的不一致：

- 原 `sim_*` 数据的战术阶段单调不减比例为 `13371/13371 = 100%`，任务实质接近模板复现；
- 原 Table 3 数字来自无数据读取的硬编码数组；
- No-CoT reasoning 曾因非唯一 key 覆盖而污染，且仓库中没有可恢复的干净备份；
- 论文写 `max_len=50`、左填充，代码实际为 `MAX_LEN=20`、右填充并读取最终 hidden state；
- 旧划分存在 root/campaign 重叠；
- 父技术折叠造成大量表面 self-transition；
- COSE 自 2024 年起暂停考虑以 AI/ML 为重要组成部分的稿件，原稿未进入学术评价。

这些问题意味着不能通过补跑几个 seed 或换期刊挽救原实验，必须重建数据与任务。

## 3. 数据重建和任务重构

### 3.1 顺序数据重建

已重建并审计三个来源：

| 来源 | 重建阶段规模 | 顺序性质 | 备注 |
|---|---:|---:|---|
| CTID | 281 prefix / 10 campaign | 战术单调约 0.625 | 真实 emulation plan |
| Attack Flow | 431 prefix / 35 flow | 战术单调约 0.650 | 真实攻击流；修复了 `latch → anchor → node` 三层边归属 |
| Stockpile | 122 prefix / 27 profile | 战术单调约 0.861 | CALDERA `atomic_ordering`，按战术设计，标为半合成 |

Attack Flow 和 Stockpile 旧 loader 全部退化为 `prefix_len=1`。重建后采用事前冻结的最长路径策略；非 action 节点可穿越但不输出，tie 按源文件节点顺序打破。

### 3.2 当前正式任务

任务已从“唯一下一项 Top-1”改为：

```text
输入：截至当前时刻的有序父技术前缀
    + 每步清洗后的 procedure/action 描述
    + 由 ATT&CK 查表得到的 multi-hot 战术

输出：未来 3 个动作内出现的唯一父技术集合中的 Top-5 排名
标签空间：固定 184 个父技术
```

当前分母：

| 数据地位 | CTID | Attack Flow | Stockpile | 合计 |
|---|---:|---:|---:|---:|
| 正式发现集 | 263 | 412 | 109 | **784** |
| 开发试点集 | 10 | 10 | 10 | **30** |
| 总计 | 273 | 422 | 119 | **814** |

正式评估采用三折 LODO（Leave-One-Data-Source-Out）：每次用两个来源训练/调参，完整留出第三来源测试。指标先在 campaign 内平均，再对 campaign 等权，最后三来源等权。主指标为 NDCG@5，同时报告 Hit@5、Precision@5 和 Recall@5。

开发试点 30 条已经用于 prompt 和方法开发，不能进入正式训练或被描述为前瞻验证。

## 4. 文献与外部数据审计

### 4.1 SECRYPT 2026 Hybrid LSTM–Markov

Raj et al. 的 Hybrid LSTM–Markov 是目前架构上最接近的工作，但其约 86% 数字不能与本研究 LODO 结果直接比较：

- 33 个 campaign 经战术分桶排列扩增为 4,849 条 chain；
- 随机 pair 划分存在大量近重复跨集合；
- 其公开说明明确把 campaign-holdout 留作后续工作；
- 本项目的 clean-room 审计表明，随机重复查表与严格 campaign/source 留出之间存在数量级差异。

本研究已实现 SECRYPT-adapted HM 基线，但在统一 future-3 LODO 下只有 `0.1245`，不是强基线。

### 4.2 Unit42

对 153 个 Unit42 campaign/bundle 的顺序审计结果为：

- 合格的技术→技术显式顺序 campaign：`0/153`；
- 2,641 条 STIX `uses` 边均是 campaign→technique 归属，不是动作顺序；
- campaign 时间区间、STIX 生命周期时间戳、`object_refs` 位置和 kill-chain 阶段均不能伪装成真实时序。

因此 Unit42 不能作为第四个序列来源。公开世界中同时具备“显式顺序 + procedure 文本”的大规模 ATT&CK 数据极其稀缺。

## 5. 正式发现集上的基线结果

下表均为 784 条发现集上的来源等权 campaign-macro NDCG@5；这些结果经过三源 LODO，但其中 B0 及后续融合已经影响方法选择，因此属于发现/探索证据，不是新来源确认。

| 方法 | NDCG@5 | 结论 |
|---|---:|---|
| **B0：DeepSeek 直接 Top-5** | **0.2028** | 当前最强单一语义信号 |
| K：单调战术过滤 | 0.1777 | 优势主要来自半合成 Stockpile |
| T：ID + 软战术先验 | 0.1586 | 战术权重跨来源不稳定 |
| A：插值转移相关度 | 0.1522 | technique 转移覆盖不足 |
| A0：频率先验 | 0.1432 | 强于多个神经/混合模型 |
| Markov-beam | 0.1377 | 候选约束会排除未见正确技术 |
| R：原始文本分支 | 0.1332 | 总体显著低于 A |
| LSTM | 0.1292 | 只在 CTID 较强，跨来源不稳 |
| HM：SECRYPT-adapted LSTM–Markov | 0.1245 | 不是统一协议下的强基线 |
| HM+S | 0.1106 | 预注册融合失败 |
| Transformer | 0.0961 | 容量增加未带来迁移收益 |
| CO：共现 | 0.0307 | 基本失效 |

重要补充：

- `HM+S − HM = −0.0139`，95% CI `[-0.0214, -0.0073]`，预注册主方法明确失败；
- `T+B0 = 0.2053`，相对 B0 仅 `+0.0025`，95% CI `[-0.0007, +0.0078]`；
- `B0−A0` 在 CTID 上会因删除 FIN6 翻转，因此不能声称 B0 在每个真实来源都稳定胜过频率先验；
- B0 相对 K/T/HM 的 CTID 方向不因删除任一单 campaign 翻转。

## 6. 已尝试的融合与机制排除

### 6.1 前一阶段的 7 类失败方向

已覆盖：

1. 原始文本编码与 ID 序列融合；
2. LLM 摘要 → BGE-M3 → MLP probe；
3. 固定 logit/rank 权重融合；
4. 战术软先验与硬级联；
5. Markov beam 约束和 LSTM–Markov 混合；
6. 直接 B0 与 HM/T 排名融合；
7. 候选级 RGAF 可学习门控。

候选级 RGAF 只使用推理时可观测的训练转移计数、支持源数、分支不确定性和前缀长度，且按 campaign-LOO 构建训练特征。结果：

```text
B0             0.2028
RGAF           0.2002
RGAF - B0     -0.0027，95% CI [-0.0044, -0.0012]
RGAF-Shuffle   0.2028
统一开启残差    0.1667
```

这证明先前基于真实目标构造的 `transition_visibility` 只能作为解盲后诊断，不能作为合法推理门控；可观测支持特征没有恢复 oracle 互补空间。

### 6.2 零费用、最多 5 个思路的本地搜索

在不调用 API 的前提下，事前冻结了 5 个新机制。每个机制先算真值条件动作空间 oracle；oracle 相对 B0 不足 `+3 pt` 就停止实现。

| 机制 | Oracle | 合法结果 | 相对 B0 / 判定 |
|---|---:|---:|---|
| F1 RR5：只重排 B0 Top-5 | 0.2712 | 0.2003 | `−0.0025`，CI 跨 0，失败 |
| F2 C1R：保守替换第 5 名 | 0.2827 | 0.2059 | `+0.0031`，CI 跨 0；CTID 为负且主要由 Stockpile 支撑，失败 |
| F3 OSER：B0/A/T/K 专家路由 | 0.3079 | 0.1852 | `−0.0177`，CI 完全低于 0，失败 |
| F4 LCB-RS：风险下界编辑 | 0.2544 | 0.1999 | `−0.0029`，CI 跨 0，失败 |
| F5 LCTR5：局部 campaign 转移 | 0.2202 | 未实现 | oracle 仅 `+0.0174`，低于事前 `+3 pt` 门 |

四个已实现方法全部配有 campaign 置换、无先验和等容量无意义内容控制；没有一个满足“CI 完全大于 0、CTID/Attack Flow 同时为正、超过置换、单 campaign 删除不翻转”的联合判据。

`F1 oracle = 0.2712` 的代码使用真实目标把命中候选事后移到前面。它只说明 B0 Top-5 内存在可重新排序的空间，不说明推理时可观测输入能够恢复该空间。

## 7. 最新 EDLR：让 LLM 直接读取转移证据

为排除“数值融合压坏 LLM 信号”的解释，最后尝试了 Evidence-Augmented Direct LLM Reranker。30 条开发样本每条运行四档：

| 档位 | 候选与证据 | 检验内容 |
|---|---|---|
| B0 | 原始直接排序 | 第一遍语义基线 |
| EA_TOP5 | 仅重排 B0 Top-5 | 第二遍 LLM 自重排 |
| UNION_LLM | B0/A/T/K 候选并集，无转移证据 | 二次推理与候选扩展 |
| EDLR | 同一并集 + 正确训练来源转移证据 | 真正的语义–转移融合 |
| EDLR_SHUFFLE | 同一并集 + 候选间旋转的证据 | 证据身份控制 |

所有请求均不含 source、campaign、样本标识、真实标签、未来步骤或未来文本；统计量只由另外两个正式训练来源计算。初次 120 请求有 3 条格式失败，按冻结 v2.1 只删除固定示例 ID、收紧摘要长度并重发 3 条；最终四档机械门全部通过。

开发集描述性结果：

| 方法 | CTID | Attack Flow | Stockpile | 来源等权 NDCG@5 |
|---|---:|---:|---:|---:|
| B0 | 0.0671 | 0.3109 | 0.2125 | 0.1968 |
| EA_TOP5 | 0.0733 | 0.3285 | 0.2204 | **0.2074** |
| UNION_LLM | 0.0671 | 0.2935 | 0.1705 | 0.1771 |
| EDLR | 0.0671 | 0.1847 | 0.1337 | **0.1285** |
| EDLR_SHUFFLE | 0.0733 | 0.2237 | 0.2091 | 0.1687 |

冻结差值：

```text
EA_TOP5 - B0             +0.0105
UNION_LLM - B0           -0.0198
EDLR - B0                -0.0683
EDLR - UNION_LLM         -0.0485
EDLR - EDLR_SHUFFLE      -0.0402
```

其他诊断：

- EA_TOP5 相对 B0 为 `5 胜 / 22 平 / 3 负`，只改变排序，Hit@5/Precision@5/Recall@5 按构造不变；`+1.05 pt` 只是小规模开发信号，不足以形成方法主张；
- EDLR 相对 B0 为 `0 胜 / 23 平 / 7 负`；
- EDLR 的 Hit@5 为 `0.3000`，B0/UNION_LLM 为 `0.4333`；
- 删除两条经过格式修复的 CTID 样本后，Attack Flow 与 Stockpile 上 `EDLR−UNION_LLM` 仍分别约为 `−10.88 pt`、`−3.68 pt`；
- 正确证据低于旋转证据，因而不能把任何变化归因于正确的序列统计。

结论：**不应把 EDLR 扩展到 784 条。** 当前转移证据会诱导 LLM 过度相信稀疏、不可迁移的计数，而不是补充语义排序。

## 8. API、费用与复现状态

主要已记录付费阶段：

| 阶段 | 请求尝试 | 输入 token | 输出 token | 估算/实测费用 |
|---|---:|---:|---:|---:|
| 784 条正式 B0 生成 | 787 | 641,862 | 127,285 | `$0.0741152384` |
| EDLR 四档试点 + v2.1 修复 | 143 | 218,016 | 10,892 | `$0.0284615744` |

两项合计约 `$0.1025768128`，不含更早的独立小型 API 试验。API key 始终只从环境变量读取，未写入代码、配置、日志或提交物。

复现措施包括：

- 协议先提交、结果后生成；
- 输入、脚本、prompt、输出 SHA-256 全部落盘；
- unique non-overwriting run directory；
- 逐请求原始 body/response、attempt、token、费用和 stdout；
- campaign 聚类 bootstrap 2,000 次；
- CTID leave-one-campaign influence；
- 本地融合完整独立重跑，核心托管输出逐字一致；
- 失败行不静默删除，格式修复另起协议和 run 保存。

## 9. 现在能写什么、不能写什么

### 可以写

1. 模板/随机划分会严重夸大技术序列预测性能；严格跨 campaign/source 留出后性能落到合理的低量级。
2. 真实跨来源 technique 转移覆盖极低，频率、Markov、LSTM、Transformer 和 SECRYPT-adapted HM 均缺乏稳定迁移收益。
3. procedure 文本驱动的直接 LLM Top-5 在发现集上优于当前转移/神经基线点估计，说明语义输入有价值。
4. 把 LLM 信号压成摘要 embedding/probe，或把稀疏转移统计通过固定、门控、局部编辑乃至 prompt 方式注入，均未形成可归因增益。
5. Unit42 不具备所需显式顺序；公开序列语料稀缺本身是任务的重要限制。

### 不能写

1. 不能声称当前 LLM–转移融合方法有效；
2. 不能把 B0 的事后发现写成预注册主方法成功；
3. 不能把 EDLR 的 30 条开发集结果做显著性或正式泛化结论；
4. 不能把真值条件 oracle、`transition_visibility` 或逐样本最优选择当作可部署方法；
5. 不能把 SECRYPT 随机划分的约 86% 与本研究 future-3 LODO 数字直接比较；
6. 不能继续在同一 784 行上增加机制并把最好一个包装成确认性结果。

## 10. 后续工作方向

### 路线 A：方法论文（推荐但需要新验证数据）

放弃跨 campaign 技术 bigram 作为序列分支，改为语义状态轨迹：

```text
prefix_t --LLM--> semantic belief/ranking q_t
prefix_{t+1} --LLM--> semantic belief/ranking q_{t+1}

sequence dynamics = q_t 到 q_{t+1} 的候选进入/退出、排名持续性、意图漂移和不确定性变化
```

这仍保留“LLM 语义理解 + 序列动态”，但序列规律来自同一 campaign 内连续观测的状态演化，而不是要求不同 campaign 重复相同 ATT&CK 技术转移。

执行边界：

1. 先在现有发现集上只做明确标注的 oracle/可学习性诊断；
2. 在看到结果前冻结特征、模型、控制和死亡条件；
3. 若没有至少可观测的机制空间，立即停止；
4. 若有空间，仍需新建未参与本轮选择的有序 procedure 数据做前瞻确认。

### 路线 B：评估论文

保留完整负结果，论文主张定位为：合成/随机划分如何夸大 ATT&CK next-step 性能、真实来源为何因覆盖率和域差异失效、哪些常见融合机制无法恢复该信号，以及公开有序语料为什么不足。

该路线证据链最完整，但不满足“必须提出一个正向新方法”的个人目标。

### 当前建议

1. 不运行 784 条 EDLR 全量；
2. 不再修补 technique-count 融合；
3. 下一项工作应是写一份全新的“语义状态轨迹”事前协议和零费用上界诊断；
4. 若该机制也没有可学习空间，就停止方法搜索，转向评估论文或投入数月自建新数据。

## 11. 关键产物索引

- 历史路线：`docs/research/2026-08-06-procedure-aware-cross-source-attack-roadmap.md`
- future-3 主协议：`project/data_v4/protocols/LLM_semantic_future3_lodo_validation_v8.md`
- 零费用融合搜索日志：`project/data_v4/local_fusion_search/trial_log_v1.md`
- EDLR v2 协议：`project/data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.md`
- EDLR 修复协议：`project/data_v4/protocols/evidence_augmented_llm_reranker_pilot_v2.1_repair.md`
- EDLR 开发集报告：`project/data_v4/results/edlr_pilot_v2_development_descriptive/report.md`
- EDLR 逐样本指标：`project/data_v4/results/edlr_pilot_v2_development_descriptive/per_sample_metrics.csv`
- Unit42 顺序审计：`project/data_v4/audits/unit42_playbook_sequence_v1/report.md`
- SECRYPT 划分审计：`project/data_v4/repro_secrypt/20260807_split_audit_v1/report.md`

## 12. 一句话状态

> LLM 直接语义排序有信号；跨来源 ATT&CK 技术转移统计没有显示可提取的互补增量，并且在最新 LLM 证据重排开发试点中的点估计表现为有害。当前融合主张未成立，下一步若坚持方法论文，必须把“序列”从稀疏技术 bigram 改成同一攻击过程中可观测的语义状态演化，并在新数据上前瞻验证。
