# GSAD：组校准、偏移感知的 ATT&CK-DAG 层级解码设计规格

日期：2026-07-30  
状态：设计与书面规格已获用户授权；用户明确要求跳过后续批准点并直接进入实验  
研究任务：MITRE ATT&CK 下一父技术预测

## 1. 研究结论与设计目标

现有 PR-HR 路由器没有证明有效：其 root-OOF Top-1 相对 GRU 仅提升 0.047 个百分点，root-cluster bootstrap 95% 区间跨越 0；普通置信度拒绝在内部受污染的 SIM 上看似有效，却在 CTID 外部数据上退化。因此，本项目不再把“重新学习一个融合权重或置信度门”作为核心创新。

GSAD（Group-calibrated Shift-aware ATT&CK-DAG Decoder）的目标是：在未知 actor、长尾技术和来源偏移下，不强迫模型始终输出一个精确技术，而是输出证据能够支持的最细可靠粒度：

\[
A(x)\in\{\{\hat y\},\;A_{\mathrm{DAG}}(x),\;\bot\},
\]

其中 \(\{\hat y\}\) 是精确父技术，\(A_{\mathrm{DAG}}\) 是由一个或多个 tactic/technique 节点组成的结构化集合，\(\bot\) 表示校准支持不足、需要人工复核。

本研究只在同时证明“可靠性、信息量和非平凡覆盖”时把 GSAD 判为有效。不能依靠全部拒识、始终输出宽泛 tactic 或反复查看最终测试集来通过门槛。

## 2. 文献边界与允许的创新声明

下列通用能力已经存在，论文不得把它们单独声称为创新：

- 类别聚类的 class-conditional conformal prediction：Ding et al., NeurIPS 2023，<https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html>。
- 层级选择分类及降低预测粒度：Goren et al., NeurIPS 2024，<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c8b100b376a7b338c84801b699935098-Abstract-Conference.html>。
- DAG/结构化 conformal prediction：Zhang et al., ICLR 2025，<https://proceedings.iclr.cc/paper_files/paper/2025/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html>。
- 多专家 learning-to-defer：Mao et al., ICML 2025，<https://proceedings.mlr.press/v267/mao25c.html>。
- 协变量漂移下的 weighted conformal prediction：Tibshirani et al., NeurIPS 2019，<https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html>。

若实验通过，GSAD 的可防守贡献限定为：

1. 面向 ATT&CK 下一技术预测，将多 tactic 归属表示为多父 DAG，而不是强制投影为单父树；
2. 用独立数据学习“ATT&CK 图邻接 + tactic 签名 + 长尾校准误差”类别簇，再进行簇条件预测集合校准；
3. 提出可精确求解的 DAG 表示压缩，使结构化显示覆盖原叶集合，并联合控制后代集合大小与显示节点数；
4. 以 actor/root 为实验交换和不确定性统计单位，利用推理时可见的偏移特征在精确技术、DAG 集合和拒识之间选择；
5. 同时报告精确技术正确性、层级覆盖、叶等价信息量、拒识率和未知 actor 压力测试，而不是只报告随机行切分的 Top-k。

本项目不得声称“首次提出 hierarchical CP”“首次将 CP 用于网络安全”或“首次进行 LLM 路由”。如果只有领域应用改写、没有通过强基线与消融，论文贡献必须降级描述。

## 3. 研究范围与非目标

### 3.1 范围

- 标签空间为当前数据中的 184 个 ATT&CK 父技术 ID；保留 fit 未见但全局词表存在的标签，用于 open-label 评估。
- 第一阶段使用 NumPy/pandas 统计序列模型产生完整 184 类概率，验证 GSAD 算法本身。
- 只有第一阶段通过预注册门槛，才在同一无泄漏协议上重训 GRU/神经主干并导出完整 logits，确认结论不依赖弱基模型。
- CTID/AEL 仅作为不参与任何选择的弱标注外部 OOD 压力测试。

### 3.2 非目标

- 不在本阶段重新设计大型语言模型或端到端 Transformer。
- 不把现有被 root 污染的 GRU/RL/LLM logits 用作正式效果证据。
- 不把包含 `matched next-step technique name/description/command summary` 的 micro-state richer 输出用于训练、调参或评估。
- 不用目标测试标签估计密度比、选择阈值、决定消融或修改算法。
- 不把当前 9-actor CTID 结果描述为确认性外部显著性证据。

## 4. 冻结数据协议

### 4.1 SIM 根定义

将 `sequence_id` 去掉 `_partNNN` 后缀得到 root。同一 root 的所有 part 必须进入同一分区。

原始三个 SIM 文件合并后有 14,128 行、789 条 sequences、162 个 roots。原 train/val/test 仅 sequence-disjoint，不是 root-disjoint：train∩test 有 65 个 roots；因此原划分和基于原划分训练的 logits 不能用于正式结论。

### 4.2 外部 actor 隔离

先从 SIM 移除与 CTID/AEL 外测相同的 9 个 actor roots：

`SIM_008, SIM_010, SIM_014, SIM_030, SIM_033, SIM_040, SIM_041, SIM_044, SIM_090`。

移除后保留 12,147 行、683 sequences、153 roots。

### 4.3 最终固定 93/20/20/20 分区

| 分区 | roots | 行 | sequences | 用途 |
|---|---:|---:|---:|---|
| fit | 93 | 7,371 | 413 | 估计序列模型、转移统计和归一化参数 |
| validation | 20 | 1,592 | 90 | 选择有限超参数、学习类别簇与偏移/正确性排序器 |
| calibration | 20 | 1,592 | 90 | 冻结 conformal 分位数与动作风险阈值 |
| untouched SIM test | 20 | 1,592 | 90 | 只对最终胜出方案揭盲一次 |

固定 validation roots：

`SIM_001, SIM_004, SIM_007, SIM_019, SIM_021, SIM_054, SIM_057, SIM_062, SIM_063, SIM_064, SIM_093, SIM_107, SIM_108, SIM_113, SIM_126, SIM_130, SIM_134, SIM_153, SIM_158, SIM_160`

固定 calibration roots：

`SIM_003, SIM_013, SIM_027, SIM_038, SIM_048, SIM_067, SIM_082, SIM_085, SIM_092, SIM_103, SIM_105, SIM_106, SIM_109, SIM_111, SIM_122, SIM_137, SIM_156, SIM_161, SIM_165, SIM_171`

固定 test roots：

`SIM_016, SIM_024, SIM_035, SIM_037, SIM_039, SIM_046, SIM_047, SIM_053, SIM_072, SIM_073, SIM_081, SIM_083, SIM_084, SIM_095, SIM_124, SIM_128, SIM_133, SIM_141, SIM_146, SIM_169`

fit 为 153 个候选 roots 扣除上述 60 个 roots 后的集合。

最终 test 中有 7 行、2 个 fit-unseen 标签。结果必须同时报告 closed-label 1,585 行、open-label 7 行和 overall 1,592 行；不得静默删除 open-label 行。

### 4.4 迭代阶段不触碰最终测试集

候选算法的失败淘汰只使用前 133 个非 test roots。具体使用 nested root cross-validation：外层 root fold 生成开发 OOF 结果，内层严格区分 fit、validation 和 calibration。算法、超参数和通过/失败决定均基于开发 OOF。

固定的 20-root test 只对一个冻结的最终胜出方案评估一次。若最终 test 未通过，不允许根据该 test 修改算法后再次声称其仍是 untouched test；必须新增独立 roots 或把结论降级为探索性。

### 4.5 ATT&CK 快照

图结构使用仓库中的 `enterprise-attack-18.1.json`，运行清单记录文件 SHA-256、ATT&CK 版本、184 类词表 SHA-256 和 technique→tactic 多父映射。不得在一次实验中混用 v18.1 与其他版本。

## 5. GSAD 算法

### 5.1 概率提供器

接口输入为父技术前缀 \(x=(t_1,\ldots,t_k)\)，输出完整的 \(p(y\mid x)\in\mathbb R^{184}\)。第一阶段实现并比较：

1. 平滑 unigram；
2. 一阶 Markov；
3. 二/三阶 interpolated n-gram，未见上下文逐级 backoff；
4. tactic→technique 层次统计模型。

所有计数仅来自 fit；n-gram order、平滑和不超过 3 个的层次权重只看 validation。calibration 和 test 不得改变概率模型。

第二阶段若启动，重训神经主干并导出完整 184 类 logits。现有仅含 Top-5 的文件不足以公平计算 APS、NLL、Brier 或完整候选集合。

### 5.2 独立学习长尾图簇

对每个标签 \(y\) 在 validation 上构造簇特征：

\[
v_y=[\operatorname{multiHot}(\tau(y)),\log(1+n_y),Q_{.25}(S_y),Q_{.5}(S_y),Q_{.75}(S_y)],
\]

其中 \(\tau(y)\) 是 tactic 多标签集合，\(n_y\) 是 fit 频次，\(S_y\) 是 validation 上真实标签为 \(y\) 时的 APS 非符合度。validation 未出现的标签只使用 tactic 签名和 fit 频次。

采用确定性的图约束凝聚合并：初始每个技术一个簇；只有共享 tactic 或在冻结 ATT&CK 图上邻接的簇可合并；按标准化特征距离从小到大合并，直到每个可合并稀有簇达到预注册的最小 validation 支持。最小支持从固定集合 `{10, 20, 30}` 中仅用 validation 选择。

类别簇函数 \(g(y)\) 在查看 calibration 前冻结。禁止在同一 calibration 样本上同时学习簇和计算阈值。

### 5.3 簇条件叶预测集合

基础使用 randomized APS 分数：

\[
S(x,y;u)=\sum_{j:p(j\mid x)>p(y\mid x)}p(j\mid x)+u\,p(y\mid x),\quad u\sim U(0,1).
\]

在 calibration 上为每个已冻结簇计算有限样本校正的 \((1-\alpha)\) 分位数 \(q_g\)：

\[
\Gamma(x)=\{y:S(x,y;u)\le q_{g(y)}\}.
\]

预注册 \(\alpha=0.10\) 为主结果，并报告 0.05、0.20 敏感性分析。若某簇 calibration 支持少于 5 个真实标签样本，使用全局阈值回退，并明确记录 fallback；不能从 test 借样本。

普通 split conformal 的理论保证依赖样本交换性，而本数据同一 root 内步骤相关。本研究不把逐行保证夸大为 actor 条件保证；主统计单位为 root，并报告 root-macro 指标和 root-cluster bootstrap。簇学习与阈值数据分离、以及下一节的覆盖保持性质，是可正式验证的算法不变量。

### 5.4 ATT&CK 多父 DAG 精确压缩

令 \(\Gamma(x)\) 为叶技术集合。可选显示节点为 14 个 tactic 节点和 184 个技术叶节点。对 tactic 子集 \(T\)，未被其后代覆盖的 \(\Gamma\) 技术作为单独叶节点加入：

\[
L_T=\Gamma\setminus D(T),\qquad A_T=T\cup L_T.
\]

在所有 tactic 子集上求：

\[
A^*(x)=\arg\min_{T\subseteq\mathcal T}
\left(|D(A_T)|+\lambda|A_T|\right),
\quad |A_T|\le r.
\]

14 个 tactic 允许枚举至多 \(2^{14}=16,384\) 个子集，因而无需近似 set cover。确定性 tie-break 顺序为：更小叶等价集合、更少显示节点、按 ATT&CK ID 字典序。

对每个输出必须满足：

\[
\Gamma(x)\subseteq D(A^*(x)).
\]

因此 DAG 压缩只能保持或扩张叶覆盖，不能丢失原 conformal 集合中的技术。单元测试在随机玩具 DAG 上把枚举解与完全 brute force 比较。

### 5.5 偏移与精确预测安全分数

仅使用推理时可见、与真值无关的低维特征：

1. 概率熵与 Top-1/Top-2 margin；
2. n-gram 实际 backoff 阶数或神经模型的预测集合大小；
3. root-balanced 技术/战术转移 surprise；
4. 前缀长度和重复率；
5. 到 fit 特征分布的稳健标准化距离。

validation 上用 root-balanced logistic model 学习“精确 Top-1 是否正确”的非单调组合分数。特征和交互总数上限为 5 个预注册主特征，不引入 14-feature PR-HR 式高自由度路由器。归一化、特征选择和系数只看 fit/validation。

CTID 无标签特征可用于报告 source→target shift distance，但 CTID 标签不得参与模型、距离尺度、阈值或特征选择。若未来实现 importance weighting，必须作为独立候选并遵守 weighted conformal 的条件，不能把任意局部权重乘进分位数后继续声称原保证。

### 5.6 三级动作策略

动作按以下固定顺序选择：

1. **精确技术**：\(\Gamma\) 为 singleton，且精确安全分数通过 calibration 上冻结的 root-balanced 风险阈值；
2. **DAG 集合**：否则输出 \(A^*\)，前提是其叶等价大小不超过 validation 冻结的 \(B_{\max}\)；
3. **拒识**：簇支持不足且全局回退仍产生过宽集合、出现未知图映射、或 \(|D(A^*)|>B_{\max}\)。

策略优化目标为：

\[
R_{\mathrm{miss}}+\lambda_I\,\mathbb E[\log(1+|D(A)|)]+\rho\,P(A=\bot),
\]

并受精确输出风险和后代集合漏失风险约束。\(B_{\max}\)、\(\lambda\)、风险阈值只在 validation/calibration 上冻结。风险曲线以每个 root 等权计算，不以长序列 root 的行数加权。

## 6. 基线与消融

### 6.1 概率与 Top-k 基线

- unigram、Markov、interpolated n-gram；
- tactic→technique 层次统计模型；
- 若进入第二阶段：干净重训的 GRU、论文原融合、固定权重 PoE。

### 6.2 不确定性与集合基线

- LAC、APS、RAPS、SAPS；
- Mondrian/class-conditional CP；
- 无图约束的 clustered CP；
- 只做 DAG 压缩的 structured CP；
- NeurIPS 2024 风格 hierarchical selective classification；
- 普通 Top-1 confidence rejection。

### 6.3 必做消融

- 去掉图约束类别簇；
- 去掉长尾簇，仅全局 APS；
- ATT&CK 多父 DAG 强制投影为单父树；
- 去掉 shift/density 特征；
- 按行加权替代 root-balanced；
- 去掉精确输出动作或去掉拒识动作；
- 使用贪心压缩替代精确枚举；
- 神经阶段去掉语义/LLM 分支，只保留干净 GRU。

## 7. 预注册成功与失败门槛

### 7.1 开发 OOF 和最终 locked test 均需满足

GSAD 被判为有效，必须满足下列 A 或 B 至少一项，并同时满足 C–G：

**A. 覆盖效率**：在匹配的平均叶等价集合大小下，层级后代覆盖比最强非 GSAD 基线提高至少 5 个百分点，2,000 次 root-cluster bootstrap 的差值 95% 区间下界大于 0。

**B. 精确产出效率**：在匹配的层级风险下，精确技术输出比例相对最强基线提高至少 10%；差值的 root-cluster bootstrap 95% 区间下界大于 0。

**C. 非平凡精确预测**：精确输出覆盖率至少 50%，其准确率相对普通置信度选择提高至少 5 个百分点，区间下界大于 0。

**D. 非平凡拒识**：总体拒识率不超过 20%。

**E. 非宽泛逃逸**：平均叶等价集合大小不得高于匹配覆盖率下的最佳 APS/RAPS/structured baseline；全集合率不得增加。

**F. 校准有效**：主目标 90% 覆盖率的逐行结果落在 88%–92%，同时报告 root-macro coverage；若逐行和 root-macro 差异超过 5 个百分点，必须解释且不能声称 actor 稳健。

**G. 消融归因**：完整 GSAD 必须显著优于至少“无 DAG”“无 shift”“无 root balancing”三项中的两项；否则不能把提升归因于核心机制。

### 7.2 外部 CTID 探索门槛

CTID 当前合并为 9 个 actor clusters，其中 Turla Carbon/Snake 视为同一 actor。所有阈值在看 CTID 标签前冻结。外部压力测试要求：

- 高置信但错误的精确技术输出比例相对普通 confidence gate 降低至少 50%；
- actor-macro hierarchical risk 或 actor-macro MRR/Hit@5 不劣于最佳干净统计基线；
- 至少 6/9 actors 的方向一致；
- 报告 9-actor cluster bootstrap，但明确标注其区间不稳定、仅探索性。

若 CTID 不通过，GSAD 不能声称外部 OOD 有效；可保留为 SIM 内部算法结果，但目标仍未完成，需要迭代或补充更可靠外部数据。

### 7.3 立即停止条件

- 90% flat APS 已接近 singleton，GSAD 没有可压缩空间；
- 图簇在独立 calibration 上大量回退，无法形成稳定阈值；
- 提升只能通过拒识率超过 20% 或输出接近全标签实现；
- 开发 OOF 的 A/B/C 区间下界不大于 0；
- 标签置换后仍出现“显著提升”；
- 任意 future-context 黑名单字段进入特征或提示词；
- 外部 actor 同名 roots 未从 SIM fit 移除；
- 发现最终 test 被用于候选选择。

## 8. 失败后的迭代顺序

迭代只使用开发 OOF，不读取 locked test：

1. **GSAD-Core**：长尾图簇 + APS + DAG 精确压缩 + root-balanced 三级策略；
2. **GSAD-Shift**：若 Core 的集合有效但精确动作排序失败，加入预注册的低维 shift 特征；
3. **Weighted GSAD**：若 source→target 显示 covariate shift 且无标签 target window 可用，严格实现 importance-weighted conformal 与 ESS 监控；
4. **终止或重新研究**：若前三者均未通过开发门槛，记录失败摘要，重新检索文献并提出新设计；不得继续堆叠特征或查看 locked test 调参。

每轮只保留：候选名称、预注册假设、数据哈希、关键指标、通过/失败原因和下一步。只有最终胜出版本交付详细代码、配置、逐样本结果、完整消融和中文报告。

## 9. 数据异常与错误处理

- fit-unseen 标签保留在 184 类空间；预测不可能命中时计入 overall 错误，并单独报告 open-label slice。
- technique 缺少 tactic 映射时，只能作为叶节点显示；运行清单列出全部缺失映射。
- calibration 簇支持不足时回退全局阈值，并计数；不自动合并到 test 上表现最好的簇。
- APS 集合为空时加入最高概率标签并标记 forced-nonempty；该比例单独报告。若超过 1%，视为实现或校准异常。
- CTID parser 的父子重复、`scenario_id=unknown` 和多技术步骤取第一个 ID 问题必须在报告中披露。正式投稿前需要修复解析器并人工抽检，或增加至少 20–30 个独立 actor/flow clusters。
- 所有随机过程使用配置中冻结的 seed；随机 APS 的 `u` 按 sample ID 派生，保证可复现。

## 10. 测试与可复现性

### 10.1 数据完整性测试

- 四个 SIM 分区 root 交集为 0；行数、sequence 数、root 数与本规格一致；
- 9 个外部同 actor roots 不在任何 SIM fit/validation/calibration/test；
- `sequence_id,prefix_len` 唯一；前缀最后状态与下一标签不发生列错位；
- 黑名单字段扫描阻止 `next_technique`、`true_label`、`matched_*next*`、`matched_description`、`matched_command_summary` 进入特征；
- CTID 标签列不会传入 fit、validation 或 calibration API。

### 10.2 算法单元测试

- n-gram backoff、概率归一化、未见上下文回退；
- APS score、有限样本分位数、簇 fallback；
- 图簇只依赖 fit/validation；
- 多父 tactic 后代并集正确；
- 14-tactic 枚举压缩在玩具图上等于 brute-force 最优解；
- `Gamma ⊆ Desc(A*)` 对每个样本成立；
- 三级动作边界和拒识条件可预测；
- root-macro 指标、cluster bootstrap 和 matched-cost 插值正确。

### 10.3 负控制

- 标签置换后所有效果门槛应失败；
- 把测试标签列改名伪装后，黑名单/接口白名单仍应阻止读取；
- 使用原污染 split 时审计器必须报错而不是只发 warning；
- 使用含下一步描述的 micro-state 文件时审计器必须硬失败。

### 10.4 运行产物

每次运行至少生成：

- `run_manifest.json`：代码版本、依赖、seed、数据/图/词表 SHA-256、分区 roots；
- `data_audit.json`：泄漏和字段审计；
- `metrics.csv`：逐行、root-macro、closed/open-label、外部 actor-macro 指标；
- `bootstrap_intervals.csv`；
- `predictions.csv`：完整概率引用、Gamma、结构化节点、后代集合、动作和原因；
- `gates.json`：每个 A–G 门槛的数值、区间和布尔结果；
- `iteration_summary.md`：非最终轮次的简要记录。

最终胜出版本另外生成完整消融表、风险—信息量曲线、外部 actor 表、可复现命令和中文详细报告。

## 11. 代码边界与预期位置

实施计划应把职责分成可独立测试的小模块，预期位于：

- `project/experiments/gsad/data_protocol.py`
- `project/experiments/gsad/probability_models.py`
- `project/experiments/gsad/conformal.py`
- `project/experiments/gsad/attack_dag.py`
- `project/experiments/gsad/shift_policy.py`
- `project/experiments/gsad/metrics.py`
- `project/experiments/gsad/run_development.py`
- `project/experiments/gsad/run_locked_evaluation.py`
- `project/tests/test_gsad_*.py`

locked evaluation 入口必须要求一个由 development 阶段产生的冻结配置和哈希，并在结果目录存在时默认拒绝覆盖，防止无意中多次查看最终测试结果。

## 12. 完成定义

只有下列条件全部满足，本研究目标才算完成：

1. 至少一个候选在开发 OOF 和一次性 locked SIM test 上达到第 7 节门槛；
2. CTID 外部探索门槛通过，或获得更可靠的独立数据并通过同等预注册门槛；
3. 强基线、必做消融、负控制、数据泄漏测试全部执行；
4. 代码、测试、配置、逐样本结果、哈希清单和复现命令齐全；
5. 中文最终文档清楚区分已证明、探索性证据、理论保证和局限；
6. 独立 review 未发现数据泄漏、测试集调参、统计单位错误或与近期文献重复的核心声明。

任何单一内部点估计提升、受污染 logits、9-actor 不稳定区间或没有消融的结果，都不足以声称 GSAD 有效或足以支撑 SCI 二区以上投稿。
