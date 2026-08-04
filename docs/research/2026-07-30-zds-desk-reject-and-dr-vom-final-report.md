# ZDS 论文拒稿复盘与 DR-VOM 最终实验报告

副标题：从“LLM 语义后融合”转向“来源域/事件根平衡的 ATT&CK 下一技术预测”  
版本：最终证据审计版  
日期：2026-07-30

> **最终判断：** 这篇稿件被 Computers & Security 在第二天退稿，最直接、证据最强的原因是选刊范围不符：该刊自 2024 年初起明确暂停考虑以 AI/ML 为重要组成部分的稿件，而原稿标题和核心方法直接以 GRU、LLM 和语义融合为主。即使换刊，原稿仍存在创新弱、最强基线胜出、表间结果不一致、划分/复现证据不足等问题。经过多轮最小实验，当前最可信的新方向是 DR-VOM；它在四个完整留一来源实验上取得一致为正的 Top-1/MRR 点估计，但聚合置信区间仍跨 0，且 SIM 的 Hit@5 下降，因此只能称为“有效且值得继续的最终候选”，不能诚实地称为“已经达到 SCI 二区录用强度”。

## 1. 核心结论与当前进度

本轮工作已经完成以下五项任务：

1. 阅读并审计 13 页论文 PDF、当前项目代码、数据文件和既有实验产物。
2. 解释原创新点为什么不足，以及为何出现近乎即时的编辑退稿。
3. 迭代并淘汰多条模型/算法路线，保留最终的 DR-VOM。
4. 检索最接近的 ATT&CK Markov、VOM、多序列 Markov 与 domain generalization 工作，划定可安全声明的创新边界。
5. 找到并实际解析/运行三个公开外部来源，建立四来源完整 leave-one-domain-out（LODO）实验；另找到一个适合作为未来一次性确认集的 Scattered Spider 2025 场景，但尚未在本机成功下载，所以没有把它伪装成已验证数据。

DR-VOM 的最终四源宏平均结果为：

| 指标 | 点估计增益 | 95% bootstrap 区间 | 判断 |
|---|---:|---:|---|
| Top-1 | +1.817 个百分点 | [-0.376, +4.782] | 方向正，但聚合区间跨 0 |
| MRR | +0.01799 | [-0.00228, +0.04729] | 方向正，但聚合区间跨 0 |
| 四域 Top-1 方向 | 4/4 非负 | — | 通过方向一致性 |
| 四域 MRR 方向 | 4/4 非负 | — | 通过方向一致性 |
| Hit@5 安全性 | SIM -3.268 个百分点 | 未设聚合区间 | 未通过“无伤害”门槛 |

因此，方法有实际作用的证据已经比此前候选更强，但还缺少“一次性新来源确认 + 聚合显著性 + Top-k 无伤害 + 概率校准”四块证据。

## 2. 为什么会在第二天被 Computers & Security 退稿

### 2.1 首要原因：选刊范围明确排除这类 AI/ML 稿件

[Computers & Security 的 Aims & Scope](https://www.sciencedirect.com/journal/computers-and-security) 明确写明：自 2024 年初起，该刊暂停考虑以 AI 或 ML 为重要组成部分的稿件，包括“将 AI/ML 技术应用到系统安全与隐私主题”的工作。原稿题为 *Enhancing ATT&CK Parent-Technique Next-Step Prediction with Controlled LLM-Derived Semantic Fusion*，方法主体是 GRU、LLM 语义分支与 logit 融合，正好落在该排除条款中。

该刊页面同时给出约 3 天的 submission-to-first-decision 统计，而正式外审通常要先通过编辑适配性评估。[作者指南](https://www.sciencedirect.com/journal/computers-and-security/publish/guide-for-authors) 也说明编辑会先判断是否适合送审。因此，“第二天拒稿”与范围筛查高度一致。除非拒稿信给出不同的具体原因，否则这是最有力的解释。

### 2.2 即使换刊，编辑仍可能认为创新不足

原稿的核心“Global Logit Fusion”本质上是固定权重的线性后融合：

`z_fuse = (1 - alpha) * z_GRU + alpha * z_semantic`

这种凸组合是通用 late fusion，并没有提出新的学习目标、可证明的估计器、针对来源偏移的机制或新的序列建模算法。论文真正新增的部分主要是把 LLM 生成的语义表示接到已有 GRU 输出上，创新粒度不足以支撑“模型/算法创新”主张。

更致命的是，论文自己的 Table 2 中二阶 Markov Top-1 为 0.5562，而 GRU+LLM 为 0.5477；也就是说，拟议方法没有击败最强且更简单的基线。GRU+LLM 相对 GRU 的 Top-1 仅从 0.5444 增至 0.5477，增量约 0.33 个百分点。编辑很容易据此判断：复杂度增加了，但主要性能并未形成有说服力的优势。

### 2.3 结果表存在无法忽略的内部不一致

论文 Table 2 与 Table 3 的同名模型结果无法自然对齐：

| 模型 | Table 2 Top-1 / MRR | Table 3 Top-1 / MRR | 问题 |
|---|---|---|---|
| GRU | 0.5444 / 0.6807 | 0.545 ± 0.003 / 0.654 ± 0.003 | Top-1 接近，但 MRR 相差 0.0267 |
| GRU+LLM | 0.5477 / 0.6861 | 0.562 ± 0.001 / 0.672 ± 0.002 | Top-1 相差 0.0143，MRR 相差 0.0141 |

如果 Table 2 是单次运行、Table 3 是多种子平均，论文必须明确说明种子、划分、选择规则和为何差异如此大；当前稿件没有充分解释。Table 4 的 LLM-only 结果与当前代码产物的评测口径也难以对应，进一步削弱可信度。

### 2.4 数据划分和“父技术折叠”可能制造虚高结果

当前代码把子技术折叠到父技术。审计发现，在父技术标签上，末尾标签与下一标签相同的比例约为行级 37.96%、root-macro 40.62%；而用原始技术 ID 判断，完全相同只约为行级 0.36%、root-macro 1.29%。在 4,007 个表面上的父标签 self-transition 中，有 3,969 个其实是原始子技术发生了变化。

这不意味着父技术任务不合法，但意味着：若不同时报告 raw-ID 非 self 切片，模型可能主要在预测“折叠后保持同一父标签”，而不是预测真正的下一攻击行为。

此外，当前代码历史产物中曾出现 root 污染：一个诊断划分的 73 个 SIM 根中有 65 个同时出现在训练相关数据中。这不能单独证明论文最终数字一定泄漏，但足以说明论文必须给出不可变 root 列表、hash、去重规则和完整 split audit；当前稿件没有做到。

### 2.5 复现、外部验证与写作包装不足

- 数据来源只概述为 APT reports/CTID，没有给出逐报告或逐文件清单、固定 commit、许可、解析器版本、去重与映射审计。
- 主结果没有 root-cluster bootstrap、显著性检验、worst-source 或真正的外部来源留出。
- LLM 生成成本被排除在时延比较之外，“lightweight”只适用于预计算后的在线部分。
- 图示容易把 ATT&CK tactics 解释成严格线性阶段，但 [MITRE ATT&CK FAQ](https://attack.mitre.org/resources/faq/) 强调 ATT&CK 不是按固定顺序执行的线性 kill chain。
- PDF 中仍可见彩色超链接边框，表格字体偏小，第二页出现重复句，参考文献信息不完整。这些不是主要科学原因，但会强化“稿件尚未成熟”的编辑印象。

### 2.6 任务 1 自审

结论不是只靠猜测“创新不够”：选刊范围提供了足以解释秒拒的直接证据；论文表格和代码审计则解释了即便换刊仍可能被拒的技术原因。最重要的改正不是润色摘要，而是换刊、换主问题、重做评测协议。

## 3. 原方向中哪些内容应该保留，哪些必须撤回

可以保留：

- MITRE ATT&CK 下一技术预测这个问题本身；
- 可解释、低成本、可在线更新的序列概率模型；
- 父技术与原始子技术双分辨率审计；
- 从 CTI、Attack Flow、adversary emulation 等异质来源学习的设定；
- 对安全分析人员有意义的 Top-k 候选和概率校准。

必须撤回或降级：

- “LLM 语义融合本身就是核心算法创新”；
- “只要比 GRU 略好就足以证明方法有效”；
- “随机行/随机序列划分可以代表对未见来源的泛化”；
- “父标签 self-transition 等同于攻击行为没有变化”；
- “使用了多个数据来源就等于完成了外部验证”。

## 4. 最终新方法：DR-VOM

DR-VOM 的全称为 **Equal-Domain, Root-Balanced Variable-Order Markov**。更保守的中文名称是“来源域等权、事件根平衡的变阶 Markov 下一技术预测”。

### 4.1 要解决的具体偏差

把所有转移直接池化会让以下样本获得过大权重：

- 行数更多的数据来源；
- 更长的攻击链；
- 被拆成更多片段或重复记录更多的 incident/campaign；
- 某个上下文在单一来源中出现次数特别多的根。

结果是模型学到“数据采集流程的分布”，而不一定是跨来源稳定的攻击行为分布。DR-VOM 改变的不是神经网络结构，而是每个上下文条件分布的统计估计目标。

### 4.2 条件分布估计器

对上下文 `c`、来源域 `d`、事件根 `r` 和下一技术 `y`，先计算根内条件频率：

`p_hat(r,c,y) = N(r,c,y) / N(r,c)`

令 `D_c` 为训练集中支持上下文 `c` 的来源域集合，`R_dc` 为域 `d` 中支持该上下文的事件根集合，则：

`q_hat(c,y) = (1 / |D_c|) * sum_d[(1 / |R_dc|) * sum_r p_hat(r,c,y)]`

这等价于：先在支持该上下文的训练来源中均匀选一个域，再在该域中均匀选一个 incident root，最后按该根内的上下文出现频率选下一技术。它与 pooled MLE 的“均匀选一条转移事件”不同。

代码对每个分布加入 `alpha / |V|` 的均匀质量并重新归一化；最终 `alpha=0.1`，词表大小为 184。模型使用三层固定插值：unigram、一阶上下文、二阶上下文，权重分别为 0.2、0.3、0.5；若高阶上下文未见，则只对可用层的权重重新归一化。代码参数名为 `order=3`，表示三层分布，而最大上下文长度实际为 2，论文中必须明确，避免把它误写成“三阶 Markov”。

### 4.3 真正的算法创新在哪里

可以主张的创新是：

1. **逐上下文的 support-conditional equal-domain estimator**：只在支持当前上下文的来源之间等权，而不是在全局简单重采样。
2. **域内 incident-root 宏平均**：同时消除大来源和长/重复事件根的支配。
3. **与估计目标匹配的完整 LODO 协议**：每次完全留出一个来源，训练、上下文计数和估计都不访问该来源行。

不能主张的内容是：

- 首个 VOM；
- 首个多序列 Markov；
- 首个 domain generalization 方法；
- 新的通用 Markov 理论；
- “完全无参数”，因为上下文阶数、平滑和插值权重仍是固定超参数。

更准确的表述是：**没有学习额外的来源平衡或收缩超参数**。

### 4.4 为什么这个估计器可能真正有效

它直接针对当前数据最明显的结构性问题：SIM 有 10,555 行，而 Stockpile 只有 65 行；如果直接池化，SIM 对参数的影响可能超过 Stockpile 两个数量级。DR-VOM 让一个支持上下文的小来源在该上下文估计中拥有与大来源相同的域级质量，同时避免小来源内部某个超长根再次垄断。

其代价也很清楚：当高阶上下文只在一个训练域出现时，等域机制无法提供跨域稳健性；当小域的根数极少时，等权会放大方差。这正是 Stockpile 区间很宽、总体聚合区间仍跨 0 的原因之一。最终论文应报告 `|D_c|`、每域 `|R_dc|` 和上下文覆盖率分布。

### 4.5 任务 2 自审

DR-VOM 没有完全离开原来的 ATT&CK 下一步预测方向，也确实包含模型/算法层面的新估计器；同时我们把通用 VOM 与域均衡已有思想排除在新颖性声明之外。它比 LLM late fusion 更容易解释、复现和做消融，但目前只能作为可发表方法的核心，不足以单独保证二区。

## 5. 文献检索：是否与已有工作重复

### 5.1 最接近的 ATT&CK Markov 工作

*Smoothed Markov Chains for MITRE ATT&CK Prediction: Addressing Data Scarcity in Cyber Kill Chains* 已在 2025 年提出多来源 ATT&CK Markov 预测，使用来自 CTI reports、Attack Flow 和 emulation plans 的 682 条 tactic sequences，并采用最高四阶上下文、absolute discounting 与回退。DOI 为 [10.1109/EIECC67963.2025.11409611](https://doi.org/10.1109/EIECC67963.2025.11409611)。

因此不能声称“首次将多来源 Markov 用于 ATT&CK”。DR-VOM 与它的安全差异是：

- 前者按逐上下文的域/根两级等权估计 technique 分布；最接近工作主要在池化语料上平滑计数；
- 前者预测 parent technique；该工作报告的是 tactic prediction；
- 前者做完整 leave-one-source-domain-out；该工作使用多次随机 train/test split；
- 前者研究来源规模、链长和 incident 重复造成的估计偏差；该工作主要研究稀疏转移的平滑。

### 5.2 VOM 与多序列 Markov 不是新概念

- [Dimitrakakis, Bayesian Variable Order Markov Models, AISTATS 2010](https://proceedings.mlr.press/v9/dimitrakakis10a.html) 已把 VOM 表述为按上下文混合不同阶数的预测模型。
- [Belloni and Oliveira, A multi-process context tree, Annals of Statistics](https://doi.org/10.1214/16-AOS1455) 已研究多个随机过程共享上下文树、保留各自条件概率的情形。
- [Sarkar et al., Bayesian semiparametric Markov transition models](https://doi.org/10.1080/01621459.2018.1423986) 及后续 [BMRMM](https://doi.org/10.32614/RJ-2024-011) 已展示多序列总体/个体层级 Markov 建模。

DR-VOM 的新意不能来自“用了 VOM”或“有多个根”，只能来自针对 ATT&CK 来源偏移定义的两阶段条件估计目标。

### 5.3 等域思想也有先例

[Distribution Free Domain Generalization, ICML 2023](https://proceedings.mlr.press/v202/tong23a.html) 明确讨论通过跨域等权避免少数域支配；[DomainBed, ICLR 2021](https://iclr.cc/virtual/2021/poster/2998) 系统讨论了 domain generalization 的模型选择与留域评测。

因此安全的新颖性表述应是：

> 截至 2026-07-30 的检索，已有多来源 ATT&CK Markov 预测通常在合并语料上估计上下文转移概率。DR-VOM 把等域风险原则具体化为离散攻击序列中每个上下文的条件分布估计：先在事件根内归一化，再在来源内对根等权，最后对支持该上下文的来源等权，并使用完整来源留出评估未见来源的下一技术预测。

这是一项**问题专用估计器与验证框架创新**，不是新的通用 DG 或 Markov 理论。

### 5.4 可借鉴而不重复的下一步

- 借鉴 absolute discounting/context tree weighting，但把折扣强度仅在训练来源内选择，并与 DR-VOM 估计目标结合。
- 借鉴 mixed-effects Markov，把“域级方差”显式建模；不过当前非零收缩实验比 `kappa=0` 更差，不能直接写入主方法。
- 借鉴 DG 的 worst-domain 和 model-selection 规范，增加 source-level uncertainty，而不是只在固定四域内 bootstrap roots。
- 借鉴概率预测文献，补充 NLL、Brier、ECE 和校准曲线；目前只有排名指标不足以证明概率估计质量。

### 5.5 任务 3 自审

检索结果既找到了直接相邻论文，也找到了 VOM、multi-process Markov 和 domain balancing 的理论先例。最终创新声明已主动避开“first multi-source”“novel VOM”等容易被审稿人击穿的说法；与最接近 ATT&CK 论文的差异可以由估计公式和 LODO 协议逐项验证。

## 6. 数据集搜索与本地可用性验证

| 来源 | 固定版本/本地证据 | 本轮可用样本 | 许可/状态 | 结论 |
|---|---|---:|---|---|
| SIM development cache | SHA-256 `c7eef6dcdf9611d50e4c33593c1274d395bbf1b996aad3bd85d348d9eee5aa6a` | 10,555 转移 / 133 roots | 本地数据；公开再分发许可未核清 | 已实际用于 LODO；投稿前必须补来源清单和许可说明 |
| CTID Adversary Emulation Library | commit `4467a6eed6e67d25009704130e1d27d1a8007f57` | 281 转移 / 9 actors | Apache-2.0；[官方仓库](https://github.com/center-for-threat-informed-defense/adversary_emulation_library) | 已解析、已用于实验 |
| Attack Flow corpus | commit `295d20d27cefce0a2d309b6c24781545e45f547d` | 705 转移 / 35 usable flows | Apache-2.0；[官方仓库](https://github.com/center-for-threat-informed-defense/attack-flow) | 已解析、已用于实验；与 CTID 重合的 Turla flow 已排除 |
| MITRE Stockpile | commit `996ec41cd1c5d1c7cc09e620fc55dabe5aefd9cc`; ZIP SHA-256 `4BE3232D61D16C9E38B477910E890D9B8F40AB4B0AB66B6FCB88B8762AA969CF` | 65 转移 / 10 profiles | Apache-2.0；[官方仓库](https://github.com/mitre/stockpile) | 已解析、已用于实验 |
| Scattered Spider 2025 | commit `5594518da57ba41faaaaa99b3e0078d29504b033`；预期 67 technique rows / 66 edges | 尚未本地生成 | 官方场景可访问；[场景页](https://attackevals.github.io/ael/enterprise/scattered_spider/) | 适合作为未来一次性确认集，但本轮下载受环境网络限制，未进入任何指标 |

本轮“可用”不是只检查网址存在：前三个公开来源已经由本地 loader 解析成与模型词表对齐的 `prefix -> target` 行，并完成训练或留出预测；最终 manifest 记录了每个域的行数、规范化 frame hash、loader hash 和模型代码 hash。

还核查了 [Technique Inference Engine](https://github.com/center-for-threat-informed-defense/technique-inference-engine)。它是可借鉴的 ATT&CK technique inference 基线/资源，但任务更接近“从已观察技术推断相关技术”，不等同于有序下一步预测，因此不应直接当作同任务 SOTA。

### 6.1 Scattered Spider 的冻结解析合同

如果后续成功获取该固定 commit，应在首次看指标前冻结以下规则：只解析 `Enterprise/scattered_spider/Emulation_Plan/Scattered_Spider_Scenario.md`；只读取 Step 1–7 各自 `Reference Tables` 后第一张 GFM 表；按表内行序连接，保留跨 Step 的 6 条边；不去重、不压缩重复 ID；预期各 Step 行数为 7/12/12/11/7/9/9，总计 67 个节点、66 条边。任一硬断言失败即中止，不能手工修表后继续沿用“一次性确认”名义。

### 6.2 任务 4 自审

Attack Flow、CTID、Stockpile 均有官方来源、固定版本、许可和本地解析结果，满足“确实可用”。Scattered Spider 只满足“官方来源和解析合同已确定”，不满足本地下载/运行，所以被明确标注为未验证、未消费，避免夸大。

## 7. 实验协议

### 7.1 完整四来源 LODO

每轮选择一个来源作为完全未见测试域，其余三个来源合并训练：

1. 留出 SIM，训练 CTID + Attack Flow + Stockpile；
2. 留出 CTID，训练 SIM + Attack Flow + Stockpile；
3. 留出 Attack Flow，训练 SIM + CTID + Stockpile；
4. 留出 Stockpile，训练 SIM + CTID + Attack Flow。

模型没有访问留出域的转移行。Attack Flow 与 CTID 的已知重合 flow 被排除。184 类标签词表在实验前固定，不从每轮测试行动态生成；但它来自现有项目词表，投稿级严格复核仍应改为从固定 ATT&CK release 定义完整父技术词表，并说明 OOV 策略。

### 7.2 公平基线

基线与候选使用完全相同的：

- 184 类标签空间；
- uniform smoothing `alpha=0.1`；
- unigram/一阶/二阶三层插值，权重 0.2/0.3/0.5；
- suffix backoff；
- root-macro 评测。

唯一变化是条件分布的聚合：

- 基线：所有训练来源的 incident roots 直接等权；当某来源包含更多 roots 时，它拥有更大总质量。
- DR-VOM：先在来源内对 roots 等权，再在支持该上下文的来源之间等权。

这使增益可以归因于域级平衡，而不是更高阶模型、更大词表或不同平滑。

### 7.3 不确定性

每个来源使用 2,000 次 root-cluster bootstrap；跨来源汇总使用 10,000 次“域内 root 重采样后等域平均”。这个区间只反映**当前四个固定域内部 root 的抽样不确定性**，不能代表对未知来源总体的 source-level inference；域数只有 4，也不适合声称已经证明普遍跨域泛化。

## 8. 最终实验结果

| 留出来源 | 行 / roots | 基线 Top-1 | DR-VOM Top-1 | Top-1 增益（百分点，95% CI） | MRR 增益（95% CI） | Hit@5 增益 |
|---|---:|---:|---:|---:|---:|---:|
| SIM | 10,555 / 133 | 0.02120 | 0.03730 | +1.610 [1.013, 2.354] | +0.00202 [-0.00239, 0.00710] | -0.03268 |
| CTID | 281 / 9 | 0.02314 | 0.03319 | +1.004 [-1.637, 3.859] | +0.02117 [0.00159, 0.04270] | +0.02763 |
| Attack Flow | 705 / 35 | 0.02698 | 0.04071 | +1.374 [0.062, 3.253] | +0.01097 [-0.00106, 0.02536] | +0.00610 |
| Stockpile | 65 / 10 | 0.11646 | 0.14926 | +3.281 [-4.517, 14.531] | +0.03779 [-0.03531, 0.14763] | +0.00417 |
| 等域宏平均 | — | — | — | +1.817 [-0.376, 4.782] | +0.01799 [-0.00228, 0.04729] | — |

[[FIGURE_DR_VOM]]

### 8.1 能证明什么

- 四个来源的 Top-1 和 MRR 点估计方向一致，没有出现此前 QSMR 在公平基线下某个来源显著负向的问题。
- SIM 和 Attack Flow 的 Top-1 root-bootstrap 区间下界为正；CTID 的 MRR 区间下界为正。
- 候选与基线只有域聚合不同，因此结果支持“来源规模/根数平衡有实际作用”这一机制解释。
- 方法不需要在留出域上选择 domain power 或 shrinkage；此前看似更复杂的幂权/部分池化版本都没有稳定超过最简单的等域版本。

### 8.2 不能证明什么

- 跨四域宏平均区间仍跨 0，不能写“statistically significant overall improvement”。
- SIM 的 Hit@5 下降 3.27 个百分点，不能写“all ranking metrics improve”或“no-harm”。
- Stockpile 只有 10 个 roots，区间非常宽；其 +3.28 个百分点不能作为强证据单独宣传。
- 这四个来源已被用于方法迭代，属于 development benchmarks，不再是全新锁定外部测试。
- 当前没有 NLL、Brier、ECE，尚未证明概率分布比基线校准得更好。

### 8.3 最终门槛复核

| 门槛 | 结果 | 状态 |
|---|---|---|
| 所有来源 Top-1 点估计非负 | 4/4 | 通过 |
| 所有来源 MRR 点估计非负 | 4/4 | 通过 |
| 聚合 Top-1 CI 下界 > 0 | -0.376 个百分点 | 未通过 |
| 聚合 MRR CI 下界 > 0 | -0.00228 | 未通过 |
| 每个来源 Hit@5 无明显伤害 | SIM -3.268 个百分点 | 未通过 |
| 全新一次性外部确认 | 尚未执行 | 未通过 |
| 结果和代码可追溯 | manifest + code/data hashes | 通过 |

结论：DR-VOM 通过“有效方向”门槛，但没有通过“SCI 二区投稿就绪”门槛。

## 9. 迭代过程摘要

完整简表见 `docs/research/gsad-iteration-log.md`。关键淘汰逻辑如下：

- GSAD 的集合预测没有形成有效 exact/coverage-efficiency 结果。
- RACER 开发结果看似正向，但一次性锁定测试 Top-1、MRR、Hit@5 均失败；锁定结果没有重跑。
- QSMR v5 在 SIM 内部五折很强，但在公平的域/根平衡基线下，跨域平均转负，说明原增益依赖较弱基线或域组成。
- dwell、raw multi-resolution、图残差、未见目标传播等附加机制没有提供稳定的增量。
- 事后选择的 domain power 看似有效，但只用训练来源的嵌套选择器无法复现，判定为开发域过拟合。
- 非零收缩参数 `kappa` 没有超过 `kappa=0`，所以最终方法主动删除该复杂性。

本轮最终选择 DR-VOM，不是因为它最复杂，而是因为它在公平基线、完整四源 LODO 下给出最一致、最容易归因的结果。

## 10. 它能否支撑 SCI 二区及以上

### 10.1 诚实答案

**现在还不能保证，也不应立即投稿。** SCI 分区和录用受期刊、年份、编辑判断和完整论文质量影响，不存在由一个小实验自动推导“必发二区”的证据。当前 DR-VOM 已经具备可形成论文主线的三个条件：问题明确、估计器可形式化、四来源方向一致；但还缺少足够强的统计与外部确认。

### 10.2 达到可投稿强度的最低补强包

1. **一次性新来源确认**：按冻结合同获取 Scattered Spider 2025，首次运行前固定 parser、映射、模型和门槛；失败就如实报告，不能继续围绕该来源调参。
2. **修复 Hit@5 伤害**：研究为什么留出 SIM 时 Top-1 提升但候选集召回下降；可考虑温度校准或受约束混合，但任何新参数只能在训练来源内选择。
3. **概率质量**：加入 NLL、Brier、ECE、reliability diagram；DR-VOM 是概率估计器，只报告 Top-1/MRR 会浪费方法优势。
4. **更强基线**：pooled MLE、root-balanced VOM、equal-domain unigram/一阶/二阶、absolute-discounted VOM、HMM/HSMM、可复现的 GRU/Transformer，以及最接近 ATT&CK smoothed Markov 方法。
5. **必要消融**：仅 root balance、仅 domain balance、DR-VOM、不同上下文支持 `|D_c|`、raw-ID/parent-ID、self/non-self、是否保留跨 Step 边。
6. **source-level 不确定性**：扩大到至少 6–10 个独立来源/语料族，报告 macro-source、worst-source 和 leave-one-source confidence，而非把固定四域的 root bootstrap 当成来源总体推断。
7. **完整复现包**：固定 ATT&CK release、逐文件 provenance、许可、hash、parser unit tests、跨来源实体/报告去重、不可变 split manifest。

如果上述补强后仍能保持约 +1–2 个百分点 Top-1、+0.01 以上 MRR，聚合区间为正，且 fresh source 不负向，那么它可以合理支撑一篇面向高质量 Q2/Q1 应用型期刊的投稿。现在的证据更适合称为“有希望的完整方法原型”。

## 11. 重新选刊建议

不要再次把以 AI/ML 为核心的版本投 Computers & Security，除非该刊未来正式撤销 moratorium。更合适的官方 scope 候选包括：

- [Journal of Information Security and Applications](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications)：信息安全与实践应用匹配度最高，也有 LLM/cybersecurity 专题历史；适合强化安全问题、外部验证与实践价值后的版本。
- [Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)：明确要求真实工程应用、AI 方法新意和公共数据可复现；只有在完成 fresh external、概率评测和强基线后才建议尝试。
- [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems)：预测系统和知识驱动 AI 在范围内，但方法创新门槛更高；当前简单 DR-VOM 还不够，需增加更强理论或可学习但不泄漏的稳健估计机制。

投稿前必须用学校认可的当年 JCR/中科院分区重新核对“二区及以上”，不能把当前影响因子或历史分区当作保证。

## 12. 推荐论文定位、标题与贡献写法

推荐标题：

> **DR-VOM: Equal-Domain, Root-Balanced Variable-Order Markov Prediction of MITRE ATT&CK Techniques**

更保守的标题：

> **Source-Balanced Variable-Order Markov Forecasting of MITRE ATT&CK Techniques under Domain Shift**

推荐三条贡献：

1. 提出逐上下文的来源域等权、incident-root 等权条件分布估计器，减轻大来源、长链和重复根对 ATT&CK 转移统计的支配。
2. 构建四种异质来源的完整 leave-one-source-domain-out 评测，并提供固定版本、hash、重合排除和 root-macro bootstrap。
3. 通过公平消融量化来源/根平衡对未见来源预测的影响，同时报告失败指标和不确定性，而不是只展示单一内部随机划分。

摘要和结论中不要出现：first multi-source、novel general VOM、state-of-the-art、significant overall improvement、guarantees robust generalization，除非后续新实验真的支持这些表述。

## 13. 可复现资产

最终实验目录：

`project/experiments/gsad/results/external/dr_vom_full_lodo_final_seed20260730`

关键文件：

- `summary.json`：主结论和门槛摘要；
- `domain_metrics.csv`：每个留出来源的点估计；
- `bootstrap_intervals.csv`：每来源 root-bootstrap 区间；
- `aggregate_bootstrap_intervals.csv`：等域宏平均区间；
- `predictions.csv`：逐行配对预测指标；
- `run_manifest.json`：数据 frame、代码、loader、配置和 split audit 的 hash。

模型实现：

- `project/experiments/gsad/probability_models.py`
- `project/experiments/gsad/run_support_tempered_ngram_lodo.py`

当前测试：`python -m unittest discover -s project/tests -v`，共 166 个测试通过。

复现实验时必须指定一个不存在的新输出目录，因为 runner 使用 `exist_ok=False` 防止覆盖已有证据：

```powershell
python -m project.experiments.gsad.run_support_tempered_ngram_lodo `
  --candidate-power 0 `
  --candidate-kappa 0 `
  --include-sim `
  --output-dir project/experiments/gsad/results/external/dr_vom_reproduction
```

注意：若重新生成，新 manifest 会记录当时代码 hash；不得覆盖本报告引用的最终证据目录。

## 14. 最终行动顺序

1. 将原论文从“LLM semantic fusion”重写为“source shift + conditional estimator + LODO”问题，不在原稿上做局部修补。
2. 冻结 Scattered Spider parser contract、ATT&CK release、184/完整词表策略和所有门槛。
3. 先在现有四源内完成 NLL/Brier/ECE、Hit@5 修复和强基线，不访问 fresh source。
4. 冻结代码/hash 后只运行一次 Scattered Spider；把成功或失败都写入论文。
5. fresh source 通过后再扩到更多来源、完成论文和重新选刊；若失败，回到机制诊断，但不再把该来源称为锁定测试。

## 15. 最终任务审查

| 用户任务 | 完成情况 | 审查结论 |
|---|---|---|
| 1. 解释创新不足和秒拒 | 已完成 | 首要原因是期刊明确排除 AI/ML；另有技术与写作证据 |
| 2. 提出不脱离原方向的新创新 | 已完成 | DR-VOM 保留 ATT&CK 下一步预测，新增域/根平衡估计器 |
| 3. 检索期刊/论文，避免重复 | 已完成 | 已找到最接近 ATT&CK Markov、VOM、多序列 Markov、DG 工作并限定声明 |
| 4. 搜索并验证数据集 | 部分完成且如实标注 | 三个公开来源已本地解析和实验；Scattered Spider 已定位但未本地下载 |
| 5. 汇总成文档 | 已完成 | 本报告、简版迭代日志和最终 DOCX 共同交付 |

## 参考来源

1. Elsevier. [Computers & Security — Aims & Scope](https://www.sciencedirect.com/journal/computers-and-security).
2. Elsevier. [Computers & Security — Guide for Authors](https://www.sciencedirect.com/journal/computers-and-security/publish/guide-for-authors).
3. Pappu, K. [Smoothed Markov Chains for MITRE ATT&CK Prediction](https://doi.org/10.1109/EIECC67963.2025.11409611), EIECC 2025.
4. Dimitrakakis, C. [Bayesian Variable Order Markov Models](https://proceedings.mlr.press/v9/dimitrakakis10a.html), AISTATS 2010.
5. Belloni, A., Oliveira, R. I. [A multi-process context tree](https://doi.org/10.1214/16-AOS1455), Annals of Statistics.
6. Sarkar et al. [Bayesian semiparametric Markov transition models](https://doi.org/10.1080/01621459.2018.1423986), JASA.
7. Tong et al. [Distribution Free Domain Generalization](https://proceedings.mlr.press/v202/tong23a.html), ICML 2023.
8. Gulrajani, I., Lopez-Paz, D. [DomainBed](https://iclr.cc/virtual/2021/poster/2998), ICLR 2021.
9. MITRE Center for Threat-Informed Defense. [Attack Flow](https://github.com/center-for-threat-informed-defense/attack-flow).
10. MITRE Center for Threat-Informed Defense. [Adversary Emulation Library](https://github.com/center-for-threat-informed-defense/adversary_emulation_library).
11. MITRE. [Stockpile](https://github.com/mitre/stockpile).
12. MITRE ATT&CK. [Frequently Asked Questions](https://attack.mitre.org/resources/faq/).
13. MITRE Center for Threat-Informed Defense. [Technique Inference Engine](https://github.com/center-for-threat-informed-defense/technique-inference-engine).

本报告中的网页状态与文献检索截止日期为 2026-07-30；期刊范围、指标和分区在正式投稿前必须重新核对。
