# LLM 语义分支在真实跨来源数据上的验证（#4）v4.2

这是核心思路的第一次真实检验，决定论文能否成立。

## v4.2 修订说明

v4 已撤下 v3 中存在数学上界错误、弱锚点反向工作及一致率无解释力问题的非对称门控 B2，改为 LLM Top-5 排名先验；B1 单独承担原方法成立判定，取消在外层测试域取 `max(B1, B2)`；同时加入 intention-to-treat 失败处理和 groundedness 盲审。

v4.1 在此基础上进一步机械化以下事项：

1. 删除将 campaign-macro 的 2 pt 错误换算为固定样本数的表格，给出正确的 campaign 贡献公式。
2. 为全部判死条件补齐“三个来源”的量词，并增加“未成立、未判死、证据不足”状态。
3. 固定可训练档与确定性档的 seed、inner-LODO 和 campaign bootstrap 聚合顺序。
4. groundedness 盲审只作为接受门，不允许在本协议内据此修改语义 prompt。
5. 补全 B0/B1/B2 的全部诊断逻辑。
6. 固定 BGE 分块后的最终归一化和 C1b 仅在有效 reasoning 内置换。

v4.2 在任何模型训练、测试或付费 API 调用前，补齐前置审计发现的两个任务定义缺口：

1. 原始三源数据保留 834 行；确认性主实验采用固定 184 类词表定义的闭集视图，只过滤 4 条 `true_label` 不在词表内的行，得到 830 行。prefix 中的词表外技术保留。
2. 4 条过滤行没有删除任何 campaign，因此主实验的 campaign-macro 分母固定为 CTID `J=10`、Attack Flow `J=35`、Stockpile `J=27`；四个受影响 campaign 的组内 `n_j` 单独冻结。
3. 对三个含环 Attack Flow 文件，固定按源文件节点顺序执行 DFS、删除指向灰色祖先的回边，再在所得 DAG 上求最长路径；禁止尝试其他断环策略挑结果。
4. 增加原始 834 行的统一失败敏感性分析，但它不参与任何正式判定。

## 0. 背景与目的

论文核心思路（不可变的底线）：**LLM 语义理解能力 + 序列转移规律能力的融合**。

原实验在 `sim_*` 合成数据上完成，战术阶段单调不减比例为 13371/13371 = 100.0%（真实攻击会反复回跳）。任务因此塌缩为“背模板”：二阶马尔可夫 0.5562 打败 GRU+LLM 0.5477，语义融合仅提升 +0.33 pt。

已验证的两个事实：

| 结论 | 数据 |
|---|---|
| 跨来源时转移统计基本失效 | 池化主干 Top-1 仅 0.0214 / 0.0371 / 0.0410 |
| 失效原因是转移未被观测，而非估计偏差 | 未见转移 85.8% / 91.6% / 73.0%；来源域等权宏平均 −1.03 pt |

上述数字使用 127 类候选空间计算，存在标签空间泄漏，见 §3.1。改用固定 184 类词表后需重跑基线，这些旧数字不得用于计算本实验的相对提升。

本实验的目的是量化语义信号的实际量级并定位瓶颈，不是证明方法有效。正负结果都有价值。

## 1. 数据

### 1.1 真实来源（主实验，三折 LODO）

| 来源 | 主实验闭集文件 | 主实验行数 | 原始行数 | campaign / J | prefix_len | 战术单调 | 性质 |
|---|---|---:|---:|---:|---:|---:|---|
| CTID | `project/data_v2/repro_external/closed_set/ctid_in184.csv` | 281 | 281 | 10 | 1–71 | 0.625 | 真实 emulation plan |
| Attack Flow | `project/data_v2/repro_external/closed_set/attack_flow_cumulative_in184.csv` | 428 | 431 | 35 | 1–31 | 0.650 | 真实报告衍生 |
| Stockpile | `project/data_v2/repro_external/closed_set/stockpile_cumulative_in184.csv` | 121 | 122 | 27 | 1–17 | 0.861 | 半合成 |

若 `cumulative/` 不存在，按附录 A 重建。重建产物、脚本、质量报告和输入哈希必须先提交，再花 API 费用。

原始文件保持 834 行且不得修改。闭集视图只按运行前固定的 184 类词表过滤输出真值；不得过滤 prefix 内的技术。闭集审计必须保存被过滤行、过滤前后 campaign 集合、各 campaign 的 `n_j` 与全部文件哈希。

### 1.2 SIM 的角色

SIM 只作为 A 档与 D 档的描述性对照，用于展示合成数据上的任务塌缩。SIM 不参与训练、不参与选择超参数、不进入主结论。

原因：SIM 仍使用旧 Qwen 缓存，且有 222 条截断，其语义输入与三源真实数据不同源，会成为混淆变量。

### 1.3 规模现实

确认性主实验三源合计 830 行、72 个 campaign，固定标签空间为 184 类，平均每类不到 5 个样本。原始 834 行只进入 §5.8 的敏感性分析。

该规模大概率无法获得统计显著性。此前在 1051 行上完成的四源 LODO 聚合 95% CI 跨 0。本实验的项目判定以预注册的方向一致性与量级为主，显著性为辅。汇报中禁止使用“接近显著”等措辞。

## 2. 不可违反的规则

1. **留出域完全隔离。** λ 与 epoch 只能通过 §4.7 的 inner-LODO 选择；其他训练超参数按 §4.6 冻结，不做任何搜索。
2. **标签空间固定。** 使用 `project/data_v2/core/rl_label_vocab.csv`，SHA-256 为 `9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738`（185 行 = 表头 + 184 类）。禁止从实验数据动态生成词表。运行前校验哈希，不匹配立即停止。
3. **闭集口径固定。** 主实验只使用 `true_label∈V` 的 830 行；训练、inner-LODO、外层测试、试点抽样和全部八档使用同一视图。4 条 OOV 真值不得训练、不得映射、不得扩词表。prefix 内 OOV token 保留。
4. **bootstrap 单位是 campaign。** 共 2000 次；同一 replicate 对全部八档使用完全相同的 campaign 重采样。seed 的聚合顺序见 §4.7。
5. **主点估计口径固定。** Top-1、Hit@5、MRR 先在每个 campaign 内平均，再对 campaign 等权，即 campaign-macro。Macro-F1 例外：在整个来源的行上使用固定 184 类重算，`zero_division=0`，CI 仍通过 campaign 聚类 bootstrap 获得。row-micro 结果只放补充材料。
6. **禁止事后调整阈值。** §6 的阈值在运行前固定。
7. **结果为负就报负。** 禁止使用“虽然主指标没提升但某子集有意义”“语义分支更易解释”“可提供可读推理理由”“接近显著”等补救话术。
8. **每一步必须落盘。** 保存逐样本预测、结果 CSV、完整 stdout 和配置快照。逐样本预测至少包含 `source`、`campaign_id`、`prefix_len`、真值、Top-20 排名及分数。B0 的例外见 §4.2。
9. **禁止使用 Superpowers 的 TDD、writing-plans 和 requesting-code-review。** 只允许 systematic-debugging 排查运行故障。
10. **API Key 只从环境变量读取。** 不得写入代码、配置、日志或任何提交物。

## 3. 主干与融合

### 3.1 主干 A

使用三层插值与后退：order-2 / order-1 / unigram，权重固定为 0.5 / 0.3 / 0.2。未见高阶上下文时只对可用层重新归一化。计数为行池化，每条训练转移等权。

uniform smoothing 的总质量固定为 `alpha_s=0.1`。对上下文 `h`：

```text
P(c | h) = (N(h,c) + alpha_s / |V|) / (N(h) + alpha_s)
```

固定词表大小 `|V|=184`。

参考实现为 `project/data_v2/scripts/lodo_backbone_check.py`，但必须先修复标签空间泄漏。旧脚本从全部三源构建 vocab，导致留出 Attack Flow 时 41 个只在留出域出现的标签进入候选集，留出 CTID 和 Stockpile 时分别有 6 个和 3 个。

### 3.2 融合形式

固定使用 log-opinion pooling：

```text
z_A    = log(P_A + eps)                 # eps = 1e-12
z_S    = log_softmax(semantic_logits)
z_fuse = (1 - lambda) * z_A + lambda * z_S
```

`lambda ∈ [0,1]`，网格步长 0.05，共 21 个取值。最终概率统一通过 `softmax(z_fuse)` 获得。

## 4. 八档对照

| 档 | 配置 | 检验内容 |
|---|---|---|
| A | 池化转移主干 | 基线 |
| B0 | LLM 直接 Top-5 | LLM 本身是否含预测信号 |
| B1 | A + reasoning→BGE→MLP，固定 λ | 原论文方法形式 |
| B2 | A + LLM Top-5 排名先验，无 MLP | 直接语义排名是否可与主干互补，以及信号是否卡在 probe 链路 |
| C1a | 确定性 ATT&CK 描述基线 | 排除“多一个编码器和 MLP 就会涨” |
| C1b | 分层置换 reasoning | 参数量和文本分布匹配的信息消融 |
| C2 | A + 二阶 campaign 等权 Markov | 排除一般异质集成效应 |
| D | A + 战术规则 | 语义是否超过战术阶段序号信息 |

### 4.1 C1a 命名

C1a 的输入是 prefix 最后一步技术对应的 ATT&CK 官方描述，因此仍是 prefix 的函数。其准确名称为“确定性 ATT&CK 描述基线（deterministic description baseline）”：文本由 prefix 唯一决定，但不含针对该 prefix 生成的推理。

ATT&CK 描述来源的版本与文件 SHA-256 必须写入 manifest。

### 4.2 B0：LLM 直接 Top-5

使用 `predicted_next_ttps` 的返回顺序作为排名。真值在第 `k` 位时 `rank=k`；不在前五时 Top-1、Hit@5 和 RR 均为 0。

两个口径例外：

1. B0 的 MRR 实际为 MRR@5，必须单独标注，不能与其他档的完整 184 类 MRR 同列比较。
2. B0 无法提供 Top-20 分数，只保存 Top-5 及其顺序，并在 manifest 中记录该例外。

### 4.3 B2：LLM Top-5 排名先验融合

B2 直接利用 B0 已有的五个有序候选构造先验，不训练 MLP，也不增加 API 调用。

事前固定：

```text
r = softmax([0, -1, -2, -3, -4])

P_rank[c]   = 0.1 / 184             # 对全部184类初始化
P_rank[T_k] = P_rank[T_k] + 0.9*r[k] # k=0..4

z_rank = log(P_rank + 1e-12)
z_B2   = (1 - lambda)*z_A + lambda*z_rank
P_B2   = softmax(z_B2)
```

λ 使用与 B1 相同的 inner-LODO 网格。B2 不依赖 probe，因此不选择 epoch，也没有训练 seed。

若某样本的 `predicted_next_ttps` 无效，B2 在该样本上回退为 A，见 §5.8。

### 4.3.1 为什么不使用 v3 的非对称门控

v3 曾尝试把 S-GRec 的 advantage-space 门控迁移到 184 类 logit 空间，但在运行前发现三个问题：

1. **全局无穷范数不能保证逐类上界。** 当 `||d_A||∞=||d_S||∞=1`、`d_A[c]=0.01`、`d_S[c]=0.5` 时，全局 magnitude 为 1，但语义贡献 0.5 大于主干量级 0.01。逐类上界至少需要逐类比值 `min(|d_A[c]|,|d_S[c]|)/(max(|d_A[c]|,|d_S[c]|)+eps)`。
2. **弱锚点下方向门控会反向工作。** S-GRec 的锚点是可靠的 business advantage，而本文 A 在旧实验中的跨来源 Top-1 仅为 0.02–0.04。若语义想提升 A 判为低概率的类别，门控会把这种修正归零，尤其会封死 transition-unseen 样本。
3. **184 类符号一致率会被尾部类别支配。** 即使两个 Top-5 完全不重叠，全部类别的符号一致率仍可能接近 50%，不能解释 Top-1 融合，也不能与 S-GRec 的 rollout advantage 一致率比较。

后续可研究 reliability-conditioned asymmetric fusion：只在 transition seen 且 A 高置信时使用方向门控，在 transition unseen 或 A 低置信时允许语义独立修正。该方案记为 B3，不在本实验执行，也不进入本实验结论。

### 4.4 C1b：只在有效 reasoning 内分层置换

在 `source × prefix_len短/中/长三分位` 内进行无固定点置换。具体规则：

1. 只有 `valid_reasoning=True` 的行进入置换池。
2. 禁止任何样本拿回自己的 reasoning。
3. 格子少于两条时，与相邻 prefix_len 格合并后再置换。
4. 无效行不进入置换池，直接按 §5.8 回退 A。
5. 禁止有效行被置换到无效或空文本。
6. 置换使用一个事前固定的随机种子；保存完整 `(target_key, donor_key)` 映射。

### 4.5 C2 与 D 的完整定义

#### C2：A + 二阶 campaign 等权 Markov

- 最高阶固定为 2，与 A 相同，但聚合方式不同。
- 对每个上下文 `h`，先在每个实际观察到 `h` 的 campaign 内按 §3.1 的公式得到平滑条件分布 `P_j(c|h)`，再只对观察到 `h` 的 campaign 等权平均。未观察到 `h` 的 campaign 不作为全零分布加入平均。
- 若所有训练 campaign 都未观察到二阶上下文，则后退到一阶；一阶仍未观察到则后退到 unigram。
- `alpha_s=0.1`，固定 184 类词表。
- `z_C2=log(P_C2+eps)`，与 A 使用相同 log-opinion pooling。
- λ 通过 inner-LODO 选择；C2 是确定性方法，没有 epoch 和训练 seed。

#### D：A + 战术规则

- 对固定词表中的每个父技术查 ATT&CK 主战术；属于多个战术时取附录 A 中 14 战术序号的最小值。
- prefix 最后一步战术序号为 `base`，候选战术序号为 `t`，`gap=t-base`。
- 固定打分：`gap==1 → 1.0`，`gap==0 → 0.8`，`gap==2 → 0.6`，`gap>2 → 0.3`，`gap<0 → 0.4`。
- 候选战术查不到时得分为 0。
- 若 prefix 最后一步的战术查不到，则 D 在该样本上直接回退为 A，不进行规则融合。
- 分数归一化为概率，`z_D=log(P_D+eps)`，再与 A 做 log-opinion pooling。
- λ 通过 inner-LODO 选择；D 是确定性方法，没有 epoch 和训练 seed。

旧 D 数字基于 127 类候选空间与 score 级线性混合，不得直接搬入新表，必须在固定 184 类词表和相同三折上重跑。

### 4.6 训练超参数

除 λ 和 epoch 外，不进行任何搜索。固定：

```text
optimizer           = Adam
learning_rate       = 0.003
weight_decay        = 1e-4
dropout             = 0.3
batch_size          = 256
max_epochs          = 150
checkpoint_interval = 5
probe               = 768 -> 256 -> 184
encoder             = BAAI/bge-base-zh-v1.5，冻结
train_seeds         = 42, 43, 44, 45, 46
```

manifest 必须记录 BGE 模型和 tokenizer 的 Hugging Face revision/commit、`transformers` 版本、`torch` 版本、tokenizer 配置和 deterministic 设置，包括 `torch.use_deterministic_algorithms`、cuDNN 设置及全部 seed 设置点。

### 4.7 inner-LODO、seed 与 bootstrap

#### inner-LODO

每个外层折留出一个来源，剩余两个来源轮流作为 inner 训练源与验证源，共两个 inner 折。按两验证来源等权的 campaign-macro Top-1 选择参数。

可训练档 B1、C1a、C1b：

- 每个 seed 独立选择 λ 与 epoch。
- 并列打破顺序：campaign-macro Top-1 最高、λ 最小、epoch 最早。
- 选定后，使用该 seed 与选定参数在两个外层训练来源的合并数据上重训，再评估留出来源。

确定性档：

- A、B0 无 λ、无 epoch、无 seed。
- B2、C2、D 只通过 inner-LODO 选择 λ；没有 epoch 和 seed。

#### 点估计

- B1、C1a、C1b：先分别计算 5 个 seed 的来源指标，再取 seed 均值作为来源点估计。
- A、B0、B2、C2、D：每个外层折只计算一次。

#### campaign 聚类 bootstrap

每个来源完成 2000 次重采样。每个 replicate：

1. 有放回地重采样完整 campaign。
2. 对全部八档使用同一组重采样 campaign。
3. 对 B1、C1a、C1b 分别计算 5 个 seed 的指标，再在 replicate 内取 seed 均值。
4. 确定性档直接计算一次。
5. 在完成 seed 聚合后计算所有配对差值。

seed 不得作为额外独立样本加入 bootstrap。CI 反映 campaign 变异，不宣称覆盖训练随机性的不确定性；5 个 seed 的单独结果和范围必须同时报告。

### 4.8 BGE 截断与分块

现有 `project/llm/train_llm_multiseed.py` 使用 `max_length=512`。若试点中的 tokenizer 截断率超过 5%，自动切换为非重叠分块编码，不修改 prompt，也不缩短 reasoning。

分块规则固定为：

```text
content_limit = 512 - tokenizer.num_special_tokens_to_add(pair=False)
按 content_limit 对原始 token IDs 做连续、无重叠分块
每块添加模型所需特殊 token
每块取 CLS embedding
每块分别做 L2 normalize
对全部块取算术平均
对平均向量再次做 L2 normalize
```

切换后必须重新运行完整 30 条试点。B1 与 C1b 使用完全相同的分块实现。

## 5. Reasoning 生成

### 5.1 数量与成本

确认性主实验三源共 830 条 prefix，预计费用约 2 元。SIM 不在本次语义生成范围内。原始 834 行敏感性分析中的 4 条 OOV 行不调用 LLM；它们按 §5.8 统一记为失败，避免为不可能输出的类别生成语义结果。

### 5.2 API 调用

| 项 | 固定值 |
|---|---|
| 模型 ID | 运行前调用 `/models`，确认并保存真实模型 ID；预期为 `deepseek-v4-flash` |
| BASE URL | `https://api.deepseek.com` |
| temperature | 0.0 |
| max_tokens | 4096 |
| 关闭思考 | `extra_body={"thinking":{"type":"disabled"}}` |
| JSON 输出 | `response_format={"type":"json_object"}`，返回后本地 JSON Schema 校验 |
| 并发 | 起步 50 |

每次响应必须保存：返回模型 ID、completion ID、system fingerprint、HTTP request ID、finish reason、`reasoning_content` 长度、attempt 次数、延迟、HTTP 状态和错误类型。

`finish_reason=="length"` 时直接标记为 `truncated`，即使 JSON 恰好可解析。429 和 5xx 使用指数退避加 jitter。空内容必须重试。

### 5.3 Prompt

System prompt：

```text
你是一个高级 APT 威胁狩猎专家与 ATT&CK 攻击图分析师。
你的任务是：基于攻击者已执行的 ATT&CK 技术序列（Prefix）以及相关的知识图谱上下文（KG Context），推断攻击者当前的【阶段性操作状态】，并直接预测下一步最可能执行的 5 个 ATT&CK Parent Technique（父技术）。
输入是真实攻击活动的技术序列。你必须遵守以下严格限制：
1. 绝对不要凭空捏造微观动作。推理必须完全基于传入的 Prefix ID 及 KG Context 进行逻辑推演。
2. 预测结果必须是纯粹的父技术 ID。
请以 json 格式输出，在 `_thinking_process` 字段中写下你的推理过程，按以下三步进行思考：
[战术阶段评估]：分析 Prefix 中最后两步，它们处于什么战术阶段？
[已获资产推演]：基于前缀技术，攻击者目前掌握了什么级别的粗粒度资产或权限？
[意图图谱映射]：结合 KG Context，前缀的最后几步操作最可能为后续攻击开启了什么逻辑攻击面？
推理完成后，请在 `predicted_next_ttps` 数组中输出恰好 5 个最可能的下一步 ATT&CK 父技术 ID。
输出格式示例：
{"_thinking_process":"[战术阶段评估]...[已获资产推演]...[意图图谱映射]...","predicted_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}
```

User prompt：

```text
### 攻击前缀序列 (Prefix) ###
{prefix}
(重点关注最后两步：{recent_ids})

### 相关的知识图谱上下文 (KG Context) ###
{kg_context}

### 任务要求 ###
请先在 `_thinking_process` 字段推演，随后在 `predicted_next_ttps` 数组输出 5 个预测的父技术 ID。
```

### 5.4 KG context 前置门

KG 构建是试点前置门。构建失败时停止，不启动 API 生成。无 KG 版本只能作为另一个预注册实验。

复用 `project/data/build_attack_kg_snippets.py` 的逻辑，context 截断到 700 字符。每条样本保存：snippet ID 列表、截断前后字符数、KG 来源文件哈希和空 KG 原因。

必须断言：KG 只能由 prefix 中的技术构造，不得读取 `true_label`、未来步骤或数据集转移统计。违反即构成标签泄漏。

若某条 prefix 的全部技术在固定 KG 中均无可用 snippet，可保留空 KG，但必须标记为 `no_prefix_snippet`；这不等同于 KG 构建流程失败。空 KG 行数和比例必须按来源报告。

### 5.5 唯一键

必须使用 `(source, campaign_id, prefix_len)`。禁止单独使用 `campaign_id` 或 `sequence_id` 作为 key。

此前 `hotfix_align.py` 使用 `dict(zip(df['sequence_id'], ...))`，因 key 不唯一导致后值覆盖前值，9919 行只剩 551 个唯一 reasoning，且原地覆盖源文件、数据不可恢复。本实验不得重复该错误。

### 5.6 输出与 manifest

写入 `project/data_v4/external_reasoning/`，不得覆盖现有文件。至少包含：

```text
source, campaign_id, prefix_len, prefix, true_label,
llm_thinking_process, predicted_next_ttps, raw_output,
generation_status, has_reasoning, valid_reasoning, valid_top5,
finish_reason, model_returned, completion_id, system_fingerprint,
request_id, reasoning_content_len, attempt, latency_ms, http_status,
kg_snippet_ids, kg_chars_before, kg_chars_after,
generated_at, input_tokens, output_tokens
```

`generation_status ∈ {ok, json_parse_failed, api_error, truncated, empty_content}`。失败最多重试 3 次，仍失败时如实保存，不得丢弃或填空。

`generation_manifest.json` 必须记录：

- `/models` 返回与最终使用的模型 ID；
- 全部 API 参数；
- system prompt、user prompt template 和 JSON Schema 各自的 SHA-256；
- KG 截断长度、来源文件哈希与空 KG 统计；
- 并发数、起止时间、各来源行数与失败计数；
- 累计 token 与实测费用；
- 生成脚本和全部输入文件 SHA-256；
- 标签词表 SHA-256；
- §4.6–§4.8 的全部训练参数、依赖版本与编码配置。

生成完成后立即 git commit 一份未经后处理的原始版本。

### 5.7 试点

每源 10 条，共 30 条。抽样算法在运行前固定如下：

1. 只从 830 行闭集 KG 文件抽样；随机种子固定为 `20260806`。
2. 在每个来源内，对 `prefix_len` 使用 nearest-rank 经验三分位：`q1=x_(ceil(N/3))`、`q2=x_(ceil(2N/3))`；`prefix_len≤q1` 为 short，`q1<prefix_len≤q2` 为 medium，其余为 long。
3. 每源固定抽取 short 3 条、medium 4 条、long 3 条，共 10 条；10 条必须来自 10 个不同 campaign。
4. 候选 campaign 与 campaign 内候选行均按 `SHA256(seed, source, stratum-slot, unique-key)` 排序，并使用确定性回溯找到满足约束的第一组。不得因样本内容、真值或 LLM 输出重新抽样。
5. 保存 30 条唯一键、分层、分位点、输入/输出/脚本 SHA-256。若任一来源无法满足约束，停止并发布新协议，不放宽后重抽。

按上述算法首次运行得到并冻结：CTID `q1=10,q2=22`，Attack Flow `q1=5,q2=10`，Stockpile `q1=2,q2=5`；`pilot_sample_30.csv` SHA-256 为 `3d0e1dbd5f27e3bd17ddf951508787760f338bfbd6a068ba9bdf07de36ad40c0`。这些数值是固定算法在已冻结闭集输入上的派生审计值，不得用于重新抽样。

| # | 检查 | 阈值 | 不通过时的固定动作 |
|---:|---|---:|---|
| 1 | JSON 解析成功率（3 次重试后） | ≥95% | 只允许检查 JSON Schema、解析器和机械格式约束；重跑试点 |
| 2 | `reasoning_content` 长度为 0 或字段缺失 | 100% | 检查关闭思考参数；重跑试点 |
| 3 | `finish_reason==length` 比例 | ≤2% | 提高 `max_tokens`，记录偏离并重跑试点 |
| 4 | 三个段落标题全部存在 | ≥95% | 失败则停止 v4.2，不得在本协议内修改语义 prompt |
| 5 | Top-5 恰好 5 个、互不重复、均属于固定词表 | ≥90% | 只允许加强 JSON Schema/格式校验，不得添加影响候选语义选择的提示 |
| 6 | reasoning 无格式模板之外的空段落 | ≥95% | 只允许修复机械格式约束；重跑试点 |
| 7 | BGE tokenizer 截断率 | ≤5% | 自动启用 §4.8 分块并重跑完整试点 |
| 8 | reasoning 字符长度均值 | 600–1100 | 只记录可比性差异，不阻塞 |

#### 5.7.1 groundedness 盲审

对全部 30 条试点进行人工盲审。审阅者不得看到 `true_label`。审阅表、审阅者标识、逐项判断和理由必须落盘。

| 检查项 | 阈值 |
|---|---:|
| ATT&CK 战术判断正确率 | ≥90% |
| 无依据的资产或权限推断比例 | ≤10% |
| KG 结论可由所列 snippet 追溯的比例 | ≥90% |
| 五个候选中明显无依据项的比例 | ≤20% |

KG 追溯率只在存在非空 KG context 的试点行上计算。对于 `no_prefix_snippet` 行，reasoning 必须明确表示没有可用 KG 证据，并且不得声称由 KG 得出了具体关系；违反时计为 groundedness 失败。分母和 `no_prefix_snippet` 行数必须同时报告。

groundedness 盲审是**接受门，不是 prompt 开发集**：

- 在 v4.2 内，盲审结果不得用于修改战术推理、资产推演、KG 使用方式、候选预测要求或其他语义指令。
- 任一 groundedness 门失败时，v4.2 停止，不生成剩余 800 条。
- 若要修改语义 prompt，必须发布新的协议版本并重新预注册，同时披露 prompt 已使用三源无标签样本进行校准。当前 30 条不得进入新版本的最终评估。仅排除 30 行不能将新版本重新描述为“从未接触留出来源分布”。
- API、解析器、JSON Schema、重试和分块编码等纯机械修复可以在 v4.2 内完成，但每次修复后必须重新运行完整 30 条试点。
- 若盲审通过且未修改语义 prompt，30 条试点保留在最终 830 行确认性评估中。

试点完成后停止并汇报，等待人工确认，再生成剩余 800 条。

### 5.8 失败行的 intention-to-treat 规则

所有主结果保持相同测试分母。有效性拆为两个标记：

```text
valid_reasoning = llm_thinking_process非空 AND generation_status==ok
valid_top5      = predicted_next_ttps恰好5个互不重复且全部属于184类词表
```

| 档 | 失败处理 |
|---|---|
| B0 | `valid_top5=False` 时，该样本 Top-1、Hit@5、RR 均记 0 |
| B1 | `valid_reasoning=False` 时，测试回退 A；训练时不进入 probe 训练 |
| B2 | `valid_top5=False` 时回退 A |
| C1b | 与 B1 相同；无效行不进入置换池 |
| A / C1a / C2 / D | 不依赖 LLM 输出，不受影响 |

另报 complete-case 敏感性分析，只在相应语义输入均有效的样本上重算。该分析不参与任何判定；若某 campaign 在过滤后为空，则从该 complete-case 描述性分析中移除并报告数量。

另做原始 834 行的闭集外敏感性分析：4 条 `true_label∉V` 的行对 A、B0、B1、B2、C1a、C1b、C2、D 全部统一记为 Top-1/Hit@5 失败、MRR=0；不得训练这些目标。该分析使用原始 campaign 组内分母，其中 FIN13 Case 1、NotPetya、Target Breach 和 Stockpile `0b73bf34-fc5b-48f7-9194-dce993b915b1` 的 `n_j` 分别为 31、16、11、5。它不参与 §6 的任何判定。

## 6. 判定标准

### 6.0 唯一判定指标

所有正式判定只使用 campaign-macro Top-1。Hit@5、MRR 和 Macro-F1 只用于描述。B0 的 MRR@5 单列，不与完整 MRR 比较。

### 6.1 B1 原论文实现获得支持的条件

B1 是唯一确认性主方法。B0 和 B2 是诊断实验，不与 B1 竞争，也不通过外层测试结果选择模型。

以下五项必须全部满足：

1. 三个真实来源上，B1 相对 A 的增益点估计全部非负。
2. 至少两个来源的增益不低于 2 pt。
3. 至少两个来源上，B1 同时优于 C1a 和 C1b，即优于同来源的 `max(C1a,C1b)`。
4. 至少两个来源上，B1 优于 C2。
5. 至少两个来源上，B1 优于 D。

#### 2 pt 门槛的正确解释

主指标为 campaign-macro：

```text
M = (1/J) * sum_j (k_j / n_j)
```

闭集主实验固定 `J_CTID=10`、`J_AttackFlow=35`、`J_Stockpile=27`。过滤没有删除 campaign。四个受影响 campaign 的闭集组内分母固定为：FIN13 Case 1 `n_j=30`、NotPetya `n_j=15`、Target Breach `n_j=10`、Stockpile `0b73bf34-fc5b-48f7-9194-dce993b915b1` `n_j=4`。其余 campaign 的 `n_j` 等于原始行数。

在 campaign `j` 中多预测对一行，来源指标变化为：

```text
Delta M = 1 / (J * n_j)
```

因此 2 pt 不对应固定数量的样本，其离散粒度取决于各 campaign 长度。旧的 0.0214 / 0.0371 / 0.0410 来自 127 类候选空间，不用于计算本实验的相对提升。重跑 A 后，可以报告 `1/(J*n_j)` 的分布作为指标粒度说明，但不得据此修改门槛。

门槛固定为 2 pt。低于门槛时必须表述为“未达到预注册的实用价值门槛”，不得表述为“无信号”或“数学上没有信号”。

### 6.2 判死条件与三状态结论

定义每个来源 `s` 上的主方法增益：

```text
Delta_s = Top1_campaign_macro(B1, s) - Top1_campaign_macro(A, s)
```

命中任一条件时，报告“B1 原论文实现命中预注册判死条件”：

1. 存在来源 `s` 使 `Delta_s<0`，且对所有其他来源 `t≠s` 都有 `Delta_t<0.01`。
2. 对全部三个真实来源 `s`，都有 `max(C1a(s),C1b(s)) >= B1(s)`。
3. 对全部三个真实来源 `s`，都有 `C2(s) >= B1(s)`。
4. 对全部三个真实来源 `s`，都有 `D(s) >= B1(s)`。

最终结论只有三种：

| 状态 | 固定表述 |
|---|---|
| §6.1 五项全部满足 | “B1 原论文实现达到预注册支持条件。” |
| §6.2 任一条件命中 | “B1 原论文实现命中预注册判死条件。” |
| §6.1 未全部满足且 §6.2 未命中 | “B1 原论文实现未达到支持条件，也未命中判死条件；当前证据不足。” |

禁止把第三种状态强行归为成立或判死。

### 6.3 B0/B1/B2 诊断逻辑

B0 和 B2 的“有信号”判据分别固定为：三个来源相对 A 的增益全部非负，且至少两个来源增益不低于 2 pt。二者独立判断，不取 max。

先按 §6.1–§6.2 判断 B1，再按以下层级解释：

#### B1 达到 §6.1 支持条件

- B0、B2 同时有信号：直接排名与融合均支持语义信号，B1 的结果具有一致机制证据。
- B0 有信号、B2 无信号：LLM 直接排名有信息，但固定排名先验与 A 的融合未达到门槛；B1 的有效性仍须由 C1a/C1b 排除架构效应。
- B0 无信号、B2 有信号：LLM 单独未达到门槛，但其排序与 A 具有互补性；B1 的有效性仍须通过 C1 控制。
- B0、B2 均无信号：这是警告信号。只有在 B1 已满足 §6.1 的 C1/C2/D 条件时，才能报告 B1 获得支持；不得额外声称 LLM 直接预测能力有效。

#### B1 未达到 §6.1 支持条件

| B0 | B2 | 固定解释 |
|---|---|---|
| 有信号 | 有信号 | 直接语义排名及其与 A 的融合均有信号，瓶颈位于 reasoning→BGE→probe 路径或 B1 的固定融合实现；不能只归因于 probe。 |
| 有信号 | 无信号 | LLM 直接排名有信号，但当前排名先验融合未达到门槛；需要重新设计融合。 |
| 无信号 | 有信号 | LLM 单独未达到门槛，但与 A 融合后产生互补增益；这是广义“语义排名 + 序列规律融合”的证据，但不是 B1 原实现成立的证据。 |
| 无信号 | 无信号 | 在当前模型、prompt、KG 与输入设定下，未检出达到预注册门槛的直接语义预测信号。不得断言任务输入客观上没有信息。 |

若 B1 只满足方向或量级条件，但未通过 C1 控制，则必须优先解释为 probe 可能学习了训练域统计，不能归因于 reasoning 内容。

### 6.4 主干与语义分支的分歧诊断

不使用全部 184 类的符号一致率。按 transition seen/unseen 分层报告：

| 指标 | 定义 |
|---|---|
| Top-1 一致率 | A 与语义分支 argmax 相同的比例 |
| Top-5 Jaccard | 两者 Top-5 集合的 Jaccard 相似度 |
| 秩相关 | 在两者 Top-20 并集上计算 Kendall tau 与 Spearman rho |
| A Top-1 在语义分支中的排名 | 中位数与完整分布 |
| 语义 Top-1 在 A 中的排名 | 中位数与完整分布 |

这里的“语义分支”对 B1 指 MLP 的 184 类分布；对 B0/B2 的直接排名诊断，使用 B0 的五个候选及其固定 positional prior。不得把这些数值与 S-GRec 在 RL rollout advantage 空间报告的方向一致率直接比较。

## 7. 分层分析

同时报告两个正交分层：

1. **转移 seen/unseen：**真值对应的 `(ctx[-1], label)` 是否在外层训练来源出现过。
2. **目标标签 seen/unseen：**`label` 本身是否在外层训练来源的标签集中出现过。

四个 `transition × label` 格子分别报告主指标和配对差值。

label-unseen 时，MLP 输出头并非完全没有训练，而是没有收到该类的正样本监督，同时在其他样本的交叉熵中持续收到负向梯度。因此该格只作诊断，不作机制结论。

任一格子少于 5 个 campaign 或少于 20 行时，只报告描述统计并标记 `NA`，不进行配对显著性解释。

预注册假设：语义增益集中在 `transition-unseen × label-seen`。若增益只集中在 label-unseen 格子，机制解释不成立，必须如实报告。

## 8. 汇报格式

必须依次报告：

1. 标签词表 SHA-256、原始/闭集输入 SHA-256、过滤的 4 个唯一键、各来源 `J` 与四个受影响 campaign 的过滤前后 `n_j`。
2. 试点 8 项自动门和 groundedness 4 项盲审的实际数值与通过情况。
3. 主结果表：3 折 × 8 档 × `{Top-1, Hit@5, MRR, Macro-F1}`，含 campaign-macro 点估计与 campaign 聚类 bootstrap 95% CI。B0 单列 MRR@5。
4. §6.1 五个条件逐项判定。
5. §6.2 判死条件与三状态结论。
6. §6.3 B0/B1/B2 诊断。
7. §6.4 分歧诊断。
8. §7 四格分层结果，含 `NA` 标记。
9. B1/C1a/C1b 每折每 seed 的 λ 与 epoch；B2/C2/D 每折的 λ；确定性档不得伪造 seed。
10. 每个可训练档的 5 个 seed 单独结果、均值与范围。
11. 实测费用、token 消耗、失败行数、失败分类、fallback 数量和 complete-case 敏感性结果。
12. 任何偏离规格的决定及原因。

## 附录 A：重建 cumulative 数据

现有 loader 将每个样本编码为“一个源技术 + 一个目标技术”，导致 705 条 Attack Flow 和 65 条 Stockpile 全部为 `prefix_len=1`，不是序列预测任务。

### Stockpile

- 从 adversary profile 的 `atomic_ordering` 构建累积前缀。
- ability→technique_id 映射从 `abilities/` 下的 YAML 读取；单个 YAML 可能是 dict 或 list，两种都要处理。
- technique 优先读取 `technique_id`，缺失时回退 `technique.attack_id`。
- profile YAML 缺少 ID 时使用文件名作为 `profile_id`。
- 183 个 ordering step 中预计约 34 个无法映射，必须如实丢弃并报告。

### Attack Flow

- `.afb` 为 JSON。
- `objects` 中 `id=="action"` 的对象，其 `properties` 包含 technique ID；`properties` 是 `[[key,value],...]` 列表，不是 dict。
- 边端点指向 latch，归属链为 `latch → anchor → node` 三层。
- 持有 `anchors` 的对象是 node；`anchors` 为“角度字符串→anchor instance”的 dict。
- 持有 `latches` 的对象是 anchor；`latches` 为 latch instance 列表。
- 只解析一层 anchor 会导致边零匹配。

### DAG 线性化

- 先按源文件节点出现顺序执行确定性 DFS；每个节点的出边按目标节点在源文件中的出现顺序遍历。
- 遇到指向当前灰色祖先节点的回边时删除该边，并把源、目标、节点类型及对应 action technique 写入审计报告。
- 必须断言删除回边后的图无环；随后使用最长路径。
- 首先最大化路径中的 action 节点数，其次最大化总图节点数；最终 ties 按源文件中节点出现顺序打破。
- 非 action 节点可以穿越但不输出。
- 禁止尝试其他断环或线性化策略后选择结果最好者。

### 累积前缀

一条 `n` 步序列生成 `n-1` 个样本：第 `i` 个样本 `prefix=steps[:i]`、`true_label=steps[i]`。全部映射到父技术，同时保存 `raw_prefix` 和 `target_raw_id`。

### 质量门

- Attack Flow 战术单调不减比例必须小于 0.85。
- Stockpile 预注册为半合成例外，不因预计 0.861 判为失败。
- 战术顺序固定为：reconnaissance、resource-development、initial-access、execution、persistence、privilege-escalation、defense-evasion、credential-access、discovery、lateral-movement、collection、command-and-control、exfiltration、impact。
- 一个技术属于多个战术时取序号最小者。

输出 `attack_flow_cumulative.csv`、`stockpile_cumulative.csv` 和 `rebuild_report.json`。CSV 必须包含：

```text
sample_id, source, campaign_id, prefix_len, prefix,
raw_prefix, true_label, target_raw_id
```

`sample_id` 必须唯一。报告行数、campaign 数、唯一标签数、prefix_len 分布、单调性分子分母、质量门结果及 Stockpile 未映射步数。

### 固定 184 类闭集视图

原始 cumulative 与 CTID loader snapshot 不得修改。使用预注册词表只过滤 `true_label` 不属于 184 类的行，生成：

```text
project/data_v2/repro_external/closed_set/ctid_in184.csv
project/data_v2/repro_external/closed_set/attack_flow_cumulative_in184.csv
project/data_v2/repro_external/closed_set/stockpile_cumulative_in184.csv
project/data_v2/repro_external/kg_context/external_prefixes_with_kg_in184.csv
```

不得过滤 prefix 内的词表外 technique。必须保存 `closed_set_report.json`，记录被过滤的唯一键、真值、过滤前后每个受影响 campaign 的 `n_j`、campaign 集合、原始/闭集文件哈希及词表哈希。预期主实验形状为 CTID 281/10、Attack Flow 428/35、Stockpile 121/27，总计 830 行/72 campaign；任一 campaign 消失时立即停止。

预期自查：

- Attack Flow：约 431 行、35 个 flow、prefix_len 1–31、单调性约 0.650。
- Stockpile：约 122 行、27 个 profile、prefix_len 1–17、单调性约 0.861。

若行数与预期差异超过 10%，停止并汇报，不得调整解析逻辑去凑数字。不得修改 `loader_snapshots/` 下的审计产物。

## 附录 B：计算量与降档预案

λ 网格只影响评估，不需要为每个 λ 重新训练模型。主要训练开销来自 B1/C1a/C1b 的 3 个外层折、2 个 inner 折、5 个 seed 和最多 30 个 epoch checkpoint。

若资源不足，必须在运行相关实验前声明降档。预注册顺序为：D → C2 → C1a。A、B0、B1、B2、C1b 不可删除。

删除某控制后，§6.1 对应条件标记为“未检验”，不得视为通过，因此不能宣布 B1 达到全部预注册支持条件。
