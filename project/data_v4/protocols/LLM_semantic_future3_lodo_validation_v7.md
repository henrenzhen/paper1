# LLM 语义分支在真实跨来源数据上的验证（#4）v7

> 状态：待实现、待运行。本文档冻结实验定义；运行后不得依据外层留出来源结果修改任务、模型、阈值或判定条件。

## v7 修订摘要

v7 不再验证“仅凭 ATT&CK ID 精确预测唯一下一项”，而验证一个更符合分析师使用方式的任务：

> 给定截至当前时刻的有序技术与事件描述，输出未来三步内最值得检查的 5 个父技术候选。

与 v4.2 相比，冻结以下变更：

1. ground truth 从唯一下一项改为冻结序列上未来三步的唯一父技术集合。
2. 语义输入从“ID + 由 ID 检索的 KG”改为“ID + 真实历史步骤 description + 多战术信息”。KG 不进入主实验。
3. 三源全部保留并执行三折 LODO；`source` 只用于划分、bootstrap 和汇报，不进入模型或 LLM prompt。
4. Attack Flow 使用 `properties.description`，禁止回退到 `name`。冻结最长路径上的 466 个步骤中，description 预期 464/466 非空、中位 76.5 字符、90.8% 超过 40 字符。
5. 三源 description 统一剥离明文 ATT&CK ID。空文本保留显式标记，不静默丢弃。
6. CTID 的 12 个多技术事件在主实验中先删除，再重建单技术事件序列；旧“每步取第一个父技术”只进入敏感性分析。
7. 输出和损失改为真正的 184 维多标签 relevance 建模；主指标改为 campaign-macro NDCG@5。
8. 使用六档主阶梯，并增加 LLM 直接排名、训练文本置换和单调战术过滤三个诊断档。
9. 新旧任务的任何绝对数字禁止直接比较。旧任务上的 `0.0873` 只能说明需要重跑对应方法，不能作为 v7 的数值基线。
10. 旧 30 条试点已经解盲，只能作为开发样本。v7 将其稳定映射到开发集，主评估排除开发集。

---

## 0. 研究问题与允许的结论

论文核心问题保持不变：

> LLM 对真实攻击活动文本的语义建模，能否为稀疏的跨来源序列规律提供互补信号？

v7 分开检验三层主张：

| 层级 | 主张 | 必要对照 |
|---|---|---|
| 文本信号 | 历史事件描述包含 ID 序列之外的预测信息 | `R` 对 `A` |
| LLM 增量 | LLM 状态摘要优于直接编码原始文本 | `S` 对 `R`，并用 `P` 排除容量效应 |
| 融合增量 | LLM 语义与序列规律互补 | `S` 对 `A` |
| 战术扩展 | 软战术先验能否继续改善核心方法 | `ST` 对 `S`、`T` |

若只证明 `R>A` 而 `S≤R`，允许的结论是“真实事件文本有用”，不得声称 LLM 必要。若 `T≥ST`，不得把增益归因于 LLM 语义。

本实验不检验：

- SIM 合成数据上的模板拟合；
- asset、process、file、tool、privilege 等来源覆盖不均的稀疏字段；
- 使用真实未来战术的 oracle 掩码；
- conformal prediction；
- KG 是否进一步改善结果。

---

## 1. 固定数据与标签空间

### 1.1 原始来源

| 来源 | 序列来源 | 原始性质 |
|---|---|---|
| CTID | adversary emulation plan 的有序步骤 | 真实 emulation plan |
| Attack Flow | 冻结断环策略后的最长路径 | 真实报告衍生 |
| Stockpile | adversary profile 的 `atomic_ordering` | 半合成计划 |

固定输出标签空间为：

```text
project/data_v2/core/rl_label_vocab.csv
```

必须校验：184 个唯一父技术，SHA-256：

```text
9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738
```

禁止从三源数据、外层测试来源或 LLM 输出动态扩展标签空间。

### 1.2 步骤级对齐表

先生成且冻结：

```text
project/data_v4/semantic_alignment/step_text_alignment.csv
project/data_v4/semantic_alignment/step_text_alignment_manifest.json
```

预期原始对齐为 908 个步骤：

| 来源 | 步骤数 | description 预期质量 |
|---|---:|---|
| CTID | 293 | 中位约 108 字符 |
| Attack Flow | 466 | 464/466 非空，中位 76.5，90.8% >40 |
| Stockpile | 149 | 剥离 ID 后中位约 51 字符 |

每行至少包含：

```text
source, campaign_id, step_index, stable_step_id,
raw_technique_ids, parent_technique_ids,
step_name, description_raw, description_clean,
description_chars, has_description, is_multitech,
source_file_sha256
```

约束：

1. `step_index` 在每个 `(source,campaign_id)` 内从 0 连续编号。
2. `(source,campaign_id,step_index)` 和 `(source,stable_step_id)` 均唯一。
3. Attack Flow 必须复用 `rebuild_cumulative_prefixes.py` 的断环与最长路径实现，不得重新实现一套近似解析器。
4. Attack Flow 同时保存 `name` 和 `description`，主实验只使用 `description`；禁止 `description or name` 回退。
5. Stockpile 保存 ability ID；Attack Flow 保存 action instance；CTID 保存 source step ID。
6. 对齐出的技术序列必须逐 campaign 精确重现已冻结 cumulative 序列；任何不一致立即停止。

### 1.3 文本清洗

三源使用同一函数，顺序固定：

1. Unicode NFKC 规范化；
2. 将换行、制表符和连续空白折叠为一个空格；
3. 大小写不敏感删除正则 `\bT\d{4}(?:\.\d{3})?\b`；
4. 再次折叠空白并去除首尾标点空白；
5. 最多保留前 2,000 个字符，并记录是否触发截断；
6. 在写入缺失标记前计算 `description_chars`；若结果为空，令其为0并写入字面量 `[NO_DESCRIPTION]`。

不得用以下内容回填 description：

- ATT&CK 技术名称；
- source、campaign、actor 或文件名；
- target/future 步骤文本；
- asset、process、file、tool、privilege、platform、executor、command。

manifest 必须按来源记录：原始/清洗后非空率、明文 ATT&CK ID 比例、字符数分位数、`[NO_DESCRIPTION]` 数量和全部文件哈希。

### 1.4 CTID 多技术事件的主实验规则

已定位 12 个 `len(parent_technique_ids)>1` 的 CTID 事件。主实验固定如下处理：

1. 在构建序列前删除这 12 个事件；
2. 保持其余事件原始顺序；
3. 删除后若相邻事件父技术相同，只保留较早事件；
4. 在投影后的单技术序列上重新计算 `step_index`、prefix 和未来三步目标；
5. 不删除整个 campaign。

这相当于在“可唯一标注的事件子序列”上建模。不得只删除以多技术事件为 target 的行，也不得删除该事件后所有 prefix。

敏感性分析 `CTID-first-parent` 复现旧解析器的字典序第一个父技术规则；它不参与任何正式判定。

### 1.5 未来三步样本

对每条重建后的 campaign 序列 `s_0,...,s_(n-1)`，对 `i=0,...,n-2` 生成一个样本：

```text
observed_steps = s_0,...,s_i
target_steps   = s_(i+1),...,s_(min(i+3,n-1))
Y_i            = stable_unique(parent(target_steps))
```

`stable_unique` 按首次出现顺序去重。主实验只保留 `Y_i` 中每个标签都属于固定 184 类词表的样本；不得对 `Y_i` 与词表取交集后假装目标完整。prefix 中的词表外历史技术保留。

预期全量 v7 数据形状：

| 来源 | 样本 | campaign | `|Y|=1` | `|Y|=2` | `|Y|=3` |
|---|---:|---:|---:|---:|---:|
| CTID | 275 | 10 | 10 | 25 | 240 |
| Attack Flow | 422 | 35 | 38 | 85 | 299 |
| Stockpile | 119 | 27 | 33 | 44 | 42 |
| 合计 | 816 | 72 | 81 | 154 | 581 |

任一来源样本数与预期不一致或任一 campaign 消失时停止，不得调整规则去凑数字。

样本级文件：

```text
project/data_v4/semantic_alignment/future3_samples_in184.csv
project/data_v4/semantic_alignment/future3_samples_manifest.json
```

至少包含：

```text
source, campaign_id, prefix_len, prefix_endpoint_step_id,
observed_step_indices, observed_parent_ids, observed_tactic_multihot,
observed_descriptions, target_step_indices, target_parent_set,
target_cardinality, last_description_chars,
is_development
```

唯一键固定为 `(source,campaign_id,prefix_len)`。所有数组使用 JSON 编码，禁止空格分隔的不可逆字符串。
`transition_visibility` 必须在每个外层折内只根据该折训练来源动态计算，不得作为全数据预计算字段写入基础样本表。

---

## 2. 开发集与三折 LODO

### 2.1 已解盲 30 条的处理

v4.2 的 30 条试点已经查看过真实标签并影响了任务重设计，不能进入 v7 主评估。

使用 `prefix_endpoint_step_id` 将旧 30 条稳定映射到 v7 数据。若某条因 CTID 多技术删除或未来窗口 OOV 不再存在，则在同来源、同 prefix 长度三分位内按以下排序补足：

```text
SHA256("v7-dev-20260806" || source || campaign_id || prefix_len)
```

每源固定 10 条，共 30 条，尽可能来自不同 campaign。映射表和补足原因必须落盘。开发集用于新 prompt/API 机械试点，主评估固定排除。因此若预期 816 行成立，主评估预期为 786 行。

这30条不得进入主评估的外层训练、inner-train、inner-validation、外层测试、频率/共现计数、战术计数或融合权重选择。三折 LODO 的全部正式拟合与评价都只使用其余786条。

预期主评估分母为 CTID 265、Attack Flow 412、Stockpile 109，共786条；campaign 仍为10/35/27。任一开发样本映射导致这些分母或 campaign 集合改变时停止并先更新协议。

全量 816 行结果只作为包含开发样本的敏感性分析，不进入 §9 判定。

### 2.2 外层 LODO

| 外层折 | 训练来源 | 测试来源 |
|---|---|---|
| 1 | Attack Flow + Stockpile | CTID |
| 2 | CTID + Stockpile | Attack Flow |
| 3 | CTID + Attack Flow | Stockpile |

`source` 不得进入：

- LLM system/user prompt；
- 文本编码内容；
- probe 特征；
- 融合器特征；
- 战术或序列打分。

`source` 只用于 LODO、分层、bootstrap 和结果表。

### 2.3 inner-LODO

每个外层折的两个训练来源轮流作为 inner-train 和 inner-validation，共两个 inner 折。epoch、融合权重和任何允许选择的参数，只能最大化两个 inner-validation 来源等权平均的 campaign-macro NDCG@5。

外层测试来源的标签、指标、文本长度分布和错误案例均不得参与选择。

---

## 3. 输入序列化与泄漏门

### 3.1 每个历史事件的统一格式

按时间顺序序列化：

```text
[STEP {relative_index}]
Technique: {parent_id}
Tactics: {sorted_tactic_ids_or_names}
Description: {description_clean}
```

不写 source、campaign、actor、文件名、目标标签或未来步骤数量之外的信息。

### 3.2 长度预算

LLM 输入字符预算固定为 12,000 字符。若完整 prefix 超出预算：

1. 从最新事件向过去选择完整事件块；
2. 不截断单个事件块；
3. 选择后恢复时间正序；
4. 至少保留最后两个事件；
5. 保存 `included_step_start`、`included_step_count` 和截断前后字符数。

原始文本编码器使用 BGE-M3 tokenizer 的 8,192 token 上限，并预留特殊 token；同样只在事件边界从旧到新删除，不在事件内部截断。

### 3.3 自动泄漏断言

数据构建必须在读入任何预测结果前通过：

1. 每条 observed step 都满足 `step_index <= prefix_endpoint_index`；
2. 每条 target step 都满足 `step_index > prefix_endpoint_index`；
3. observed 与 target step ID 集合交集为空；
4. prompt 不包含 target step 的 `description_raw/clean`；
5. description_clean 不含 ATT&CK ID 正则；
6. prompt 不含 source/campaign/actor/文件路径；
7. target set 只由冻结序列未来三步构建，不读取模型排名；
8. 同一唯一键只出现一次。

任一断言失败即停止。

---

## 4. 多标签无语义主干

### 4.1 A0：全局频率

在每个外层训练集内，对每个标签 `c` 统计它出现在训练样本目标集合中的次数：

```text
N_c = sum_i 1[c in Y_i]
```

按 `N_c` 降序排列；并列按固定词表的 `label_id` 升序。A0 不读取 prefix。

### 4.2 A：多标签上下文 relevance 主干

对 order-2、order-1 和 unigram 三层分别统计：

```text
N(h)   = 具有上下文 h 的训练样本数
N(h,c) = 上下文为 h 且 c in Y 的训练样本数
p_uni(c) = (N(c) + 0.5) / (N + 1.0)
p(c|h) = (N(h,c) + alpha_s * p_uni(c)) / (N(h) + alpha_s)
```

固定 `alpha_s=0.1`。order-2/order-1/unigram 权重为 `0.5/0.3/0.2`；未见上下文时只对可用层重新归一化。得到 184 维独立 relevance，允许总和大于 1。

用于融合前：

```text
z_A(c) = logit(clip(p_A(c), 1e-6, 1-1e-6))
```

### 4.3 K：单调战术过滤诊断

K 是本地、受既有工作启发的确定性诊断，不得把旧 `0.0873` 写成文献成绩。

对最后一个历史父技术的全部战术集合 `M_last` 和候选标签的战术集合 `M_c`：

```text
compatible(c) = any(order(t_c) >= order(t_last)
                    for t_c in M_c for t_last in M_last)
```

K 保持 A 的原始排序，但先排列 `compatible=True` 的候选，再排列其余候选；组内保持 A 的顺序。K 在 v7 的未来三步真值与新指标下重跑，只作诊断。

---

## 5. 多战术软先验

### 5.1 技术到战术的固定映射

从 MITRE ATT&CK Enterprise v18 STIX 构建 `technique_id -> set(tactic)`，保留全部合法战术，禁止 `split("||")[0]`、最小序号或任意单战术压平。14战术顺序仅用于稳定序列化和 K 的单调诊断：

```text
reconnaissance, resource-development, initial-access, execution,
persistence, privilege-escalation, defense-evasion, credential-access,
discovery, lateral-movement, collection, command-and-control,
exfiltration, impact
```

运行前保存：ATT&CK release/version、原文件 SHA-256、生成脚本 SHA-256、184 类中无战术映射的标签列表。映射缺失的候选战术分数取训练集战术边际均值，不得取 0。

### 5.2 未来战术 relevance

每个训练样本的战术目标为：

```text
U_i = union(M(c) for c in Y_i)
```

在 14 个战术维度上使用与 §4.2 相同的 order-2/order-1/unigram relevance 计数，得到 `q(t|prefix)`。候选技术的软战术得分：

```text
p_T(c) = mean(q(t|prefix) for t in M(c))
z_T(c) = logit(clip(p_T(c), 1e-6, 1-1e-6))
```

它只使用训练来源的历史上下文与目标战术计数，不得读取测试来源标签，不得使用真实未来战术。

---

## 6. 文本与 LLM 语义分支

### 6.1 统一冻结编码器

原始 description 为英文，LLM 状态摘要要求中文，因此所有文本分支统一使用多语编码器：

```text
encoder = BAAI/bge-m3
dimension = 1024
max_length = 8192
encoder frozen = true
```

运行前固定 Hugging Face revision/commit，禁止使用浮动 `main`。BGE-M3 官方模型卡说明其为多语、1024 维、最长 8192 token：<https://huggingface.co/BAAI/bge-m3>。

取 dense CLS embedding，L2 normalize。超过上限按 §3.2 在事件边界截断；不得根据标签选择保留事件。

### 6.2 共享 probe

R、P、S 使用完全相同的 probe：

```text
1024 -> 256 -> 184
hidden_activation = GELU
dropout = 0.3
loss = BCEWithLogitsLoss
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 32
max_epochs = 100
checkpoint_interval = 5
train_seeds = 42,43,44,45,46
```

每个实际训练运行（inner-train 或外层两源合并重训）按该运行训练集的标签赋值总量计算一个统一正类权重：

```text
pos_weight = min(100,
  negative_label_assignments / positive_label_assignments)
```

同一折全部184类使用同一个 `pos_weight`，不为稀有类单独调权。epoch 由 inner-LODO 选择。

### 6.3 R：原始文本编码

R 将 §3.1 的完整历史事件序列直接送入 BGE-M3，再训练共享 probe。它拥有与 LLM 相同的历史 ID、战术和 description，但没有 LLM 转换，是“LLM 是否必要”的关键控制。

### 6.4 S：LLM 状态摘要编码

LLM 只根据历史事件生成三个不含 ATT&CK ID 的字段：

```text
stage_assessment
observed_capabilities
likely_next_intents
```

按上述顺序拼接后送入 BGE-M3。`predicted_next_ttps` 单独用于 B0，绝不拼入 S 的编码文本。

### 6.5 P：训练摘要置换控制

P 与 S 结构、参数量、训练次数完全相同，但只在训练数据中置换 LLM 状态摘要与标签的对应关系：

1. 在 `source × prefix_len三分位` 内置换；
2. 禁止固定点；
3. 小于2条的格与相邻长度格合并；
4. permutation seed 为 `9000 + train_seed`；
5. inner-validation 和外层测试使用各自真实对齐摘要，不置换；
6. 保存每个训练运行的 `(recipient_key, donor_key)` 映射。

若 `S` 不能超过 `P`，不得把提升归因于摘要内容。

### 6.6 B0：LLM 直接 Top-5

B0 使用 `predicted_next_ttps` 返回顺序直接计算未来三步集合指标。恰好5个、互不重复、全部属于184类词表才有效；否则该样本的 B0 四项指标全部记0，不允许回退 A。

---

## 7. 六档主阶梯与三个诊断档

### 7.1 主阶梯

| 档 | 配置 | 回答的问题 |
|---|---|---|
| A0 | 全局目标频率 | 不看输入能做到什么 |
| A | 多标签上下文主干 | ID 序列/共现是否有信息 |
| T | A + 软战术先验 | 战术归纳偏置是否有用 |
| R | A + 原始文本 probe | 真实文本是否有增量 |
| S | A + LLM 状态摘要 probe | LLM 语义转换是否有增量 |
| ST | A + LLM 状态摘要 + 软战术 | 软战术扩展是否继续增益 |

### 7.2 诊断档

| 档 | 配置 | 用途 |
|---|---|---|
| P | A + 训练摘要置换 probe | 等容量、等文本分布控制 |
| B0 | LLM 直接 Top-5 | LLM 排名本身是否有信号 |
| K | A + 单调战术过滤 | 重跑旧强规则基线 |

不得根据外层结果从 R、S、ST 中挑最好的作为“本文方法”。对应原始核心主张的主方法事前固定为 S；ST 是加入软战术先验后的预注册扩展，不得用 ST 的结果替换 S 的核心判定。

### 7.3 分数标准化与融合

对每个样本、每个分支的184维分数做：

```text
standardize(z) = (z - mean(z)) / max(std(z), 1e-6)
```

两分支档使用：

```text
z_fuse = (1-lambda)*standardize(z_A)
         + lambda*standardize(z_branch)
lambda in {0.0,0.1,...,1.0}
```

ST 使用非负三分支 simplex：

```text
z_ST = w_A*standardize(z_A)
     + w_S*standardize(z_S)
     + w_T*standardize(z_T)
w_A,w_S,w_T in {0.0,0.1,...,1.0}
w_A+w_S+w_T = 1
```

权重只由 inner-LODO campaign-macro NDCG@5 选择。并列时依次选择：语义权重更小、战术权重更小、主干权重更大的组合，避免在无增益时偏向复杂分支。

最终按 `z` 降序输出 Top-20；并列按固定 `label_id` 升序。

---

## 8. LLM 生成协议

### 8.1 API 与权限

v7 会把清洗后的真实事件 description 发送给外部 DeepSeek API。此前授权只覆盖旧30条及旧字段，不能自动扩展到 v7。

执行任何 v7 API 调用前必须获得新的明确授权，说明：发送字段、开发集30条或全量数量、模型和按 token 计费。

API Key 仅从环境变量读取，禁止写入代码、配置、日志或提交物。

### 8.2 固定参数

| 项 | 值 |
|---|---|
| BASE URL | `https://api.deepseek.com` |
| 模型 | 调用 `/models` 后冻结实际可用的 `deepseek-v4-flash` ID |
| temperature | `0.0` |
| max_tokens | `2048` |
| 思考模式 | `extra_body={"thinking":{"type":"disabled"}}` |
| JSON 输出 | `response_format={"type":"json_object"}` |
| 并发 | 起步 50；只因429/5xx机械降并发 |

DeepSeek 官方文档确认思考模式默认开启、可显式 disabled，JSON Output 需设置 `json_object` 并在 prompt 中说明 JSON：

- <https://api-docs.deepseek.com/guides/thinking_mode>
- <https://api-docs.deepseek.com/guides/json_mode/>

### 8.3 固定 prompt

System prompt：

```text
你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻已经观察到的攻击事件，
每个事件只包含 ATT&CK 父技术、该技术可能所属的战术，以及事件的真实描述。

任务：基于已观察历史，概括攻击者当前状态，并给出未来三次攻击动作范围内最值得检查的
5 个 ATT&CK Parent Technique 候选。未来三步是一个无序目标集合；你的5个候选仍须按
可能性从高到低排列。

严格限制：
1. 只能使用输入中的历史事件，不能声称看到了未来事件。
2. 不得从来源名称、campaign 名称或文件格式推断答案；这些字段不会提供。
3. stage_assessment、observed_capabilities、likely_next_intents 中不得出现任何 Txxxx
   或 Txxxx.xxx 标识，也不要逐字重复 predicted_next_ttps。
4. predicted_next_ttps 必须恰好包含5个互不重复的 ATT&CK 父技术 ID。
5. 输出必须是 JSON，不要输出 JSON 之外的文本。

JSON 示例：
{
  "stage_assessment":"对已观察阶段的简洁判断",
  "observed_capabilities":"仅由历史描述支持的能力概括",
  "likely_next_intents":"对下一阶段意图的概括，不写ATT&CK ID",
  "predicted_next_ttps":["T1059","T1078","T1021","T1003","T1105"]
}
```

User prompt：

```text
### 已观察攻击事件（按时间顺序）
{serialized_observed_events}

### 输出要求
请根据以上历史事件输出 JSON。预测目标是未来三次动作内的技术集合，候选数组按可能性排序。
```

system prompt、user template、JSON Schema 必须分别保存 SHA-256。运行中不得改字。

### 8.4 JSON Schema 与有效性

四个字段均 required，`additionalProperties=false`：

- 三个摘要字段：非空字符串；
- `predicted_next_ttps`：长度恰好5、元素唯一、匹配父技术正则并属于固定词表。

另定义：

```text
valid_summary = generation_status==ok
                AND 三个摘要字段非空
                AND 三个摘要字段不含ATT&CK ID

valid_top5 = generation_status==ok
             AND predicted_next_ttps恰好5个唯一词表内父技术
```

失败最多重试3次。不得静默修复、丢行或让另一个模型补写。

### 8.5 新30条开发试点

先只对 §2.1 的30条开发集运行。自动门：

| 检查 | 阈值 | 失败动作 |
|---|---:|---|
| JSON 解析成功率 | ≥95% | 只修解析/schema后重跑完整30条 |
| `reasoning_content` 为空 | 100% | 检查显式关闭思考后重跑 |
| `finish_reason==length` | ≤2% | 提高 max_tokens，发布协议偏离后重跑 |
| valid_summary | ≥95% | 停止，不得按内容改 prompt 后继续 |
| 摘要字段含 ATT&CK ID | 0% | 停止，发布新协议版本 |
| valid_top5 | ≥90% | 只允许加强机械 JSON 约束后重跑 |
| prompt 泄漏断言 | 100% | 停止并修数据构建 |

摘要字符长度、输入/输出 token、费用只记录，不设置事后“质量”阈值。试点完成后停止，报告结果并等待人工确认后再全量。

开发集已经排除于主评估，因此允许基于开发集修复 prompt；但任何语义修改都必须升级协议小版本、保存旧输出，并重新跑完整30条。不得查看主评估输出后改 prompt。

### 8.6 全量输出

原始响应写入新的 timestamp run 目录，不得覆盖旧试点：

```text
project/data_v4/external_reasoning/future3/runs/{timestamp}_{run_id}/
```

逐样本至少保存：唯一键、prompt、输入事件 step ID、原始响应、解析字段、有效性、模型返回 ID、completion/request ID、finish reason、HTTP 状态、attempt、latency、token、时间戳。

生成完成后，先提交未经后处理的原始版本，再进行编码或训练。

### 8.7 intention-to-treat

主评估分母不因 API 失败改变：

| 档 | 失败处理 |
|---|---|
| B0 | `valid_top5=False` 时四项指标全记0 |
| S | `valid_summary=False` 时回退 A |
| ST | `valid_summary=False` 时回退 T |
| P | 训练无效摘要不入 probe；测试无效摘要回退 A |
| A0/A/T/R/K | 不依赖 LLM，不受影响 |

另报 valid-only 敏感性分析，但不参与判定。

---

## 9. 评估、统计与判定

### 9.1 样本指标

令预测前5为 `R_5`，真值集合为 `Y`：

```text
Hit@5       = 1[|R_5 ∩ Y| > 0]
Precision@5 = |R_5 ∩ Y| / 5
Recall@5    = |R_5 ∩ Y| / |Y|
DCG@5       = sum_(k=1..5) 1[R_k in Y] / log2(k+1)
IDCG@5      = sum_(k=1..min(5,|Y|)) 1 / log2(k+1)
NDCG@5      = DCG@5 / IDCG@5
```

唯一正式判定指标为 NDCG@5。其余三项用于解释实际候选集表现。

### 9.2 campaign-macro 与来源聚合

1. 先在每个 campaign 内对样本指标取均值；
2. 再对测试来源内 campaign 等权平均；
3. 总体指标对三个来源等权平均，不按样本数加权。

同时报告 row-micro，但不得用于判定。

### 9.3 campaign 聚类 bootstrap

固定2000次，seed `20260807`。每个 replicate：

1. 在每个来源内有放回重采样完整 campaign；
2. 所有方法使用同一组重采样结果；
3. 可训练方法先在 replicate 内对5个 seed 的指标取均值；
4. 再计算配对方法差值；
5. 总体差值为三个来源差值等权平均。

报告 percentile 95% CI。seed 不得作为独立样本加入 bootstrap。另报告每个 seed 的单独结果和范围。

### 9.4 核心方法 S 的支持条件

核心方法固定为 S。定义每个来源：

```text
Delta_s = NDCG5_campaign_macro(S,s) - NDCG5_campaign_macro(A,s)
```

“LLM语义+序列融合达到预注册方向与量级条件”要求全部满足：

1. 三个来源 `Delta_s >= 0`；
2. 至少两个来源 `Delta_s >= 0.02`；
3. 至少两个来源 `S > R`；
4. 至少两个来源 `S > P`；
5. 三来源等权总体 `S > A`。

若以上全部满足，且总体 `S-A` 的95% CI下界大于0，记为“强支持”；CI跨0则记为“方向与量级支持，但统计不确定”。禁止使用“接近显著”。

### 9.5 核心方法 S 的判死与证据不足

命中任一条件，报告“核心 LLM 融合在当前任务设定下缺乏证据”：

1. 某来源 `S-A<0`，且其余两个来源均 `<0.01`；
2. 三个来源全部 `R>=S`；
3. 三个来源全部 `P>=S`；
4. 三来源等权总体 `S<=A`。

未满足 §9.4 且未命中本节时，固定表述：

> 未达到预注册支持条件，也未命中判死条件；当前证据不足。

### 9.6 软战术扩展 ST 的独立判定

ST 不改变 §9.4–§9.5 对核心方法 S 的结论。只有同时满足以下条件，才允许声称“软战术先验与 LLM 语义互补”：

1. 至少两个来源 `ST>S`；
2. 至少两个来源 `ST>T`；
3. 三来源等权总体 `ST>S`；
4. 三个来源均有 `ST-A>=0`。

否则固定表述为“未检出软战术扩展相对核心语义融合的稳定增量”。不得因此否定已经由 S 获得的核心结论。

### 9.7 分层诊断

必须报告以下四项，均不替代主结果：

#### 转移可见性

对每个目标 `y in Y`，检查 `(last_observed_parent,y)` 是否在外层训练样本中出现。样本分为：

- `all_seen`：全部目标转移见过；
- `mixed`：部分见过；
- `all_unseen`：全部未见。

#### 目标标签可见性

对每个 `y in Y`，检查该标签是否曾出现在外层训练样本的目标集合中，同样分为 `all_seen/mixed/all_unseen`。转移未见与标签本身未见必须分开报告；若语义增益只出现在 label-unseen，不得解释为模型学会了可迁移的标签监督。

#### 文本长度

按最后一个历史事件的 `description_clean` 字符数分为 `<40` 与 `>=40`。每层同时报告来源、campaign、样本数；不得把来源构成差异解释为文本长度因果效应。

#### 目标集合大小

按 `|Y|=1/2/3` 分层。它用于解释新旧任务数字为何不可比，也用于检查方法是否只在更容易命中的大目标集合上有效。

任一分层格少于5个 campaign 或20条样本时，只报描述统计和 `NA`，不作显著性解释。

### 9.8 固定机制解释

| 结果模式 | 允许的解释 |
|---|---|
| `R>A`，但 `S<=R` | 真实文本有用，但未证明 LLM 转换必要 |
| `S>P` 且 `S>R` | LLM 状态摘要提供超出容量与直接编码的信号 |
| `ST>S` 且 `ST>T` | 语义与战术先验具有互补性 |
| `B0` 有信号、`S/ST` 无 | LLM 排名有信息，瓶颈在编码/probe/融合 |
| 增益只在 all_seen | “语义补未见转移”的机制不成立 |
| 增益集中在 all_unseen | 支持语义提供不依赖转移计数的信号 |

禁止根据表现最好的来源、文本长度子集或单个 seed 改写主结论。

---

## 10. 必须落盘的产物

每次运行使用不可复用的 `run_id`。至少保存：

1. 步骤对齐 CSV、审计 JSON、输入文件哈希；
2. v7 样本 CSV、开发集映射、过滤行、target cardinality 报告；
3. technique→multi-hot tactic 映射与 ATT&CK 版本/哈希；
4. prompt、JSON Schema、生成脚本与依赖版本；
5. 原始 API 请求/响应及 generation manifest；
6. BGE-M3 revision、tokenizer、截断审计和 embedding 文件哈希；
7. 每个外层折、inner折、seed、epoch、融合权重；
8. 逐样本 Top-20 排名和分数；
9. 每个 campaign 的四项指标；
10. bootstrap replicate 或足以完全复算 CI 的固定随机索引；
11. 主表、分层表、失败/fallback 表、费用表；
12. 所有偏离及发生时间，特别标明偏离发生在查看外层结果之前还是之后。

逐样本预测至少包含：

```text
source, campaign_id, prefix_len, is_development,
target_parent_set, target_cardinality,
method, seed, top20_labels, top20_scores,
hit5, precision5, recall5, ndcg5,
valid_summary, valid_top5, fallback_used,
transition_visibility, label_visibility, last_description_chars
```

---

## 11. 执行顺序与停止门

严格按顺序执行：

1. 修复/引入 `audit_step_text_alignment.py`，生成908步对齐表；
2. 运行三源序列重现、文本清洗、ID剥离和泄漏断言；
3. 删除12个CTID多技术事件并生成预期816行 v7 样本；
4. 映射旧30条为开发集，冻结预期786行主评估；
5. 构建并冻结 multi-hot tactic 映射；
6. 先跑不收费的 A0、A、K、T；
7. 下载并固定 BGE-M3 revision，跑 R；
8. 获得新的外部发送授权；
9. 对开发30条运行新版 API 试点，停下汇报；
10. 人工确认后生成剩余样本，立即提交原始响应；
11. 冻结全部 embedding；
12. 跑 P、S、ST、B0 的 inner-LODO 和外层评估；
13. 一次性生成主结果、bootstrap、分层和判定报告；
14. 不根据外层结果重跑 prompt、换编码器、换窗口或改阈值。

停止条件：

- 任一输入哈希异常；
- 对齐不能精确重现冻结序列；
- 预期816行/72 campaign 不成立；
- 目标或文本泄漏断言失败；
- 开发试点机械门失败且需要语义修改；
- 模型/编码器 revision 无法固定；
- 外层测试域被用于任何超参数选择。

停止不等于判死。修复数据或协议后必须升级版本并完整披露。

---

## 12. 汇报模板

最终报告依次包含：

1. 数据、文本、标签词表、ATT&CK 映射与全部 SHA-256；
2. 908步对齐审计、12步删除、未来三步闭集过滤与最终样本数；
3. 开发30条自动门、token、费用与失败分类；
4. 三折 × 六档主阶梯 × 四指标主表；
5. P、B0、K 三个诊断档；
6. 每来源及三来源等权的配对 bootstrap 95% CI；
7. §9.4 五个核心支持条件逐条判定；
8. §9.5 核心判死条件与最终三状态结论；
9. §9.6 软战术扩展的独立判定；
10. transition 与 label 的 all-seen/mixed/all-unseen 分层；
11. 文本长度和 `|Y|` 分层；
12. 每个训练 seed、epoch、融合权重和置换映射；
13. API 失败、fallback、valid-only 与包含开发集816行敏感性分析；
14. CTID-first-parent 敏感性分析；
15. 所有协议偏离及原因。

## 13. 禁止事项

1. 禁止把未来三步任务的更高 Hit@5 描述为模型相对旧任务的提升。
2. 禁止用外层测试结果选择文本字段、编码器、prompt、epoch、融合权重或战术规则。
3. 禁止把 source 输入模型，或用缺失模式替代 source。
4. 禁止 Attack Flow `description` 为空时回退到技术名称。
5. 禁止只在表现好的来源、长文本或 seen/unseen 子集宣布方法成立。
6. 禁止逐样本挑融合权重。
7. 禁止按 prefix 行 bootstrap。
8. 禁止把 K 的本地重跑数字写成 Kuwano 论文原始成绩。
9. 禁止把 oracle 真实战术结果作为可实现方法。
10. 禁止静默删除 API 失败、OOV 目标或空 description。
11. 禁止用“某子集有意义”“更可解释”“接近显著”等话术覆盖主结果为负或证据不足。
12. 实验实现禁止套用 Superpowers 的 TDD、writing-plans 或 requesting-code-review 流程；运行故障只允许使用 systematic-debugging。
