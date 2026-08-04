# T0.1 代码与数据盘点

- 执行日期：2026-08-04
- 范围：只读盘点现有代码、数据、缓存、权重和论文 Table 2 / 3 / 4 / 6；未运行训练、推理或 T0.2 复现实验。
- T0.1 判定：**判死**
- 流水线状态：**立即停止；不得进入 T0.2，不得重新生成 reasoning，等待人工决策。**

## 0. 判死依据

T0.1 的硬条件已经命中：主 CoT reasoning 缓存不完整。

| Split | 应有样本 | 有效 `llm_thinking_process` | 空 reasoning | 有效覆盖率 |
|---|---:|---:|---:|---:|
| Train | 9,919 | 9,773 | 146 | 98.528077% |
| Validation | 2,102 | 2,068 | 34 | 98.382493% |
| Test | 2,107 | 2,065 | 42 | 98.006645% |
| 合计 | 14,128 | 13,906 | **222** | **98.428652%** |

对 222 条缺失记录的进一步只读检查表明：

- `raw_output` 均非空，但内容是停在中途的 JSON/推理文本；
- `llm_thinking_process` 为空；
- `predicted_next_ttps` 为 `[]`；
- 从文本终止位置和脚本的 `max_tokens=1024` 看，最可能原因是输出被截断，JSON 解析失败后脚本返回空字段。这里是基于文件证据的推断，不是重新运行得到的结论。
- 下游代码对空 CoT 有两种不一致处理：有的编码空字符串，有的替换成 `"No reasoning."` 或 `"None"`，因此同一批缺失样本在不同表格脚本中的实际输入并不一致。

此外，只找到一个主 reasoning 生成脚本，且其输入/输出被硬编码为 validation：

- `project/llm/run_llm_kg_context_test_pipeline.py`
- 输入：`sim_val_parent_min3_kg_context.csv`
- 输出：`sim_val_llm_cot.csv`

没有找到生成 train/test 主缓存时使用的独立脚本、命令日志或配置快照。现有脚本可能曾通过手工修改常量复用，但修改历史未记录。

依据用户规定：“reasoning 缓存不完整或无法定位生成脚本，停下汇报，不要自行重新生成”，因此本轮结论必须是 **判死**。

## 1. 论文与代码的总体对应关系

论文任务是 184 类 ATT&CK parent-technique 下一步预测。现有主数据正好是 14,128 行：

- Train：9,919
- Validation：2,102
- Test：2,107
- 标签表：`project/data/rl_label_vocab.csv`
- ATT&CK taxonomy：`project/data/enterprise-attack-18.1.json`

但论文与代码存在一项已静态确认的实现矛盾：

- 论文第 4.1/4.4 节写最大长度为 50、短序列左填充、使用最后一个非 padding 时刻的隐藏状态；
- `project/rl/train_rl_multiseed.py` 和 `project/rl/train_rl_baseline_v2.py` 实际设置 `MAX_LEN=20`，短序列右填充，并直接使用 GRU 处理完整 20 步后的最终 hidden state；
- 对 `rl_baseline_v2_seed42.pt` 的 pickle 元数据只读解析也确认 `max_len=20`；
- Transformer seed42 checkpoint 同样记录 `max_len=20`。

因此论文中的实现描述不能代表当前 checkpoint 的真实配置。

## 2. 数据资产与构建入口

### 2.1 主数据文件

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `project/data/sim_train_parent_min3.csv` | 9,919 | `c306208764b8702cf479124d5b6751ea3c389b1b1b3e62d6088ed632d9f57755` |
| `project/data/sim_val_parent_min3.csv` | 2,102 | `2ffc7c91d7ac7ad3654d3055ebd1886b7db5e83892546b5460ed773e2dc3d5ae` |
| `project/data/sim_test_parent_min3.csv` | 2,107 | `8e3ec3f13351e236c827ac8769b03a0e277e4c1422dccb8390961f0c3b35f213` |
| `project/data/sim_train_parent_min3_kg_context.csv` | 9,919 | `75d6f57e2386bf611b888d949a58c50e3ed4582ed661786eb9b63e464a13b69c` |
| `project/data/sim_val_parent_min3_kg_context.csv` | 2,102 | `d61163e15f8c45b9a588b606a21a0ad621adf2c0bddee6675fb34143681d9260` |
| `project/data/sim_test_parent_min3_kg_context.csv` | 2,107 | `f7ecb9a67dc4c7d18bd4f1d40aaa8803a057451415394bd9e5c51bf421827fdc` |

主序列文件包含：

`sequence_id, prefix_len, prefix_techniques, prefix_tactics, next_technique, next_technique_name, next_technique_id, prefix_technique_names, prefix_technique_ids, next_technique_id_parent, prefix_technique_ids_parent`

### 2.2 数据构建代码

| 模块 | 入口 | 作用 | 盘点结论 |
|---|---|---|---|
| ATT&CK parent lookup | `project/data/build_attack_parent_lookup.py` | 从 Enterprise ATT&CK JSON 构建父技术映射 | 已定位 |
| KG snippet | `project/data/build_attack_kg_snippets.py` | 构建技术/KG 检索片段 | 已定位 |
| Prefix KG context | `project/data/build_sim_test_kg_context.py` | 给现有 prefix CSV 添加 recent IDs、检索片段和 KG 文本 | 已定位，但脚本当前硬编码读取 validation，文件名与实际用途不一致 |
| 数据一致性检查 | `project/data/DataSet_check.py`、`project/data/CheckCot.py` | 行数、标签、sequence ID 和 CoT 检查 | 已定位 |
| 原始报告/CTID 导入 | **未记录** | 应负责从 APT 报告和 CTID emulation plan 恢复有序攻击轨迹 | 仓库中没有主数据的原始导入脚本和完整来源清单 |
| parent 映射、prefix slicing、min-frequency=3、主 split | **未记录** | 应产出三个 `sim_*_parent_min3.csv` | 现有代码无法从原始资料原样重建主 split |

仓库后续新增的 `project/data_v2/external_*` 不能证明它们就是论文主数据生成时使用的原始快照；主数据没有来源 URL/commit 到 `SIM_###` 的不可变映射。

### 2.3 campaign_id / threat group

主序列 CSV 和主 reasoning CSV 中均没有：

- `campaign_id`
- `campaign`
- `threat_group`
- `intrusion_set`
- `actor`

`sequence_id` 也不是每样本唯一键：一个 sequence 会产生多个 prefix 样本。当前可用的逐样本唯一组合是：

`(sequence_id, prefix_len)`

reasoning CSV 没有 `prefix_len`，但当前文件中：

`(sequence_id, state, true_label)`

在各 split 内唯一。

可恢复方案：

1. 临时 campaign surrogate：从 `sequence_id` 去掉 `_partNNN`，例如 `SIM_014_part021 -> SIM_014`。
2. 真正的 campaign/threat group：必须回到原始 APT 报告或 CTID plan，在生成 prefix 前建立 `sequence_id -> campaign_id, threat_group, source_document, chronology` 映射，并将字段传播到每个 prefix。
3. 仅凭当前处理后 CSV，无法把 `SIM_###` 可靠映射回真实 threat group；该映射 **未记录**。

按上述 surrogate 检查，当前 split 并非 root/campaign 隔离：

| Split | sequence_id 数 | root surrogate 数 |
|---|---:|---:|
| Train | 552 | 142 |
| Validation | 118 | 75 |
| Test | 119 | 73 |

root surrogate 重叠：

- Train vs Validation：62
- Train vs Test：65
- Validation vs Test：42
- Test 的 73 个 root surrogate 中有 65 个也在 Train

论文写“sequence/campaign-level split”。代码数据只保证完整 `sequence_id` 不跨 split；同一 `SIM_###` 的不同 `part` 大量跨 split。若 `SIM_###` 对应原始 campaign/root，则论文的 campaign-level split 描述不成立。最终定义仍需原始数据构建记录确认。

## 3. 模型与融合模块入口

| 模块 | 入口文件 | 作用 | 状态 |
|---|---|---|---|
| 单 seed GRU | `project/rl/train_rl_baseline_v2.py` | 128-d embedding + 单层 128-d GRU；reward-weighted CE；按 Val Top-5、再按 MRR 保存 | 脚本使用相对数据路径，不能从仓库根目录直接按原样运行 |
| 5-seed GRU | `project/rl/train_rl_multiseed.py` | seeds 42–46 训练 GRU 并输出测试指标 | 已定位；论文建议/后续规则的 0–4 尚未冻结 |
| Global Prior / Markov | `project/Ablation experiment/run_markov_baseline.py` | 0/1/2 阶计数排名；二阶未见时回退一阶，再以 global ranking 补齐 | 已定位；只输出 stdout，不保存逐样本预测 |
| Markov + LLM | `project/Ablation experiment/run_markov_llm_fusion.py` | 二阶 Markov probability 与 CoT probe 做 probability/logit fusion；Val MRR 选 alpha | 已定位 |
| Transformer baseline | `project/Ablation experiment/train_transformer_baseline.py` | 2 层、4 头 Transformer，masked mean pooling；Val MRR early stopping | 主入口只运行 seed42；只有 seed42 checkpoint |
| Transformer + LLM | `project/Ablation experiment/run_transformer_llm_fusion.py` | Transformer logits 与 CoT logits；Val MRR 网格选择 alpha | 只有 seed42 资产 |
| LLM reasoning 生成 | `project/llm/run_llm_kg_context_test_pipeline.py` | Qwen 本地 vLLM 生成 `_thinking_process` 和 Top-5 TTP | 只保留 validation 常量；train/test 运行快照未记录 |
| Semantic encoder + probe | `project/llm/train_llm_multiseed.py` | fixed BAAI/bge-base-zh-v1.5 CLS embedding + 768→256→184 MLP probe；seeds 42–46 | BGE revision 未固定；CoT 5 个 checkpoint 存在 |
| No-CoT probe | `project/Ablation experiment/train_nocot_probe.py` | No-CoT semantic probe | 只有 seed42 checkpoint |
| Empty probe | `project/Ablation experiment/make_and_train_empty.py` | 构造 `No reasoning.` 控制并训练 probe | 只有 seed42 checkpoint |
| CoT/No-CoT/Empty ablation | `project/Ablation experiment/run_3way_ablation.py` | 三种语义输入的 LLM-only 与 GRU global-logit fusion；各自 Val MRR 选 alpha | seed42；只输出 stdout |
| Seed42 late-fusion variants | `project/fusion/run_late_fusion_search.py` | GRU only、global probability、global logit、adaptive routing、dual recall | Table 2 的 GRU/global-fusion 候选产出脚本 |
| Multi-seed global fusion | `project/fusion/run_final_fusion.py` | seeds 42–46 的 GRU、global probability/logit、dual-recall | 有 5 组 GRU/CoT checkpoint；结果未落盘 |
| Feature/direct/gated fusion | `project/fusion/train_gated_fusion.py` | LLM-only、direct concat、scalar gate、vector gate | seed42 主入口；checkpoint 存在但没有统一原始预测 |
| Regularized adaptive gate | `project/adaptive fusion/run_ AdaptiveFusion.py` | logistic gate、rule gate、global fusion 对照和显著性 | 只运行 seed42 |
| 后续外部分布探索 | `project/fusion/run_dynamic_rl_llm_fusion_on_56.py`、`run_gated_rl_llm_rerank_on_56.py`、`run_cmab_dynamic_fusion.py` | CTID/56 样本上的后续探索 | 不属于论文 Table 2/3/4/6 主产出链 |

## 4. Reasoning 缓存、语言和生成配置

### 4.1 主 CoT 缓存

| 文件 | 行数 | 空 reasoning | 语言 | SHA-256 |
|---|---:|---:|---|---|
| `project/data/sim_train_llm_cot.csv` | 9,919 | 146 | 中文为主，包含英文 ATT&CK 名称/缩写 | `bd5c04819d4017392a48ffa87e3787ef17e0f8951ff25ca5579b78e2a650e6c6` |
| `project/data/sim_val_llm_cot.csv` | 2,102 | 34 | 中文为主，包含英文 ATT&CK 名称/缩写 | `4feaa6fb34472b08f1d3b27fc27a1faf2d8c04577cfc0730115a1aa4efa74f7d` |
| `project/data/sim_test_llm_cot.csv` | 2,107 | 42 | 中文为主，包含英文 ATT&CK 名称/缩写 | `ce7c3d4ba2437530521847dd13354ea5d9726a2e722a2784f2e2529af35b5cf3` |

对全部非空 reasoning 进行字符统计，没有发现英文占主导的记录；输出是中文推理，夹杂 TID、ATT&CK 技术名和英文战术术语。

字段：

`sequence_id, state, true_label, llm_thinking_process, predicted_next_ttps, raw_output`

缓存没有保存：

- campaign_id
- prefix_len
- LLM 模型 revision/hash
- vLLM 版本
- prompt 文件 hash
- request ID
- 生成时间
- 每行生成配置

### 4.2 LLM 与生成参数

从 `run_llm_kg_context_test_pipeline.py` 和 `start_vllm_clean.sh` 能找到：

- 模型路径/名称：`/model-storage/model/Qwen3.5-35B-A3B-FP8`
- 精确模型 revision/commit/hash：**未记录**
- tokenizer revision：**未记录**
- vLLM 版本：**未记录**
- temperature：`0.0`
- max_tokens：`1024`
- `enable_thinking=False`
- response format：JSON schema，要求 `_thinking_process` 和 `predicted_next_ttps`
- 并发数：30
- vLLM：tensor parallel 2、max model length 2048、FP8 模型目录
- sampling seed：**未记录**
- train/test 实际生成命令：**未记录**

因此可以确定模型目录名和采样温度，但不能确定可复现所需的精确模型版本。

### 4.3 完整主 prompt 模板

System prompt：

```text
你是一个高级 APT 威胁狩猎专家与 ATT&CK 攻击图分析师。
你的任务是：基于攻击者已执行的 ATT&CK 技术序列（Prefix）以及相关的知识图谱上下文（KG Context），推断攻击者当前的【阶段性操作状态】，并直接预测下一步最可能执行的 5 个 ATT&CK Parent Technique（父技术）。

由于输入数据是模拟的宏观序列，你必须遵守以下严格限制：
1. 绝对不要凭空捏造微观动作。推理必须完全基于传入的 Prefix ID 及 KG Context 进行逻辑推演。
2. 预测结果必须是纯粹的父技术 ID。

请在 JSON 的 `_thinking_process` 字段中写下你的推理过程，按以下三步进行思考：
[战术阶段评估]：分析 Prefix 中最后两步，它们处于什么战术阶段？
[已获资产推演]：基于前缀技术，攻击者目前掌握了什么级别的粗粒度资产或权限？
[意图图谱映射]：结合 KG Context，前缀的最后几步操作最可能为后续攻击开启了什么逻辑攻击面？

推理完成后，请在 `predicted_next_ttps` 数组中输出恰好 5 个最可能的下一步 ATT&CK 父技术 ID。
```

User prompt：

```text
### 攻击前缀序列 (Prefix) ###
{prefix}
(重点关注最后两步：{recent_ids})

### 相关的知识图谱上下文 (KG Context) ###
{kg_context_truncated_to_700_chars}

### 任务要求 ###
请先在 `_thinking_process` 字段推演，随后在 `predicted_next_ttps` 数组输出 5 个预测的父技术 ID。
```

### 4.4 No-CoT 缓存的额外问题

`project/Ablation experiment/hotfix_align.py` 用 `sequence_id` 建字典，再覆盖 No-CoT 文件。由于每个 `sequence_id` 有多个 prefix，该字典只保留每个 sequence 的最后一条 reasoning，并把同一文本复制给该 sequence 的所有 prefix。

静态统计：

| Split | No-CoT 行数 | sequence_id 数 | No-CoT reasoning 唯一数 | 同一 sequence 内 reasoning 变化 |
|---|---:|---:|---:|---|
| Train | 9,919 | 552 | 551 | 0 |
| Validation | 2,102 | 118 | 117 | 0 |
| Test | 2,107 | 119 | 119 | 0 |

因此 Table 4 的 No-CoT 不是严格的逐 prefix 对照；它受错误对齐脚本影响。该问题可能影响 Empty > No-CoT 的非单调结果，但 T0.1 不对根因作最终实验判定，留待人工决定是否允许后续修复。

## 5. Checkpoint 资产

| 模型 | 现有 seed | 结论 |
|---|---|---|
| GRU | 42, 43, 44, 45, 46 | 5 个 checkpoint 存在 |
| CoT semantic probe | 42, 43, 44, 45, 46 | 5 个 checkpoint 存在 |
| No-CoT probe | 42 | 缺少 43–46 |
| Empty probe | 42 | 缺少 43–46 |
| Transformer | 42 | 缺少 43–46 |
| Dynamic/regularized gate | 仅单个 best 文件或 seed42 入口 | 没有 Table 3 所需的五个逐 seed 资产 |

主要 checkpoint SHA-256：

- GRU seed42：`8c344a899e517f669de471c340b532fba99be0edb96d9c68d577cd4f8460c3eb`
- GRU seed43：`f793ed09ced4933dd09cdbb7c597be4b55cfb1cad44a136bc74cfb3dfb7b7c85`
- GRU seed44：`9188b497455b5cd454b7c83e4a7bda18e129ba2098125cd2e5f984771557afe8`
- GRU seed45：`ba2e9853e683f59b08867333b504aa26d5ac95001e798587dca92d506773810e`
- GRU seed46：`bd2626a15baef84efb9d4a96b7e38b70672a56cd312f53acec0d0d0bfe3e8e60`
- Transformer seed42：`a461d144ec65648d5d58d8864204b6fce97e6bb73ce72f3cabf8eedce48f2498`
- CoT probe seed42：`ead9df1ae62c954af15377d53f9cc1bab61adc4e9f8ed456adf1a2b2ec567b10`
- No-CoT probe seed42：`64c375336aebedda5671f69800389c984d1eaab02e6ef3c4bc4b109c45b6783d`
- Empty probe seed42：`5b11657fc94cefe1e001c604cf5cbd495e0146164777b6770721eb31c3b97dd6`

仓库没有主项目的 `requirements.txt`、environment lock、PyTorch/CUDA/cuDNN/transformers 版本快照。BGE 只按模型名加载，未固定 revision。

## 6. Table 2 每个数字的代码来源

论文 Table 2：

| Method | Top-1 | Top-5 | MRR | Macro-F1 | Weighted-F1 | 候选产出脚本与命令 | 当前可复现性 |
|---|---:|---:|---:|---:|---:|---|---|
| Global Prior (0-Order) | 0.0574 | 0.1951 | 0.1475 | 0.0007 | 0.0062 | `python "project/Ablation experiment/run_markov_baseline.py"` | 代码链已定位；T0.2 未执行 |
| 1st-Order Markov | 0.4794 | 0.8434 | 0.6319 | 0.2664 | 0.4374 | 同上 | 代码链已定位；T0.2 未执行 |
| 2nd-Order Markov | 0.5562 | 0.8543 | 0.6833 | 0.3718 | 0.5350 | 同上 | 代码链已定位；T0.2 未执行 |
| 2nd-Order Markov+LLM (Prob Fusion) | 0.5558 | 0.8719 | 0.6901 | 0.3753 | 0.5317 | `python "project/Ablation experiment/run_markov_llm_fusion.py"` | 依赖不完整 CoT 与未固定 BGE revision |
| 2nd-Order Markov+LLM (Logit Fusion) | 0.5553 | 0.8747 | 0.6895 | 0.3846 | 0.5306 | 同上 | 同上 |
| GRU Baseline | 0.5444 | 0.8695 | 0.6807 | 0.2969 | 0.5237 | `python "project/fusion/run_late_fusion_search.py"` 的 seed42 baseline；训练入口为 `project/rl/train_rl_multiseed.py` | seed42 checkpoint 存在；T0.2 未执行 |
| GRU+LLM (Global Fusion) | 0.5477 | 0.8780 | 0.6861 | 0.3049 | 0.5215 | `python "project/fusion/run_late_fusion_search.py"`，Val MRR 选 alpha | 依赖不完整 CoT；T0.2 未执行 |
| Transformer Encoder | 0.4813 | 0.8619 | 0.6436 | 0.2815 | 0.4651 | `python "project/Ablation experiment/train_transformer_baseline.py"` | seed42 checkpoint 存在；T0.2 未执行 |
| Transformer+LLM (Logit Fusion) | 0.5040 | 0.8757 | 0.6617 | 0.2815 | 0.4763 | `python "project/Ablation experiment/run_transformer_llm_fusion.py"` | 依赖不完整 CoT；T0.2 未执行 |

现有脚本没有为这些 Table 2 行保存统一的逐样本 logits/top-20、配置快照和完整历史 stdout。因此“代码入口已定位”不等于数字已经复现。

## 7. Table 3 每个数字的代码来源

论文 Table 3：

| Model/Variant | N Seeds | Top-1 | MRR | 文件/命令 | 证据结论 |
|---|---:|---:|---:|---|---|
| GRU Baseline | 5 | 0.545 ± 0.003 | 0.654 ± 0.003 | `project/data/Multi-SeedAggregator.py`；原运行位置推定为 `project/data` 后执行 `python Multi-SeedAggregator.py` | 数组被手工硬编码；脚本不读取 checkpoint、预测或日志 |
| Global Logit Fusion (No-CoT) | 5 | 0.550 ± 0.002 | 0.660 ± 0.002 | 同上 | **现有代码无法复现为五 seed 实验**：只有 No-CoT seed42 checkpoint |
| Global Logit Fusion (CoT) | 5 | 0.562 ± 0.001 | 0.672 ± 0.002 | 同上；真正可重算候选入口为 `python "project/fusion/run_final_fusion.py"` | published summary 来自硬编码数组；5 个 GRU/CoT checkpoint 可用于另行重算，但没有原 stdout |
| Transformer Baseline | 5 | 0.531 ± 0.011 | 0.639 ± 0.009 | 同上 | **现有代码无法复现为五 seed 实验**：只有 Transformer seed42 checkpoint，主入口只运行 42 |
| Dynamic Gating Fusion | 5 | 0.541 ± 0.003 | 0.649 ± 0.002 | 同上；候选单 seed 入口为 `python "project/adaptive fusion/run_ AdaptiveFusion.py"` | **现有代码无法复现为五 seed 实验**：入口只运行 seed42，缺少逐 seed 产物 |

`Multi-SeedAggregator.py` 第 85–87 行明确说明应填入“真实多 seed 数据”，但第 89–109 行直接写入五组数值；它是排版/汇总脚本，不是实验结果读取器。因此 Table 3 的 exact mean/std 没有可追溯的逐 seed 原始产物。该事实直接影响 T0.2 的 GRU MRR 与 Fusion Top-1 矛盾调查。

## 8. Table 4 每个数字的代码来源

候选命令：

`python "project/Ablation experiment/run_3way_ablation.py"`

脚本固定 seed42，对 CoT、No-CoT、Empty 分别在 validation 上以 MRR 从 `alpha=0.00..1.00`、步长 0.02 选择 alpha，然后评估 test。

| Variant | LLM-only Top-1 | Top-5 | MRR | GRU+LLM Top-1 | Top-5 | MRR | 当前可复现性 |
|---|---:|---:|---:|---:|---:|---:|---|
| Empty | 0.0574 | 0.2041 | 0.1432 | 0.5439 | 0.8709 | 0.6803 | seed42 checkpoint 存在；未保留原 run.log/逐样本预测 |
| No-CoT | 0.1058 | 0.2648 | 0.1955 | 0.5420 | 0.8662 | 0.6793 | seed42 checkpoint 存在，但 No-CoT cache 已被错误的 sequence-level hotfix 覆盖 |
| CoT | 0.3916 | 0.7632 | 0.5511 | 0.5477 | 0.8780 | 0.6861 | seed42 checkpoint 存在，但 CoT 缺 42 条 test reasoning |

Table 4 的 18 个数都有候选计算入口，但当前资产不能证明它们来自正确的逐-prefix No-CoT 对照，也没有合规原始产物。

## 9. Table 6 每个数字的代码来源

### 9.1 Calibration

命令：

`python "project/data/Calibrate_analysis.py"`

脚本使用 seed42 GRU/CoT probe、test split，并把 alpha **硬编码为 0.18**；脚本内部没有读取 validation alpha 搜索结果或其配置快照。

| Model | ECE (%) | Brier Score | 输出文件 | 当前可复现性 |
|---|---:|---:|---|---|
| GRU Baseline | 13.46 | 0.6318 | `project/data/calibration_summary.csv` | 可用现有 seed42 checkpoint 候选重算；T0.2 未执行 |
| LLM Probe | 16.82 | 0.7931 | 同上 | 依赖不完整 CoT 与未固定 BGE revision |
| Global Fusion, precomputed embedding | 12.14 | 0.6187 | 同上 | alpha 来源未落盘；依赖不完整 CoT |

### 9.2 Latency / Relative Cost

命令：

`python "project/data/Cost_Alasysis.py"`

脚本固定 sample_size=256、batch_size=32、seed42、alpha=0.18。

| Model/Setting | Avg. Latency/Sample (ms) | Relative Cost | 输出文件 | 当前可复现性 |
|---|---:|---:|---|---|
| GRU Baseline | 0.0093 | 1.00× | `project/data/cost_breakdown_system.csv` | **不能精确复现**：GPU、CUDA、PyTorch、transformers、BGE revision 和负载环境未记录 |
| LLM Probe | 13.8807 | 1495.91× | 同上；CSV 实际命名为 LLM Semantic Stack，包含 encoding+probe+fusion | **不能精确复现**，且论文行名弱化了其实际组件 |
| Global Fusion, precomputed embedding | 0.0158 | 1.71× | 同上 | **不能精确复现** |
| Global Fusion, online encoding | 15.3036 | 1649.25× | 同上 | **不能精确复现**；不含在线 Qwen reasoning 生成成本 |

## 10. 当前明确无法由现有代码/产物复现的数字

1. **Table 3 Global Logit Fusion (No-CoT) 的全部五-seed mean/std**：缺少 4 个 seed checkpoint。
2. **Table 3 Transformer Baseline 的全部五-seed mean/std**：只有 seed42。
3. **Table 3 Dynamic Gating Fusion 的全部五-seed mean/std**：入口和资产仅支持 seed42。
4. **Table 3 所有模型的 published exact mean/std provenance**：`Multi-SeedAggregator.py` 使用手工数组，没有读取原始结果；即使 GRU/CoT 可重算，也不能证明 published 数字来自现存 checkpoints。
5. **Table 6 的四组精确 latency 与 relative cost**：运行环境未冻结。
6. **任何依赖“完整 14,128 条 CoT reasoning”的结论**：实际缺 222 条。
7. **任何要求精确 Qwen checkpoint/version 的 reasoning 复现**：只记录本地目录名，revision/hash 未记录。
8. **从原始 APT/CTID 资料重建主数据、主 split 和 campaign/threat-group 字段**：主构建脚本和来源映射未记录。

## 11. T0.1 最终结论

**判死。**

原因不是“结果暂时不理想”，而是已经命中预先规定的硬条件：

- 主 reasoning 缓存不完整：222/14,128 条没有可用的 `llm_thinking_process` 和 Top-5；
- train/test 的实际 reasoning 生成命令与配置快照未记录；
- 精确 Qwen revision 未记录；
- Table 3 的 published multi-seed 数字来自手工数组，且三种模型缺少五-seed 资产；
- No-CoT cache 存在按非唯一 `sequence_id` 覆盖的对齐错误。

按执行规则，本 Agent 在 T0.1 后停止，不进入 T0.2，不补生成 reasoning，不修改 split，不训练或重算任何论文数字。下一步必须由人工决定如何处置缓存缺失和实验 provenance。

