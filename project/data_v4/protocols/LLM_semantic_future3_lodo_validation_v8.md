# Procedure-Aware Cross-Source ATT&CK Recommendation v8

> 状态：预注册、待实现、待运行。
>
> 冻结日期：2026-08-07。
>
> 本文档取代 v7 作为后续正式实验协议；v7 保留为历史记录。任何在查看外层测试结果后发生的修改都必须升级大版本，并把原结果与修改原因一并保留。

## 0. 研究问题与论文主张

主研究问题：

> 给定截至当前时刻已观察到的 ATT&CK 父技术序列与对应 procedure-level 事件描述，LLM 生成的攻击状态语义能否在严格跨来源验证下，为稀疏、未见的技术转移提供超出非语义序列模型的未来短窗口候选排序信号？

主任务不是“精确猜中唯一下一项”，而是：

> 输出未来至多三次动作范围内最值得分析师检查的 5 个父技术候选。

v8 将结论拆为四层，禁止混写：

| 层级 | 可检验主张 | 必要比较 |
|---|---|---|
| 现实泛化 | 随机 pair 划分会高估同源排列数据的泛化 | `P-pair` 对 `P-campaign`，仅限同一 `H=1` 任务 |
| 文本信号 | procedure text 含 ID 历史之外的预测信息 | `HM+R` 对 `HM` |
| LLM 增量 | LLM 状态摘要优于原始文本和等容量置换控制 | `HM+S` 对 `HM+R`、`HM+P` |
| 机制 | 语义主要补充训练折未见转移 | unseen-transition 分层 |

若 `HM+R>HM` 但 `HM+S<=HM+R`，只允许声称“真实事件文本有用”，不得声称 LLM 必要。若提升只来自 Stockpile，不得声称已在真实来源上稳定成立。

本实验不检验 KG、稀疏 asset/process/tool/privilege 字段、真实未来战术 oracle、风险评分或 conformal prediction。

## 1. 三种协议必须分开报告

| 协议 | 数据与标签 | 目标 | 独立单位 | 主指标 | 角色 |
|---|---|---|---|---|---|
| `P-pair` | Raj et al. 公开 ATT&CK v16 构造数据，技术名称词表 | `H=1` 单标签 next-step | 随机 prefix-target pair | row-micro Accuracy | 复核已发表随机划分的含义与重复污染 |
| `P-campaign` | 与 `P-pair` 完全相同 | `H=1` 单标签 next-step | MITRE campaign | campaign-macro Accuracy | 去除同 campaign 排列跨划分后的泛化审计 |
| `P-source` | CTID、Attack Flow、Stockpile，固定184父技术 | 未来至多3步唯一标签集合 | 数据来源与 campaign | campaign-macro NDCG@5 | 论文主实验 |

只允许直接比较：

- `P-pair` 与 `P-campaign` 中同一方法的 Accuracy，因为数据构造、词表和 `H=1` 相同；
- `P-source` 内各方法的同口径 future-3 指标。

禁止把 SECRYPT 的约 86% Accuracy、旧唯一下一项 Top-1、旧 `0.0873`、future-3 Hit@5/NDCG@5 互相做数值升降比较。

## 2. SECRYPT 2026 审计协议

### 2.1 外部对象与归因边界

参考工作：Raj et al., *MITRE ATT&CK-based Attack Chain Prediction Using Hybrid LSTM-Markov Models for Cyber Risk Assessment*, SECRYPT 2026, DOI `10.5220/0015075400004103`。

正式论文报告与本研究独立复算必须分列。下列数字是正式论文/公开仓库事实：

- 33 个有效 MITRE campaign 经战术分桶、桶内排列/采样和跨桶组合扩展为 4,849 条链；
- 论文报告 132,621 个 occurrence；
- 论文表格报告 128,413 个 pair；
- 公开实现采用随机 pair 划分，并把 campaign-holdout 列为后续方向。

下列数字只能标成“本研究对公开材料的独立审计”：

- 根据公开链重建，`sum(chain_length-1)=127,772`，与论文 128,413 相差641；
- 随机划分的完全重复率、prefix 覆盖率和字典基线；
- 任意 campaign-holdout 结果。

公开仓库顶层未提供明确代码许可证。v8 不复制、改写或分发其源码；审计与基线均根据论文算法描述 clean-room 独立实现。公开数据只作外部输入，不提交到本仓库。

### 2.2 输入与可复现环境

`P-pair/P-campaign` 固定输入为公开仓库 commit：

```text
e188cc6ec96df0288470380dbafccda1591e2c95
```

使用其中 ATT&CK Enterprise v16 工作簿的 `techniques`、`relationships`、`campaigns` 三张表。运行 manifest 必须记录：原文件 SHA-256、若发生无语义格式转换则同时记录转换后 SHA-256、脚本 SHA-256、Git commit、Python/NumPy/pandas 版本、操作系统、开始/结束时间。

释放版构造审计固定：

```text
random seed        = 42
numpy seed         = 42
PYTHONHASHSEED     = 0
min chain length   = 3
max permutations per tactic bucket = 25
max chains per campaign             = 200
tactic order       = ATT&CK Enterprise 14 tactics
```

脚本启动时若 `PYTHONHASHSEED!=0` 必须退出，不能在进程内假装修改该变量。另运行 `canonical` 敏感性分析：所有集合与去重结果按 Unicode 字典序排序；它不替换 released-order 主审计。

### 2.3 链与 pair 唯一键

每个 campaign 先取得其 `uses` 的技术集合；每个技术只按 ATT&CK 列出的第一个战术进入一个战术桶。桶大小不超过6时，按当前桶顺序取前25个全排列；超过6时用 seed 42 随机抽取最多25个唯一排列。按战术顺序做笛卡尔积，每 campaign 最多保留200条长度至少3的链。

稳定键：

```text
chain_key = (campaign_id, chain_ordinal, tuple(chain))
pair_key  = (campaign_id, chain_ordinal, prefix_len)
content_key = (tuple(prefix), target)
```

`pair_key` 是行身份；`content_key` 用于测完全重复。禁止把 `content_key` 当独立观测单位。

### 2.4 P-pair 划分

主 `P-pair` 复核正式 80/20 场景。为对齐公开程序的 RNG 消耗顺序，使用 `np.random.RandomState(42)`，依次生成：

1. 初始 90/10 permutation；
2. 100% scenario permutation；
3. 50/50 scenario permutation；
4. 80/20 scenario permutation，并以该次 permutation 的前80%为训练、后20%为测试。

切点固定为 `int(0.8*N)`。初始90/10仅作附录诊断，不参与主结论。

必须报告：总链数、campaign 数、occurrence、总 pair、唯一 `content_key`、训练/测试行数、测试 `content_key` 在训练中完全出现比例、测试 prefix 在训练中出现比例、测试 campaign 在训练中出现比例、每个 campaign 对 pair 的贡献分布。

### 2.5 P-campaign 划分

主 `P-campaign` 固定为 33 折 leave-one-campaign-out（LOCO）。每折全部留出一个 campaign 的所有生成链和所有 prefix pair，其余 campaign 训练。不得把同一 campaign 的不同排列分到两侧。

主结果先在每个留出 campaign 内算 Accuracy，再对33个 campaign 等权平均；同时报告全部测试 pair 合并后的 pooled row-micro Accuracy。若某折测试标签在训练词表中从未出现，仍保留该行并计错，不缩小分母。

### 2.6 冻结简单审计基线

所有并列先按训练频率降序，再按完整词表 Unicode 字典序。未见条件统一回退训练目标全局频率第一名。

| 方法 | 定义 |
|---|---|
| `FREQ` | 总是预测训练目标全局频率第一名 |
| `PREFIX` | 完整 prefix 对应的训练目标众数；未见 prefix 回退 `FREQ` |
| `M1` | 最后一个历史技术对应的训练目标众数；未见回退 `FREQ` |
| `M2` | 最后两个历史技术对应的训练目标众数；未见 M2 时回退 M1，再回退 `FREQ` |

这些基线用于量化重复查表，不等同于 SECRYPT Hybrid 模型。

## 3. P-source 数据冻结

### 3.1 标签与来源

固定父技术词表：

```text
project/data_v2/core/rl_label_vocab.csv
SHA-256 = 9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738
|V| = 184
```

来源：

| 来源 | 性质 | 预期 future-3 样本 | campaign |
|---|---|---:|---:|
| CTID | 真实 emulation plan | 275 | 10 |
| Attack Flow | 真实报告衍生最长路径 | 422 | 35 |
| Stockpile | 半合成 adversary profile | 119 | 27 |
| 合计 | — | 816 | 72 |

旧30条已解盲样本稳定映射为开发集，每源10条；正式主评估完全排除，预期分母为 CTID 265、Attack Flow 412、Stockpile 109，共786条且 campaign 数仍为10/35/27。任何计数或 campaign 消失都必须停止。

### 3.2 步骤表与文本

必须先冻结：

```text
project/data_v4/semantic_alignment/step_text_alignment.csv
project/data_v4/semantic_alignment/step_text_alignment_manifest.json
```

预期908个原始步骤：CTID 293、Attack Flow 466、Stockpile 149。键 `(source,campaign_id,step_index)` 和 `(source,stable_step_id)` 均唯一。

规则：

1. Attack Flow 复用已验证的断环和最长路径解析；只用 `properties.description`，禁止回退 `name`；
2. Stockpile 使用 ability description；
3. 三源文本执行 Unicode NFKC、空白折叠、大小写不敏感删除 `\bT\d{4}(?:\.\d{3})?\b`、再清理；
4. 最多保留前2,000字符，空值写 `[NO_DESCRIPTION]` 并显式标记；
5. 不使用 source、actor、campaign、文件名、技术名称回填、target/future 文本或稀疏资产字段；
6. CTID 12个多技术事件在主实验中先删除，再对剩余单技术事件重建序列；`first-parent` 仅作敏感性分析；
7. `source` 只用于划分、bootstrap 和报告，不进入任何模型或 prompt。

### 3.3 future-3 目标

对重建序列 `s_0,...,s_(n-1)` 的每个非末尾位置 `i`：

```text
observed = s_0,...,s_i
future   = s_(i+1),...,s_(min(i+3,n-1))
Y_i      = stable_unique(parent_technique(future))
```

只保留 `Y_i` 全部属于184类闭集的样本；不得取交集后伪装完整目标。prefix 中词表外历史技术保留。样本唯一键固定为 `(source,campaign_id,prefix_len)`。

预期目标基数：`|Y|=1/2/3` 分别为81/154/581。

### 3.4 泄漏门

在读取任何预测结果前必须断言：历史 step index 不晚于 endpoint；target 全部晚于 endpoint；历史/target step ID 无交集；prompt 不含未来文本、source、campaign、actor或文件路径；清洗文本不含 ATT&CK ID；基础表不预计算使用全数据训练统计的 visibility 字段；唯一键无重复。

## 4. P-source 外层与内层验证

三折外层 LODO：

| 折 | 训练来源 | 测试来源 |
|---|---|---|
| 1 | Attack Flow + Stockpile | CTID |
| 2 | CTID + Stockpile | Attack Flow |
| 3 | CTID + Attack Flow | Stockpile |

每个外层折的两个训练来源轮流作 inner-train/inner-validation，共两个 inner-LODO 折。所有 epoch、融合权重和允许的超参数只最大化两个 inner-validation 来源等权的 campaign-macro NDCG@5。外层测试来源不得影响任何选择。

## 5. 非语义基线阶梯

### 5.1 简单基线

| 代号 | 方法 | 固定定义 |
|---|---|---|
| `A0` | 目标频率 | 训练目标集合出现次数排序，并列按固定词表 |
| `CO` | 集合共现 | 对历史中每个唯一技术累加训练目标共现的 PPMI，未见回退 A0 |
| `A` | order-2/order-1/unigram relevance | v7 的0.5/0.3/0.2插值、`alpha_s=0.1` |
| `K` | 单调战术过滤 | 兼容候选优先、组内保持 A 顺序，仅诊断 |
| `T` | A + multi-hot 软战术 relevance | 所有战术保留，禁止压成单战术 |

`TIE-local` 与 `KUW` 仅在完成原论文公式核验并另存 clean-room 方法卡后运行；不得凭名称自行补全实现。它们未完成前报告“待实现”，不能用未经核验的近似版本占位。

### 5.2 ID-only 神经模型

LSTM 固定结构：

```text
embedding=128, hidden=256, layers=2, dropout=0.3
padding=right, pack_padded_sequence=true, readout=last valid hidden
head=184 independent logits, loss=BCEWithLogitsLoss
optimizer=AdamW, weight_decay=1e-4, batch_size=32
learning_rate in {3e-4,1e-3}
epoch in {20,40,60,80,100}
seeds=42,43,44,45,46
```

Transformer 固定结构：embedding 128、4 heads、2 encoder layers、FFN 512、dropout 0.3、causal mask、最后一个有效 token readout；其余优化器、网格和 seeds 与 LSTM 相同。

### 5.3 Markov-only beam 与 SECRYPT-adapted HM

共同参数：

```text
horizon = 3
beam width = 50
per-state branch cap = 20
transition floor = 1e-12
path dedup key = tuple(full generated future path)
```

一阶 Markov 只从训练折观测到的出边扩展；无出边时使用训练目标频率前20。概率采用 `count(last,next)+0.1` 在可用候选集合内归一化。

LSTM 在每个 beam step 对“历史+已生成路径”计算 next-step softmax。Hybrid 路径分数事前固定为 log-opinion pooling：

```text
log P_H(path) = (1-beta) * sum_h log P_M(x_h | history_h)
              + beta     * sum_h log P_LSTM(x_h | history_h)
beta in {0.0,0.1,...,1.0}
```

`MB` 固定 `beta=0`；`HM` 的 beta 由 inner-LODO 选择。不得在看到外层结果后改成纯重排或概率直接相乘。

将保留路径分数在 beam 内 softmax 归一化，未来标签 marginal 为：

```text
score_H(c) = sum_path P_H(path) * 1[c appears at least once in path]
```

按分数输出 Top-20，并列按固定词表。HM 是 clean-room 的 `SECRYPT-adapted` 基线，不称 exact reproduction。

## 6. 文本、LLM 与融合

### 6.1 编码器与 probe

原始描述为英文、LLM 摘要为中文，统一使用固定 revision 的 `BAAI/bge-m3`：1024维、最大8192 token、encoder冻结、dense CLS 后 L2 normalize。超过长度只在完整事件边界从旧到新删除，至少保留最后两步。

共享 probe：`1024→256→184`、GELU、dropout 0.3、BCEWithLogitsLoss、AdamW、lr `1e-3`、weight decay `1e-4`、batch 32、epoch `{20,40,60,80,100}`、seeds 42–46。

| 代号 | 语义输入 |
|---|---|
| `R` | ID、multi-hot tactics 与清洗后的原始历史 description |
| `S` | LLM 的 `stage_assessment`、`observed_capabilities`、`likely_next_intents` |
| `P` | 与 S 等容量，但训练摘要在 `source × prefix_len三分位` 内无固定点置换 |
| `B0` | LLM 返回的直接 Top-5，不参与编码 |

### 6.2 主方法与消融

v8 在查看任何正式外层结果前固定：

> `HM+S` 是论文候选主方法；`A/S` 是机制消融，不得用外层表现更好的方法事后替换主方法。

必须运行：`HM`、`HM+R`、`HM+S`、`HM+P`、`HM+T`、`HM+ST`。另报告 `A0/CO/A/K/T/LSTM/TR/MB` 和 v7 的 `R/S/P/B0`。

每个184维分支先做样本内标准化：

```text
std(z) = (z-mean(z))/max(sd(z),1e-6)
```

二分支融合：

```text
z = (1-lambda)*std(logit(score_H)) + lambda*std(z_semantic)
lambda in {0.0,0.1,...,1.0}
```

三分支 `HM+ST` 使用步长0.1的非负 simplex 权重，和为1。并列时优先语义权重更小、战术权重更小、HM权重更大，避免无收益时偏向复杂模型。

## 7. LLM 生成协议

任何 v8 外部 API 请求前必须重新获得明确授权；旧30条、旧字段或旧任务的授权不自动延伸。授权须说明发送清洗后的历史 description、样本数量、DeepSeek 模型及按 token 计费。API key 只读环境变量，不写日志或提交物。

固定 API：DeepSeek OpenAI-compatible endpoint、调用 `/models` 后冻结实际模型 ID、temperature 0、max_tokens 2048、显式 `thinking.type=disabled`、JSON object。prompt 沿用 v7 的未来三步版本，且三个摘要字段禁止 ATT&CK ID、source、campaign 和未来文本；`predicted_next_ttps` 恰好5个唯一父技术。

先只跑排除于主评估的30条开发集。机械门：JSON成功率≥95%、`reasoning_content` 为空100%、length finish≤2%、valid summary≥95%、摘要含 ATT&CK ID=0%、valid Top-5≥90%、泄漏断言100%。任何语义 prompt 修改升级协议小版本并重跑完整30条。开发试点不估计正式效果。

全量生成必须获得再次确认，写入新 run 目录；保存逐请求 prompt、step ID、原始响应、解析字段、HTTP/attempt/latency/token/时间戳；先提交未经后处理的原始版本，再编码训练。

intention-to-treat：B0无效记0；`HM+S` 无效回退 HM；`HM+ST` 无效回退 HM+T；`HM+P` 测试无效回退 HM。不得缩小分母。

## 8. 指标与统计

对 Top-5 `R_5` 与目标集合 `Y`：

```text
Hit@5       = 1[|R_5 ∩ Y|>0]
Precision@5 = |R_5 ∩ Y|/5
Recall@5    = |R_5 ∩ Y|/|Y|
NDCG@5      = DCG@5/IDCG@5, binary relevance
```

`P-source` 主指标为 campaign-macro NDCG@5：先 campaign 内平均，再来源内 campaign 等权，再三来源等权。row-micro 只作补充。

campaign cluster bootstrap：2000次、seed `20260807`；各来源内有放回抽完整 campaign；全部方法共享索引；每 replicate 先平均5个训练 seed，再算配对差；总体为三来源等权；报告 percentile 95% CI。seed 不当独立样本。

必须分层：transition visibility `all_seen/mixed/all_unseen`、target-label visibility、最后一步文本 `<40/≥40`、目标基数1/2/3。少于5 campaign或20行的格只报描述统计，不作推断。

## 9. 预注册判定

主差值：

```text
Delta_s = NDCG5_campaign_macro(HM+S,s) - NDCG5_campaign_macro(HM,s)
```

### 9.1 LLM 语义主张获得方向支持

以下全部满足：

1. 至少两个来源 `Delta_s>0`；
2. CTID 与 Attack Flow 的等权平均差值 `>0`，排除只靠半合成 Stockpile；
3. 三来源等权总体 `HM+S>HM`；
4. 至少两个来源且三来源总体 `HM+S>HM+P`；
5. 三来源总体 `HM+S>HM+R`。

若同时总体 `HM+S-HM` campaign-bootstrap 95% CI 下界大于0，记“强支持”；CI跨0只记“方向支持但统计不确定”，禁止写“接近显著”。

若前四项成立但第5项不成立，只能结论“procedure text 有增量，未证明 LLM 转换优于直接文本编码”。

### 9.2 缺乏证据

命中任一即报告“LLM语义融合在当前任务设定下缺乏证据”：

1. 三来源总体 `HM+S<=HM`；
2. CTID 与 Attack Flow 等权平均 `HM+S<=HM`，且正结果只来自 Stockpile；
3. 三来源总体 `HM+P>=HM+S`；
4. 三个来源全部 `HM+R>=HM+S`。

其余情况固定为“未达到预注册支持条件，也未命中判死条件；当前证据不足”。

`HM+ST` 独立判定：至少两个来源 `HM+ST>HM+S`、至少两个来源 `HM+ST>HM+T`、总体同时超过二者，才允许声称战术先验与语义互补；它不改变 `HM+S` 主结论。

unseen 机制只有在 `all_unseen` 的总体 `HM+S-HM` 大于 `all_seen` 且前者为正时才记方向支持；否则不得声称语义专门补未见转移。

## 10. 执行顺序与停止门

1. 提交 v7 与路线历史基线；
2. 发布并提交本 v8；
3. 运行零费用 SECRYPT `P-pair/P-campaign` 数据与简单基线审计；
4. 生成908步对齐表、816 future-3 样本、30条开发映射和全部 manifest；
5. 运行 `P-source` 零费用 A0/CO/A/K/T 与 coverage；
6. 完成 LSTM/TR/MB/HM 及 SECRYPT `P-pair/P-campaign` 神经模型；
7. 固定 BGE-M3 revision，运行 R 与 HM+R；
8. 获得新的外部发送授权后运行30条 v8 开发试点并停止汇报；
9. 人工确认后全量生成，立即提交原始响应；
10. 运行 P/S/ST 与 HM+P/HM+S/HM+ST；
11. 一次性生成主表、bootstrap、分层和预注册判定。

任一输入哈希异常、计数/campaign 分母不符、泄漏断言失败、模型 revision 无法固定、外层测试参与选择，立即停止。修复后升级协议并披露，禁止为凑预期数字调整规则。

## 11. 必须落盘

每次运行使用唯一 `run_id`，至少保存：输入与脚本哈希、外部 commit、配置、依赖、stdout；链/pair/重复率审计；步骤对齐与future-3样本；开发集映射；每折每seed超参数；逐样本Top-20及分数；campaign指标；bootstrap索引；分层；API原始请求响应、失败与费用；所有偏离及发生在查看外层结果前后。

SECRYPT 审计产物固定目录：

```text
project/data_v4/repro_secrypt/{run_id}/
  manifest.json
  summary.csv
  campaign_loco.csv
  report.md
  stdout.log
```

不提交外部 ATT&CK 工作簿、SECRYPT 源码、模型权重或其生成链明细。

## 12. 禁止事项

1. 不把随机 pair 的86%写成跨 campaign 或跨来源能力；
2. 不把本研究复算数字归因给 Raj et al.；
3. 不在脚本、输入哈希、`PYTHONHASHSEED`、tie-break、fallback 未冻结时引用精确字典或 campaign-holdout 数字；
4. 不把42.3% tactic-level coverage与13.3% technique bigram直接比较；只有统一词表、折和分母后才比较；
5. 不把来源标识或不均匀缺失模式输入模型；
6. 不用测试来源训练任何 Markov、LSTM、文本、战术或外部统计；
7. 不逐样本选择融合权重，不按 prefix 行 bootstrap；
8. 不用表现最好的来源、文本长度层、目标基数层或单个seed替换主结论；
9. 不把未来三步更高 Hit@5写成相对旧 next-step 方法提升；
10. 不静默删除 OOV、API失败或空文本；
11. 不以“更可解释”“某子集有意义”或“接近显著”挽救负结果；
12. 不在获得 v8 新授权前发送任何新字段或新样本到外部 API；
13. 实验实现不使用 Superpowers 的 TDD、writing-plans 或 requesting-code-review 流程；故障只采用 systematic-debugging。

## 13. 来源

- Raj et al. 正式论文：<https://doi.org/10.5220/0015075400004103>
- SECRYPT 2026 论文集：<https://www.scitepress.org/ProceedingsDetails.aspx?ID=GrF7dp3BzeY%3D&t=1>
- 官方报告日程：<https://www.insticc.org/node/TechnicalProgram/secrypt/2026/presentationDetails/150754>
- 公开仓库：<https://github.com/mayank02raj/MITRE-ATTACK-based-Attack-Chain-Prediction>
- BGE-M3：<https://huggingface.co/BAAI/bge-m3>
