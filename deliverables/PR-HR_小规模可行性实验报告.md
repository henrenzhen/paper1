# PR-HR 与选择性预测：小规模可行性实验报告

日期：2026-07-29  
性质：诊断性生死实验，不是可直接写入论文主表的最终结果

## 1. 结论先行

本轮实验得到的是“内部工件有信号，但核心算法和跨域可靠性均未成立”：

1. **内部SIM工件中，LLM候选与GRU存在非常小的互补性。** GRU Top-5 覆盖率为 87.71%，加入LLM Top-5候选后的并集上限为 88.32%，净增 **0.617个百分点、13条样本**；SIM根级bootstrap 95%区间为 **[0.299, 0.985]个百分点**。达到预设的内部最低门槛，但效应太小，而且GRU骨干存在同根训练污染，不能单独支撑论文贡献。
2. **当前轻量成对重排器没有证明有效。** OOF Top-1 从GRU的54.960%变为55.007%，仅增加 **0.047个百分点、净多1条正确样本**；根级bootstrap 95%区间为 **[-1.074, 1.139]个百分点**，跨过0。重排器救回93条、同时破坏92条。因此，不能声称该算法已经有效。
3. **内部SIM工件中有强选择性信号，但不能外推。** 阈值只使用其他元折校准；目标覆盖80%时，实际覆盖79.92%，被接受样本准确率为 **62.173%**，相对54.960%净增 **7.214个百分点**，根级bootstrap 95%区间为 **[5.858, 8.732]个百分点**。但预计算GRU的训练集覆盖了65/73个诊断SIM根，所以这不是未见根泛化证据。
4. **真正独立的CTID压力测试给出了反例。** 10个组织、281条样本上，GRU Top-1只有2.847%；跨组织折校准到80.78%实际覆盖后，接受样本准确率降到2.203%，变化为 **-0.644个百分点**，10组织cluster bootstrap区间约为 **[-2.266, 0.516]个百分点**。普通softmax置信度在强域偏移下没有可靠拒识能力。

因此，当前更值得投入的主线不是“直接把LLM接到GRU后面做动态路由”，而是：

> **以分歧/OOD感知的选择性预测与校准预测集合为主贡献，以候选重排为受约束的次级模块。**

这组结果支持“可靠性与OOD校准是必须解决的问题”，但不支持“当前路由器或普通置信度拒识已经有效”，更没有证明它足以发表SCI二区以上论文。

## 2. 实验问题与预先固定的判据

实验开始前固定了三个判据，结果出来后没有降低门槛：

| 判据 | 预设通过条件 | 结果 | 判定 |
|---|---:|---:|---|
| A：候选互补性 | 并集Top-5相对GRU Top-5提升≥0.5个百分点，且95%区间下界>0 | +0.617；CI [0.299, 0.985] | 通过 |
| B：可利用的成对路由 | OOF Top-1相对GRU提升≥1.0个百分点，且95%区间下界>0 | +0.047；CI [-1.074, 1.139] | **失败** |
| C：内部选择性信号 | 约80%覆盖时准确率提升≥5.0个百分点，且95%区间下界>0 | +7.214；CI [5.858, 8.732] | 内部通过；不可作OOD证据 |

A/B/C是实验前固定的内部诊断门槛。审查后发现GRU骨干本身存在根级训练污染，因此“通过”仅表示当前工件值得继续研究，不表示通过论文级泛化验证。CTID检查是在看到内部结果后追加的探索性压力测试，不伪装成预注册Gate D。

## 3. 数据、算法和防泄漏设计

### 3.1 输入工件

- GRU逐样本Top-5及概率：`project/data/rl_v2_test_predictions_top5.csv`
- 完整LLM候选输出：`project/data/sim_test_llm_cot.csv`
- 仅用于转移先验的原训练序列：`project/data/sim_train_parent_min3.csv`

GRU和LLM均有2,107条记录。LLM行顺序与GRU不同，因此实验没有按行号拼接，而是按 `(sequence_id, prefix_len, true_label)` 一对一对齐，并逐条比较规范化后的前缀状态。结果为：2,107/2,107对齐、0个重复键、0个状态不一致。

LLM输出中的sub-technique先折叠到parent-technique标签空间，例如 `T1021.001 → T1021`；42条空候选保留为空，没有用标签补全。

### 3.2 分组和先验

- 将 `SIM_xxx_partyyy` 合并为 `SIM_xxx` 根；共有73个诊断根。
- 五折元排序器按根分组，任何同源part都不会跨越元训练折和元测试折；五折最大根重叠为0。
- 原训练CSV中与这73个诊断根重合的6,941行全部从转移先验中删除，只保留2,978行、77个不重合根；先验训练根与诊断根重叠为0。
- 置信区间以SIM根而非2,107个嵌套前缀为抽样单位，固定随机种子进行2,000次bootstrap。

这些区间是对已经生成的OOF预测和接受掩码做的**条件cluster percentile bootstrap**；每次重采样没有重新训练排序器或重新估计阈值，所以不包含完整训练/校准过程的不确定性。论文级实验应使用nested root bootstrap或多次完整重训。

### 3.3 必须披露的GRU骨干污染

`train_rl_baseline_v2.py`明确从 `sim_train_parent_min3.csv` 训练GRU。该训练文件与诊断池没有完全相同的 `sequence_id`，但按本实验采用的依赖单位 `SIM_xxx` 检查后：

- 73个诊断根中有65个也出现在GRU训练文件，重合率89.04%；
- GRU训练文件9,919行中有6,941行来自这些诊断根。

因此，元排序器和转移先验虽然根隔离，输入给它们的GRU概率却不是root-out-of-sample。内部bootstrap只能描述这些工件的稳定性，不能修复上游训练污染。正式实验必须重新按根/actor/source切分并重新训练GRU和所有支路。

### 3.4 轻量成对重排器

候选集合是GRU Top-5和LLM Top-5的去重并集。对每个候选使用推理时可获得的固定特征：

- GRU概率、GRU/LLM倒数排名、两支是否一致；
- GRU置信度与margin的候选交互项；
- 前缀长度、候选在前缀中的频率、是否等于最后一步；
- 只用严格非重合训练根计算的转移和全局频率先验。

模型是固定L2系数的NumPy线性成对排序器；每折只在其余根上训练，然后给当前根生成OOF预测。没有根据结果搜索L2、特征或判据。

### 3.5 选择性预测

选择性实验使用GRU Top-1概率。对每一元折，80%目标覆盖率的阈值只由另外四折计算，再应用到当前折；因此不是先查看当前折正确标签再挑阈值。它是“交叉拟合分位数拒识”的内部可行性验证，还不是具有限样本覆盖保证的Mondrian/weighted conformal算法；上游GRU根污染仍然存在。

### 3.6 探索性CTID外部压力测试

将10个CTID adversary-emulation组织的281条既有GRU预测合并，按组织分组做五折阈值校准。该数据与SIM训练根独立，因而比内部SIM曲线更接近真实域偏移检查。但只有10个cluster，bootstrap区间很粗；这项检查是审查后追加的探索性分析，不是预注册显著性检验。

## 4. 完整结果

| 方法/上限 | Top-1或覆盖率 |
|---|---:|
| GRU Top-1 | 54.960% |
| LLM原始候选 Top-1 | 1.851% |
| 等权倒数排名融合 Top-1 | 52.112% |
| 线性成对重排器 OOF Top-1 | 55.007% |
| GRU/LLM Top-1专家oracle | 55.909% |
| GRU Top-5覆盖 | 87.708% |
| LLM Top-5覆盖 | 9.350% |
| GRU+LLM Top-5并集oracle | 88.325% |

风险—覆盖关系也具有单调实用趋势：

| 保留比例 | GRU接受样本准确率 | 风险 |
|---:|---:|---:|
| 100% | 54.960% | 45.040% |
| 90% | 58.830% | 41.170% |
| 80% | 62.100% | 37.900% |
| 70% | 65.559% | 34.441% |
| 50% | 72.296% | 27.704% |
| 30% | 81.517% | 18.483% |

上表是完整样本上的描述性risk–coverage曲线；用于正式判据C的62.173%来自跨折校准阈值，两者口径不同，不能混用。

输出文件还包含pairwise margin的risk–coverage曲线，但不同外折的margin没有做共同尺度校准，因此该曲线只用于描述，不能直接给出部署阈值。

## 5. 这些结果具体证明了什么

### 5.1 仅在当前内部工件中被支持的命题

- 当前SIM工件中的GRU概率含有错误排序信号：低置信度区富集错误，因此“允许拒识、输出预测集合或交给分析师”值得继续研究。
- 两个分支不是完全冗余：LLM候选在13条GRU Top-5遗漏样本上补入了真标签。
- 元学习阶段按SIM根分组后信号仍存在，说明它不是由元折内同根前缀直接泄漏造成；但它仍可能来自GRU训练阶段的同根污染。

### 5.2 没有被支持的命题

- 不能说“PR-HR成对路由算法有效”。净提升只有1条，区间跨0；93次救回几乎被92次破坏完全抵消。
- 不能说“LLM专家很强”。当前原始LLM候选Top-1只有1.85%，Top-5只有9.35%；它最多证明有少量独特信息。
- 不能说“选择性预测算法具有创新性”。本轮只是验证置信度是否值得继续做；普通分位数拒识本身不是新算法。
- 不能说“普通置信度拒识能够跨域工作”。CTID上准确率没有随80%拒识改善，反而下降0.644个百分点。
- 不能把本结果直接写成最终测试结果。元排序器使用原测试池做了根分组交叉拟合；后续论文必须另设从未参与开发的外部测试集。
- 本轮没有使用BGE/MLP的完整184类semantic logits，只使用已保存的LLM Top-5候选。因此它不能替代对现有语义分支的最终审判。

## 6. 对新创新点的修正建议

### 主创新：可靠的分歧感知选择性融合

保留现有GRU与LLM/BGE语义支路，但研究目标从“无条件提高Top-1”改为：在给定覆盖率下最小化选择性风险，并在actor、source、ATT&CK版本变化时提供可校准预测集合或拒识。

必须增加的算法内容：

1. 分别校准GRU和semantic logits；
2. 门控输入显式包含两支entropy、margin、Jensen–Shannon分歧、序列稀有度和OOD分数；
3. 采用tactic/class-conditional或weighted conformal构造预测集合；
4. 当集合过大、两支冲突或发生版本/actor shift时拒识；
5. 优化目标同时考虑错误代价、拒识代价和集合大小，而不是只优化准确率。

本轮实验只提供了“内部可筛错、外部会失效”的需求证据：拒绝约20%的低置信SIM样本可以过滤大量错误，但同一规则在CTID域偏移下失败。这正说明第1至第4点必须作为一个整体实现，不能把最大softmax概率包装成可靠不确定性。

### 次创新：候选条件化的near-miss重排

当前通用LLM Top-5候选互补性太弱，不能直接作为强专家。若保留重排方向，应改为：只对GRU Top-k中的候选逐个生成结构化证据，训练candidate-conditioned listwise/pairwise损失，并以同tactic、同parent邻域或图近邻构造hard negatives。还必须做rationale交换、删除和矛盾注入，验证解释是否真正影响排序。

只有当新的候选条件化语义分支在严格根分组验证上超过本轮的1个百分点预设门槛、且在外部OOD上不退化，才能重新把重排器提升为主创新。

## 7. 能否据此判断SCI二区以上可行

当前答案是：**方向有继续做的价值，但证据不足以承诺SCI二区以上。**

积极证据是内部选择性风险改善大；否定证据是GRU根污染、CTID性能崩溃、当前LLM候选很弱且成对重排没有显著收益。要形成二区以上稿件，最低还需要：

1. 新建严格的actor/source/time/ATT&CK-version切分，并锁定一个完全未参与开发的外部测试集；
2. 实现真正的calibration/conformal模块，报告coverage、set size、selective risk、ECE、Brier和OOD结果；
3. 用完整GRU与semantic logits训练分歧门控，与固定α、learned gate、RRF、Markov、Transformer等比较；
4. 在Attack Flow和AEL等真实有序外部场景复验；
5. 对LLM rationale做反事实faithfulness消融；
6. 多随机种子、根/actor级置信区间及配对显著性检验。

## 8. 可复现文件

- 实验代码：`project/experiments/pr_hr_feasibility/pr_hr_small_experiment.py`
- 单元测试：`project/tests/test_pr_hr_small_experiment.py`
- 实施计划：`docs/superpowers/plans/2026-07-29-pr-hr-small-feasibility.md`
- 逐样本OOF结果：`project/experiments/pr_hr_feasibility/results/aligned_oof_predictions.csv`
- 指标汇总：`project/experiments/pr_hr_feasibility/results/metric_summary.csv`
- 风险—覆盖曲线：`project/experiments/pr_hr_feasibility/results/risk_coverage.csv`
- 根级bootstrap区间：`project/experiments/pr_hr_feasibility/results/bootstrap_intervals.csv`
- 五折泄漏审计：`project/experiments/pr_hr_feasibility/results/fold_audit.csv`
- 判据结果：`project/experiments/pr_hr_feasibility/results/gates.csv`
- 输入哈希、参数与审计清单：`project/experiments/pr_hr_feasibility/results/run_manifest.json`
- CTID逐样本外部结果：`project/experiments/pr_hr_feasibility/results/ctid_external_predictions.csv`
- CTID风险—覆盖曲线：`project/experiments/pr_hr_feasibility/results/ctid_external_risk_coverage.csv`
- CTID选择性汇总：`project/experiments/pr_hr_feasibility/results/ctid_external_selective_summary.csv`

复现实验命令：

```powershell
$py='C:\Users\z\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m project.experiments.pr_hr_feasibility.pr_hr_small_experiment `
  --gru 'project\data\rl_v2_test_predictions_top5.csv' `
  --llm 'project\data\sim_test_llm_cot.csv' `
  --train 'project\data\sim_train_parent_min3.csv' `
  --output-dir 'project\experiments\pr_hr_feasibility\results' `
  --folds 5 --bootstrap 2000 --seed 20260729 --l2 1.0 `
  --ctid-dir 'project\rl\all_ctid_eval'
```
