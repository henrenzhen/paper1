# Evidence-augmented direct LLM reranker: development protocol v1

Status: frozen after the negative preregistered v8 fusion results, B0/T+B0
diagnostics, and the negative candidate-level RGAF audit, but before any API
request using this reranking prompt.

This is post-result method development. Results on the already observed 784
formal rows are exploratory even if the implementation follows this document
exactly. A primary method claim requires prospective confirmation on data that
did not influence this design.

## 1. Motivation and method boundary

Direct DeepSeek ranking B0 retains a cross-source signal, whereas summary
embedding probes, fixed-weight fusion, and a shallow candidate-support gate do
not. The next test must therefore preserve the direct semantic decision and
expose transition evidence to the LLM itself, rather than compressing the LLM
output into an embedding or adding a numeric residual after generation.

For every prefix, the **Evidence-Augmented Direct LLM Reranker (EDLR)** receives:

1. the exact observed event history used by B0;
2. a bounded candidate union from B0, A, T, and K;
3. candidate-wise ranks and transition evidence computed only from the outer-
   training sources.

It returns five candidates selected only from that union. The prompt treats B0
as the semantic prior and transition counts as reliability-qualified evidence:
positive support may promote a candidate, while zero support is explicitly
described as missing evidence rather than evidence of impossibility.

## 2. Frozen outer-fold construction

The three existing source folds remain CTID, Attack Flow, and Stockpile LODO.
For a row in held-out source `s`, all statistical values are fit using rows
whose source is not `s`.

Frozen input anchors:

```text
future3_samples.csv
  af7d9a6b6358939697730c5de206c51f1b63f7c16954d5dce5a0a4006bbca724
technique_tactic_multihot.csv
  dfe0aa810207eac80a91a8f37452c475445e3f8a323139e090f7a343fd85e0b4
attack_lookup_dedup.csv
  a8368252e37ba485a8dd7429fd4544935343bd96062ea78d71befb782b5dd1b6
b0_rankings.csv
  f4b03b95db516a46db231f4fec9e505019c41cbb60b7372b52c984f1077b9216
nonsemantic_future3_lodo_v1/predictions.csv
  75b53e71e166139c1e42d29ab40a2252d8fa6bf80879ec625cc91e1d825d640c
future3_dev_prompts_v1/development_prompt_index.csv
  f8b860839d5ad0a43b339b1f1356374c29d6df3d22d341019a44c45bbcdb5dd6
future3_dev_prompts_v1/raw_semantic_inputs.jsonl
  00820b1d401ac61be04a72658e904d8c3f67911a4f663ad1807ecf341e00d876
future3/pilot/runs/20260807T073425Z_1ec8fb8d/pilot_raw_results.csv
  41bb231e1e82c53be5e4799255e444cd15bf645f9d2d61db93fbf0c090408315
```

Technique names use the unique name associated with each parent ID in the
frozen lookup; duplicate rows caused by multi-tactic membership must agree on
the name or preflight stops.

The candidate union is:

```text
C(x) = unique(B0_top5 union A_top5 union T_top5 union K_top5)
```

- fixed 184-parent-technique vocabulary;
- minimum size 5 because B0 always contributes five unique labels;
- maximum size 20;
- prompt presentation order is lexicographic ATT&CK parent ID, not model rank;
- no candidate may be added from the target or future text;
- B0, A, T, and K rankings must reproduce their committed Top-5 exactly before
  a request payload is created.

For candidate `c`, expose only:

- parent technique ID and frozen ATT&CK name;
- all mapped tactics from the frozen multi-hot map;
- B0/A/T/K rank in 1--5, otherwise `not_top5`;
- training unigram count and the frozen A-model smoothed relevance;
- `(last_parent,c)` count, conditional smoothed relevance, and number of supporting
  outer-training sources;
- `(last2_parent,last_parent,c)` count, conditional smoothed relevance, and number of
  supporting outer-training sources; use explicit `context_unavailable` when
  the observed prefix has length one;
- total order-1 and order-2 context support, so that small conditional counts
  are not presented without their denominator.

Counts and probabilities are calculated from unique future-3 target sets in
the same frozen relevance model as A. Repeated appearance of a target inside
one future set contributes at most once. Training source identity is collapsed
to a support count in `{0,1,2}`; source names are never transmitted.

Forbidden payload fields include held-out source, campaign, actor, sample ID,
file name, targets, `transition_visibility`, target-label visibility, future
descriptions, any target-derived metric, and any statistic fit outside the
outer-training sources.

## 3. Frozen prompts

System prompt:

```text
你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻已经观察到的攻击事件、一个候选技术表，
以及只由训练语料计算的序列统计。任务是在候选表内选出未来三次攻击动作范围内最值得检查的
5 个 ATT&CK Parent Technique，并按可能性排序。

判断规则：
1. 历史事件的行为描述是语义主证据；B0排名是基于这些历史得到的语义先验。
2. 序列统计只表示训练语料中的经验支持。出现次数和跨训练源支持较高时，可提升与当前历史
   语义一致的候选。
3. 统计为0表示训练语料没有观察到，不表示该技术不可能。不得仅因0次观察删除语义上合理的
   B0候选。
4. 不得使用或猜测来源、campaign、未来事件；这些信息不会提供。
5. 最终数组必须恰好包含5个互不重复的候选表内父技术ID，不得新增候选。
6. 输出必须是JSON，不要输出JSON之外的文本，也不要输出内部思维链。
```

User prompt template:

```text
### 已观察攻击事件（按时间顺序）
{serialized_observed_events}

### 候选技术与训练语料证据（按技术ID排序）
{serialized_candidate_table}

### 输出要求
请综合历史语义、B0语义先验和训练语料支持，在候选表内重排并输出：
{
  "evidence_summary":"不超过120个中文字符，只概括语义与统计是否一致，不写内部思维链",
  "reranked_next_ttps":["T1059","T1078","T1021","T1003","T1105"]
}
```

The response schema requires exactly these two keys, a nonempty
`evidence_summary` of at most 120 Chinese characters, and exactly five unique
`^T\d{4}$` IDs. Every output ID must occur in the transmitted candidate table.

## 4. Controls that isolate the contribution

The eventual development comparison requires three second-pass arms with the
same candidate union and observed history:

1. `EDLR`: the real evidence table above;
2. `Union-LLM`: identical candidates, names, tactics, and B0 rank, but A/T/K
   ranks and all transition counts/relevances are replaced by the literal
   `not_provided`;
3. `EDLR-Shuffle`: the complete A/T/K ranks and transition evidence records are
   rotated by `37 mod |C(x)|` positions within the lexicographically ordered
   candidate table, while IDs, names, tactics, B0 ranks, and observed history
   stay fixed.

`EDLR > B0` alone is insufficient because a second LLM pass and a larger
candidate union may explain the change. Evidence attribution requires EDLR to
exceed both Union-LLM and EDLR-Shuffle. All arms use the same model, prompt
structure, temperature, retry policy, and candidate order. They receive no
arm-specific prompt tuning after results are seen.

## 5. Thirty-row mechanical pilot

The first network stage uses only the 30 development rows already excluded
from formal evaluation. It runs EDLR only; controls are not needed until the
payload is mechanically valid.

Frozen pilot gates after at most three retries:

- 30/30 request payloads pass temporal and outer-training-only assertions;
- 0 payloads contain source/campaign/sample identifier values, targets, or
  future text;
- B0 Top-5 exactly reproduces the frozen 30-row pilot response; A/T/K are
  deterministically recomputed from the two formal outer-training sources
  with the already selected outer-fold tactic weight, then frozen in the
  preflight manifest before networking;
- candidate-union size is within 5--20 on 30/30 rows;
- JSON parsing success at least 95%;
- exactly five unique in-union parent IDs at least 95%;
- empty `evidence_summary` at most 3%;
- no response contains an ID outside the fixed 184-class vocabulary;
- output ranking differs from B0 on at least 10% but no more than 90% of valid
  rows; this is a non-degeneracy check, not an effectiveness threshold.

Pilot targets may be used only after generation for a clearly labelled
development diagnostic. No pilot metric becomes a paper result or a threshold
for choosing prompt variants. If a mechanical gate fails, only the failure's
direct schema/API cause may be repaired and the protocol version must change.

## 6. API configuration and authorization boundary

- `GET https://api.deepseek.com/models` before exact/family model selection;
- actual `deepseek-v4-flash` family model returned by the endpoint;
- temperature 0, JSON output, thinking explicitly disabled;
- max tokens 1024, stream false, concurrency initially 30;
- initial attempt plus at most three retries;
- API key only from `DEEPSEEK_API_KEY`, never persisted;
- raw request, raw response, safe request ID, attempts, usage, token cost,
  prompt hashes, input hashes, script hash, and model response from `/models`
  are stored in a unique non-overwriting run directory.

No earlier authorization covers this prompt because it transmits an additional
candidate/evidence table and creates new token-billed requests. Before the
30-row pilot, the user must explicitly authorize sending:

- 30 development rows;
- observed parent techniques, tactics, and cleaned historical descriptions;
- B0/A/T/K candidate IDs and training-only rank/count/probability evidence;
- no source, campaign, identifier, target, or future text;
- the actual available `deepseek-v4-flash`, thinking disabled, with token-
  billed requests.

Running controls or any remaining rows requires a separate authorization that
states the exact number of rows and requests.

## 7. Evaluation boundary

On development data report NDCG@5, Recall@5, Precision@5, and Hit@5 only as
exploratory diagnostics. If later run on the already observed 784 rows, use the
same source-equal campaign-macro aggregation and paired 2,000-campaign
bootstrap, but label every result post-result exploratory.

The evidence mechanism receives exploratory support only if:

1. EDLR exceeds B0 in CTID and Attack Flow and source-equal overall;
2. EDLR exceeds Union-LLM and EDLR-Shuffle source-equal overall;
3. the direction is not supported solely by Stockpile;
4. no single CTID campaign deletion reverses every real-source advantage.

Even when all four hold on the existing rows, the paper must say that the
method was developed after those rows had influenced the research decisions.
Confirmatory evidence requires a newly constructed or independently held
ordered procedure-text source with a protocol frozen before its outcomes are
observed.
