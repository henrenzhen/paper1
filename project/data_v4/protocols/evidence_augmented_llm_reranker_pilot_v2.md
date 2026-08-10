# Evidence-augmented LLM reranker pilot v2

Status: frozen before constructing any v2 payload and before any API request
using these prompts. This document extends, and does not silently replace,
`evidence_augmented_llm_reranker_development_v1.md`.

All outcomes on the 30 development rows are mechanical/development evidence.
Any later outcome on the already observed 784 rows remains post-result
exploratory. A primary method claim requires prospective confirmation on data
that did not influence this design.

## 1. Question and arms

The pilot separates four explanations that were conflated by a proposed
second LLM pass:

| arm | candidates | transition evidence | question |
|---|---|---|---|
| `EA_TOP5` | frozen B0 Top-5 only | none | can the same LLM improve its own ordering with a second pass? |
| `UNION_LLM` | unique B0/A/T/K Top-5 union | none | is any change caused only by a second pass and candidate expansion? |
| `EDLR` | the same union | correct outer-training evidence | does sequence evidence help the direct semantic decision? |
| `EDLR_SHUFFLE` | the same union | deterministically rotated evidence | does correct evidence outperform equally shaped meaningless evidence? |

The existing B0 ranking is not regenerated. It is the frozen first-pass
reference. The pilot therefore prepares exactly 120 new completions: 30 rows x
4 arms. A `/models` lookup is a separate non-completion request.

`EA_TOP5 > B0` is evidence only for inference-time self-reranking. It is not
evidence for semantic--sequence fusion. Evidence attribution to transition
statistics requires `EDLR > UNION_LLM` and `EDLR > EDLR_SHUFFLE`.

## 2. Frozen rows and training boundary

Use the same 30 development rows already excluded from the 784-row formal
table: 10 CTID, 10 Attack Flow, and 10 Stockpile rows. For a development row
from source `s`, every count, relevance, rank A/T/K, and support-source count is
fit only on the 784 formal rows whose source is not `s`. Development rows never
enter a fitted statistic.

Input anchors are those frozen in v1. Preflight must verify their SHA-256
values against v1 before emitting a payload.

The B0 Top-5 for each development row must be recovered from the committed raw
pilot output and must contain exactly five unique IDs in the frozen 184-label
vocabulary. A/T/K are recomputed mechanically with the already frozen outer
fold tactic weights: CTID `0.0`, Attack Flow `0.0`, Stockpile `0.1`.

## 3. Candidate table and evidence

`EA_TOP5` exposes exactly the five B0 candidates. The other arms expose:

```text
C(x) = unique(B0_top5 union A_top5 union T_top5 union K_top5)
```

The union is sorted lexicographically by parent technique ID before being put
in a prompt and must contain 5--20 candidates.

For every candidate all union arms expose:

- parent technique ID, frozen ATT&CK parent name, and all frozen tactics;
- B0 rank in 1--5 or `not_top5`.

`UNION_LLM` replaces A/T/K ranks and every numeric evidence value with the
literal `not_provided`.

`EDLR` additionally exposes A/T/K rank in 1--5 or `not_top5`; unigram count and
smoothed A relevance; order-1 target count, conditional smoothed relevance,
supporting training-source count, and total context count; and the same
order-2 fields, with `context_unavailable` for prefixes shorter than two.

`EDLR_SHUFFLE` rotates each candidate's complete A/T/K and transition-evidence
record by `37 mod |C(x)|` over the lexicographically ordered candidate list.
Candidate ID/name/tactics, B0 rank, and observed history stay fixed. The
rotation is computed before networking and saved in the preflight artifact.

Zero support means missing empirical evidence, never impossibility.

## 4. Frozen prompts

### 4.1 `EA_TOP5`

System prompt:

```text
你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻的已观察攻击事件，以及同一模型首次分析
给出的5个候选。请重新核对历史语义，只能重新排列这5个候选，输出未来三次攻击动作范围内最
值得检查的5个 ATT&CK Parent Technique。

限制：
1. 历史描述是唯一新证据；候选原始排名只是先验，不保证正确。
2. 不得新增、删除或重复候选。
3. 不得使用或猜测来源、campaign、未来事件或真实答案。
4. 输出必须是JSON，不输出内部思维链或JSON之外的文本。
```

User template:

```text
### 已观察攻击事件（按时间顺序）
{serialized_observed_events}

### 首次候选（按原始顺序）
{serialized_b0_candidates}

### 输出
{"evidence_summary":"不超过120个中文字符","reranked_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}
```

### 4.2 union arms

System prompt:

```text
你是一名 APT 威胁狩猎分析师。你将看到截至当前时刻的已观察攻击事件、一个候选技术表，以及
可能提供的训练语料序列统计。任务是在候选表内选出未来三次攻击动作范围内最值得检查的5个
ATT&CK Parent Technique，并按可能性排序。

判断规则：
1. 历史事件的行为描述是语义主证据；B0排名是首次语义分析形成的先验。
2. 仅当候选表提供了序列统计时才使用它。出现次数、上下文分母和跨训练源支持必须结合判断；
   0次观察只表示训练语料未见，不表示不可能。
3. 不得使用或猜测来源、campaign、未来事件或真实答案。
4. 最终数组必须恰好包含5个互不重复的候选表内父技术ID，不得新增候选。
5. 输出必须是JSON，不输出内部思维链或JSON之外的文本。
```

User template:

```text
### 已观察攻击事件（按时间顺序）
{serialized_observed_events}

### 候选技术表（按技术ID排序）
{serialized_candidate_table}

### 输出
{"evidence_summary":"不超过120个中文字符","reranked_next_ttps":["T1059","T1078","T1021","T1003","T1105"]}
```

The JSON schema for all arms has exactly two keys: a nonempty
`evidence_summary` of at most 120 Chinese characters and five unique
`^T\d{4}$` IDs. Every output ID must occur in the transmitted candidate set.

## 5. Payload exclusions and mechanical gates

No request may contain source, campaign, actor, sample/development identifier,
file/path, target labels or steps, future descriptions, target-conditioned
visibility, or a metric computed from the target. Audit keys are stored outside
`request_payload`.

Before authorization, the offline preflight must establish:

- exactly 30 development rows and 120 `NOT_SENT` payloads;
- 10 rows per source and exactly four arms per row;
- B0 is reproduced on 30/30 rows and all candidates belong to the frozen 184;
- each union is 5--20 unique candidates and is identical across its three arms;
- A/T/K/evidence use only the two allowed formal training sources;
- the shuffle is a non-identity rotation when union size is greater than one;
- no forbidden key or exact forbidden literal occurs in any transmitted body;
- prompt, input, script, and output hashes are recorded.

After at most three retries per arm, the online mechanical gates are:

- JSON parsing success at least 95% per arm;
- exactly five unique in-candidate IDs at least 95% per arm;
- empty summary at most 3% per arm;
- reasoning/internal-thinking content length is zero on all successful rows;
- output differs from the arm's input B0 order on 10--90% of valid `EA_TOP5`
  rows (non-degeneracy only).

Pilot effectiveness metrics are descriptive only. No prompt, threshold, arm,
or row subset may be selected from their labels.

## 6. API boundary

- call `GET https://api.deepseek.com/models` before model selection;
- select the actually available `deepseek-v4-flash` family model;
- temperature 0, JSON output, thinking explicitly disabled;
- maximum 1024 output tokens, stream false, concurrency initially 30;
- one initial attempt plus at most three retries;
- key only from `DEEPSEEK_API_KEY`, never persisted;
- unique non-overwriting run directory with raw bodies/responses, usage, cost,
  request IDs, attempts, prompt/input/script hashes, and `/models` response.

The prior future-3 authorization does not cover these new prompts or the added
candidate/evidence table. No API request may occur until the user explicitly
authorizes the `/models` call and all 120 billed completions, including the
fields transmitted and the exclusion of source/campaign/targets/future text.

## 7. Interpretation

The 30-row pilot decides only whether the machinery is safe and nondegenerate.
If a later exploratory 784-row run is authorized, the frozen hierarchy is:

1. self-reranking: `EA_TOP5 - B0`;
2. candidate expansion/second pass: `UNION_LLM - B0`;
3. evidence contribution: `EDLR - UNION_LLM`;
4. evidence identity: `EDLR - EDLR_SHUFFLE`.

Only conditions 3 and 4 address semantic--sequence fusion. All results retain
the development/post-result label until a new prospective source is evaluated.
