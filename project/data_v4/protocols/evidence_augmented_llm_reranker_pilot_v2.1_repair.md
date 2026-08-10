# EDLR pilot v2.1 mechanical repair

Status: frozen after the v2 mechanical gate failed, after inspection of only
the three terminally invalid raw responses, and before any target/effectiveness
metric was computed or any v2.1 payload was constructed.

This repair is limited to a directly observed output-format defect. It cannot
change candidate sets, evidence, semantic history, model family, temperature,
arm membership, effectiveness metric, or interpretation.

## 1. Frozen failure set

The immutable v2 run is `20260810T031013Z_1338c419`. It made 140 billed
completion attempts and ended with 117 `ok` rows and three terminal failures:

| development slot | arm | failure | direct cause |
|---|---|---|---|
| `ctid:long:1` | `EDLR` | `invalid_summary` | valid ranking, but summary length 195 exceeded the frozen 120-character limit |
| `ctid:medium:3` | `UNION_LLM` | `invalid_top5` | output copied `T1003` from the fixed JSON example even though it was absent from the candidate union |
| `ctid:medium:3` | `EDLR` | `invalid_top5` | the same fixed-example `T1003` defect |

No target labels or effectiveness scores were inspected when defining this
repair. The v2 outputs and gate failure remain reported and are never
overwritten.

## 2. Only permitted prompt edit

For exactly the three rows above, preserve the full v2 system prompt, history,
candidate table, evidence, and payload settings. Replace only the final output
section of the user message:

```text
### 输出格式（硬限制）
只输出一个JSON对象，且只能有两个键：
- evidence_summary：1至80个中文字符；
- reranked_next_ttps：恰好5个互不重复的ID，必须逐字复制自上面的候选表。
输出前检查数组中每个ID均在候选表内。不要输出示例ID，不要输出JSON之外的文本。
```

The fixed example IDs are removed. The 80-character instruction is stricter
than the unchanged 120-character validator, so the validation threshold is not
relaxed post hoc.

## 3. Requests and authorization

Prepare exactly three `NOT_SENT` payloads. Revalidate their hashes, candidate
sets, source/campaign/identifier/target/future exclusions, and thinking-disabled
configuration. A new authorization is required for `/models` plus three billed
completions, each with at most three retries (maximum 12 billed attempts).

## 4. Merge and gate

If a v2.1 row becomes valid, replace only the corresponding invalid row for
mechanical-gate and later descriptive evaluation. Preserve both raw versions
and record the replacement provenance. If it remains invalid after four total
v2.1 attempts, stop; do not patch, truncate, or accept an out-of-union output.

Recompute the original v2 mechanical gates without changing a threshold. Only
after all gates pass may target labels be opened for the frozen descriptive
pilot evaluation. The merged pilot remains development evidence, never formal
or prospective confirmation.
