# Future-3 DeepSeek formal generation v1

Status: frozen before any of the 784 formal requests were sent.

## Preconditions

The 30-row development pilot `20260807T073425Z_1ec8fb8d` passed every frozen
mechanical gate with 30/30 first-attempt successes.  The formal request payloads
are frozen in `data_v4/semantic_preflight/future3_full_prompts_v1` and reproduce
the formal denominator exactly:

| Source | Rows | Campaigns |
|---|---:|---:|
| CTID | 263 | 10 |
| Attack Flow | 412 | 35 |
| Stockpile | 109 | 27 |
| Total | 784 | 72 |

The development rows are not regenerated and remain excluded from formal
evaluation.  Before networking, the runner must reproduce the preflight hashes
and all 784 temporal, provenance, structured-key, source/campaign literal, and
target-step-ID leakage assertions.

## Separate authorization boundary

The 30-row authorization does not authorize this run.  Execution requires a
new explicit user confirmation covering all of the following:

- 784 formal samples;
- observed parent techniques, possible tactics, cleaned historical
  descriptions, and the frozen instructions;
- exclusion of source, campaign, audit identifiers, future labels, and future
  descriptions;
- the actually available `deepseek-v4-flash` model with thinking explicitly
  disabled;
- token-billed DeepSeek requests.

The command additionally requires `--authorized-full-generation`.  This flag
records that the external authorization was obtained; it is not itself a
substitute for authorization.

## Frozen execution

- `GET https://api.deepseek.com/models`, then deterministic exact/family model
  selection inherited from the successful pilot;
- temperature 0, max tokens 2048, JSON object, stream false;
- top-level `thinking={"type":"disabled"}`;
- concurrency 30, timeout 300 seconds;
- at most three retries after the initial attempt;
- API key only from `DEEPSEEK_API_KEY` and never persisted;
- same response schema, validation, retry reasons, safe response headers,
  attempt-level token accounting, and official price snapshot as the pilot.

The expected no-retry cost inferred from the pilot is USD 0.126318; the
mechanical four-attempt ceiling is USD 0.505272.  These are estimates, not a
billing cap.  Actual cost is calculated from every returned usage record.

## Outputs and stop point

A unique directory under
`data_v4/external_reasoning/future3/full/runs/{run_id}` stores `/models`, every
transmitted request body and raw response, parsed per-row CSV, stdout, quality
gates, token/cost totals, hashes, and a generation manifest.  The unprocessed
run directory is committed before any embedding or training.

Generation is not evidence of method effectiveness.  After committing raw
responses, the next distinct stage is frozen S/P encoding and HM+S/HM+P/HM+ST
inner-LODO and outer LODO evaluation.

