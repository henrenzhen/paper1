# Future-3 DeepSeek 30-row development pilot v1

Status: frozen before the first request to DeepSeek `/models`.

## Scope and authorization

This run is limited to the 30 frozen development rows in
`data_v4/semantic_preflight/future3_dev_prompts_v1`.  The user explicitly
authorized, in the Codex task on 2026-08-07, (1) a DeepSeek `/models` request,
(2) transmission of the 30 rows, and (3) token-billed completion requests.
The transmitted content is limited to observed parent techniques, possible
tactics, cleaned historical descriptions, and the frozen instructions.  It
must not contain source fields, campaign identifiers, audit keys, target step
identifiers, or future-step text.

This pilot is a format and quality gate only.  Its 30 rows remain excluded from
all formal outer-fold evaluation, and its predictions must not be used to
estimate method effect.  Full-corpus generation requires a separate approval.

## Frozen request configuration

- Base URL: `https://api.deepseek.com`
- Requested family: `deepseek-v4-flash`
- Actual model ID: selected deterministically from `/models`; exact match first,
  otherwise the lexicographically first ID containing the normalized family
  string.  Absence of such an ID stops the run.
- `temperature=0.0`, `max_tokens=2048`, `stream=false`
- `thinking={"type":"disabled"}` at the top level of the transmitted body
- `response_format={"type":"json_object"}`
- concurrency: 30
- retries: at most 3 after the initial attempt
- timeout: 300 seconds per completion

The API key is read only from `DEEPSEEK_API_KEY`.  It is never written to a
request artifact, manifest, stdout log, command line, or Git-tracked file.

## Transmission transform and leakage gate

The preflight format stores `thinking` under `extra_body`, matching an OpenAI
client call.  The runner performs exactly these transformations before raw
REST transmission:

1. remove local `audit_key_not_sent` entirely;
2. move `extra_body.thinking` to top-level `thinking`;
3. add the model ID returned by the deterministic selection rule and
   `stream=false`;
4. case-insensitively replace an exact source identifier or exact campaign
   identifier if it occurs naturally inside historical text.

Step 4 is required because the frozen preflight audit found two natural-text
matches: `WhisperGate` and `SolarWinds`.  The replacement token is
`[REDACTED_CAMPAIGN_ENTITY]`; no other semantic rewrite is allowed.  The runner
records the redaction count and SHA-256 of every transmitted body.

Before network access, all 30 rows must pass: preflight hash reproduction,
unique development slot and sample ID, exact observed-step provenance,
observed/target step disjointness, absence of forbidden structured keys,
absence of source/campaign/audit identifiers after redaction, and absence of
target step IDs.  Future-text isolation is checked by provenance: only observed
step IDs may be serialized.  A historical and future step may legitimately
have identical description text, so raw string collision is not treated as a
leak.

## Output schema and retry rule

Each completion must be a JSON object with non-empty strings
`stage_assessment`, `observed_capabilities`, and `likely_next_intents`, plus
`predicted_next_ttps`, an ordered list of exactly five unique IDs from the
frozen 184-label vocabulary.  The three summaries must contain no ATT&CK ID.
Any transient HTTP error, truncation, non-empty API `reasoning_content`, JSON
failure, invalid summary, or invalid Top-5 is retried within the frozen limit.
Non-429 4xx errors are not retried.

## Frozen gates

| Gate | Threshold |
|---|---:|
| HTTP-successful content parse to JSON object | >= 95% |
| API `reasoning_content` empty | 100% |
| `finish_reason=length` | <= 2% |
| three non-empty summary fields | >= 95% |
| ATT&CK ID in summary fields | 0% |
| valid five unique in-vocabulary parent IDs | >= 90% |
| pre-send leakage assertions | 100% |

The run stops after reporting these gates.  No full generation is implied.

## Cost accounting

The manifest records prompt cache-hit, prompt cache-miss, and output tokens for
every billed attempt.  The frozen price snapshot is the DeepSeek official
pricing page retrieved 2026-08-07 for DeepSeek-V4-Flash: USD 0.0028 / 1M cached
input tokens, USD 0.14 / 1M uncached input tokens, and USD 0.28 / 1M output
tokens.  Source: `https://api-docs.deepseek.com/quick_start/pricing`.

