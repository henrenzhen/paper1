# EDLR pilot v2 implementation addendum

Status: frozen after the v2 method protocol and before the payload-builder was
implemented or any v2 payload/result was produced. It resolves serialization
details only and does not change an arm, candidate set, evidence field, gate,
or interpretation rule.

## 1. Deterministic ordering and shuffle

- Development rows are ordered by `development_slot`; arms use the order
  `EA_TOP5`, `UNION_LLM`, `EDLR`, `EDLR_SHUFFLE`.
- Union candidates are lexicographically sorted by ATT&CK parent ID.
- For an ordered union `c[0:n]`, the shuffled table displayed beside `c[i]`
  receives the complete evidence/rank record originally belonging to
  `c[(i + 37) mod n]`. Identity/name/tactics and B0 rank remain those of
  `c[i]`. Because all union sizes are at most 20, this is non-identity unless
  `n` divides 37; preflight must fail if a union nevertheless yields an
  identity evidence assignment.
- Numeric probabilities/relevances are serialized with eight decimal places;
  integer counts/ranks stay integers.

## 2. Candidate serialization

`EA_TOP5` serializes one block per original rank:

```text
候选 {rank}: {id} | 名称: {name} | 可能战术: {tactics}
```

Every union arm serializes one JSON object per line, without an enclosing
array. Common fields are `candidate_id`, `name`, `possible_tactics`, and
`b0_rank`. `UNION_LLM` uses a single additional field
`sequence_evidence: "not_provided"`.

`EDLR` and `EDLR_SHUFFLE` add `a_rank`, `t_rank`, `k_rank`,
`unigram_target_count`, `unigram_relevance`, `a_smoothed_relevance`, and nested
`order1`/`order2` objects. Each available order object contains
`target_count`, `context_total`, `conditional_relevance`, and
`supporting_training_sources`; unavailable order-2 context is the literal
`"context_unavailable"`.

## 3. Target-blind construction audit

Target parent IDs may naturally occur in B0/A/T/K candidate lists, so literal
coincidence with a target parent ID is not a leakage failure. Instead:

- the builder may read target sets only from the 784 formal rows used to fit
  the two allowed outer-training sources;
- the development row's `target_parent_ids`, `target_step_ids`, and any future
  description are never passed to a scoring, ranking, union, evidence, prompt,
  or payload function;
- their hashes and exact values may appear only inside `audit_key_not_sent`;
- the transmitted-body literal gate covers sample ID, source, campaign,
  development slot, target step IDs, and file/path values;
- exact source/campaign strings accidentally present in observed descriptions
  are replaced case-insensitively by `[REDACTED_SOURCE]` and
  `[REDACTED_CAMPAIGN_ENTITY]` before hashing the payload.

## 4. Payload envelope

Every preflight body contains only `temperature`, `max_tokens`,
`response_format`, `extra_body`, and `messages`. Model selection is added only
by the authorized runner after `/models`. Frozen values are temperature `0.0`,
max tokens `1024`, JSON object response, and
`extra_body.thinking.type = "disabled"`.

The preflight JSONL stores `audit_key_not_sent`, `request_payload`, payload
SHA-256, and `network_status = "NOT_SENT"`; the runner must transmit only the
inner `request_payload` plus selected model.
