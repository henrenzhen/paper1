# EDLR pilot v2 descriptive evaluation protocol

Status: frozen after all merged mechanical gates passed and before the merged
predictions were joined to any development target or any effectiveness metric
was computed.

This evaluation is descriptive development evidence only. The 30 rows were
selected for prompt development and cannot support a formal, confirmatory, or
publication-level effectiveness claim.

## 1. Methods and denominator

Evaluate the same 30 development rows for `B0`, `EA_TOP5`, `UNION_LLM`,
`EDLR`, and `EDLR_SHUFFLE`. The merged v2/v2.1 file must contain exactly 120
valid arm rows and its saved mechanical gate must be fully passing. Recover one
frozen B0 Top-5 per sample from the local audit field. No row may be dropped,
imputed, truncated, or repaired during evaluation.

## 2. Metrics and aggregation

For each row, use the frozen unordered future-3 unique parent-technique set and
compute binary-relevance `NDCG@5`, `Hit@5`, `Precision@5`, and `Recall@5` with
the same formulas as the formal future-3 experiment.

Aggregate in this order:

1. mean over prefix rows inside each campaign;
2. equal mean over campaigns inside each source;
3. equal mean over CTID, Attack Flow, and Stockpile.

Report every source and the source-equal value. Do not report row-micro values
as the main pilot summary. Do not bootstrap, test significance, or use phrases
such as significant, near-significant, validated, effective, or failed from
this 30-row development diagnostic.

## 3. Frozen contrasts

Report, without a pass/fail declaration:

- `EA_TOP5 - B0`: second-pass self-reranking;
- `UNION_LLM - B0`: second pass plus candidate expansion;
- `EDLR - B0`: total evidence-augmented pipeline difference;
- `EDLR - UNION_LLM`: incremental correct transition evidence;
- `EDLR - EDLR_SHUFFLE`: identity-specific evidence versus equally shaped
  rotated evidence.

For each contrast report CTID, Attack Flow, Stockpile, and source-equal metric
differences. `EA_TOP5` has the same candidate set as B0, so Hit/Precision/Recall
equality is a construction check; only its ordering can change NDCG.

An EDLR gain over B0 alone cannot be attributed to transition evidence. Such
attribution requires positive EDLR differences over both `UNION_LLM` and
`EDLR_SHUFFLE`, but this pilot can only motivate or reject a later exploratory
full run; it cannot establish the mechanism.

## 4. Outputs and boundary

Write per-sample predictions/targets/metrics, campaign aggregates, source
aggregates, frozen contrasts, a report, and a manifest covering script/input/
output hashes. Mark every artifact `development_descriptive_only` and record
that target labels were first opened only after the merged mechanical gate
passed.
