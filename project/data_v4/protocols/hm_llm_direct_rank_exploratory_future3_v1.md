# Exploratory HM + direct LLM rank fusion v1

Status: frozen after the preregistered B0 fold-level result was observed and
before any HM+B0 fused outer result was computed.

This analysis is explicitly post-result exploratory.  It cannot replace the
negative preregistered HM+S conclusion, and any positive fusion claim requires
prospective confirmation on a fresh source such as Unit42.

## Motivation and fixed score

The preregistered diagnostic B0 showed cross-source signal while the
summary-embedding probe did not.  No LLM probabilities are available.  Convert
the returned five-item order to a 184-dimensional score without labels:

```text
score_B0(c) = 1/rank(c), rank in {1,2,3,4,5}
score_B0(c) = 0, otherwise
```

No candidate outside the returned list is inferred or filled from ground
truth.  Ties outside the Top-5 follow frozen vocabulary order.

## Fusion and selection

For each HM seed/sample:

```text
z = (1-lambda) * standardize(logit(score_HM))
  + lambda     * standardize(score_B0)
lambda in {0.0,0.1,...,1.0}
```

Each outer fold uses its two training sources as alternating inner-validation
sources.  Lambda maximizes campaign-macro NDCG@5 averaged equally across both
sources and all five frozen HM seeds; ties prefer smaller lambda.  The held-out
source is never used for selection.  Lambda 0 must reproduce HM Top-20 and
lambda 1 must reproduce the exact LLM Top-5.

## Interpretation

- HM+B0 > B0 with selected lambda below 1 on prospectively validated data
  would support complementary transition and semantic signals.
- Selected lambda 1 or HM+B0 <= B0 means transition statistics add no evidence
  beyond direct LLM ranking.
- B0 itself remains a diagnostic/recommendation baseline, not proof of fusion.
- All comparisons retain 784 rows, campaign-macro aggregation, source-equal
  overall reporting, and paired 2,000-replicate campaign bootstrap with seed
  20260807.

