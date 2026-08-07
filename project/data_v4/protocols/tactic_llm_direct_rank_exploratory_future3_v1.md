# T + direct LLM rank fusion exploratory future-3 v1

Status: frozen after all preregistered HM+S results and the exploratory HM+B0
result were observed, but before any T+B0 outer-fold result was computed.

This is explicitly post-result exploratory analysis.  It cannot replace the
negative preregistered HM+S conclusion.  A positive result requires prospective
confirmation on a fresh source before it can support a primary method claim.

## Frozen branches

`T` is rebuilt exactly from the nonsemantic future-3 method card:

1. the order-2/order-1/unigram `A` relevance branch;
2. the 14-dimensional multi-hot tactic relevance branch;
3. candidate tactic relevance is the mean over all mapped tactics;
4. the two clipped-logit branches are sample-standardized and mixed with
   tactic weight `lambda_T`.

The direct DeepSeek Top-5 uses the same fixed score as the earlier HM+B0
exploration:

```text
score_B0(c) = 1/rank(c), rank in {1,2,3,4,5}
score_B0(c) = 0, otherwise
```

No probability is invented for candidates outside the returned list.

## Joint inner-LODO selection

For each outer held-out source, the two outer-training sources alternate as
inner train and inner validation.  Jointly search:

```text
lambda_T  in {0.0,0.1,...,1.0}
lambda_B0 in {0.0,0.1,...,1.0}

z_TB0 = (1-lambda_B0) * standardize(z_T)
      + lambda_B0     * standardize(score_B0)
```

The pair maximizes the source-equal mean of the two inner-validation
campaign-macro NDCG@5 values.  Exact ties prefer smaller `lambda_B0`, then
smaller `lambda_T`, avoiding unsupported complexity.  The held-out source is
never used for either weight.

Joint selection is required: freezing `lambda_T` after using both inner
validation sources and then tuning `lambda_B0` on the same rows would hide a
two-stage reuse of validation labels.

## Reproduction, statistics, and interpretation

- Rebuilding T with its already frozen outer-fold `lambda_T` must reproduce
  all 784 committed T Top-20 rankings.
- Setting `lambda_B0=1` must reproduce all 784 exact DeepSeek Top-5 rankings.
- All methods retain the same 784 rows and 10/35/27 campaign denominators.
- Uncertainty is paired 2,000-replicate campaign bootstrap with seed 20260807,
  and the overall value gives the three sources equal weight.

`T+B0 > B0` would only show exploratory evidence that soft tactic/ID relevance
can complement the direct LLM ranking.  `T+B0 <= B0`, or an interval crossing
zero, means no incremental benefit is detected under this fusion rule.  It
does not prove that all possible nonsemantic information is useless.
