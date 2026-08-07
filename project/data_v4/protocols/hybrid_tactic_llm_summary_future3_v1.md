# HM + tactic + LLM-summary future-3 completion v1

Status: frozen after the formal HM/HM+S outer results were observed, but before
any HM+T or HM+ST outer result was computed.  This card completes the methods
already preregistered in v8; it cannot alter the negative HM+S primary result.

## Branches

The three independently standardized 184-dimensional branches are:

- `H`: clipped-logit HM probabilities from the committed `hm_future3_v1`
  cache;
- `S`: logits from the frozen BGE-M3 summary probe defined in
  `llm_summary_semantic_future3_v1.md`;
- `T`: clipped-logit candidate tactic relevance.  A 14-output relevance model
  uses the same order-2/order-1/unigram counts, smoothing, and complete
  multi-hot target tactic unions as the frozen nonsemantic `T` baseline.  A
  candidate's score is the mean over all of its mapped tactics.

`T` here means the raw tactic branch, not the already A+tactic-fused baseline
ranking.  This follows the three-branch equation frozen in v7/v8 and prevents
double-counting another sequence backbone inside HM+T.

## Selection

For each outer held-out source, its two training sources alternate as inner
train and inner validation.  All selection maximizes campaign-macro NDCG@5
equally averaged over both validation sources and five HM/probe seeds.

```text
HM+T  = (1-lambda_T) * std(H) + lambda_T * std(T)
lambda_T in {0.0,0.1,...,1.0}

HM+ST = w_H*std(H) + w_S*std(S) + w_T*std(T)
w_H,w_S,w_T in {0.0,0.1,...,1.0}; w_H+w_S+w_T=1
```

The S probe uses the frozen epoch grid 20/40/60/80/100 and seeds 42--46.
Exact score ties follow v8: smaller semantic weight, then smaller tactic
weight, then larger HM weight.  If weights are also identical, fewer epochs
is the final deterministic tie-break, inherited from the frozen HM+S method
card.  No held-out-source metric is used for selection.

## Reproduction gates and interpretation

- `w_T=0` at each fold's already selected HM+S epoch/weight must reproduce the
  committed HM+S Top-20 for every outer seed/sample row.
- `lambda_T=0` must reproduce the committed HM Top-20 for every outer
  seed/sample row.
- All 784 intention-to-treat rows remain in every method.

The v8 complementarity rule is unchanged: HM+ST must exceed HM+S in at least
two sources, exceed HM+T in at least two sources, and exceed both in the
source-equal overall result.  Passing or failing this rule does not change the
already determined HM+S primary conclusion.  Uncertainty uses paired
2,000-replicate campaign bootstrap with seed 20260807.
