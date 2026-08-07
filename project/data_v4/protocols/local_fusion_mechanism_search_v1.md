# Zero-cost local fusion mechanism search v1

Status: frozen on 2026-08-07 before any mechanism-specific oracle or outer result in this search was computed.

This is post-result method development on the already contaminated 784-row discovery set. It performs no network request, imports no remote model, and incurs no token or API cost. A positive discovery result requires prospective confirmation and cannot replace the negative v8 conclusion.

## 1. Immutable data and evaluation

- Task: future-three unique parent-technique set, fixed 184 labels, Top-5.
- Folds: CTID / Attack Flow / Stockpile source LODO.
- Primary metric: source-equal campaign-macro NDCG@5.
- Uncertainty: paired 2,000-replicate campaign bootstrap, seed 20260807.
- Denominators: 784 rows and 10/35/27 campaigns; Stockpile is semi-synthetic.
- B0 is the committed DeepSeek Top-5. A/A0/K/T are frozen local outputs.

Frozen input SHA-256 values:

```text
future3_samples.csv                         af7d9a6b6358939697730c5de206c51f1b63f7c16954d5dce5a0a4006bbca724
b0_rankings.csv                             f4b03b95db516a46db231f4fec9e505019c41cbb60b7372b52c984f1077b9216
nonsemantic_future3_lodo_v1/predictions.csv 75b53e71e166139c1e42d29ab40a2252d8fa6bf80879ec625cc91e1d825d640c
rgaf_candidate_gate.../predictions.csv      be6924ab6a36d0fa1333d8903e6c38320a893d803172d8fad764583ccacb8859
rl_label_vocab.csv                          9a4f0c09b86969ef33dd4532ec315e6e00d542d2483c6f5b9b0e9709b9b35738
```

No method may read test targets while constructing features, selecting a hyperparameter, fitting, routing, or ranking. Targets enter only training loss, oracle calculation, and final metrics. `transition_visibility` and `target_label_visibility` are forbidden features.

## 2. Shared legal features and controls

All test-row transition statistics are fit on the other two sources. Training examples use campaign-LOO transition statistics. For history `h` and candidate `c`:

```text
n1/n2       context counts for h[-1] / h[-2:]
k1/k2       context rows whose unique future-three target set contains c
p0(c)       (global target count(c)+0.5)/(training rows+1)
p1(c|h)     (k1+0.1*p0)/(n1+0.1), when n1>0
p2(c|h)     (k2+0.1*p0)/(n2+0.1), when n2>0
lr1/lr2     clip(log((p_context+1e-9)/(p0+1e-9)),-4,4)
```

Other legal features are support-source counts; prefix length; A/T/K/B0 ranks; Top-5 Jaccard/RBO agreements; A entropy; union size; and fractions of B0 candidates with support. Source, campaign, actor, sample/file IDs, future text, targets, and future tactics are never features.

Every implemented mechanism must report:

1. exact B0;
2. a deterministic within-source campaign-rotation control for transition evidence, with the same capacity;
3. a no-prior control replacing likelihood ratios by raw conditional evidence without division by `p0`;
4. an equal-capacity control replacing transition content by SHA-256-derived pseudo-features.

## 3. Global oracle and pass/fail rules

At most five mechanisms are explored. Each first computes its frozen action-space oracle. It is abandoned before fitting if oracle source-equal campaign-macro NDCG@5 is below 0.2328 (B0 0.2028 plus 0.03).

An implemented mechanism is discovery-positive only if all hold:

- main minus B0 95% campaign-bootstrap CI is entirely above zero;
- CTID and Attack Flow point deltas are both positive;
- main exceeds its campaign-permutation control;
- deleting any one CTID campaign does not reverse the overall sign.

Method-specific dead conditions below are additional and immutable.

## 4. F1: reliability-ratio Top-5 reranker (RR5)

**Hypothesis.** The low-risk space is inside B0's existing Top-5. Reordering those five candidates avoids the 5.5% precision of union-only additions.

**Difference.** RR5 never scores the other 179 labels and optimizes pairwise order inside B0. RGAF scored all 184 with a residual; T+B0 mixed whole score vectors.

**Model.** Pairwise logistic ranker. Candidate features: B0-rank one-hot; `log1p(n1,n2,k1,k2)`; support-source counts; `lr1/lr2`; A/T/K reciprocal ranks; A/T/K Top-5 vote count; A entropy; prefix length. Interactions are limited to B0-rank by `lr1`, `lr2`, and vote count. Full-batch Adam, 80 epochs, learning rate 0.03. L2 grid `{0.001,0.01,0.1,1.0}` by inner source LODO. Exact ties preserve B0 order.

**Oracle/expectation/death.** Oracle perfectly permutes B0 Top-5. Expected delta +0.01 to +0.03. Abandon if oracle delta is below +0.03; after fitting any failure of the global rule is final.

## 5. F2: conservative one-slot consensus replacement (C1R)

**Hypothesis.** Expert consensus plus cross-source support can select rare useful union additions while a one-slot budget caps downside.

**Difference.** C1R is a discrete rank-5-only replacement, not a 184-way residual, global mixture, or wholesale transition ranking.

**Action.** Outside candidates must appear in the Top-10 of at least two of A/T/K. Their score combines reciprocal-rank votes, `lr1/lr2`, support-source counts, and log counts. Only B0 rank 5 may be replaced; ranks 1--4 stay fixed.

Joint inner-LODO grid:

```text
minimum supporting experts {2,3}
minimum support-source count {0,1,2}
margin {0.0,0.25,0.5,1.0}
order2 weight {0.0,0.5,1.0}
```

Ties prefer more experts, more source support, larger margin, then smaller order2 weight.

**Oracle/expectation/death.** Oracle keeps B0 or makes the best target-aware rank-5 replacement among broad-eligible candidates (`>=2` experts, support-source `>=0`). Expected delta +0.01 to +0.04. Abandon below +0.03. Kill if fewer than 1% or more than 90% of outer rows are replaced.

## 6. F3: observable selective expert router (OSER)

**Hypothesis.** Complete expert rankings are useful in different observable regimes. Cross-fitted disagreement and support can predict competence without target-conditioned visibility.

**Difference.** OSER selects one frozen ranking from `{B0,A,T,K}`. It neither mixes scores nor gates candidate residuals and is multivariate unlike context-seen switches.

**Model.** Four-way multinomial logistic competence model. Training label is the sample-best expert; ties prefer B0, T, K, A. Features: `log1p(n1/n2)`; A entropy; prefix length; union size; B0-vs-A/T/K Top-5 Jaccard and RBO; A/T/K mutual agreement; fractions of B0 candidates with `k1>0`, `k2>0`, or two-source support; mean/max `lr1/lr2` over B0; maximum outside-candidate vote count. Training features are campaign-LOO. Full-batch Adam, 100 epochs, learning rate 0.03; L2 `{0.001,0.01,0.1,1.0}` by inner source LODO.

**Oracle/expectation/death.** Oracle chooses the best of B0/A/T/K per row. Expected delta +0.02 to +0.05. Abandon below +0.03. Kill if OSER selects B0 on over 99% of rows or either real source has no non-B0 selections.

## 7. F4: lower-confidence-bound rank surgery (LCB-RS)

**Hypothesis.** An unreliable transition teacher should be used only when training campaigns provide positive lower-bound evidence for a bounded edit. Explicit abstention is safer than a continuous gate.

**Difference.** LCB-RS uses a finite edit library and defaults exactly to B0 unless a campaign-level lower confidence bound is positive. It learns no neural/numeric fusion function.

Frozen edits:

```text
identity
reorder B0 Top-5 by A / T / K order
replace B0 rank 5 by highest outside A / T / K candidate
```

Cells are the Cartesian bins of: `n2>0`; any B0 candidate with two-source order1 support; B0/expert Jaccard `<0.25`, `0.25--0.5`, `>0.5`; and proposed outside-candidate `lr1<=0` or `>0`. Training deltas are averaged inside campaign. Apply only if at least three campaigns populate the cell and `mean_delta - gamma*sd/sqrt(campaigns)>0`. Select `gamma {0,0.5,1.0,1.64}` by inner source LODO. Multiple passing edits choose the largest lower bound; ties prefer identity, reorder A/T/K, then replace A/T/K.

**Oracle/expectation/death.** Oracle chooses the best frozen edit per row. Expected delta +0.01 to +0.03. Abandon below +0.03. Kill if fewer than three CTID and three Attack Flow campaigns contain an applied edit.

## 8. F5: local-campaign transition reranking (LCTR5)

**Hypothesis.** Pooled transitions average incompatible campaign regimes. A local teacher from training campaigns resembling the observed prefix may be more reliable; restricting it to B0 Top-5 controls noise.

**Difference.** LCTR5 changes the per-instance transition estimator. It is not domain-equal pooling, global A, or a learned residual gate; B0 candidate membership is unchanged.

Represent each training campaign by its observed parent-technique and tactic sets. Similarity is `0.7*Jaccard(parent IDs)+0.3*Jaccard(tactics)`. Source and text are excluded. Select k nearest campaigns with stable similarity/hash ties and build local `p0/p1/p2`. Reorder B0 Top-5 by `-log(B0_rank)+lambda1*local_lr1+lambda2*local_lr2`.

Joint inner-LODO grid: `k {3,5,10,all}`, `lambda1 {0,0.25,0.5,1.0}`, `lambda2 {0,0.25,0.5,1.0}`. Ties prefer larger k then smaller weights. Campaign-LOO applies to training/validation rows.

**Oracle/expectation/death.** After constructing local scores without test labels, the oracle chooses the best frozen grid action per row. Expected delta +0.01 to +0.03. Abandon below +0.03. Kill if inner selection chooses identity in at least two outer folds.

## 9. Mechanism sources

These sources motivate mechanisms, not transferable performance claims:

- Jacobs et al., adaptive mixtures of local experts, DOI `10.1162/neco.1991.3.1.79`: competence routing for OSER.
- Wolpert, stacked generalization, DOI `10.1016/S0893-6080(05)80023-1`: cross-fitted meta-learning.
- Geifman and El-Yaniv, selective classification, arXiv `1705.08500`: abstain-to-B0 behavior in LCB-RS.
- Sagawa et al., group robustness, arXiv `1911.08731`: requiring gains beyond the semi-synthetic group.
- Cao et al., ListNet, DOI `10.1145/1273496.1273513`: ranking a fixed candidate list rather than 184-way classification.
- Rendle et al., Bayesian Personalized Ranking, arXiv `1205.2618`: pairwise ranking under sparse relevance.

No external performance number is imported. Identifier verification is deferred because this run forbids external access.

## 10. API-dependent candidates not executed

- EDLR: let DeepSeek read the B0/A/T/K union plus training-only evidence.
- Obtain full LLM candidate log probabilities/confidence instead of reciprocal-rank proxy scores.
- LLM self-consistency or multi-prompt uncertainty for selective fusion.
- LLM extraction for a new prospective ordered-procedure source.

These are listed only. No request may be made by this search.
