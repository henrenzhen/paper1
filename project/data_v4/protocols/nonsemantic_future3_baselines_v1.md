# Non-semantic future-3 baselines v1

Status: frozen before the first outer-fold baseline run.

This method card operationalizes v8/v8.1 baselines `A0`, `CO`, `A`, `K`, and
`T`. All fitting uses the 784 non-development rows in
`future3_samples.csv`; the 30 development rows are excluded from outer and
inner folds.

## 1. Ranking and ties

Every method ranks the fixed 184-label vocabulary. Ties are broken by the
frozen `label_id` order in `rl_label_vocab.csv`. Top-20 IDs and numeric scores
are stored for every test row; metrics use the first five.

## 2. A0 target frequency

For training samples `i` with target set `Y_i`:

```text
score_A0(c) = sum_i 1[c in Y_i]
```

No prefix field is read.

## 3. CO positive-PMI set co-occurrence

Let `H_i` be the unique parent techniques in the observed history of sample
`i`, `N` the number of training samples, and:

```text
N_H(h)   = sum_i 1[h in H_i]
N_Y(c)   = sum_i 1[c in Y_i]
N_HY(h,c)= sum_i 1[h in H_i and c in Y_i]
PPMI(h,c)= max(0, ln((N_HY(h,c)*N)/(N_H(h)*N_Y(c))))
score_CO(c|H) = sum_(h in unique(H)) PPMI(h,c)
```

Terms with a zero count contribute zero. If every candidate score is zero, CO
uses the A0 ranking and scores for that row. No smoothing or tuned parameter is
used.

## 4. A context relevance

Use the exact v7 definition with `alpha_s=0.1`:

```text
p_uni(c) = (N(c)+0.5)/(N+1.0)
p(c|h)   = (N(h,c)+0.1*p_uni(c))/(N(h)+0.1)
```

Order-2, order-1, and unigram weights are `0.5/0.3/0.2`. An order-2
or order-1 layer is available only when its exact context occurs in training;
weights are renormalized over the available layers. The unigram layer is always
available. Independent relevance scores need not sum to one.

## 5. K monotonic-tactic diagnostic

Using all tactics for the last observed parent and each candidate, a candidate
is compatible when any candidate tactic is no earlier than any last-step tactic
in the frozen 14-tactic order. Compatible candidates are placed before
incompatible candidates; A order is retained within each partition. The stored
numeric score is `compatible + p_A(c)`, which preserves that ordering.

## 6. T soft multi-tactic fusion

For each training target set, the target tactic set is the union of all tactics
of its labels. Fourteen tactic relevance scores are estimated using the same
order-2/order-1/unigram counts and smoothing as A. Candidate tactic relevance
is the mean over all of its mapped tactics.

For each sample, convert A and tactic probabilities to clipped logits, then
standardize each 184-vector independently using population standard deviation:

```text
z_T = (1-lambda)*standardize(logit(p_A))
    + lambda*standardize(logit(p_tactic))
lambda in {0.0,0.1,...,1.0}
```

For each outer fold, the two training sources alternate as inner-train and
inner-validation. Lambda maximizes the source-equal mean of the two inner
campaign-macro NDCG@5 values. Exact ties select the smaller lambda. The chosen
lambda is then applied after refitting counts on both outer-training sources.

## 7. Metrics, visibility strata, and uncertainty

Sample metrics are Hit@5, Precision@5, Recall@5, and NDCG@5 as defined in v8.
The reported primary value first averages rows within each campaign, then
campaigns within each held-out source, then the three sources equally.

Outer-test target transition visibility is computed only from outer-training
rows. For the last observed label `h`, each `(h,c)` for `c in Y` is marked seen
if it occurred as a last-history/target-set pair in training. Rows are
`all_seen`, `mixed`, or `all_unseen`. Target-label visibility, final-description
length (`<40`/`>=40`), and target-set cardinality are also frozen strata.

Uncertainty uses 2,000 paired campaign-cluster bootstrap replicates with seed
`20260807`. Cells with fewer than 5 campaigns or 20 rows are descriptive only.
