# SECRYPT-adapted Hybrid Markov--LSTM future-3 baseline v1

Status: frozen before the first `HM` result is produced.

This card operationalizes v8 method `HM`. It is a clean-room adaptation, not an
exact reproduction of Raj et al. It uses only ordered parent-technique IDs and
training-fold transition counts. It uses no descriptions, tactics, LLM, or
external API. The 30 development rows are excluded everywhere.

## Components

The Markov component, horizon, beam width, branch cap, smoothing, fallback, path
deduplication, and marginalization are exactly those frozen in
`markov_beam_future3_baseline_v1.md` and implemented by the same `MarkovBeam`
class. A beta of zero must reproduce every frozen outer `MB` Top-20 ranking.

The LSTM architecture and optimizer are exactly those in
`id_neural_future3_baselines_v1.md`. For each outer fold, its learning rate and
epoch are read from the already frozen ID-only LSTM inner-selection result. The
same setting is used in both inner beta selection and outer fitting. Each outer
LSTM's direct logits on the observed test histories must reproduce the frozen
ID-only LSTM Top-20 ranking for the same seed.

The LSTM input vocabulary contains the 120 technique IDs actually present in
observed histories. If beam generation appends one of the 64 output labels that
never occurs in an observed history, it is encoded as a zero embedding. This is
an explicit unknown-generated-state representation; it preserves the exact
already-frozen LSTM component instead of silently changing its embedding table
and random initialization after seeing the baseline result.

## Hybrid beam

At each expansion step the LSTM evaluates `observed_history + generated_path`
and its 184 logits are converted with log-softmax. For a path of length `h`:

```text
log P_H(path) = (1-beta) * sum_{j=1..h} log P_M(x_j | state_j)
              + beta     * sum_{j=1..h} log P_LSTM(x_j | history_j)

beta in {0.0,0.1,...,1.0}
```

After every step, the best 50 paths under that beta are retained. Final paths
are softmax-normalized under the same hybrid score and converted to candidate
marginals exactly as for MB.

## Selection and reporting

For an outer held-out source, the other two sources alternate as inner-train
and inner-validation. Beta maximizes campaign-macro NDCG@5 averaged equally
over both validation sources and five seeds. Exact ties prefer smaller beta.
Five outer models are trained from scratch. Metrics are averaged over seeds at
sample level before campaign aggregation. Paired campaign bootstrap uses 2,000
replicates and seed `20260807`.

Primary comparisons are `HM-MB`, `HM-LSTM`, `HM-A`, and `HM-K`. A gain over MB
alone does not establish the paper's semantic claim; HM is the strongest
non-semantic architecture control that the future LLM method must beat.
