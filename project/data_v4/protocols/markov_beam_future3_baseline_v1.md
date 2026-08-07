# Markov-beam future-3 baseline v1

Status: frozen before the first `MB` result is produced.

This card operationalizes v8 method `MB`. It uses no neural model, description,
tactic feature, LLM, or external API. The 30 development rows are excluded from
transition counts, fallback counts, and evaluation.

For every training sample, the observed pair from the final history technique
to the immediate next technique is counted once. The immediate next technique
is the first ordered member of `target_parent_ids`; data construction preserves
first occurrence order when later duplicates are removed from the future-three
target set.

Beam expansion is fixed as follows:

```text
horizon = 3
beam width = 50
per-state branch cap = 20
additive transition smoothing = 0.1
transition floor = 1e-12
path deduplication key = exact generated three-technique tuple
```

At a state with observed outgoing edges, candidates are the 20 most frequent
outgoing techniques; ties follow the frozen 184-label order. Probabilities are
`(edge_count+0.1) / sum_candidate(edge_count+0.1)`. At a state without an
observed edge, candidates are the top 20 training target-frequency labels and
probabilities are `(target_count+0.1)` normalized within that fallback set.

After each step, the 50 paths with largest cumulative log probability are
retained. Final beam log scores are softmax normalized. A candidate's relevance
is the total probability of final paths containing it at least once. Top-20
labels are ranked by this marginal; zero-score ties follow the fixed vocabulary.

The model is fit independently on the two outer-training sources in each LODO
fold. Campaign-cluster bootstrap uses 2,000 paired replicates and seed
`20260807`. Primary diagnostics are `MB-A`, `MB-A0`, and `MB-K`.
