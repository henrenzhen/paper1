# HM + raw-description semantic future-3 control v1

Status: frozen before the first `HM+R` probe result is produced.

This card operationalizes v8 method `HM+R`. It tests whether the negative raw
description result was caused by pairing the semantic probe with `A` rather
than with the SECRYPT-adapted hybrid baseline. It uses no LLM or external API.
The 30 development rows are excluded throughout.

The HM branch is the already frozen full 184-dimensional marginal score from
`hybrid_markov_lstm_future3_v1.md`. Inner and outer scores are loaded from the
reproducible `hm_future3_v1` score cache; outer Top-20 rankings were checked
against all 3,920 frozen HM seed-sample predictions before the cache was
written.

The raw semantic branch uses the frozen BGE-M3 embeddings and the same probe as
`raw_semantic_probe_future3_v1.md`:

```text
1024 -> 256 -> 184
GELU, dropout=0.3, BCEWithLogitsLoss
AdamW, lr=1e-3, weight_decay=1e-4, batch=32
epochs in {20,40,60,80,100}
seeds = {42,43,44,45,46}
```

For every sample:

```text
z = (1-lambda) * standardize(logit(score_HM))
  + lambda     * standardize(z_raw_probe)
lambda in {0.0,0.1,...,1.0}
```

Each outer fold alternates its two training sources as inner-train and
inner-validation. `(epoch, lambda)` maximizes campaign-macro NDCG@5 averaged
equally over two validation sources and five seeds. Exact ties prefer smaller
lambda and then fewer epochs. Five outer probes are trained from scratch.

Primary comparison is `HM+R - HM`. Secondary comparisons are `HM+R - R` and
`HM+R - A`. Metrics are averaged over neural seeds at sample level before
campaign aggregation. Paired campaign bootstrap uses 2,000 replicates and seed
`20260807`.
