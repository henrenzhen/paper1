# Raw-description semantic probe for future-3 v1

Status: frozen before the first probe training run.

This card operationalizes v8 method `R`: the frozen context relevance branch
`A` fused with a probe trained on frozen BGE-M3 embeddings of observed raw event
histories. It does not use an LLM or any future event text.

## Inputs

- Model: `BAAI/bge-m3`, revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- Pooling: `last_hidden_state[:,0]` (CLS), then L2 normalization.
- Maximum tokens: 8192. Complete oldest events are removed only if necessary;
  at least the final two complete events must remain.
- Probe input contains observed parent IDs, all mapped tactics, and cleaned
  observed descriptions. Source/campaign/file metadata is excluded.
- The 30 development rows are excluded from all probe fitting, inner
  validation, and outer evaluation.

## Probe

```text
1024 -> 256 -> 184
hidden activation = GELU
dropout = 0.3
loss = BCEWithLogitsLoss (no class weights)
optimizer = AdamW
learning rate = 1e-3
weight decay = 1e-4
batch size = 32
epoch candidates = {20,40,60,80,100}
seeds = {42,43,44,45,46}
```

The encoder is never updated. Training runs on CPU with deterministic PyTorch
algorithms and no data-loader workers.

## Inner selection

For each outer held-out source, its two training sources alternate as
inner-train and inner-validation. At every candidate epoch, each of five seeds
produces validation logits. For each semantic fusion weight:

```text
z_R = (1-lambda)*standardize(logit(p_A))
    + lambda*standardize(z_probe)
lambda in {0.0,0.1,...,1.0}
```

The selection score is campaign-macro NDCG@5 averaged equally over the two
inner validation sources and five seeds. Exact ties prefer smaller lambda, then
fewer epochs. Outer-test results cannot affect selection.

After selection, five probes are trained from scratch on both outer-training
sources for the selected epoch count. Results are reported per seed and as the
mean of the five seed-level sample metrics; seeds are not treated as independent
campaigns.

## Comparison and uncertainty

The script independently reconstructs A within each outer fold and asserts its
Top-20 ranking equals the already frozen non-semantic baseline row by row.
Primary comparison is `R-A` on campaign-macro NDCG@5. Campaign-cluster paired
bootstrap uses 2,000 replicates and seed `20260807`; R metrics are averaged over
the five training seeds before campaign resampling.

Transition visibility, target-label visibility, final-description length, and
target-set cardinality strata use the already frozen outer-fold annotations.
Cells below 5 campaigns or 20 rows are descriptive only.
