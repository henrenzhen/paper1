# LLM summary semantic future-3 branch v1

Status: frozen after raw API responses were committed and before summary
embeddings, probe fitting, or any formal semantic outer-fold metric was
produced.

## Frozen input and serialization

The only LLM input to branch `S` is the three validated fields from formal run
`20260807T075707Z_310c2c11`, serialized exactly as:

```text
[阶段评估]
{stage_assessment}

[已观察能力]
{observed_capabilities}

[可能后续意图]
{likely_next_intents}
```

`predicted_next_ttps`, raw prompts, reasoning channels, source, campaign,
sample metadata, and targets are not included in the encoder text.  All 784
formal summaries must be non-empty, ATT&CK-ID-free, and exact-source/campaign-
literal-free.  Any invalid summary remains in the intention-to-treat denominator
and falls back according to v8; the committed run currently has no invalid row.

`B0` is a separate diagnostic that uses `predicted_next_ttps` directly.  It is
never concatenated into `S`.

## Encoder and probe

- encoder: `BAAI/bge-m3`
- revision: `5617a9f61b028005a4858fdac845db406aefb181`
- pooling: dense CLS (`last_hidden_state[:,0]`) then L2 normalization
- encoder frozen; 1024-dimensional vectors
- maximum 8192 tokens; summary rows exceeding it stop instead of being
  silently truncated
- probe: `1024 -> 256 -> 184`, GELU, dropout 0.3
- BCEWithLogitsLoss without per-class weights
- AdamW, learning rate 1e-3, weight decay 1e-4, batch 32
- epoch grid 20/40/60/80/100
- seeds 42/43/44/45/46
- deterministic CPU algorithms and zero data-loader workers

## S and HM+S selection

For each outer held-out source, the other two sources alternate as inner train
and inner validation.  `(epoch, lambda)` maximizes the two-source-equal,
five-seed mean campaign-macro NDCG@5.  Lambda is 0.0 to 1.0 in steps of 0.1.
Ties prefer smaller lambda and then fewer epochs.

```text
HM+S = (1-lambda) * standardize(logit(score_HM))
     + lambda     * standardize(logit_probe_S)
```

The HM branch is loaded from the already frozen full-score cache.  Lambda zero
must reproduce all frozen HM outer Top-20 rows.

## Permutation control HM+P

P has identical embeddings, probe, optimizer, epoch grid, seeds, and fusion.
Only embeddings assigned to training labels are permuted; inner-validation and
outer-test summaries remain correctly aligned.

For each actual probe training run:

1. within each training source, sort rows by `(prefix_len, sample_id)`;
2. assign equal-count terciles using `floor(3*rank/n)`, capped at 2;
3. groups smaller than two merge with the nearest adjacent tercile (ties to
   the lower tercile); the frozen data have no such group;
4. initialize one RNG with seed `9000 + train_seed`, visit `(source, tercile)`
   groups in lexical/numeric order, shuffle each group's sorted sample IDs,
   and map every recipient to the next donor in the circular order;
5. assert zero fixed points and save every recipient/donor mapping.

Thus P preserves source, length regime, vector distribution, architecture, and
training count while destroying summary-label correspondence only in training.

## Statistics and stop rule

The primary comparison is `HM+S - HM`; content attribution additionally
requires `HM+S > HM+P`.  Metrics are averaged across the five probe seeds at
sample level, then campaign-macro aggregated.  Paired campaign bootstrap uses
2,000 replicates and seed 20260807.  No result may change this serialization,
permutation, fusion, or tie-break protocol.

