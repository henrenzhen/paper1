# ID-only neural future-3 baselines v1

Status: frozen before the first LSTM or Transformer result is produced.

This card operationalizes the v8 `LSTM` and `TR` baselines. It uses only the
ordered parent-technique prefix. It does not use descriptions, tactics, source
metadata, an LLM, or an external API. The 30 development rows are excluded from
all fitting, inner selection, and outer evaluation.

## Input and target

- Output vocabulary is the frozen 184-parent-technique vocabulary.
- Input vocabulary is the deterministic sorted union of parent-technique IDs
  present in observed histories before any result is computed. This retains the
  three observed history-only IDs that are outside the output vocabulary.
- Index 0 is padding. Sequences are right padded; lengths are carried
  separately.
- The target is a 184-dimensional multi-hot vector for the unique parent
  techniques occurring in the next three observed steps.
- Loss is `BCEWithLogitsLoss` without class weights.

## Architectures

LSTM:

```text
embedding = 128
hidden = 256
layers = 2
dropout = 0.3
pack_padded_sequence = true
readout = final valid hidden state
head = 184 independent logits
```

Transformer:

```text
embedding = 128
learned positional embedding = maximum observed history length
heads = 4
encoder layers = 2
feed-forward = 512
dropout = 0.3
activation = GELU
causal mask = true
padding mask = true
readout = final valid token
head = 184 independent logits
```

Both use AdamW, weight decay `1e-4`, batch size 32, learning-rate candidates
`{3e-4,1e-3}`, epoch candidates `{20,40,60,80,100}`, and seeds 42--46.

## Inner selection and outer evaluation

For every outer held-out source, the two training sources alternate as
inner-train and inner-validation. For each architecture, `(learning_rate,
epoch)` maximizes campaign-macro NDCG@5 averaged equally over the two validation
sources and five seeds. Exact ties prefer the smaller learning rate and then
the earlier epoch. Inner models trained on one source may be cached and
evaluated on both other sources; this is a computational reuse only and does
not change any fold membership.

Five outer models are trained from scratch on both outer-training sources with
the selected hyperparameters. Sample metrics are averaged over seeds before
campaign aggregation. Campaign-cluster bootstrap uses 2,000 replicates and
seed `20260807`. Neural seeds are not treated as independent campaigns.

The primary diagnostic comparisons are `LSTM-A`, `TR-A`, `LSTM-A0`, and
`TR-A0` on source-equal campaign-macro NDCG@5. These baselines do not establish
or refute the LLM claim by themselves; they establish whether ID-only sequence
capacity explains the available signal.
