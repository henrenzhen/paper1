# ID-only neural future-3 LODO results

No text, tactic feature, LLM, or external API was used.

## Campaign-macro NDCG@5 (five-seed mean)

| Method | CTID | Attack Flow | Stockpile | Source-equal overall |
|---|---:|---:|---:|---:|
| LSTM | 0.1991 | 0.1255 | 0.0630 | 0.1292 |
| TR | 0.1097 | 0.1165 | 0.0622 | 0.0961 |

## Inner-selected hyperparameters

| Method | Held-out | Learning rate | Epoch |
|---|---|---:|---:|
| LSTM | ctid | 0.001 | 40 |
| LSTM | attack_flow | 0.001 | 100 |
| LSTM | stockpile | 0.0003 | 40 |
| TR | ctid | 0.001 | 100 |
| TR | attack_flow | 0.0003 | 80 |
| TR | stockpile | 0.0003 | 20 |

## Source-equal paired NDCG@5 differences

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| LSTM-A | -0.0230 | [-0.0499, +0.0013] |
| TR-A | -0.0561 | [-0.0813, -0.0338] |
| LSTM-A0 | -0.0140 | [-0.0275, -0.0013] |
| TR-A0 | -0.0471 | [-0.0708, -0.0221] |
| TR-LSTM | -0.0331 | [-0.0550, -0.0102] |

Metrics average five neural seeds at sample level before campaign aggregation. The 30 development rows are excluded.
