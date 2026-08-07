# SECRYPT-adapted Hybrid Markov--LSTM future-3 LODO results

No text, tactic feature, LLM, or external API was used.

## Campaign-macro metrics (five-seed mean)

| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | LR | Epoch | Beta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctid | 0.1202 | 0.3710 | 0.0845 | 0.1537 | 0.001 | 40 | 0.1 |
| attack_flow | 0.1470 | 0.4010 | 0.0835 | 0.1668 | 0.001 | 100 | 0.2 |
| stockpile | 0.1063 | 0.2978 | 0.0600 | 0.1575 | 0.0003 | 40 | 0.9 |
| **Source-equal overall** | **0.1245** | **0.3566** | **0.0760** | **0.1593** | — | — | — |

## Source-equal paired NDCG@5 differences

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| HM-MB | -0.0132 | [-0.0448, +0.0173] |
| HM-LSTM | -0.0047 | [-0.0283, +0.0183] |
| HM-A | -0.0277 | [-0.0510, -0.0070] |
| HM-K | -0.0532 | [-0.0842, -0.0244] |

All outer direct-LSTM and beta=0 Markov-beam Top-20 reproduction gates passed.
