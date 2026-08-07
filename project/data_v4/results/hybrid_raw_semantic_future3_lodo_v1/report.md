# HM + raw-description future-3 LODO results

No LLM or external API was used.

## Campaign-macro metrics (five-seed mean)

| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | Epoch | Lambda |
|---|---:|---:|---:|---:|---:|---:|
| ctid | 0.1202 | 0.3710 | 0.0845 | 0.1537 | 20 | 0.0 |
| attack_flow | 0.1470 | 0.4010 | 0.0835 | 0.1668 | 20 | 0.0 |
| stockpile | 0.0634 | 0.2047 | 0.0409 | 0.1135 | 20 | 1.0 |
| **Source-equal overall** | **0.1102** | **0.3256** | **0.0697** | **0.1447** | — | — |

## Source-equal paired NDCG@5 differences

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| HM+R-HM | -0.0143 | [-0.0221, -0.0077] |
| HM+R-R | -0.0230 | [-0.0378, -0.0078] |
| HM+R-A | -0.0420 | [-0.0669, -0.0200] |

All lambda=0 outer rankings reproduced the frozen HM Top-20 rows.
