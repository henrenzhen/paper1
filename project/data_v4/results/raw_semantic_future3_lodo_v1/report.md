# Raw-description semantic R results

BGE-M3 is frozen. No LLM or external generation API is used.

## Campaign-macro metrics (five-seed mean)

| Source | NDCG@5 | Hit@5 | Precision@5 | Recall@5 | Selected epoch | Selected lambda |
|---|---:|---:|---:|---:|---:|---:|
| ctid | 0.1597 | 0.4303 | 0.1012 | 0.1836 | 40 | 0.1 |
| attack_flow | 0.1765 | 0.4543 | 0.1042 | 0.2157 | 20 | 0.0 |
| stockpile | 0.0634 | 0.2047 | 0.0409 | 0.1135 | 20 | 1.0 |
| **Source-equal overall** | **0.1332** | **0.3631** | **0.0821** | **0.1710** | — | — |

## Paired R - A NDCG@5

| Scope | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| ctid | +0.0038 | [+0.0006, +0.0070] |
| attack_flow | +0.0000 | [+0.0000, +0.0000] |
| stockpile | -0.0608 | [-0.1216, -0.0172] |
| source_equal_overall | -0.0190 | [-0.0393, -0.0044] |

Raw-description fusion does not satisfy a cross-source support claim when the source-equal R-A difference is negative or any held-out source degrades materially. This result does not evaluate the LLM-normalized S branch.

The 30 development rows were excluded from inner fitting, hyperparameter selection, and outer evaluation.
