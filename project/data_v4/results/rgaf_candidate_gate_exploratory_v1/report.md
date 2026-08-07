# Candidate-level RGAF learnability audit v1

**Post-result exploratory method development; prospective confirmation is required.**

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| B0 | 0.1943 | 0.2401 | 0.1741 | **0.2028** |
| A | 0.1559 | 0.1765 | 0.1241 | **0.1522** |
| UniformResidual | 0.1410 | 0.1939 | 0.1652 | **0.1667** |
| RGAF | 0.1893 | 0.2375 | 0.1737 | **0.2002** |
| RGAF-Shuffle | 0.1943 | 0.2401 | 0.1741 | **0.2028** |

Selected L2: ctid=0.1, attack_flow=0.1, stockpile=0.001

| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |
|---|---:|---:|
| RGAF-B0 | -0.0027 | [-0.0044, -0.0012] |
| RGAF-RGAF-Shuffle | -0.0027 | [-0.0044, -0.0012] |
| RGAF-UniformResidual | +0.0334 | [+0.0157, +0.0533] |
| UniformResidual-B0 | -0.0361 | [-0.0559, -0.0183] |

Frozen decision: **no learnability evidence**.

Exact B0 Top-5 and A Top-20 reproduction gates passed for all 784 rows.
