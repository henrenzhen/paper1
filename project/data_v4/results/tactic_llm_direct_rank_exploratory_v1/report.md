# Exploratory T + direct LLM rank fusion

**Post-result exploratory analysis; prospective confirmation is required.**

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| A0 | 0.1755 | 0.1897 | 0.0644 | **0.1432** |
| A | 0.1559 | 0.1765 | 0.1241 | **0.1522** |
| K | 0.1450 | 0.1637 | 0.2244 | **0.1777** |
| T | 0.1559 | 0.1765 | 0.1434 | **0.1586** |
| B0 | 0.1943 | 0.2401 | 0.1741 | **0.2028** |
| HM+B0 | 0.1943 | 0.2393 | 0.1558 | **0.1965** |
| T+B0 | 0.1944 | 0.2401 | 0.1815 | **0.2053** |

Selected weights: ctid=(T 0.1, B0 0.8), attack_flow=(T 0.0, B0 0.9), stockpile=(T 0.0, B0 0.4)

| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |
|---|---:|---:|
| T+B0-B0 | +0.0025 | [-0.0007, +0.0078] |
| T+B0-T | +0.0467 | [+0.0094, +0.0891] |
| T+B0-HM+B0 | +0.0089 | [-0.0040, +0.0240] |
| B0-T | +0.0442 | [+0.0068, +0.0875] |
| B0-K | +0.0251 | [-0.0163, +0.0746] |

Frozen T Top-20 and exact B0 Top-5 reproduction gates passed for all 784 rows.
This result does not alter the negative preregistered HM+S conclusion.
