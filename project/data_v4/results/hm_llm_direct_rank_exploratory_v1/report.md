# Exploratory HM + direct LLM rank fusion

**Post-result exploratory analysis; prospective confirmation is required.**

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| A0 | 0.1755 | 0.1897 | 0.0644 | **0.1432** |
| A | 0.1559 | 0.1765 | 0.1241 | **0.1522** |
| K | 0.1450 | 0.1637 | 0.2244 | **0.1777** |
| LSTM | 0.1991 | 0.1255 | 0.0630 | **0.1292** |
| MB | 0.1000 | 0.1191 | 0.1939 | **0.1377** |
| HM | 0.1202 | 0.1470 | 0.1063 | **0.1245** |
| B0 | 0.1943 | 0.2401 | 0.1741 | **0.2028** |
| HM+B0 | 0.1943 | 0.2393 | 0.1558 | **0.1965** |

Selected lambda: ctid=0.9, attack_flow=0.6, stockpile=0.4

| Comparison | Delta NDCG@5 | 95% campaign-bootstrap CI |
|---|---:|---:|
| HM+B0-B0 | -0.0064 | [-0.0216, +0.0066] |
| HM+B0-HM | +0.0720 | [+0.0455, +0.1025] |
| B0-A0 | +0.0596 | [+0.0157, +0.1077] |
| B0-K | +0.0251 | [-0.0163, +0.0746] |
| B0-LSTM | +0.0736 | [+0.0326, +0.1187] |

Lambda 0 reproduced HM and lambda 1 reproduced the exact B0 Top-5 for all outer seed-sample rows.
