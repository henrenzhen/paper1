# Future-3 cross-source validation status v1

All values use the same 784-row main set, campaign-macro aggregation, and source-equal outer summary.

## Campaign-macro NDCG@5

| Method | CTID | Attack Flow | Stockpile | Source-equal |
|---|---:|---:|---:|---:|
| A0 | 0.1755 | 0.1897 | 0.0644 | 0.1432 |
| CO | 0.0357 | 0.0486 | 0.0080 | 0.0307 |
| A | 0.1559 | 0.1765 | 0.1241 | 0.1522 |
| K | 0.1450 | 0.1637 | 0.2244 | 0.1777 |
| T | 0.1559 | 0.1765 | 0.1434 | 0.1586 |
| R | 0.1597 | 0.1765 | 0.0634 | 0.1332 |
| LSTM | 0.1991 | 0.1255 | 0.0630 | 0.1292 |
| TR | 0.1097 | 0.1165 | 0.0622 | 0.0961 |
| MB | 0.1000 | 0.1191 | 0.1939 | 0.1377 |
| HM | 0.1202 | 0.1470 | 0.1063 | 0.1245 |

## All-unseen transition stratum: source-equal NDCG@5 delta vs A

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| T-A | -0.0009 | [-0.0028, +0.0000] |
| R-A | +0.0190 | [+0.0012, +0.0418] |
| LSTM-A | +0.0538 | [+0.0268, +0.0857] |
| TR-A | +0.0125 | [-0.0112, +0.0422] |
| MB-A | -0.0021 | [-0.0205, +0.0209] |
| HM-A | +0.0084 | [-0.0097, +0.0316] |

Current evidence rejects stable gains from raw-description probing, ID-only neural capacity, Markov beam restriction, and the SECRYPT-adapted hybrid. It does not evaluate the LLM-normalized S branch.
