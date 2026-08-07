# HM + LLM summary future-3 LODO results

## Campaign-macro metrics (five-seed mean for probe methods)

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| HM | 0.1202 | 0.1470 | 0.1063 | **0.1245** |
| HM+R | 0.1202 | 0.1470 | 0.0634 | **0.1102** |
| HM+S | 0.1202 | 0.1470 | 0.0645 | **0.1106** |
| HM+P | 0.1202 | 0.1470 | 0.0632 | **0.1101** |
| B0 | 0.1943 | 0.2401 | 0.1741 | **0.2028** |

## Selected S/P hyperparameters

| Method | Held out | Epoch | Lambda |
|---|---|---:|---:|
| HM+S | ctid | 20 | 0.0 |
| HM+S | attack_flow | 20 | 0.0 |
| HM+S | stockpile | 20 | 1.0 |
| HM+P | ctid | 20 | 0.0 |
| HM+P | attack_flow | 20 | 0.0 |
| HM+P | stockpile | 20 | 1.0 |

## Source-equal paired NDCG@5 differences

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| HM+S-HM | -0.0139 | [-0.0214, -0.0073] |
| HM+S-HM+P | +0.0005 | [-0.0000, +0.0012] |
| HM+S-HM+R | +0.0004 | [-0.0000, +0.0009] |
| HM+P-HM | -0.0144 | [-0.0219, -0.0080] |

All lambda=0 outer rankings reproduced the frozen HM Top-20 rows.
All P training mappings were derangements within source and prefix-length tercile; validation and test summaries were never permuted.
The 30 development rows were excluded throughout.
