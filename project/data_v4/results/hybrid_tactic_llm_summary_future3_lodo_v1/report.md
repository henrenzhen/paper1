# HM + tactic + LLM-summary future-3 LODO results

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| HM | 0.1202 | 0.1470 | 0.1063 | **0.1245** |
| HM+S | 0.1202 | 0.1470 | 0.0645 | **0.1106** |
| HM+T | 0.1202 | 0.1470 | 0.1063 | **0.1245** |
| HM+ST | 0.1414 | 0.1470 | 0.0645 | **0.1176** |

## Selected hyperparameters

| Held out | HM+T w_T | HM+ST epoch | w_H | w_S | w_T |
|---|---:|---:|---:|---:|---:|
| ctid | 0.0 | 60 | 0.7 | 0.2 | 0.1 |
| attack_flow | 0.0 | 20 | 1.0 | 0.0 | 0.0 |
| stockpile | 0.0 | 20 | 0.0 | 1.0 | 0.0 |

## Source-equal paired NDCG@5 differences

| Comparison | Delta | 95% campaign-bootstrap CI |
|---|---:|---:|
| HM+T-HM | +0.0000 | [+0.0000, +0.0000] |
| HM+ST-HM | -0.0068 | [-0.0156, +0.0014] |
| HM+ST-HM+S | +0.0071 | [+0.0027, +0.0117] |
| HM+ST-HM+T | -0.0068 | [-0.0156, +0.0014] |

## Frozen complementarity rule

- HM+ST > HM+S sources: 1/3
- HM+ST > HM+T sources: 1/3
- Overall exceeds both: False
- Complementarity claim: NOT SUPPORTED

HM and HM+S Top-20 reproduction gates passed for all 3,920 outer seed/sample rows.
The negative preregistered HM+S primary result remains unchanged.
