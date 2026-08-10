# EDLR pilot v2 development-only descriptive results

> These 30 rows are prompt-development data. Numbers below are descriptive and cannot establish effectiveness or significance.

## NDCG@5

| Method | CTID | Attack Flow | Stockpile | Source-equal |
|---|---:|---:|---:|---:|
| B0 | 0.0671 | 0.3109 | 0.2125 | 0.1968 |
| EA_TOP5 | 0.0733 | 0.3285 | 0.2204 | 0.2074 |
| UNION_LLM | 0.0671 | 0.2935 | 0.1705 | 0.1771 |
| EDLR | 0.0671 | 0.1847 | 0.1337 | 0.1285 |
| EDLR_SHUFFLE | 0.0733 | 0.2237 | 0.2091 | 0.1687 |

## Frozen NDCG@5 contrasts

| Contrast | CTID | Attack Flow | Stockpile | Source-equal |
|---|---:|---:|---:|---:|
| EA_TOP5_minus_B0 | +0.0061 | +0.0176 | +0.0079 | +0.0105 |
| UNION_LLM_minus_B0 | +0.0000 | -0.0173 | -0.0420 | -0.0198 |
| EDLR_minus_B0 | +0.0000 | -0.1262 | -0.0788 | -0.0683 |
| EDLR_minus_UNION_LLM | +0.0000 | -0.1088 | -0.0368 | -0.0485 |
| EDLR_minus_EDLR_SHUFFLE | -0.0061 | -0.0390 | -0.0753 | -0.0402 |

Mechanism attribution would require EDLR to exceed both UNION_LLM and EDLR_SHUFFLE; this pilot can only guide whether a separately authorized exploratory full run is worth considering.
