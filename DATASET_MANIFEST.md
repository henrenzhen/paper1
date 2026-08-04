# Dataset manifest

This manifest records the principal processed datasets and semantic controls committed with the repository. Byte counts and SHA-256 hashes describe the exact files in the accompanying commit.

## Main sequence splits

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `project/data/sim_train_parent_min3.csv` | 9,919 | 8,848,233 | `c306208764b8702cf479124d5b6751ea3c389b1b1b3e62d6088ed632d9f57755` |
| `project/data/sim_val_parent_min3.csv` | 2,102 | 1,870,580 | `2ffc7c91d7ac7ad3654d3055ebd1886b7db5e83892546b5460ed773e2dc3d5ae` |
| `project/data/sim_test_parent_min3.csv` | 2,107 | 1,848,765 | `8e3ec3f13351e236c827ac8769b03a0e277e4c1422dccb8390961f0c3b35f213` |

## CoT reasoning caches

| File | Rows | Empty reasoning | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `project/data/sim_train_llm_cot.csv` | 9,919 | 146 | 38,416,223 | `bd5c04819d4017392a48ffa87e3787ef17e0f8951ff25ca5579b78e2a650e6c6` |
| `project/data/sim_val_llm_cot.csv` | 2,102 | 34 | 8,125,098 | `4feaa6fb34472b08f1d3b27fc27a1faf2d8c04577cfc0730115a1aa4efa74f7d` |
| `project/data/sim_test_llm_cot.csv` | 2,107 | 42 | 8,173,538 | `ce7c3d4ba2437530521847dd13354ea5d9726a2e722a2784f2e2529af35b5cf3` |

The CoT caches contain 14,128 rows in total, of which 13,906 have non-empty reasoning and 222 have empty reasoning. These files are preserved as-is. No missing reasoning has been regenerated. See `experiments/INVENTORY.md` for the audit evidence and the stopped T0.1 decision.

## Semantic controls

| Variant | Train rows | Validation rows | Test rows |
|---|---:|---:|---:|
| No-CoT | 9,919 | 2,102 | 2,107 |
| Empty | 9,919 | 2,102 | 2,107 |

The matching files are `sim_{train,val,test}_llm_no_cot.csv` and `sim_{train,val,test}_llm_empty.csv`. KG-context CSV files, lookup tables, label vocabularies, calibration tables, and micro-state CSV files are also versioned in their respective data directories.

## Excluded material

The following remain excluded because they are large third-party sources or generated binary artifacts rather than the processed research datasets:

- `project/data_v2/external_*`
- model checkpoints and serialized weights
- logs and experiment result directories
- nested Git histories, attack-simulation binaries, and test certificates
