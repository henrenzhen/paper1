# Future-3 semantic dataset audit

All count, key, closed-set, text-cleaning, and temporal leakage gates passed.

## Frozen counts

| Source | Logical steps | Future-3 rows | Development-only | Main rows | Main campaigns |
|---|---:|---:|---:|---:|---:|
| ctid | 283 | 273 | 10 | 263 | 10 |
| attack_flow | 466 | 422 | 10 | 412 | 35 |
| stockpile | 149 | 119 | 10 | 109 | 27 |
| **Total** | **898** | **814** | **30** | **784** | **72** |

## Text coverage

| Source | Has description | Missing | Median cleaned chars | <40 | >=40 | Raw text with removed ATT&CK ID |
|---|---:|---:|---:|---:|---:|---:|
| ctid | 282 | 1 | 69 | 45 | 238 | 3 |
| attack_flow | 464 | 2 | 76.5 | 40 | 426 | 0 |
| stockpile | 149 | 0 | 38 | 85 | 64 | 0 |

## Target and development diagnostics

- Target-set cardinalities: {1: 81, 2: 153, 3: 580}.
- Old pilot rows mapped directly by exact corrected prefix: 26/30.
- Deterministic same-source/same-length-stratum replacements: 4/30.
- Development selection never uses a target label.

## CTID correction

The legacy CTID loader recursively visited nested YAML dictionaries and regex-extracted all ATT&CK-looking IDs from the entire object. This created child `technique` objects as extra events and made 12 ordinary single-label abilities appear multi-label when their name or description mentioned an older ID. The corrected representation uses one top-level ability as one logical event and reads only `technique.attack_id`.
