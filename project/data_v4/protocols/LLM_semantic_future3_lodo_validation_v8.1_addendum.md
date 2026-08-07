# LLM semantic future-3 LODO validation v8.1 addendum

Status: frozen before any `P-source` outer-fold prediction was produced.

This addendum supersedes only the CTID parsing rule and the derived row-count
expectations in v8. All model definitions, inner/outer separation, primary
metric, development-only exclusion, leakage rules, and negative-result rules in
v8 remain unchanged.

## 1. Why v8's CTID count cannot be used

The legacy parser in
`project/data_v2/scripts/parse_all_ctid_yaml_orgs.py` recursively walks every
dictionary inside a top-level CTID YAML ability. It then regex-extracts every
ATT&CK-looking ID from the serialized object and chooses the lexicographically
first parent ID. This has two consequences:

1. a nested `technique` dictionary is emitted as another chronological event;
2. an obsolete ID mentioned in an ability name or description is treated as an
   additional ground-truth label.

The legacy long tables therefore contain 702 recursive records and yield 293
deduplicated pseudo-steps. The 12 rows previously classified as multi-technique
events are ordinary single-label abilities: their structured
`technique.attack_id` contains exactly one ATT&CK ID, while another ID appears
only in free text. Deleting those 12 rows would remove valid observations and
would not repair the recursive child-event problem.

This defect was found while building the source/text alignment, before running
any outer-fold baseline or semantic model. The correction is consequently a
pre-result data-integrity correction, not a result-driven protocol change.

## 2. Corrected CTID logical-step rule

For CTID only:

1. one top-level YAML ability record is one candidate logical event;
2. the label is read only from the structured `technique.attack_id` field;
3. values that do not match `T\d{4}(\.\d{3})?` are not guessed from names,
   descriptions, commands, or procedure identifiers;
4. after invalid structured labels are removed, consecutive equal parent
   techniques are deduplicated while retaining the earlier event;
5. the top-level `description` is the only semantic text; the technique name is
   retained for audit but never used as a description fallback.

Twenty-six top-level Turla records have invalid structured labels (`x`,
`7.A.5`, or `8.A.2`) and are excluded without imputation. The full list is
stored in `step_text_alignment_manifest.json`.

## 3. Superseded count gates

The following gates replace v8 Sections 3.1-3.3:

| Source | Corrected logical steps | Future-3 rows | Development-only | Main rows | Main campaigns |
|---|---:|---:|---:|---:|---:|
| CTID | 283 | 273 | 10 | 263 | 10 |
| Attack Flow | 466 | 422 | 10 | 412 | 35 |
| Stockpile | 149 | 119 | 10 | 109 | 27 |
| **Total** | **898** | **814** | **30** | **784** | **72** |

The corrected target-set cardinalities are:

```text
|Y| = 1: 81
|Y| = 2: 153
|Y| = 3: 580
```

The old 30-row pilot is still development-only and remains 10 rows per source.
Mapping first attempts an exact `(source, campaign_id, prefix_len, prefix)`
match under the corrected sequence. If that endpoint is unavailable, differs
because of the legacy CTID defect, or would remove the only evaluable row of a
campaign, replacement is deterministic within the same source and prefix-length
tertile using:

```text
SHA256("v7-dev-20260806" || source || campaign_id || prefix_len)
```

Selection never uses a target label. The frozen mapping contains 26 direct
matches and 4 deterministic replacements. The replacement that avoids a
single-row Stockpile campaign is required to preserve the campaign-macro
denominator at 27.

## 4. Tactic mapping provenance

Multi-hot tactic labels are read from the repository's frozen
`project/data/attack_lookup_dedup.csv`. The associated layer-source metadata
identifies ATT&CK Enterprise v18. The exact lookup-file SHA-256 and the canonical
14-tactic order are stored in the alignment manifest. All 184 candidate labels
have at least one tactic mapping.

## 5. Required artifacts and stop rule

The corrected frozen artifacts are:

```text
project/data_v4/semantic_alignment/step_text_alignment.csv
project/data_v4/semantic_alignment/future3_samples.csv
project/data_v4/semantic_alignment/development_mapping.csv
project/data_v4/semantic_alignment/technique_tactic_multihot.csv
project/data_v4/semantic_alignment/step_text_alignment_manifest.json
project/data_v4/semantic_alignment/audit_report.md
```

If any corrected count, uniqueness check, temporal check, cleaned-text ATT&CK
ID check, or main-evaluation campaign count changes, construction stops before
model evaluation.
