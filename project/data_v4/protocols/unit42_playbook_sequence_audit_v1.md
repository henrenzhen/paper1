# Unit42 Playbook Viewer sequence-sufficiency audit v1

Status: frozen after schema inventory but before the full per-bundle and
per-campaign eligibility result was computed.

## Scope and immutable source

Audit the archived public `pan-unit42/playbook_viewer` repository at an exact
Git commit.  The repository is an external input and is not copied into this
project.  The report records the Git commit, `playbooks.json` SHA-256, every
`playbook_json/*.json` SHA-256, and the aggregate tree digest.

The audit decides only whether the archived Unit42 bundles contain a defensible
within-campaign ATT&CK technique order for this project's next-step/future-3
task.  It does not assess the general CTI value of the playbooks.

## What qualifies as observed order

A campaign is sequence-eligible only if its bundle contains at least one of:

1. campaign-specific technique occurrences with an explicit execution
   timestamp or ordinal;
2. an explicit action/technique list whose schema documents list position as
   execution order;
3. directed technique-to-technique edges with an explicitly temporal relation
   such as `precedes`, `follows`, or `next`.

The following do **not** qualify:

- STIX object `created` or `modified`, which records CTI object lifecycle;
- campaign `first_seen`/`last_seen`, which bounds the campaign as a whole;
- report `published` or indicator `valid_from`;
- report/campaign `object_refs` position without documented temporal semantics;
- `campaign uses attack-pattern`, which expresses membership rather than the
  order of use;
- ATT&CK or Lockheed kill-chain phase order;
- JSON/bundle object order;
- sorting by tactic, ATT&CK ID, curation time, or file position;
- DFS over a membership star rooted at a campaign/report.

No permutation, tactic bucket, arbitrary tie-break, or inferred edge may be
introduced to make a campaign eligible.

## Mechanical checks

For all indexed and unindexed JSON bundles, record:

- valid JSON/bundle status and object-type counts;
- relationship-type × source-object-type × target-object-type counts;
- campaign and campaign-report counts;
- campaign→attack-pattern `uses` membership;
- attack-pattern→attack-pattern edges and their relationship types;
- campaign-level and relationship-level time fields;
- field names containing sequence-like tokens (`order`, `sequence`, `step`,
  `next`, `preced`, `follow`, `before`, `after`, `execution_time`,
  `observed_time`);
- per-campaign technique membership size and number of distinct relationship
  `created`/`modified` values;
- whether a campaign meets the frozen eligibility rule.

The report must distinguish “campaign has multiple ATT&CK techniques” from
“campaign has an ordered technique sequence.”  The former is not evidence for
the latter.

## Decision

- If at least one campaign is eligible, report exactly which schema/edge/field
  carries order and audit whether enough independent campaigns remain for a
  fourth LODO fold.  Do not build the fold in this task.
- If none is eligible, Unit42 is rejected as a direct sequential fourth source.
  It may still be used as an unordered recommendation corpus or reconstructed
  later from underlying narrative reports, but any such reconstruction is a
  new dataset and requires a separate frozen protocol.
