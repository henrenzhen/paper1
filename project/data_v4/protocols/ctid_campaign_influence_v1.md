# CTID campaign influence audit v1

Status: frozen after the aggregate three-source results were observed, but
before any CTID leave-one-campaign-out influence values were computed.

This is a post-result sensitivity audit. It does not replace the frozen LODO
evaluation or create a new significance test.

## Question

CTID has only ten independent campaigns. The audit asks whether the positive
CTID point estimate of the direct DeepSeek ranking (`B0`) is caused by one
campaign, and whether the exploratory `T+B0` increment has the same problem.

## Frozen inputs and comparisons

Read campaign-level metrics already committed by:

- `tactic_llm_direct_rank_exploratory_v1` for `A0`, `K`, `T`, `B0`, and
  `T+B0`;
- `hybrid_tactic_llm_summary_future3_lodo_v1` for `HM` and `HM+S`.

The primary metric is campaign-macro NDCG@5. Hit@5, Precision@5, and Recall@5
are reported as secondary diagnostics. The comparisons are fixed as:

```text
B0 - A0
B0 - K
B0 - T
B0 - HM
B0 - HM+S
T+B0 - B0
```

All methods must contain the same ten CTID campaign IDs and the same row count
within every campaign. Any mismatch is a hard failure.

## Influence calculation

For campaign `j`, let `d_j` be the campaign-level metric of the first method
minus the second method. The full CTID effect is the unweighted mean of all ten
`d_j` values. For each campaign, recompute that mean after omitting `j`.

Report:

- the full effect;
- the minimum and maximum leave-one-campaign-out effect;
- the campaign causing each extreme;
- the number of positive, zero, and negative campaign-level differences;
- whether omitting any one campaign reverses the sign of a positive full
  effect.

An advantage is called **single-campaign sign-stable** only when its full
effect and every leave-one-campaign-out effect are strictly positive. If any
omission makes it zero or negative, call it **single-campaign fragile**. This
label is a sensitivity description, not a claim of statistical significance.

## Reproducibility

The script must refuse to overwrite an existing output directory and record
SHA-256 hashes for the script, this protocol, every input, and every output.

