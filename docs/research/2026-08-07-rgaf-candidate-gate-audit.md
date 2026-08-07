# Candidate-level RGAF learnability audit

> Date: 2026-08-07
>
> Status: post-result exploratory method development; not prospective confirmation
>
> Frozen protocol: `project/data_v4/protocols/rgaf_candidate_gate_learnability_v1.md`

## 1. Why the earlier oracle cannot be a gate

The stored `transition_visibility` value is computed by checking every true
future target against transitions observed in the outer-training sources. It
is therefore a valid **post-hoc diagnostic stratum**, but it is not observable
when a new prefix arrives and cannot be passed to an inference-time gate.

Recomputing the truth-conditioned routing with source-equal campaign-macro
NDCG@5 gives:

| Diagnostic router | NDCG@5 | Delta from B0 |
|---|---:|---:|
| B0 | 0.202830 | -- |
| use T on `all_seen`, otherwise B0 | 0.209848 | +0.007018 |
| use T on `all_seen`, K on `mixed`, otherwise B0 | 0.222082 | +0.019253 |
| per-sample oracle over B0/A/T/K | 0.307906 | +0.105076 |
| per-sample oracle over B0/A/T/K/A0 | 0.329901 | +0.127071 |

The first two routers still read the true-target stratum, and the final two
select the best expert after seeing the target. They are upper-bound
diagnostics only. They cannot support a deployable RGAF claim. In particular,
the previously discussed values 0.2411 and 0.2919 are not reproduced by the
frozen source-equal campaign-macro aggregation.

## 2. Legal candidate-level gate

The frozen audit replaces target-conditioned routing with a candidate-level
asymmetric residual gate:

```text
e(c|h) = clip(log((p_A(c|h)+1e-9)/(p_0(c)+1e-9)), -4, 4) / 4
g(x,c) = sigmoid(w^T phi(x,c))
score(c) = reciprocal_rank_B0(c) + g(x,c) * e(c|h)
```

`phi(x,c)` contains only inference-observable prefix support, candidate
transition counts computed from outer-training data, cross-training-source
support, A uncertainty/rank, B0 rank, and prefix length. It excludes the test
target, target-conditioned visibility, source/campaign/sample identifiers, and
future text. Training labels are legitimately used to fit transition counts
and the gate on training campaigns; **held-out test labels** enter only final
metrics. Gate-training evidence is campaign-LOO to avoid in-sample stacking
leakage, and L2 is selected by inner-LODO.

## 3. Result

| Method | CTID | Attack Flow | Stockpile | Source-equal NDCG@5 |
|---|---:|---:|---:|---:|
| B0 | 0.1943 | 0.2401 | 0.1741 | **0.2028** |
| A | 0.1559 | 0.1765 | 0.1241 | **0.1522** |
| UniformResidual | 0.1410 | 0.1939 | 0.1652 | **0.1667** |
| RGAF | 0.1893 | 0.2375 | 0.1737 | **0.2002** |
| RGAF-Shuffle | 0.1943 | 0.2401 | 0.1741 | **0.2028** |

Primary paired differences use 2,000 campaign bootstrap replicates:

| Comparison | Delta NDCG@5 | 95% CI |
|---|---:|---:|
| RGAF - B0 | -0.002676 | [-0.004357, -0.001183] |
| RGAF - RGAF-Shuffle | -0.002676 | [-0.004357, -0.001183] |
| UniformResidual - B0 | -0.036101 | [-0.055903, -0.018294] |

The frozen decision is **no learnability evidence**. RGAF is below B0 in both
real sources and overall, and the candidate-shuffled control exactly retains
B0's NDCG@5. The trained real gate opens only weakly on average; its small
perturbations change 30 CTID, 30 Attack Flow, and 77 Stockpile Top-5 lists, but
the net effect is harmful. Opening the transition residual for every candidate
is substantially worse.

This result rejects this linear candidate-support gate. It does not prove that
every possible LLM--transition interaction is impossible. It does show that
the large truth-conditioned oracle gap cannot be treated as evidence that an
observable count-based gate can recover that gap.

## 4. Mechanical and reproducibility audit

- 784 unique formal samples: CTID 263, Attack Flow 412, Stockpile 109;
- campaign denominators unchanged: 10 / 35 / 27;
- five methods each contain exactly 784 rows (3,920 prediction rows total);
- all metric and gate fields are finite;
- exact B0 Top-5 reproduction: 784/784;
- exact A Top-20 reproduction: 784/784;
- all four input hashes, protocol hash, script hash, and eight output hashes
  match `results_manifest.json`;
- an independent clean rerun in a different output directory produced
  byte-identical hashes for all eight managed outputs and identical manifest
  content apart from its generation timestamp.

Frozen hashes:

```text
protocol  5b5f900529682cd03ca8cb83072adc2458e41a2016655ff645b17ff746e5f25b
script    7c15c813859132359e6040a65fda63643a9eb24108b6424d263af387f9ffa9c8
predictions be6924ab6a36d0fa1333d8903e6c38320a893d803172d8fad764583ccacb8859
```

## 5. Consequence for method development

Further tuning candidate-count gates on these same 784 rows would be repeated
post-result search, not confirmation. The next technically distinct fusion
route is to keep the successful direct LLM ranking and let the LLM itself
reason over training-only transition evidence for a bounded candidate union,
instead of converting that evidence into a global weight or a shallow numeric
gate. Such a reranker needs a new frozen prompt/protocol, new API authorization,
and prospective data for a confirmatory method claim.
