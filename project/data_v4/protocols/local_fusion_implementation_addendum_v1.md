# Zero-cost local fusion implementation addendum v1

Status: frozen after the five oracle screens but before any legal F1--F4 outer result or control result was computed. It resolves only implementation details left open by `local_fusion_mechanism_search_v1.md`; it does not change an oracle gate, feature family, action space, grid, or pass/fail rule.

## Shared nested construction

- Outer-test features and expert rankings are fit on both outer-training sources.
- Outer meta-training rows use evidence and A/T/K rankings fit after excluding the row's whole campaign.
- Each inner validation source is scored by models fit only on the other outer-training source. Inner meta-training rows use campaign-LOO fits inside that source.
- T uses the already frozen outer-fold tactic weight from `nonsemantic_future3_lodo_v1`: CTID 0.0, Attack Flow 0.0, Stockpile 0.1. This fixed base-expert setting is not retuned for any new mechanism or control.
- Feature standardization uses only the current meta-training bundles. Zero-variance dimensions use scale 1.
- All score ties preserve B0 order unless a mechanism specifies another written tie order.
- Hyperparameter objective is the source-equal mean of the two inner-validation campaign-macro NDCG@5 values. Exact ties choose larger L2 for F1/F3 and the conservative written order for F2/F4.

## Controls

`campaign-permuted` rotates the evidence donor campaign by one position inside each evaluation source after sorting campaign IDs. Rows within recipient and donor campaigns are ordered by `(prefix_len,sample_id)`; recipient row `i` takes donor row `floor(i*n_donor/n_recipient)`. The recipient keeps its B0 list, observed prefix length, and targets for evaluation. Candidate-aligned A/T/K ranks, entropy, context counts, transition counts, support-source counts, and likelihood evidence come from the donor bundle. This is run separately for meta-training, inner validation, and outer test and never crosses a source boundary.

`equal-capacity` replaces every transition/expert value by a deterministic value in `[-1,1]` derived from SHA-256 of `control-v1|sample_id|candidate|feature_index`. B0 rank one-hot and prefix length remain real. For sample-level agreement features, deterministic pseudo-rankings over all 184 candidates are derived separately for pseudo A/T/K by SHA-256. Sample IDs are used only to construct this explicitly meaningless control and never enter a main method.

`no-prior` uses `log(p1+1e-9)` and `log(p2+1e-9)` in the two dimensions occupied by `lr1/lr2`; all other dimensions and capacity remain unchanged.

## F1 details

The score is a learned linear function of the frozen candidate feature vector. Pairwise loss includes every relevant-vs-irrelevant pair inside B0 Top-5; rows without both classes contribute no pair. The intercept and five B0-rank indicators are unregularized; all other weights receive the selected L2 penalty. Initial weights are zero. Adam uses betas 0.9/0.999 and epsilon `1e-8`.

## F2 details

For candidate `c`:

```text
vote_rr(c) = sum(1/rank_e(c) for e in A/T/K when rank_e(c)<=10)
support(c) = max(order1_support_sources, order2_support_sources)
score(c) = vote_rr(c) + lr1(c) + order2_weight*lr2(c)
```

The corresponding no-prior score substitutes raw log conditional evidence. The rank-5 B0 candidate is scored by the same formula. Replace only when the best eligible outside candidate exceeds rank 5 by the selected margin. Candidate ties choose more expert votes, more support sources, then lexical parent ID. Hyperparameter ties use the conservative order already written in the main protocol.

## F3 details

RBO over Top-5 uses `p=0.9` and the finite formula `(1-p)*sum_{d=1..5}(overlap_at_d/d)*p^(d-1)`. Multinomial softmax weights and intercepts initialize to zero. Cross-entropy is source-balanced: the mean loss is computed within each meta-training source and then sources are equally weighted. All non-intercept weights receive L2. Adam uses the frozen epochs/rate and betas 0.9/0.999, epsilon `1e-8`. A prediction tie follows `B0,T,K,A`.

## F4 details

For reorder actions, `outside_lr_sign` is `na`. For replacement actions it is the proposed candidate's evidence sign. The cell key is:

```text
(action, n2_positive, any_b0_two_source_order1_support,
 jaccard_bin_for_that_action_expert, outside_lr_sign)
```

Jaccard boundaries are left-closed: `<0.25`, `[0.25,0.5]`, and `>0.5`. Training row deltas are averaged first within `(cell,campaign)`, and the LCB uses those campaign means. Standard deviation is sample SD with denominator `campaigns-1`; cells with fewer than three campaigns never pass. Identity has fixed lower bound 0 and is chosen unless an edit has strictly positive LCB. Equal positive LCB ties follow the action order frozen in the main protocol.

## Outputs and reproducibility

Each mechanism writes separate per-sample predictions for main, B0, campaign-permuted, no-prior, and equal-capacity; inner selections; campaign/fold summaries; 2,000-replicate paired bootstrap; CTID leave-one-campaign influence; parameters; stdout; and a manifest with all hashes. Output directories must not already exist. No script imports a networking library or reads an API key.
