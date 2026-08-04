# GSAD Minimal Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a leakage-controlled, root-disjoint minimum experiment that can falsify or support GSAD for ATT&CK next-technique prediction without reusing contaminated neural logits.

**Architecture:** A frozen data protocol produces fit/validation/calibration/locked-test partitions and hard-fails on leakage. NumPy/pandas probability models feed clustered APS, an exact ATT&CK multi-parent DAG compressor, and a root-balanced three-action policy. Development uses nested root OOF; the locked test is callable once only after configuration freeze, and CTID is scored afterward as an exploratory nine-actor OOD set.

**Tech Stack:** Python 3.12, standard library, NumPy, pandas, `unittest`; repository-local ATT&CK v18.1 JSON and CSV data.

## Global Constraints

- Use the 184-parent-technique vocabulary and freeze `enterprise-attack-18.1.json` by SHA-256.
- Remove `SIM_008, SIM_010, SIM_014, SIM_030, SIM_033, SIM_040, SIM_041, SIM_044, SIM_090` before every SIM split.
- Final split is exactly 93/20/20/20 roots with 7,371/1,592/1,592/1,592 rows.
- Never use CTID labels for fit, validation, calibration, feature selection, thresholds, or candidate selection.
- Never use existing contaminated GRU/RL/LLM logits as publication-valid inputs.
- Hard-fail if future-context fields enter features: `next_technique*`, `true_label`, `matched_technique_name`, `matched_description`, or `matched_command_summary`.
- Candidate iteration uses only the 133 non-test roots; the 20-root locked test is evaluated once for one frozen winner.
- Use 2,000 whole-root bootstrap replicates with a fixed seed for final intervals.
- Do not overwrite a locked result directory or unlock it through a command-line flag.
- This workspace is not a Git repository. Replace commit steps with a passing-test checkpoint plus SHA-256 manifest; do not claim commits were made.

---

## File Map

- `project/experiments/gsad/data_protocol.py`: root extraction, fixed partitioning, field allowlist, leakage audit, CTID actor normalization.
- `project/experiments/gsad/probability_models.py`: full-vocabulary unigram, Markov, interpolated n-gram, tactic-aware models.
- `project/experiments/gsad/conformal.py`: APS scores, graph-constrained label clustering, finite-sample thresholds, prediction sets.
- `project/experiments/gsad/attack_dag.py`: frozen ATT&CK parsing, multi-parent descendants, exact 14-tactic compression.
- `project/experiments/gsad/shift_policy.py`: inference-only features, root-balanced logistic fit, exact/DAG/abstain action selection.
- `project/experiments/gsad/metrics.py`: row/root/actor metrics, matched-cost comparison, cluster bootstrap, gates A–G.
- `project/experiments/gsad/artifacts.py`: deterministic JSON/CSV writers, hashes, freeze token, no-overwrite guard.
- `project/experiments/gsad/run_development.py`: nested root OOF candidate evaluation and winner freeze.
- `project/experiments/gsad/run_locked_evaluation.py`: one-shot locked SIM and CTID evaluation.
- `project/tests/test_gsad_*.py`: unit, integration, leakage, and negative-control tests.

---

### Task 1: Freeze and Audit the Root-Disjoint Data Protocol

**Files:**
- Create: `project/experiments/gsad/__init__.py`
- Create: `project/experiments/gsad/data_protocol.py`
- Create: `project/tests/test_gsad_data_protocol.py`

**Interfaces:**
- Produces: `sim_root(sequence_id: str) -> str`
- Produces: `build_frozen_split(frames: Sequence[pd.DataFrame]) -> FrozenSplit`
- Produces: `audit_feature_columns(columns: Sequence[str], target_columns: Sequence[str]) -> None`
- Produces: `normalize_ctid_actor(name: str) -> str`
- `FrozenSplit` fields: `fit`, `validation`, `calibration`, `test`, `excluded_roots`, `audit`.

- [ ] **Step 1: Write failing root and split tests**

```python
class FrozenSplitTests(unittest.TestCase):
    def test_root_suffix_is_removed(self):
        self.assertEqual(sim_root("SIM_010_part042"), "SIM_010")

    def test_external_actor_roots_are_removed_and_partitions_disjoint(self):
        split = build_frozen_split(load_core_frames())
        self.assertEqual([len(x) for x in (split.fit, split.validation,
                                           split.calibration, split.test)],
                         [7371, 1592, 1592, 1592])
        root_sets = [set(x["root"]) for x in
                     (split.fit, split.validation, split.calibration, split.test)]
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(root_sets)
                            for b in root_sets[i + 1:]))
        self.assertTrue(set(EXTERNAL_SIM_ROOTS).isdisjoint(set().union(*root_sets)))
```

- [ ] **Step 2: Run the tests and confirm missing-module failure**

Run: `python -m unittest project.tests.test_gsad_data_protocol -v`  
Expected: `ModuleNotFoundError: No module named 'experiments.gsad'`.

- [ ] **Step 3: Implement exact constants and deterministic construction**

```python
VALIDATION_ROOTS = frozenset({
    "SIM_001", "SIM_004", "SIM_007", "SIM_019", "SIM_021", "SIM_054",
    "SIM_057", "SIM_062", "SIM_063", "SIM_064", "SIM_093", "SIM_107",
    "SIM_108", "SIM_113", "SIM_126", "SIM_130", "SIM_134", "SIM_153",
    "SIM_158", "SIM_160",
})
CALIBRATION_ROOTS = frozenset({
    "SIM_003", "SIM_013", "SIM_027", "SIM_038", "SIM_048", "SIM_067",
    "SIM_082", "SIM_085", "SIM_092", "SIM_103", "SIM_105", "SIM_106",
    "SIM_109", "SIM_111", "SIM_122", "SIM_137", "SIM_156", "SIM_161",
    "SIM_165", "SIM_171",
})
TEST_ROOTS = frozenset({
    "SIM_016", "SIM_024", "SIM_035", "SIM_037", "SIM_039", "SIM_046",
    "SIM_047", "SIM_053", "SIM_072", "SIM_073", "SIM_081", "SIM_083",
    "SIM_084", "SIM_095", "SIM_124", "SIM_128", "SIM_133", "SIM_141",
    "SIM_146", "SIM_169",
})
EXTERNAL_SIM_ROOTS = frozenset({"SIM_008", "SIM_010", "SIM_014", "SIM_030",
                                "SIM_033", "SIM_040", "SIM_041", "SIM_044",
                                "SIM_090"})

def sim_root(sequence_id: str) -> str:
    return re.sub(r"_part\d+$", "", str(sequence_id))
```

Load only `sequence_id`, `prefix_len`, `prefix_technique_ids_parent`, and `next_technique_id_parent`; rename the last column to the explicit target field `target` before any feature construction.

- [ ] **Step 4: Add field allowlist and hard-failure tests**

```python
def test_future_context_is_rejected_even_when_renamed_as_feature(self):
    for name in ["next_technique_id_parent", "true_label", "matched_description",
                 "matched_command_summary", "matched_technique_name"]:
        with self.subTest(name=name), self.assertRaises(ValueError):
            audit_feature_columns(["prefix_len", name], target_columns=["target"])
```

- [ ] **Step 5: Implement the allowlist, CTID actor normalization, and full audit**

The feature API accepts only `prefix_ids`, `prefix_len`, `sequence_id`, `root`, and model-derived values explicitly registered by later modules. Map `turla_carbon` and `turla_snake` to `turla`; normalize other actor names case-insensitively.

- [ ] **Step 6: Run the task tests and record a checkpoint**

Run: `python -m unittest project.tests.test_gsad_data_protocol -v`  
Expected: all tests pass; print the four partition counts and zero pairwise root overlaps.

---

### Task 2: Implement Full-Vocabulary Statistical Probability Models

**Files:**
- Create: `project/experiments/gsad/probability_models.py`
- Create: `project/tests/test_gsad_probability_models.py`

**Interfaces:**
- Consumes: `FrozenSplit.fit`, `FrozenSplit.validation`, 184-label vocabulary.
- Produces: `UnigramModel.fit/predict_proba`
- Produces: `InterpolatedNGram.fit/predict_proba_with_meta`
- Produces: `TacticAwareModel.fit/predict_proba_with_meta`
- Common output: `(probs: np.ndarray[n, 184], metadata: pd.DataFrame)`.

- [ ] **Step 1: Write failing normalization, backoff, and root-balance tests**

```python
def test_unseen_trigram_backs_off_and_probabilities_sum_to_one(self):
    model = InterpolatedNGram(vocab=("T1", "T2", "T3"), order=3,
                              alpha=0.5, interpolation=(0.2, 0.3, 0.5))
    model.fit([("T1", "T2")], ["T3"], groups=["G1"])
    p, meta = model.predict_proba_with_meta([["T9", "T2"]])
    self.assertAlmostEqual(float(p.sum()), 1.0)
    self.assertLessEqual(int(meta.loc[0, "used_order"]), 1)

def test_root_balanced_counts_do_not_let_long_group_dominate(self):
    prefixes = [("T1",)] * 101
    targets = ["T2"] * 100 + ["T3"]
    groups = ["G1"] * 100 + ["G2"]
    model = InterpolatedNGram(vocab=("T1", "T2", "T3"), order=2,
                              alpha=0.0, interpolation=(0.0, 1.0))
    model.fit(prefixes, targets, groups=groups)
    p, _ = model.predict_proba_with_meta([["T1"]])
    self.assertAlmostEqual(float(p[0, 1]), float(p[0, 2]), places=12)
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_probability_models -v`  
Expected: import failure for `probability_models`.

- [ ] **Step 3: Implement root-balanced counts and smoothed unigram/Markov**

For each root, normalize transition counts by that root's total; average normalized matrices across roots; add Dirichlet `alpha / K`; renormalize every row. Empty contexts back off to unigram.

- [ ] **Step 4: Implement interpolated n-gram and tactic-aware scoring**

```python
score_y = (w0 * p_unigram[y] + w1 * p_bigram[y]
           + w2 * p_trigram[y] + wt * p_tactic[y])
score_y = np.maximum(score_y, 1e-15)
probs = score_y / score_y.sum()
```

Grid choices are fixed before locked evaluation: order `{1,2,3}`, alpha `{0.1,0.5,1.0}`, and at most three interpolation weights chosen on validation by root-macro MRR, then NLL, then smaller order.

- [ ] **Step 5: Add real-data smoke test for 184 classes**

Assert that fit has 93 roots, output has shape `(1592, 184)`, every row sums to one within `1e-10`, and no validation target was passed to `predict_proba_with_meta`.

- [ ] **Step 6: Run tests and checkpoint model-selection output**

Run: `python -m unittest project.tests.test_gsad_probability_models -v`  
Expected: all tests pass and the selected model configuration serializes without NumPy scalar errors.

---

### Task 3: Build ATT&CK Multi-Parent DAG and Exact Compressor

**Files:**
- Create: `project/experiments/gsad/attack_dag.py`
- Create: `project/tests/test_gsad_attack_dag.py`

**Interfaces:**
- Produces: `AttackDAG.from_stix(path: Path, vocab: Sequence[str]) -> AttackDAG`
- Produces: `AttackDAG.tactics_for(technique_id: str) -> frozenset[str]`
- Produces: `compress_leaf_set(gamma: frozenset[str], dag: AttackDAG, lam: float, max_nodes: int) -> StructuredSet`
- `StructuredSet` fields: `nodes`, `descendants`, `leaf_equivalent_size`, `objective`, `coverage_preserved`.

- [ ] **Step 1: Write a multi-parent toy graph and optimality tests**

```python
def test_multi_parent_membership_is_preserved(self):
    dag = AttackDAG.from_edges({"TA1": {"T1", "T2"}, "TA2": {"T2", "T3"}})
    self.assertEqual(dag.tactics_for("T2"), frozenset({"TA1", "TA2"}))

def test_compressor_is_exact_and_never_drops_gamma(self):
    dag = AttackDAG.from_edges({"TA1": {"T1", "T2"}, "TA2": {"T2", "T3"}})
    out = compress_leaf_set(frozenset({"T1", "T2"}), dag, lam=1.0, max_nodes=3)
    self.assertTrue({"T1", "T2"}.issubset(out.descendants))
    self.assertEqual(out.nodes, frozenset({"TA1"}))
    self.assertEqual(out.objective, 3.0)
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_attack_dag -v`  
Expected: import failure for `attack_dag`.

- [ ] **Step 3: Parse STIX without collapsing multi-tactic edges**

Resolve parent technique IDs, exclude revoked/deprecated objects, and derive tactic IDs from `kill_chain_phases`. Keep every valid tactic membership. Missing vocab mappings remain leaf-only and are returned in `mapping_audit`.

- [ ] **Step 4: Implement exact 14-tactic enumeration**

```python
for mask in range(1 << len(dag.tactic_ids)):
    tactics = selected_tactics(mask)
    covered = dag.descendants_of_tactics(tactics)
    leaves = gamma - covered
    nodes = tactics | leaves
    if len(nodes) <= max_nodes:
        descendants = dag.descendants(nodes)
        candidate = (len(descendants) + lam * len(nodes),
                     len(descendants), len(nodes), tuple(sorted(nodes)))
        best = min(best, candidate)
```

- [ ] **Step 5: Compare enumeration with full brute force on randomized toy DAGs**

Use 100 deterministic random graphs with 3 tactics/6 techniques; assert equal objective and `gamma <= descendants` for every case.

- [ ] **Step 6: Run tests and record ATT&CK hash/mapping checkpoint**

Run: `python -m unittest project.tests.test_gsad_attack_dag -v`  
Expected: all tests pass; audit reports the exact tactic count and all missing mappings.

---

### Task 4: Implement Graph-Clustered APS Without Calibration Leakage

**Files:**
- Create: `project/experiments/gsad/conformal.py`
- Create: `project/tests/test_gsad_conformal.py`

**Interfaces:**
- Consumes: validation probabilities/targets, frozen `AttackDAG`.
- Produces: `fit_graph_clusters(validation_probs: np.ndarray, validation_targets: Sequence[str], fit_counts: Mapping[str, int], dag: AttackDAG, vocab: Sequence[str], min_support: int) -> LabelClusters`
- Produces: `fit_clustered_aps(cal_probs, cal_targets, clusters, alpha, sample_ids) -> ClusteredAPS`
- Produces: `ClusteredAPS.predict_sets(probs, sample_ids) -> list[frozenset[str]]`
- Produces: per-cluster threshold/support/fallback audit table.

- [ ] **Step 1: Write failing APS score and finite-quantile tests**

```python
def test_aps_true_score_uses_mass_before_label_plus_randomized_mass(self):
    p = np.array([0.6, 0.3, 0.1])
    self.assertAlmostEqual(aps_score(p, label_index=1, u=0.5), 0.75)

def test_cluster_function_is_frozen_before_calibration(self):
    fixture = validation_fixture()
    clusters = fit_graph_clusters(
        fixture.probs, fixture.targets, fixture.fit_counts,
        fixture.dag, fixture.vocab, min_support=10,
    )
    before = clusters.digest()
    fit_clustered_aps(cal_probs(), cal_targets(), clusters, alpha=0.1,
                      sample_ids=cal_ids())
    self.assertEqual(before, clusters.digest())
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_conformal -v`  
Expected: import failure for `conformal`.

- [ ] **Step 3: Implement deterministic sample-ID randomization and APS**

Derive `u` as the first 53 bits of `SHA256(f"{seed}|{sample_id}|{label}")`, divided by `2**53`. Use stable descending sort with vocabulary order as tie-break. The calibrated index is `ceil((n + 1) * (1 - alpha)) - 1`, clipped to `[0, n-1]`.

- [ ] **Step 4: Implement graph-constrained agglomerative label clusters**

Features are tactic multi-hot, `log1p(fit_count)`, and validation APS quartiles. Only clusters sharing a tactic/edge may merge. Choose min support from `{10,20,30}` using validation-only mean class-conditional coverage gap, then mean set size, then larger support.

- [ ] **Step 5: Implement per-cluster thresholds and global fallback**

Clusters with fewer than five calibration true-label rows use the global threshold and set `fallback=True`. Prediction sets are forced nonempty by adding Top-1; forced additions can only expand coverage and are counted.

- [ ] **Step 6: Run tests including a calibration-label mutation sentinel**

Mutate calibration feature columns while keeping probabilities/targets fixed and assert clusters do not change; mutate validation targets and assert clusters do change. Run: `python -m unittest project.tests.test_gsad_conformal -v`.

---

### Task 5: Implement Root-Balanced Shift Score and Three-Action Policy

**Files:**
- Create: `project/experiments/gsad/shift_policy.py`
- Create: `project/tests/test_gsad_shift_policy.py`

**Interfaces:**
- Produces: `build_inference_features(probs, metadata, prefixes, fit_reference) -> np.ndarray`
- Produces: `fit_root_balanced_logistic(X, correct, roots, l2) -> LogisticScore`
- Produces: `calibrate_exact_threshold(scores, correct, roots, target_risk) -> ExactThreshold`
- Produces: `choose_action(gamma, structured, safety_score, threshold, max_leaf_size, support_ok) -> Action`.

- [ ] **Step 1: Write failing action-boundary tests**

```python
def test_singleton_safe_set_outputs_exact(self):
    action = choose_action(frozenset({"T1"}), structured_t1(), 0.9,
                           threshold=0.8, max_leaf_size=20, support_ok=True)
    self.assertEqual(action.kind, "exact")

def test_wide_or_unsupported_set_abstains(self):
    action = choose_action(frozenset({"T1", "T2"}), structured_wide(), 0.2,
                           threshold=0.8, max_leaf_size=20, support_ok=False)
    self.assertEqual(action.kind, "abstain")
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_shift_policy -v`  
Expected: import failure for `shift_policy`.

- [ ] **Step 3: Implement exactly five inference-only features**

Return entropy, Top-1/Top-2 margin, transition surprise, backoff/set-size signal, and robust distance from fit median/MAD. Reject a feature matrix whose provenance contains target or future-context fields.

- [ ] **Step 4: Implement root-balanced logistic optimization**

Assign each row weight `1 / (number_of_roots * rows_in_its_root)`. Fit standardized coefficients by deterministic Newton/IRLS with L2 and a 50-iteration cap; return convergence diagnostics. Choose L2 from `{0.1,1,10}` on validation root-macro Brier, then AUC, then larger L2.

- [ ] **Step 5: Calibrate exact-risk threshold and fixed action logic**

Search unique safety scores on calibration. Select the lowest threshold satisfying accepted coverage maximum subject to equal-root empirical risk upper confidence bound and target risk. If no nonempty threshold is safe, exact action is disabled; this cannot count as a successful result because Gate C requires at least 50% exact coverage.

- [ ] **Step 6: Run tests and root-weight invariance check**

Duplicate every row in one root ten times and assert fitted coefficients/policy metrics are unchanged within tolerance. Run: `python -m unittest project.tests.test_gsad_shift_policy -v`.

---

### Task 6: Implement Metrics, Matched-Cost Comparisons, and Gates A–G

**Files:**
- Create: `project/experiments/gsad/metrics.py`
- Create: `project/tests/test_gsad_metrics.py`

**Interfaces:**
- Produces: `evaluate_predictions(frame: pd.DataFrame) -> MetricBundle`
- Produces: `cluster_bootstrap_difference(frame, metric_fn, group_col, n_boot, seed) -> Interval`
- Produces: `matched_cost_gain(candidate_curve, baseline_curve) -> float`
- Produces: `evaluate_gates(metrics, intervals, ablations) -> dict[str, GateResult]`.

- [ ] **Step 1: Write failing root-macro and bootstrap tests**

```python
def test_root_macro_is_not_row_weighted(self):
    frame = pd.DataFrame({"root": ["A"] * 100 + ["B"],
                          "correct": [True] * 100 + [False]})
    self.assertAlmostEqual(root_macro_mean(frame, "correct"), 0.5)

def test_gate_d_rejects_trivial_abstention(self):
    gates = evaluate_gates(metrics={"abstain_rate": 0.21}, intervals={}, ablations={})
    self.assertFalse(gates["D"].passed)
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_metrics -v`  
Expected: import failure for `metrics`.

- [ ] **Step 3: Implement row, root-macro, closed/open-label, and actor-macro metrics**

Metrics include Top-1, Hit@5, MRR, leaf coverage, tactic/descendant coverage, leaf-equivalent size, display-node count, exact coverage/accuracy, abstain rate, empty/full-set rate, NLL/Brier where full probabilities exist.

- [ ] **Step 4: Implement matched-cost interpolation and whole-group bootstrap**

Bootstrap roots/actors with replacement, include every row in each selected group, and preserve duplicate sampled groups by a bootstrap instance ID. Return point estimate, 2.5/97.5 percentiles, valid replicate count, and seed.

- [ ] **Step 5: Encode gates exactly from the specification**

Gate A OR B is required; C–G are all required. Return observed value, comparator, threshold, interval, and failure reason. A missing metric is a hard failure, never an implicit pass.

- [ ] **Step 6: Run tests and a synthetic all-abstain negative control**

Run: `python -m unittest project.tests.test_gsad_metrics -v`  
Expected: all tests pass; all-abstain and all-tactic fixtures fail D/E/C.

---

### Task 7: Deterministic Artifacts and Locked-Run Guard

**Files:**
- Create: `project/experiments/gsad/artifacts.py`
- Create: `project/tests/test_gsad_artifacts.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `write_manifest(path, inputs, config, split_audit) -> dict`
- Produces: `freeze_candidate(config, development_gates, path) -> FreezeToken`
- Produces: `claim_locked_run(token, results_dir) -> Path`.

- [ ] **Step 1: Write failing hash, canonical JSON, and no-overwrite tests**

```python
def test_locked_run_cannot_be_claimed_twice(self):
    token = fixture_freeze_token()
    claim_locked_run(token, self.results_dir)
    with self.assertRaisesRegex(FileExistsError, "locked evaluation already claimed"):
        claim_locked_run(token, self.results_dir)
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_artifacts -v`  
Expected: import failure for `artifacts`.

- [ ] **Step 3: Implement atomic, canonical writers**

Write to a sibling temporary file, `fsync`, then `os.replace`. Canonical JSON uses sorted keys, UTF-8, no NaN, and converts NumPy scalars to Python scalars.

- [ ] **Step 4: Implement freeze token and irreversible local claim**

Freeze token contains SHA-256 of candidate config, split root lists, source files, label vocab, ATT&CK JSON, development predictions, and gate results. `claim_locked_run` creates `LOCKED_EVALUATION_CLAIMED.json` with exclusive mode `x`; no CLI override exists.

- [ ] **Step 5: Run tests and verify deterministic hashes**

Run the same fixture twice in different temporary directories and assert identical manifest digests. Run: `python -m unittest project.tests.test_gsad_artifacts -v`.

---

### Task 8: Build Nested Root-OOF Development Runner

**Files:**
- Create: `project/experiments/gsad/run_development.py`
- Create: `project/tests/test_gsad_development_integration.py`

**Interfaces:**
- Consumes: the 133 non-test roots and ATT&CK/vocab paths.
- Produces: `run_development(config: DevelopmentConfig) -> DevelopmentResult`.
- Produces artifacts under `project/experiments/gsad/results/development/<candidate_id>/`.

- [ ] **Step 1: Write an integration test that spies on root access**

```python
def test_development_never_reads_locked_test_rows(self):
    result = run_development(tiny_config(), frames=synthetic_133_plus_20_roots())
    self.assertTrue(set(result.seen_roots).isdisjoint(TEST_ROOTS))
    self.assertEqual(set(result.oof_roots), set(result.development_roots))
```

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_development_integration -v`  
Expected: import failure for `run_development`.

- [ ] **Step 3: Implement deterministic outer/inner root folds**

Balance folds by row count with stable root-ID tie-break. For every outer fold, learn probability counts on inner fit, select model/cluster settings on inner validation, fit thresholds on inner calibration, and predict only the outer fold. Store a fold audit with zero overlaps for every role pair.

- [ ] **Step 4: Implement candidate registry and predetermined order**

Candidate IDs are `gsad_core`, `gsad_shift`, and `weighted_gsad`. The first run implements/evaluates `gsad_core`; `gsad_shift` is enabled only if the set component passes but Gate C fails; `weighted_gsad` is enabled only if measured source/target shift and effective sample size satisfy its preregistered eligibility check.

- [ ] **Step 5: Evaluate baselines, ablations, negative control, and gates**

Generate OOF predictions for unigram, Markov, n-gram, tactic-aware, LAC, APS, RAPS, global clustered CP, structured-only, confidence rejection, full GSAD, and required ablations. Run a label-permutation control with the same folds; any passing primary gate under permutation invalidates the implementation.

- [ ] **Step 6: Write the development artifacts and freeze only a passing winner**

Always write `run_manifest.json`, `data_audit.json`, `predictions.csv`, `metrics.csv`, `bootstrap_intervals.csv`, `gates.json`, and `iteration_summary.md`. Call `freeze_candidate` only when A-or-B and C–G pass and the negative control fails all positive gates.

- [ ] **Step 7: Run integration and all GSAD tests**

Run: `python -m unittest discover -s project/tests -p "test_gsad*.py" -v`  
Expected: all tests pass with no locked-test access.

---

### Task 9: Execute GSAD-Core and Apply the Failure Rule

**Files:**
- Read: `project/data_v2/core/sim_train_parent_min3.csv`
- Read: `project/data_v2/core/sim_val_parent_min3.csv`
- Read: `project/data_v2/core/sim_test_parent_min3.csv`
- Read: `project/data_v2/core/attack_parent_lookup_for_llm.csv`
- Read: `project/data/enterprise-attack-18.1.json`
- Read: `project/data_v2/core/rl_label_vocab.csv`
- Create: `project/experiments/gsad/results/development/gsad_core/*`

**Interfaces:**
- Consumes the Task 8 CLI and produces a freeze token only on a true pass.

- [ ] **Step 1: Run the complete test suite before real data**

Run: `python -m unittest discover -s project/tests -p "test_gsad*.py" -v`  
Expected: all tests pass.

- [ ] **Step 2: Run the frozen GSAD-Core development command**

Run: `python -m project.experiments.gsad.run_development --candidate gsad_core --seed 20260730 --bootstrap 2000`  
Expected: process exits 0 for a valid experiment regardless of scientific pass/fail; `gates.json` records the scientific result.

- [ ] **Step 3: Independently inspect leakage and fold evidence**

Assert from artifacts: 153 eligible roots after exclusions, 133 development roots, zero role overlaps, no blacklisted features, no CTID labels, correct hashes, and no locked-test root in predictions.

- [ ] **Step 4: Apply the preregistered branch without looking at locked test**

- If GSAD-Core passes A-or-B and C–G, freeze it and proceed to Task 11.
- If set coverage/efficiency passes but Gate C fails, execute Task 10 with `gsad_shift`.
- If A/B/E fail, write the short negative iteration record and return to literature/design; do not rescue it with shift features.
- If any integrity or permutation check fails, fix the implementation and rerun under the same candidate ID; this is a software correction, not scientific iteration.

---

### Task 10: Execute the Pre-Registered GSAD-Shift Variant if Eligible

**Files:**
- Modify: `project/experiments/gsad/run_development.py`
- Modify: `project/experiments/gsad/shift_policy.py`
- Modify: `project/tests/test_gsad_shift_policy.py`
- Create: `project/experiments/gsad/results/development/gsad_shift/*`

**Interfaces:**
- Consumes the same OOF folds and frozen GSAD-Core set component.
- Produces a new candidate result with only the five preregistered inference features.

- [ ] **Step 1: Add a failing test that core set outputs are bit-identical**

```python
def test_shift_variant_changes_only_action_policy(self):
    core, shift = run_core_and_shift_same_fixture()
    self.assertEqual(core.gamma_digest, shift.gamma_digest)
    self.assertEqual(core.structured_digest, shift.structured_digest)
```

- [ ] **Step 2: Run the test and confirm it fails before wiring the variant**

Run: `python -m unittest project.tests.test_gsad_shift_policy -v`  
Expected: failure because the shift candidate runner is not wired.

- [ ] **Step 3: Wire only the five-feature root-balanced policy**

Do not change probability model, clusters, alpha, DAG compressor, validation roots, or calibration roots. Record coefficients, convergence, score distribution, exact threshold, and per-root accepted risk.

- [ ] **Step 4: Run GSAD-Shift development OOF**

Run: `python -m project.experiments.gsad.run_development --candidate gsad_shift --seed 20260730 --bootstrap 2000`.

- [ ] **Step 5: Apply the same gates**

Freeze only if A-or-B and C–G pass. If it fails, retain only `iteration_summary.md` plus machine-readable gate evidence and begin a new literature/design cycle for Weighted GSAD; do not touch locked test.

---

### Task 11: One-Shot Locked SIM and External CTID Evaluation

**Files:**
- Create: `project/experiments/gsad/run_locked_evaluation.py`
- Create: `project/tests/test_gsad_locked_evaluation.py`
- Create: `project/experiments/gsad/results/final/<freeze_digest>/*`

**Interfaces:**
- Consumes: `FreezeToken`, frozen 93/20/20/20 protocol, frozen candidate config.
- Produces: immutable locked SIM and CTID predictions, metrics, intervals, gates, manifest.

- [ ] **Step 1: Write failing token and one-shot tests**

Reject a token whose config/data/source hash differs; reject a second evaluation; reject any invocation without a passing development gate digest.

- [ ] **Step 2: Verify red state**

Run: `python -m unittest project.tests.test_gsad_locked_evaluation -v`  
Expected: import failure for `run_locked_evaluation`.

- [ ] **Step 3: Implement final fit/validation/calibration reconstruction**

Fit probability counts on 93 roots, select the already-bounded settings on fixed 20 validation roots, fit clusters on validation, calibrate thresholds on fixed 20 calibration roots, and predict the fixed 20 test roots exactly once.

- [ ] **Step 4: Score closed-label, open-label, overall, and root-macro results**

Report all 1,592 rows, the 1,585 closed-label rows, and seven open-label rows. Run 2,000 root bootstraps and gates A–G without changing configuration.

- [ ] **Step 5: Score CTID only after SIM artifacts are sealed**

Normalize 10 plans to nine actors, use no CTID label during inference, report actor-macro metrics and nine-actor bootstrap, and label every table `exploratory_weak_label_external_ood=true`.

- [ ] **Step 6: Execute one-shot evaluation**

Run: `python -m project.experiments.gsad.run_locked_evaluation --freeze-token <development freeze_token.json>`  
Expected: first invocation writes final artifacts; any second invocation exits nonzero before reading locked labels.

- [ ] **Step 7: Apply final completion rule**

If locked SIM and CTID gates pass, proceed to Task 12. If either fails, record the honest negative result; do not modify and rerun against this locked test. Continue only after obtaining a new independent test source or a newly approved design with new untouched roots.

---

### Task 12: Independent Review, Reproduction, and Final Chinese Deliverables

**Files:**
- Create: `deliverables/GSAD_最终创新与实验报告.md`
- Create: `deliverables/GSAD_复现实验说明.md`
- Create: `deliverables/GSAD_迭代摘要.md`
- Create: `deliverables/GSAD_最终创新与实验报告.docx` only after Markdown evidence is final.

**Interfaces:**
- Consumes all final artifacts and source files.
- Produces the detailed final report requested by the user; failed rounds remain brief.

- [ ] **Step 1: Re-run all tests in a fresh process**

Run: `python -m unittest discover -s project/tests -p "test_gsad*.py" -v` and preserve stdout with timestamp and dependency versions in the final manifest.

- [ ] **Step 2: Reproduce development artifacts without locked evaluation**

Run the winning development command in a new result directory; assert deterministic config, folds, predictions, metrics, and gate digests.

- [ ] **Step 3: Perform a requirement-by-requirement evidence audit**

Check every design-spec item: data isolation, field blacklist, ATT&CK/version hash, baselines, ablations, permutation control, A–G, closed/open labels, CTID actor grouping, external caveats, and one-shot lock evidence. Missing evidence means incomplete.

- [ ] **Step 4: Write the detailed final report**

The report must separate: algorithm definition, nearest literature and exact difference, data audit, preregistered gates, primary results, confidence intervals, ablations, negative controls, external results, limitations, and the precise SCI Q2+ claim that evidence supports. It must explicitly state that publication tier is never guaranteed by an experiment.

- [ ] **Step 5: Write brief failed-iteration history and reproduction guide**

For each failed round record only hypothesis, changed component, data digest, primary gate numbers, failure reason, and next branch. The reproduction guide lists exact commands, expected files, hashes, Python/package versions, and no-overwrite behavior.

- [ ] **Step 6: Render and visually inspect the DOCX**

Use the documents workflow to render every page to images, inspect tables/equations/page breaks, fix layout defects, and deliver both Markdown and DOCX plus clickable code/result paths.

---

## Execution Decision

The user explicitly authorized autonomous execution and requested no further approval prompts. Execute this plan inline with `superpowers:executing-plans`, applying self-review checkpoints after each task. Do not dispatch subagents unless a later independent-review step explicitly requires one and remains within the user's authorization.
