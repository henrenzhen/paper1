# DR-VOM Validation and Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Falsify or support DR-VOM's domain/root-balancing mechanism with ablations, real-data perturbation tests, proper scoring rules, and a frozen fresh-source evaluation; if it fails, audit current primary literature and iterate to a stronger estimator.

**Architecture:** Keep the frozen four-domain result immutable. Add a separate validation module with pure metric/perturbation functions and a separate runner that fits four preregistered ablations on identical LODO folds. Store every run in a new non-overwriting directory with data/code hashes. Fresh-source evaluation uses an independent loader with hard structural assertions and is never used to tune the method.

**Tech Stack:** Python 3.12, NumPy, pandas, unittest, existing `InterpolatedNGram`, root/domain bootstrap utilities, canonical JSON manifests.

## Global Constraints

- Do not overwrite `project/experiments/gsad/results/external/dr_vom_full_lodo_final_seed20260730`.
- Primary four-domain metrics are equal-source macro Top-1, MRR, Hit@5, NLL, and multiclass Brier score.
- The four ablations are row-pooled VOM, root-balanced VOM, equal-domain row-balanced VOM, and DR-VOM.
- All ablations use vocabulary 184, `order=3` (unigram plus context lengths 1 and 2), `alpha=0.1`, and interpolation weights `0.2/0.3/0.5`.
- Primary success requires DR-VOM to exceed both single-balancing ablations in macro MRR, retain nonnegative Top-1/MRR on every held-out source, and satisfy both macro and worst-source Hit@5 non-inferiority above -1 percentage point.
- Fresh-source parser, mapping, model, and gates are frozen before the first metric is read.
- No shell-created source files; use `apply_patch`. No destructive or overwrite operations.
- This workspace is not a Git repository; replace commit checkpoints with SHA-256 manifests and review notes.

---

### Task 1: Pure probability and ranking diagnostics

**Files:**
- Create: `project/experiments/gsad/dr_vom_validation.py`
- Test: `project/tests/test_dr_vom_validation.py`

**Interfaces:**
- Consumes: probability matrices shaped `(n_rows, n_classes)`, ordered target IDs, and the fixed vocabulary.
- Produces: `prediction_diagnostics(probabilities, targets, vocabulary) -> pd.DataFrame`, `root_macro_diagnostics(frame, prefix) -> dict[str, float]`, and `expected_calibration_error(frame, prefix, n_bins=10) -> float`.

- [ ] **Step 1: Write failing tests for exact NLL, multiclass Brier, reciprocal rank, Hit@1/3/5/10, and ECE.**

```python
def test_prediction_diagnostics_computes_proper_scores_and_ranking():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    frame = prediction_diagnostics(probabilities, ["A", "A"], ["A", "B"])
    np.testing.assert_allclose(frame["nll"], [-np.log(0.8), -np.log(0.4)])
    np.testing.assert_allclose(frame["brier"], [0.08, 0.72])
    assert frame["hit1"].tolist() == [1.0, 0.0]
    assert frame["rr"].tolist() == [1.0, 0.5]
```

- [ ] **Step 2: Run the focused test and confirm it fails because the module/functions do not exist.**

Run: `python -m unittest project.tests.test_dr_vom_validation -v`

- [ ] **Step 3: Implement deterministic diagnostics with stable vocabulary-index tie breaking and probability validation.**

```python
def prediction_diagnostics(probabilities, targets, vocabulary):
    probs = np.asarray(probabilities, dtype=float)
    target_index = np.asarray([vocabulary.index(str(t)) for t in targets])
    true_probability = probs[np.arange(len(probs)), target_index]
    one_hot = np.eye(len(vocabulary), dtype=float)[target_index]
    ranks = np.argsort(-probs, axis=1, kind="stable")
    inverse = np.argsort(ranks, axis=1, kind="stable")
    target_rank = inverse[np.arange(len(probs)), target_index] + 1
    return pd.DataFrame({
        "nll": -np.log(np.clip(true_probability, 1e-15, 1.0)),
        "brier": np.square(probs - one_hot).sum(axis=1),
        "rr": 1.0 / target_rank,
        "hit1": (target_rank <= 1).astype(float),
        "hit3": (target_rank <= 3).astype(float),
        "hit5": (target_rank <= 5).astype(float),
        "hit10": (target_rank <= 10).astype(float),
        "confidence": probs.max(axis=1),
        "correct": (target_rank == 1).astype(float),
    })
```

- [ ] **Step 4: Run the focused tests and review edge cases: invalid row sums, unknown targets, non-finite probabilities, and empty inputs.**

---

### Task 2: Frozen ablation and perturbation runner

**Files:**
- Create: `project/experiments/gsad/run_dr_vom_validation.py`
- Modify: `project/tests/test_dr_vom_validation.py`

**Interfaces:**
- Consumes: `load_multisource_domains(project_root, vocab)` and the diagnostic functions from Task 1.
- Produces: `run_validation(project_root, output_dir, bootstrap=2000) -> tuple[pd.DataFrame, dict]` plus CSV/JSON/manifest artifacts.

- [ ] **Step 1: Add failing tests proving the four model factories differ only in aggregation.**

```python
def test_ablation_specs_freeze_identical_ngram_hyperparameters():
    specs = validation_model_specs(["A", "B"])
    assert list(specs) == ["row_pooled", "root_balanced", "domain_row_balanced", "dr_vom"]
    assert {m.order for m in specs.values()} == {3}
    assert {m.alpha for m in specs.values()} == {0.1}
```

- [ ] **Step 2: Add failing invariance tests.**

Within-root row replication must leave root-balanced and DR-VOM probabilities unchanged. Cloning every root in one source under new root IDs must leave equal-domain and DR-VOM probabilities unchanged while permitting row-pooled/root-balanced drift.

- [ ] **Step 3: Implement model fitting roles without changing `InterpolatedNGram`.**

```python
row_ids = pd.Series([f"row:{i}" for i in range(len(train))], index=train.index)
row_pooled.fit(train.prefix_ids, train.target, row_ids)
root_balanced.fit(train.prefix_ids, train.target, train.root)
domain_row_balanced.fit(train.prefix_ids, train.target, row_ids, domains=train.domain)
dr_vom.fit(train.prefix_ids, train.target, train.root, domains=train.domain)
```

- [ ] **Step 4: Implement real-data perturbations at factors 2, 5, and 10.**

For each LODO fold, choose each training source in turn. Produce two perturbed frames: duplicate every row while preserving root IDs (within-root repetition), and duplicate all source rows with deterministic cloned root IDs (source-root cloning). Score the unchanged held-out frame and record maximum absolute probability drift, macro Top-1/MRR drift, and model name.

- [ ] **Step 5: Implement four-domain ablation evaluation.**

Save paired row predictions for every model; compute root-macro Top-1/MRR/Hit@k/NLL/Brier and row ECE; bootstrap DR-VOM differences against each ablation using the same resampled roots.

- [ ] **Step 6: Write immutable artifacts.**

Output `domain_ablation_metrics.csv`, `aggregate_ablation_intervals.csv`, `probability_metrics.csv`, `stress_metrics.csv`, `predictions.csv`, `summary.json`, and `run_manifest.json`. Create the output directory with `exist_ok=False`.

- [ ] **Step 7: Run focused and full tests.**

Run: `python -m unittest project.tests.test_dr_vom_validation -v`

Run: `python -m unittest discover -s project/tests -v`

---

### Task 3: Execute and review the current four-domain validation

**Files:**
- Create: `project/experiments/gsad/results/validation/dr_vom_validation_seed20260730/*`
- Modify: `docs/research/gsad-iteration-log.md`

**Interfaces:**
- Consumes: the frozen validation runner and existing four source frames.
- Produces: a gate decision that either advances DR-VOM to fresh-source confirmation or triggers literature-backed redesign.

- [ ] **Step 1: Run once into the fixed new output directory.**

Run: `python -m project.experiments.gsad.run_dr_vom_validation --output-dir project/experiments/gsad/results/validation/dr_vom_validation_seed20260730`

- [ ] **Step 2: Review gates without changing them.**

Pass only if: all four source Top-1/MRR differences versus root-balanced are nonnegative; macro MRR exceeds root-balanced, domain-row-balanced, and row-pooled; both macro and worst-source Hit@5 differences versus the strongest baseline are greater than -0.01; NLL or Brier improves against the strongest baseline; and exact invariance checks pass at factors 2/5/10.

- [ ] **Step 3: Record a concise iteration decision and hashes.**

If any primary gate fails, mark DR-VOM as not validated and proceed to Task 5 after completing the diagnostic slices. Do not weaken a failed gate.

---

### Task 4: Frozen Scattered Spider 2025 confirmation

**Files:**
- Create: `project/experiments/gsad/scattered_spider_dataset.py`
- Create: `project/tests/test_scattered_spider_dataset.py`
- Create: `project/experiments/gsad/results/external/scattered_spider_2025_frozen/*`

**Interfaces:**
- Consumes: the exact file at commit `5594518da57ba41faaaaa99b3e0078d29504b033`, path `Enterprise/scattered_spider/Emulation_Plan/Scattered_Spider_Scenario.md`.
- Produces: exactly 67 ordered raw technique rows and 66 adjacent edges, plus a source manifest.

- [ ] **Step 1: Retrieve the fixed raw file and license without accessing a moving branch.**

- [ ] **Step 2: Write failing parser tests for Step counts `7/12/12/11/7/9/9`, 67 nodes, 66 edges, six cross-Step edges, strict technique-ID syntax, and preservation of duplicates/self-loops.**

- [ ] **Step 3: Implement the parser with hard failures for any contract mismatch.**

- [ ] **Step 4: Freeze source bytes, hashes, mapping audit, model hashes, and pass/fail gates before scoring.**

- [ ] **Step 5: Score DR-VOM and every ablation exactly once.**

Report the result regardless of sign. Do not retune on this source. Because 66 edges are small, treat it as directional confirmation, not standalone significance evidence.

---

### Task 5: Literature-backed redesign when a primary gate fails

**Files:**
- Create or modify only after the failed mechanism is identified: `docs/research/gsad-iteration-log.md`, one focused model module, one focused test module, and one new non-overwriting result directory.

**Interfaces:**
- Consumes: the failed gate and its diagnostic slices.
- Produces: one mechanism-matched candidate, not a broad hyperparameter search.

- [ ] **Step 1: Search primary sources and official journal pages for the failed mechanism.**

If Hit@5/calibration fails, audit calibrated Markov/context-tree and constrained probability-pooling work. If single-domain contexts fail, audit hierarchical/mixed-effects context trees and distributionally robust source weighting. If both balancing components are redundant, simplify the estimator instead of adding complexity.

- [ ] **Step 2: State the borrowed result, non-overlap, new estimator, and falsification gate before coding.**

- [ ] **Step 3: Add one failing unit test for the claimed mechanism, implement the minimum estimator, and run nested source-only selection when any parameter is learned.**

- [ ] **Step 4: Repeat Tasks 2 and 3 in a new result directory.**

Stop only when a candidate passes the frozen mechanism, probability, Top-k, and source-direction gates, or when evidence shows the current data cannot support a defensible claim.

---

### Task 6: Final evidence document and verification

**Files:**
- Modify: `docs/research/2026-07-30-zds-desk-reject-and-dr-vom-final-report.md`
- Modify: `deliverables/ZDS论文拒稿复盘与DR-VOM实验报告_最终版.docx`

- [ ] **Step 1: Update the report with exact new results, failed gates, literature boundary, data hashes, and the final decision.**

- [ ] **Step 2: Rebuild the DOCX, run the document-builder test and accessibility audit, export with Word, and visually inspect all affected pages.**

- [ ] **Step 3: Run the full project test suite and verify every reported number directly from final CSV/JSON artifacts.**

## Self-Review

- Spec coverage: mechanism, ablations, stress invariance, probability quality, Top-k safety, fresh source, failure-triggered literature iteration, and final documentation all have explicit tasks.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: diagnostic and runner interfaces use probability matrices, ordered targets, the fixed vocabulary, pandas frames, and canonical JSON artifacts consistently.
- Scope: only DR-VOM validation and evidence-triggered replacement are included; unrelated model or repository refactors are excluded.
