# PR-HR Small-Scale Feasibility Experiment Implementation Plan

> **For Codex:** Execute task by task with red-green-refactor checks. This is a diagnostic experiment, not a paper result.

**Goal:** Test whether the proposed pairwise-regret routing direction has measurable signal in the existing 2,107 aligned SIM test-prefix artifacts without training the original deep models again.

**Architecture:** Align existing GRU Top-5 probabilities with raw LLM Top-5 candidates by `(sequence_id, prefix_len, true_label)` after normalizing state delimiters. Evaluate a fixed-feature, NumPy-only linear pairwise candidate ranker with five-fold cross-fitting grouped by SIM root. Separately evaluate candidate complementarity and confidence-based selective prediction. Estimate uncertainty by bootstrapping whole SIM roots, never individual nested prefixes.

**Tech Stack:** Python 3.12 standard library, NumPy, pandas, `unittest`.

**Pre-registered interpretation gates:**

- Gate A — complementarity exists: the GRU+LLM Top-5 union must improve GRU Top-5 by at least 0.5 percentage points and its 95% root-bootstrap lower bound must be above zero.
- Gate B — usable routing signal: out-of-fold pairwise ranking Top-1 must improve GRU Top-1 by at least 1.0 percentage point and its 95% root-bootstrap lower bound must be above zero.
- Gate C — selective signal: at 80% coverage, accepted-sample accuracy must exceed full-coverage accuracy by at least 5.0 percentage points, with a positive 95% root-bootstrap lower bound.
- Any failed gate is reported as a negative result. No threshold or feature tuning on held-out folds is allowed.

---

## Task 1: Define contracts with failing tests

**Files:**

- Create: `project/tests/test_pr_hr_small_experiment.py`
- Test: `project/tests/test_pr_hr_small_experiment.py`

1. Add a test showing shuffled LLM rows align exactly to GRU rows and mismatched states raise an error.
2. Add a test showing all samples sharing a SIM root remain in one fold.
3. Add a synthetic test showing the pairwise linear ranker gives the larger score to a candidate whose feature consistently identifies the correct label.
4. Add a test showing risk-coverage evaluation accepts highest-confidence samples first.
5. Run `python -m unittest project.tests.test_pr_hr_small_experiment -v` and confirm failure because the implementation module does not exist.

## Task 2: Implement the experiment core

**Files:**

- Create: `project/experiments/pr_hr_feasibility/__init__.py`
- Create: `project/experiments/pr_hr_feasibility/pr_hr_small_experiment.py`
- Modify: `project/tests/test_pr_hr_small_experiment.py`

1. Implement strict parsing, state normalization, key-based alignment, and duplicate/mismatch checks.
2. Implement deterministic balanced root-group folds.
3. Implement fixed candidate features: GRU probability/rank, LLM reciprocal rank, model agreement, prefix length, GRU margin/entropy, candidate repetition in the prefix, and transition priors learned only from non-test SIM roots.
4. Implement a regularized NumPy linear pairwise ranker with a fixed regularization constant.
5. Implement out-of-fold ranking, risk-coverage tables, and root-level bootstrap confidence intervals.
6. Run unit tests until green.

## Task 3: Run the real-data diagnostic

**Files:**

- Read: `project/data/rl_v2_test_predictions_top5.csv`
- Read: `project/data/sim_test_llm_cot.csv`
- Read: `project/data/sim_train_parent_min3.csv`
- Create: `project/experiments/pr_hr_feasibility/results/aligned_oof_predictions.csv`
- Create: `project/experiments/pr_hr_feasibility/results/metric_summary.csv`
- Create: `project/experiments/pr_hr_feasibility/results/risk_coverage.csv`
- Create: `project/experiments/pr_hr_feasibility/results/bootstrap_intervals.csv`
- Create: `project/experiments/pr_hr_feasibility/results/run_manifest.json`

1. Verify 2,107/2,107 records align and normalized states match.
2. Exclude every SIM root present in the diagnostic test pool when estimating transition priors from the original training CSV.
3. Generate five-fold root-grouped out-of-fold predictions using fixed features and hyperparameters.
4. Calculate GRU, LLM, reciprocal-rank fusion, pairwise-ranker, candidate-union oracle, and selective-prediction metrics.
5. Bootstrap 2,000 times over 73 SIM roots using a fixed random seed.
6. Write machine-readable results and a manifest containing input hashes, parameters, and leakage checks.

## Task 4: Review and report

**Files:**

- Create: `deliverables/PR-HR_小规模可行性实验报告.md`

1. Re-run unit tests and the real-data experiment from a clean command.
2. Inspect fold disjointness, alignment assertions, candidate coverage, fold-level variance, and all three pre-registered gates.
3. Review whether any feature uses the true label or information unavailable at inference time.
4. State separately: what is supported, what failed, what remains unproven, and the next decisive experiment needed for an SCI Q2+ submission.

