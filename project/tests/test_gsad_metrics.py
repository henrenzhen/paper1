import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.metrics import (  # noqa: E402
    Interval,
    cluster_bootstrap_difference,
    domain_root_bootstrap_difference,
    evaluate_gates,
    evaluate_predictions,
    matched_cost_gain,
    root_macro_mean,
)


class RootMetricTests(unittest.TestCase):
    def test_exact_accuracy_scores_the_emitted_exact_label(self):
        frame = pd.DataFrame(
            {
                "root": ["A"],
                "target": ["T2"],
                "top1_pred": ["T1"],
                "exact_label": ["T2"],
                "top5": [("T1", "T2")],
                "gamma": [frozenset({"T2"})],
                "descendants": [frozenset({"T2"})],
                "action_kind": ["exact"],
                "leaf_equivalent_size": [1],
                "display_node_count": [1],
            }
        )
        bundle = evaluate_predictions(frame)
        self.assertAlmostEqual(bundle.row["top1_accuracy"], 0.0)
        self.assertAlmostEqual(bundle.row["exact_accuracy"], 1.0)

    def test_root_macro_is_not_row_weighted(self):
        frame = pd.DataFrame(
            {"root": ["A"] * 100 + ["B"], "correct": [True] * 100 + [False]}
        )
        self.assertAlmostEqual(root_macro_mean(frame, "correct"), 0.5)

    def test_prediction_metrics_distinguish_exact_dag_and_abstain(self):
        frame = pd.DataFrame(
            {
                "root": ["A", "A", "B"],
                "target": ["T1", "T2", "T3"],
                "top1_pred": ["T1", "T9", "T8"],
                "top5": [("T1",), ("T2", "T9"), ("T8",)],
                "gamma": [frozenset({"T1"}), frozenset({"T2", "T4"}), frozenset({"T8"})],
                "descendants": [frozenset({"T1"}), frozenset({"T2", "T4"}), frozenset()],
                "action_kind": ["exact", "dag", "abstain"],
                "leaf_equivalent_size": [1, 2, 0],
                "display_node_count": [1, 1, 0],
                "fit_seen": [True, True, False],
                "vocab_size": [10, 10, 10],
            }
        )
        bundle = evaluate_predictions(frame)
        self.assertAlmostEqual(bundle.row["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(bundle.row["hit_at_5"], 2 / 3)
        self.assertAlmostEqual(bundle.row["leaf_coverage"], 2 / 3)
        self.assertAlmostEqual(bundle.row["descendant_coverage"], 2 / 3)
        self.assertAlmostEqual(bundle.row["exact_coverage"], 1 / 3)
        self.assertAlmostEqual(bundle.row["exact_accuracy"], 1.0)
        self.assertAlmostEqual(bundle.row["abstain_rate"], 1 / 3)
        self.assertEqual(bundle.slices["open_label"]["n"], 1)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_roots_and_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "root": ["A", "A", "B", "B", "C", "C"],
                "candidate": [1, 1, 0, 0, 1, 1],
                "baseline": [0, 0, 0, 0, 0, 0],
            }
        )

        def difference(sample):
            return root_macro_mean(sample.assign(delta=sample.candidate - sample.baseline), "delta")

        first = cluster_bootstrap_difference(
            frame, difference, group_col="root", n_boot=200, seed=9
        )
        second = cluster_bootstrap_difference(
            frame, difference, group_col="root", n_boot=200, seed=9
        )
        self.assertEqual(first, second)
        self.assertEqual(first.valid_replicates, 200)
        self.assertGreater(first.point, 0)

    def test_domain_root_bootstrap_equal_weights_domains_and_roots(self):
        frame = pd.DataFrame(
            {
                "domain": ["large"] * 100 + ["small"],
                "root": ["large:r1"] * 100 + ["small:r1"],
                "candidate": [1.0] * 100 + [0.0],
                "baseline": [0.0] * 101,
            }
        )

        interval = domain_root_bootstrap_difference(
            frame,
            candidate_col="candidate",
            baseline_col="baseline",
            domain_col="domain",
            root_col="root",
            n_boot=50,
            seed=13,
        )

        self.assertAlmostEqual(interval.point, 0.5)
        self.assertAlmostEqual(interval.lower, 0.5)
        self.assertAlmostEqual(interval.upper, 0.5)


class MatchedCostTests(unittest.TestCase):
    def test_baseline_is_interpolated_at_candidate_cost(self):
        candidate = pd.DataFrame({"cost": [2.0], "value": [0.9]})
        baseline = pd.DataFrame({"cost": [1.0, 3.0], "value": [0.7, 0.8]})
        self.assertAlmostEqual(matched_cost_gain(candidate, baseline), 0.15)


def passing_metrics():
    return {
        "coverage_gain_pp_matched_size": 6.0,
        "exact_output_gain_relative": 0.12,
        "exact_coverage": 0.55,
        "exact_accuracy_gain_pp": 6.0,
        "abstain_rate": 0.10,
        "mean_leaf_size": 4.0,
        "baseline_mean_leaf_size": 4.5,
        "full_set_rate": 0.01,
        "baseline_full_set_rate": 0.02,
        "row_coverage": 0.90,
        "root_macro_coverage": 0.89,
        "ablation_wins": 2,
    }


def passing_intervals():
    return {
        "coverage_gain_pp_matched_size": Interval(6.0, 1.0, 10.0, 2000),
        "exact_output_gain_relative": Interval(0.12, 0.02, 0.20, 2000),
        "exact_accuracy_gain_pp": Interval(6.0, 1.0, 11.0, 2000),
    }


class GateTests(unittest.TestCase):
    def test_all_preregistered_gates_pass_for_valid_fixture(self):
        gates = evaluate_gates(passing_metrics(), passing_intervals(), ablations={})
        self.assertTrue(gates["PRIMARY"].passed)
        self.assertTrue(all(gates[name].passed for name in "ABCDEFG"))

    def test_gate_d_rejects_trivial_abstention(self):
        metrics = passing_metrics()
        metrics["abstain_rate"] = 0.21
        gates = evaluate_gates(metrics, passing_intervals(), ablations={})
        self.assertFalse(gates["D"].passed)
        self.assertFalse(gates["PRIMARY"].passed)

    def test_all_abstain_fixture_fails_nontrivial_gates(self):
        metrics = passing_metrics()
        metrics.update(
            {
                "exact_coverage": 0.0,
                "exact_accuracy_gain_pp": 0.0,
                "abstain_rate": 1.0,
                "mean_leaf_size": 0.0,
            }
        )
        intervals = passing_intervals()
        intervals["exact_accuracy_gain_pp"] = Interval(0.0, -1.0, 1.0, 2000)
        gates = evaluate_gates(metrics, intervals, ablations={})
        self.assertFalse(gates["C"].passed)
        self.assertFalse(gates["D"].passed)
        self.assertFalse(gates["PRIMARY"].passed)

    def test_missing_metric_is_hard_failure(self):
        metrics = passing_metrics()
        del metrics["row_coverage"]
        gates = evaluate_gates(metrics, passing_intervals(), ablations={})
        self.assertFalse(gates["F"].passed)
        self.assertIn("missing", gates["F"].reason)


if __name__ == "__main__":
    unittest.main()
