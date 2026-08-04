import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.attack_dag import AttackDAG  # noqa: E402
from experiments.gsad.conformal import (  # noqa: E402
    aps_score,
    deterministic_uniform,
    finite_sample_quantile,
    fit_clustered_aps,
    fit_graph_clusters,
)


class ApsPrimitiveTests(unittest.TestCase):
    def test_true_score_is_mass_before_label_plus_randomized_mass(self):
        probability = np.array([0.6, 0.3, 0.1])
        self.assertAlmostEqual(aps_score(probability, label_index=1, u=0.5), 0.75)

    def test_ties_use_label_index_as_stable_order(self):
        probability = np.array([0.5, 0.5])
        self.assertAlmostEqual(aps_score(probability, label_index=0, u=0.0), 0.0)
        self.assertAlmostEqual(aps_score(probability, label_index=1, u=0.0), 0.5)

    def test_finite_sample_quantile_uses_ceil_n_plus_one_correction(self):
        scores = np.arange(1, 10, dtype=float) / 10
        # ceil((9 + 1) * .8) - 1 = 7, so the selected value is .8.
        self.assertAlmostEqual(finite_sample_quantile(scores, alpha=0.2), 0.8)

    def test_sample_id_randomization_is_reproducible(self):
        first = deterministic_uniform("sample-1", "T1001", seed=7)
        second = deterministic_uniform("sample-1", "T1001", seed=7)
        other = deterministic_uniform("sample-2", "T1001", seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)


class GraphClusterTests(unittest.TestCase):
    def setUp(self):
        self.vocab = ("T1", "T2", "T3", "T4")
        self.dag = AttackDAG(
            {"TA1": {"T1", "T2"}, "TA2": {"T3"}, "TA3": {"T4"}},
            vocab=self.vocab,
        )
        self.probs = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.2, 0.6, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                [0.1, 0.1, 0.7, 0.1],
            ]
        )
        self.targets = ("T1", "T2", "T2", "T3")

    def test_rare_labels_merge_only_when_graph_adjacent(self):
        clusters = fit_graph_clusters(
            self.probs,
            self.targets,
            fit_counts={"T1": 2, "T2": 2, "T3": 5, "T4": 0},
            dag=self.dag,
            vocab=self.vocab,
            min_support=3,
        )
        self.assertEqual(clusters.cluster_for("T1"), clusters.cluster_for("T2"))
        self.assertNotEqual(clusters.cluster_for("T1"), clusters.cluster_for("T3"))
        self.assertNotEqual(clusters.cluster_for("T3"), clusters.cluster_for("T4"))

    def test_cluster_digest_does_not_change_during_calibration(self):
        clusters = fit_graph_clusters(
            self.probs,
            self.targets,
            fit_counts={label: 2 for label in self.vocab},
            dag=self.dag,
            vocab=self.vocab,
            min_support=2,
        )
        before = clusters.digest()
        fit_clustered_aps(
            cal_probs=self.probs,
            cal_targets=self.targets,
            clusters=clusters,
            alpha=0.25,
            sample_ids=("c1", "c2", "c3", "c4"),
            min_calibration_support=2,
            seed=11,
        )
        self.assertEqual(before, clusters.digest())


class ClusteredApsTests(unittest.TestCase):
    def setUp(self):
        self.vocab = ("T1", "T2", "T3")
        self.dag = AttackDAG(
            {"TA1": {"T1", "T2"}, "TA2": {"T3"}}, vocab=self.vocab
        )
        validation_probs = np.array(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        )
        self.clusters = fit_graph_clusters(
            validation_probs,
            ("T1", "T2", "T3"),
            fit_counts={"T1": 10, "T2": 10, "T3": 10},
            dag=self.dag,
            vocab=self.vocab,
            min_support=1,
        )

    def test_sparse_cluster_uses_global_threshold(self):
        cal_probs = np.array(
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]]
        )
        predictor = fit_clustered_aps(
            cal_probs,
            ("T1", "T1", "T3"),
            self.clusters,
            alpha=0.2,
            sample_ids=("c1", "c2", "c3"),
            min_calibration_support=2,
            seed=3,
        )
        t3_cluster = self.clusters.cluster_for("T3")
        self.assertTrue(predictor.audit.loc[t3_cluster, "fallback"])
        self.assertEqual(predictor.thresholds[t3_cluster], predictor.global_threshold)

    def test_prediction_sets_are_nonempty_and_use_only_vocabulary(self):
        cal_probs = np.array(
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        )
        predictor = fit_clustered_aps(
            cal_probs,
            ("T1", "T2", "T3"),
            self.clusters,
            alpha=0.2,
            sample_ids=("c1", "c2", "c3"),
            min_calibration_support=1,
            seed=3,
        )
        sets = predictor.predict_sets(
            np.array([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33]]),
            sample_ids=("x1", "x2"),
        )
        self.assertEqual(len(sets), 2)
        self.assertTrue(all(prediction_set for prediction_set in sets))
        self.assertTrue(all(prediction_set.issubset(set(self.vocab)) for prediction_set in sets))

    def test_mismatched_sample_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_clustered_aps(
                np.array([[0.7, 0.2, 0.1]]),
                ("T1",),
                self.clusters,
                alpha=0.2,
                sample_ids=(),
            )


if __name__ == "__main__":
    unittest.main()
