import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.attack_dag import StructuredSet  # noqa: E402
from experiments.gsad.shift_policy import (  # noqa: E402
    build_inference_features,
    calibrate_exact_threshold,
    choose_action,
    fit_feature_reference,
    fit_root_balanced_logistic,
)


def structured(nodes, descendants):
    return StructuredSet(
        nodes=frozenset(nodes),
        descendants=frozenset(descendants),
        leaf_equivalent_size=len(descendants),
        objective=float(len(descendants) + len(nodes)),
        coverage_preserved=True,
    )


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.probs = np.array(
            [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [0.9, 0.05, 0.05]]
        )
        self.metadata = pd.DataFrame({"used_order": [2, 0, 1]})
        self.prefixes = [("T1", "T2"), ("T3",), ("T1", "T1")]

    def test_exactly_five_inference_features_are_built(self):
        reference = fit_feature_reference(self.probs, self.metadata, self.prefixes)
        features = build_inference_features(
            self.probs, self.metadata, self.prefixes, reference
        )
        self.assertEqual(features.values.shape, (3, 5))
        self.assertEqual(
            features.names,
            (
                "entropy",
                "margin",
                "transition_surprise",
                "backoff_signal",
                "fit_distance",
            ),
        )

    def test_future_context_metadata_is_rejected(self):
        bad = self.metadata.assign(true_label=["T1", "T2", "T3"])
        with self.assertRaisesRegex(ValueError, "future"):
            fit_feature_reference(self.probs, bad, self.prefixes)


class RootBalancedLogisticTests(unittest.TestCase):
    def test_model_learns_correctness_signal(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        correct = np.array([False, False, True, True])
        roots = np.array(["A", "B", "C", "D"])
        model = fit_root_balanced_logistic(features, correct, roots, l2=0.1)
        scores = model.predict_proba(features)
        self.assertGreater(float(scores[3]), float(scores[0]))

    def test_duplicating_all_rows_of_one_root_does_not_change_coefficients(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        correct = np.array([False, False, True, True])
        roots = np.array(["A", "A", "B", "B"])
        base = fit_root_balanced_logistic(features, correct, roots, l2=1.0)
        duplicated = fit_root_balanced_logistic(
            np.concatenate([np.repeat(features[:2], 10, axis=0), features[2:]]),
            np.concatenate([np.repeat(correct[:2], 10), correct[2:]]),
            np.concatenate([np.repeat(roots[:2], 10), roots[2:]]),
            l2=1.0,
        )
        np.testing.assert_allclose(base.coefficients, duplicated.coefficients, atol=1e-8)
        self.assertAlmostEqual(base.intercept, duplicated.intercept, places=8)


class ThresholdTests(unittest.TestCase):
    def test_threshold_prefers_maximum_coverage_that_meets_risk_bound(self):
        scores = np.array([0.95, 0.9, 0.8, 0.7, 0.2, 0.1])
        correct = np.array([True, True, True, False, False, False])
        roots = np.array(["A", "B", "C", "D", "E", "F"])
        threshold = calibrate_exact_threshold(
            scores, correct, roots, target_risk=0.2, confidence_z=0.0
        )
        self.assertTrue(threshold.enabled)
        self.assertEqual(threshold.threshold, 0.8)
        self.assertAlmostEqual(threshold.coverage, 0.5)
        self.assertEqual(threshold.empirical_risk, 0.0)

    def test_no_safe_nonempty_region_disables_exact_action(self):
        threshold = calibrate_exact_threshold(
            np.array([0.9, 0.8]),
            np.array([False, False]),
            np.array(["A", "B"]),
            target_risk=0.1,
            confidence_z=0.0,
        )
        self.assertFalse(threshold.enabled)
        self.assertTrue(np.isinf(threshold.threshold))


class ActionTests(unittest.TestCase):
    def test_singleton_safe_set_outputs_exact(self):
        action = choose_action(
            gamma=frozenset({"T1"}),
            structured=structured({"T1"}, {"T1"}),
            safety_score=0.9,
            threshold=0.8,
            max_leaf_size=20,
            support_ok=True,
        )
        self.assertEqual(action.kind, "exact")
        self.assertEqual(action.nodes, frozenset({"T1"}))

    def test_non_singleton_supported_set_outputs_dag(self):
        action = choose_action(
            gamma=frozenset({"T1", "T2"}),
            structured=structured({"TA1"}, {"T1", "T2"}),
            safety_score=0.9,
            threshold=0.8,
            max_leaf_size=20,
            support_ok=True,
        )
        self.assertEqual(action.kind, "dag")

    def test_wide_or_unsupported_set_abstains(self):
        wide = structured({"TA1"}, {f"T{i}" for i in range(30)})
        for support_ok in [False, True]:
            with self.subTest(support_ok=support_ok):
                action = choose_action(
                    gamma=frozenset({"T1", "T2"}),
                    structured=wide,
                    safety_score=0.2,
                    threshold=0.8,
                    max_leaf_size=20,
                    support_ok=support_ok,
                )
                self.assertEqual(action.kind, "abstain")


if __name__ == "__main__":
    unittest.main()
