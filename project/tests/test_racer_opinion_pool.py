import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.opinion_pool import pool_probabilities  # noqa: E402


class OpinionPoolTests(unittest.TestCase):
    def test_linear_pool_is_convex_and_normalized(self):
        left = np.asarray([[0.8, 0.2]])
        right = np.asarray([[0.2, 0.8]])
        pooled = pool_probabilities(left, right, weight=0.75, kind="linear")
        np.testing.assert_allclose(pooled, np.asarray([[0.65, 0.35]]))
        np.testing.assert_allclose(pooled.sum(axis=1), np.ones(1))

    def test_log_pool_is_normalized_geometric_opinion(self):
        left = np.asarray([[0.8, 0.2]])
        right = np.asarray([[0.5, 0.5]])
        pooled = pool_probabilities(left, right, weight=0.5, kind="log")
        expected = np.sqrt(left * right)
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(pooled, expected)

    def test_endpoints_recover_corresponding_expert(self):
        left = np.asarray([[0.7, 0.3]])
        right = np.asarray([[0.4, 0.6]])
        for kind in ("linear", "log"):
            np.testing.assert_allclose(
                pool_probabilities(left, right, weight=1.0, kind=kind), left
            )
            np.testing.assert_allclose(
                pool_probabilities(left, right, weight=0.0, kind=kind), right
            )

    def test_invalid_geometry_weight_and_probabilities_are_rejected(self):
        valid = np.asarray([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            pool_probabilities(valid, valid, weight=0.5, kind="learned_gate")
        with self.assertRaises(ValueError):
            pool_probabilities(valid, valid, weight=1.1, kind="linear")
        with self.assertRaises(ValueError):
            pool_probabilities(valid, np.asarray([[0.8, 0.3]]), weight=0.5, kind="log")


if __name__ == "__main__":
    unittest.main()
