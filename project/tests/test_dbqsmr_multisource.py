import unittest

import pandas as pd

from project.experiments.gsad.run_dbqsmr_nested_lodo import select_robust_config
from project.experiments.gsad.run_adaptive_power_ngram_lodo import select_power
from project.experiments.gsad.run_multiresolution_vom_lodo import mix_raw_residual

import numpy as np


class DBQSMRNestedLodoTests(unittest.TestCase):
    def test_selector_prefers_nonnegative_worst_domain_over_higher_mean(self):
        rows = pd.DataFrame(
            [
                {
                    "config_id": "fragile",
                    "domain": "a",
                    "top1_gain_pp": 8.0,
                    "mrr_gain": 0.08,
                },
                {
                    "config_id": "fragile",
                    "domain": "b",
                    "top1_gain_pp": -1.0,
                    "mrr_gain": -0.01,
                },
                {
                    "config_id": "robust",
                    "domain": "a",
                    "top1_gain_pp": 1.0,
                    "mrr_gain": 0.01,
                },
                {
                    "config_id": "robust",
                    "domain": "b",
                    "top1_gain_pp": 0.2,
                    "mrr_gain": 0.002,
                },
            ]
        )

        selected = select_robust_config(rows)

        self.assertEqual(selected, "robust")

    def test_selector_uses_deterministic_lexicographic_tie_break(self):
        rows = pd.DataFrame(
            [
                {
                    "config_id": config,
                    "domain": domain,
                    "top1_gain_pp": 1.0,
                    "mrr_gain": 0.01,
                }
                for config in ("b", "a")
                for domain in ("x", "y")
            ]
        )

        self.assertEqual(select_robust_config(rows), "a")


class AdaptivePowerNGramTests(unittest.TestCase):
    def test_selector_requires_nonnegative_inner_domains_before_mean_gain(self):
        rows = pd.DataFrame(
            [
                {"power": 0.0, "domain": "a", "top1_gain_pp": 5.0, "mrr_gain": 0.05},
                {"power": 0.0, "domain": "b", "top1_gain_pp": -0.1, "mrr_gain": 0.03},
                {"power": 0.1, "domain": "a", "top1_gain_pp": 1.0, "mrr_gain": 0.01},
                {"power": 0.1, "domain": "b", "top1_gain_pp": 0.2, "mrr_gain": 0.002},
            ]
        )

        self.assertEqual(select_power(rows), 0.1)


class MultiResolutionVOMTests(unittest.TestCase):
    def test_raw_residual_only_changes_supported_subtechnique_rows(self):
        parent = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        raw = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        metadata = pd.DataFrame({"context_root_support": [10, 10, 0]})

        mixed, weights = mix_raw_residual(
            parent,
            raw,
            parent_prefixes=[("T1",), ("T1",), ("T1",)],
            raw_prefixes=[("T1.001",), ("T1",), ("T1.001",)],
            raw_metadata=metadata,
            maximum_weight=0.5,
            kappa=5.0,
        )

        self.assertGreater(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[2], 0.0)
        self.assertGreater(mixed[0, 1], parent[0, 1])
        np.testing.assert_allclose(mixed[1:], parent[1:])


if __name__ == "__main__":
    unittest.main()
