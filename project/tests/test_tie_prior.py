import unittest

import numpy as np

from project.experiments.gsad.tie_prior import (
    ReportCooccurrencePrior,
    support_adaptive_prior_pool,
)


class ReportCooccurrencePriorTests(unittest.TestCase):
    def test_each_report_contributes_unit_mass_independent_of_frequency(self):
        reports_a = [
            {"mitre_techniques": {"A": 100, "B": 1}},
            {"mitre_techniques": {"A": 1, "C": 1}},
        ]
        reports_b = [
            {"mitre_techniques": {"A": 1, "B": 1}},
            {"mitre_techniques": {"A": 1, "C": 1}},
        ]
        first = ReportCooccurrencePrior(("A", "B", "C"), alpha=0.1).fit_reports(
            reports_a
        )
        second = ReportCooccurrencePrior(("A", "B", "C"), alpha=0.1).fit_reports(
            reports_b
        )
        first_probability, _ = first.predict_proba_with_meta([("A",)])
        second_probability, _ = second.predict_proba_with_meta([("A",)])
        np.testing.assert_allclose(first_probability, second_probability)

    def test_probabilities_are_normalized_and_unseen_context_uses_global_prior(self):
        model = ReportCooccurrencePrior(("A", "B", "C"), alpha=0.1).fit_reports(
            [
                {"mitre_techniques": {"A": 1, "B": 1}},
                {"mitre_techniques": {"B": 1, "C": 1}},
            ]
        )
        probabilities, meta = model.predict_proba_with_meta([("A",), ("Z",)])
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertGreater(int(meta.iloc[0]["report_support"]), 0)
        self.assertEqual(int(meta.iloc[1]["report_support"]), 0)
        np.testing.assert_allclose(probabilities[1], model.global_prior_)


class SupportAdaptivePoolTests(unittest.TestCase):
    def test_external_prior_weight_decreases_with_local_root_support(self):
        local = np.asarray([[0.8, 0.2], [0.8, 0.2]])
        prior = np.asarray([[0.2, 0.8], [0.2, 0.8]])
        pooled, weights = support_adaptive_prior_pool(
            local,
            prior,
            local_support=np.asarray([1.0, 100.0]),
            prior_available=np.asarray([True, True]),
            strength=1.0,
            kappa=5.0,
        )
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(pooled[0, 1], pooled[1, 1])
        np.testing.assert_allclose(pooled.sum(axis=1), np.ones(2))

    def test_unavailable_prior_has_zero_weight(self):
        local = np.asarray([[0.7, 0.3]])
        prior = np.asarray([[0.1, 0.9]])
        pooled, weights = support_adaptive_prior_pool(
            local,
            prior,
            local_support=np.asarray([0.0]),
            prior_available=np.asarray([False]),
            strength=1.0,
            kappa=5.0,
        )
        np.testing.assert_allclose(pooled, local)
        np.testing.assert_allclose(weights, np.zeros(1))


if __name__ == "__main__":
    unittest.main()
