import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.rank_conformal import (  # noqa: E402
    finite_sample_rank_quantile,
    fit_rank_union,
    minimum_expert_ranks,
    stable_rank_matrix,
)


class StableRankTests(unittest.TestCase):
    def test_probability_ties_use_vocabulary_index(self):
        ranks = stable_rank_matrix(np.asarray([[0.4, 0.4, 0.2]]))
        np.testing.assert_array_equal(ranks, np.asarray([[1, 2, 3]]))

    def test_minimum_rank_uses_best_expert(self):
        first = np.asarray([[0.4, 0.3, 0.2, 0.07, 0.03]])
        second = np.asarray([[0.5, 0.1, 0.3, 0.08, 0.02]])
        minimum = minimum_expert_ranks((first, second))
        self.assertEqual(int(minimum[0, 2]), 2)


class FiniteSampleRankQuantileTests(unittest.TestCase):
    def test_quantile_uses_ceil_n_plus_one_formula(self):
        self.assertEqual(finite_sample_rank_quantile([1, 2, 3, 4], alpha=0.25), 4)
        self.assertEqual(finite_sample_rank_quantile([1, 2, 3, 4], alpha=0.50), 3)

    def test_empty_calibration_and_invalid_alpha_are_rejected(self):
        with self.assertRaises(ValueError):
            finite_sample_rank_quantile([], alpha=0.10)
        with self.assertRaises(ValueError):
            finite_sample_rank_quantile([1], alpha=1.0)


class RankUnionPredictorTests(unittest.TestCase):
    def test_union_deduplicates_overlapping_top_q_labels(self):
        vocab = ("A", "B", "C", "D")
        calibration = (
            np.asarray([[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2]]),
            np.asarray([[0.4, 0.2, 0.3, 0.1], [0.1, 0.3, 0.4, 0.2]]),
        )
        predictor = fit_rank_union(
            calibration, targets=("B", "C"), vocab=vocab, alpha=0.5
        )
        self.assertEqual(predictor.threshold, 2)
        prediction = predictor.predict_sets(
            (
                np.asarray([[0.4, 0.3, 0.2, 0.1]]),
                np.asarray([[0.4, 0.2, 0.3, 0.1]]),
            )
        )[0]
        self.assertEqual(prediction, frozenset({"A", "B", "C"}))
        self.assertEqual(len(prediction), 3)

    def test_positive_threshold_makes_every_prediction_set_nonempty(self):
        predictor = fit_rank_union(
            (np.asarray([[0.6, 0.4]]),),
            targets=("A",),
            vocab=("A", "B"),
            alpha=0.1,
        )
        sets = predictor.predict_sets((np.asarray([[0.2, 0.8], [0.9, 0.1]]),))
        self.assertTrue(all(prediction_set for prediction_set in sets))

    def test_expert_shape_and_probability_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_rank_union((), targets=("A",), vocab=("A",), alpha=0.1)
        with self.assertRaises(ValueError):
            fit_rank_union(
                (np.asarray([[0.8, 0.3]]),),
                targets=("A",),
                vocab=("A", "B"),
                alpha=0.1,
            )


if __name__ == "__main__":
    unittest.main()
