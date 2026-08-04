import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pr_hr_feasibility.pr_hr_small_experiment import (  # noqa: E402
    _top_candidate,
    align_predictions,
    assign_group_folds,
    build_strict_transition_priors,
    crossfit_quantile_acceptance,
    fit_pairwise_ranker,
    parse_candidates,
    parse_probabilities,
    risk_coverage,
)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.gru = pd.DataFrame(
            {
                "sequence_id": ["SIM_001_part001", "SIM_002_part001"],
                "prefix_len": [2, 1],
                "state": ["T1 T2", "T3"],
                "true_label": ["T4", "T5"],
                "top1_pred": ["T4", "T9"],
                "top1_prob": [0.8, 0.6],
                "top5_labels": ["T4 || T9", "T9 || T5"],
                "top5_probs": ["0.8 || 0.2", "0.6 || 0.4"],
            }
        )
        self.llm = pd.DataFrame(
            {
                "sequence_id": ["SIM_002_part001", "SIM_001_part001"],
                "state": ["T3", "T1 || T2"],
                "true_label": ["T5", "T4"],
                "predicted_next_ttps": ['["T5", "T9"]', '["T8", "T4"]'],
            }
        )

    def test_alignment_reorders_llm_rows_and_preserves_states(self):
        aligned = align_predictions(self.gru, self.llm)
        self.assertEqual(aligned["sequence_id"].tolist(), self.gru["sequence_id"].tolist())
        self.assertEqual(aligned["llm_candidates"].tolist(), [["T8", "T4"], ["T5", "T9"]])
        self.assertTrue(aligned["state_match"].all())

    def test_alignment_rejects_same_length_but_different_state(self):
        bad = self.llm.copy()
        bad.loc[bad["sequence_id"] == "SIM_001_part001", "state"] = "T1 || T7"
        with self.assertRaisesRegex(ValueError, "state mismatch"):
            align_predictions(self.gru, bad)

    def test_llm_subtechniques_are_collapsed_to_parent_label_space(self):
        parsed = parse_candidates('["T1021.001", "T1021", "T1059.003"]')
        self.assertEqual(parsed, ["T1021", "T1059"])

    def test_invalid_probabilities_are_rejected(self):
        for invalid in ["0.8 || nan", "1.2 || 0.1", "-0.1 || 0.2"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_probabilities(invalid)


class GroupFoldTests(unittest.TestCase):
    def test_sim_root_never_crosses_folds(self):
        sequence_ids = pd.Series(
            [
                "SIM_001_part001",
                "SIM_001_part002",
                "SIM_002_part001",
                "SIM_003_part001",
                "SIM_003_part002",
            ]
        )
        folds = assign_group_folds(sequence_ids, n_splits=3)
        roots = sequence_ids.str.replace(r"_part\d+$", "", regex=True)
        for root in roots.unique():
            self.assertEqual(len(set(folds[roots == root])), 1)
        self.assertGreater(len(set(folds)), 1)

    def test_transition_priors_remove_every_excluded_root_row(self):
        train = pd.DataFrame(
            {
                "sequence_id": ["SIM_001_part001", "SIM_002_part001"],
                "prefix_technique_ids_parent": ["T1001", "T1002"],
                "next_technique_id_parent": ["T1003", "T1004"],
            }
        )
        priors = build_strict_transition_priors(train, {"SIM_001"})
        self.assertEqual(priors.retained_rows, 1)
        self.assertEqual(priors.excluded_rows, 1)
        self.assertNotIn("T1001", priors.transition_counts)
        self.assertEqual(priors.transition_counts["T1002"]["T1004"], 1)


class PairwiseRankerTests(unittest.TestCase):
    def test_ranker_learns_feature_that_marks_correct_candidate(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        sample_ids = np.array([0, 0, 1, 1])
        is_correct = np.array([True, False, True, False])
        weights = fit_pairwise_ranker(features, sample_ids, is_correct, l2=0.1)
        scores = features @ weights
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[3])

    def test_score_tie_prefers_present_gru_rank_over_missing_rank(self):
        candidates = pd.DataFrame(
            {
                "sample_idx": [0, 0],
                "candidate": ["T1001", "T1002"],
                "score": [0.5, 0.5],
                "gru_rank": [1, 0],
                "llm_rank": [0, 1],
            }
        )
        top = _top_candidate(candidates, "score")
        self.assertEqual(top.iloc[0]["candidate"], "T1001")

class SelectivePredictionTests(unittest.TestCase):
    def test_risk_coverage_accepts_highest_confidence_first(self):
        table = risk_coverage(
            correct=np.array([1, 0, 1], dtype=bool),
            confidence=np.array([0.9, 0.1, 0.8]),
            coverages=[2 / 3, 1.0],
        )
        selective = table.iloc[0]
        self.assertEqual(int(selective["accepted_n"]), 2)
        self.assertAlmostEqual(float(selective["accuracy"]), 1.0)
        self.assertAlmostEqual(float(selective["risk"]), 0.0)

    def test_crossfit_threshold_uses_other_folds_only(self):
        accepted, thresholds = crossfit_quantile_acceptance(
            confidence=np.array([0.9, 0.8, 0.7, 0.1]),
            folds=np.array([0, 0, 1, 1]),
            target_coverage=0.5,
        )
        np.testing.assert_array_equal(accepted, np.array([True, True, False, False]))
        np.testing.assert_allclose(thresholds, np.array([0.7, 0.7, 0.9, 0.9]))


if __name__ == "__main__":
    unittest.main()
