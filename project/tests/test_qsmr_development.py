import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.experiments.gsad.run_mrct_development import load_multires_development
from project.experiments.gsad.run_qsmr_development import (
    QSMRConfig,
    _rootwise_permuted_targets,
    select_qsmr_model,
    summarize_qsmr,
)


class QSMRDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_selector_stays_inside_fixed_grid(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        roots = sorted(frame["root"].unique())
        fit = frame.loc[frame["root"].isin(roots[:8])].reset_index(drop=True)
        validation = frame.loc[frame["root"].isin(roots[8:10])].reset_index(drop=True)
        base = np.full((len(validation), len(vocab)), 1.0 / len(vocab))
        config = QSMRConfig(
            kappas=(2.0, 5.0), destination_fractions=(0.25, 0.5), bootstrap=10
        )

        model, selected, candidate = select_qsmr_model(
            fit, validation, base, vocab, config
        )

        self.assertIn(selected["kappa"], config.kappas)
        self.assertIn(selected["destination_fraction"], config.destination_fractions)
        self.assertTrue(model.fitted_)
        self.assertEqual(candidate.shape, base.shape)

    def test_summary_rejects_when_factor_ablation_has_no_increment(self):
        predictions = pd.DataFrame(
            {
                "root": ["g1", "g1", "g2", "g2"],
                "fold": [0, 0, 1, 1],
                "is_self": [False, False, True, True],
                "baseline_correct": [False, False, False, False],
                "candidate_correct": [True, True, True, True],
                "hazard_only_correct": [True, True, True, True],
                "destination_only_correct": [True, True, True, True],
                "no_raw_correct": [True, True, True, True],
                "baseline_rr": [0.1, 0.1, 0.1, 0.1],
                "candidate_rr": [1.0, 1.0, 1.0, 1.0],
                "hazard_only_rr": [1.0, 1.0, 1.0, 1.0],
                "destination_only_rr": [1.0, 1.0, 1.0, 1.0],
                "no_raw_rr": [1.0, 1.0, 1.0, 1.0],
                "baseline_hit5": [False, False, False, False],
                "candidate_hit5": [True, True, True, True],
                "exit_hazard": [0.5, 0.5, 0.5, 0.5],
            }
        )

        _, _, gates = summarize_qsmr(predictions, n_boot=20, seed=11)

        self.assertFalse(gates["factor_increment"])
        self.assertFalse(gates["raw_increment"])
        self.assertFalse(gates["PRIMARY"])

    def test_negative_control_permutation_preserves_each_root_multiset(self):
        frame = pd.DataFrame(
            {
                "root": ["g1", "g1", "g1", "g2", "g2", "g2"],
                "target": ["A", "B", "C", "A", "C", "B"],
            }
        )

        permuted = _rootwise_permuted_targets(frame, seed=17)

        for root, group in frame.groupby("root"):
            original = sorted(group["target"].astype(str))
            changed = sorted(permuted.loc[group.index, "target"].astype(str))
            self.assertEqual(original, changed)
        self.assertFalse(permuted["target"].equals(frame["target"]))


if __name__ == "__main__":
    unittest.main()
