import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.experiments.gsad.run_mrct_development import load_multires_development
from project.experiments.gsad.run_qhsm_development import (
    QHSMConfig,
    select_qhsm_model,
    summarize_qhsm,
)


class QHSMDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_selector_stays_inside_fixed_grid(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        roots = sorted(frame["root"].unique())
        fit = frame.loc[frame["root"].isin(roots[:8])].reset_index(drop=True)
        validation = frame.loc[frame["root"].isin(roots[8:10])].reset_index(drop=True)
        base = np.full((len(validation), len(vocab)), 1.0 / len(vocab))
        config = QHSMConfig(
            kappas=(2.0, 5.0), hazard_fractions=(0.5, 1.0), bootstrap=10
        )

        model, selected, candidate = select_qhsm_model(
            fit, validation, base, vocab, config
        )

        self.assertIn(selected["kappa"], config.kappas)
        self.assertIn(selected["hazard_fraction"], config.hazard_fractions)
        self.assertTrue(model.fitted_)
        self.assertEqual(candidate.shape, base.shape)

    def test_summary_requires_raw_increment(self):
        predictions = pd.DataFrame(
            {
                "root": ["g1", "g1", "g2", "g2"],
                "fold": [0, 0, 1, 1],
                "is_self": [False, False, True, True],
                "baseline_correct": [False, False, False, False],
                "candidate_correct": [True, True, True, True],
                "no_raw_correct": [True, True, True, True],
                "baseline_rr": [0.1, 0.1, 0.1, 0.1],
                "candidate_rr": [1.0, 1.0, 1.0, 1.0],
                "no_raw_rr": [1.0, 1.0, 1.0, 1.0],
                "baseline_hit5": [False, False, False, False],
                "candidate_hit5": [True, True, True, True],
                "exit_hazard": [0.5, 0.5, 0.5, 0.5],
            }
        )

        _, _, gates = summarize_qhsm(predictions, n_boot=20, seed=13)

        self.assertFalse(gates["raw_increment"])
        self.assertFalse(gates["PRIMARY"])


if __name__ == "__main__":
    unittest.main()
