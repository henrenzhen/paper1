import unittest
from pathlib import Path

from project.experiments.gsad.data_protocol import TEST_ROOTS
from project.experiments.gsad.run_mrct_development import (
    MRCTConfig,
    load_multires_development,
    select_mrct_model,
)
from project.experiments.gsad.run_qmrct_development import (
    QMRCTConfig,
    select_qmrct_model,
)


class MRCTDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_loader_retains_raw_history_and_excludes_locked_roots(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        self.assertEqual(len(frame), 10555)
        self.assertEqual(frame["root"].nunique(), 133)
        self.assertFalse(set(frame["root"]) & set(TEST_ROOTS))
        self.assertTrue(
            (frame["raw_prefix_ids"].map(len) == frame["prefix_len"]).all()
        )
        self.assertEqual(len(vocab), 184)

    def test_selection_uses_only_supplied_fit_and_validation_frames(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        roots = sorted(frame["root"].unique())
        fit = frame.loc[frame["root"].isin(roots[:8])].reset_index(drop=True)
        validation = frame.loc[frame["root"].isin(roots[8:10])].reset_index(drop=True)
        model, selected, probabilities = select_mrct_model(
            fit,
            validation,
            vocab,
            MRCTConfig(
                parent_contexts=(1,),
                raw_contexts=(1,),
                backoff_strengths=(5.0,),
                raw_backoff_strengths=(5.0,),
                bootstrap=10,
            ),
        )
        self.assertEqual(probabilities.shape, (len(validation), len(vocab)))
        self.assertEqual(selected["max_parent_context"], 1)
        self.assertEqual(selected["max_raw_context"], 1)
        self.assertTrue(model.fitted_)

    def test_qmrct_selection_uses_preregistered_grid(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        roots = sorted(frame["root"].unique())
        fit = frame.loc[frame["root"].isin(roots[:8])].reset_index(drop=True)
        validation = frame.loc[frame["root"].isin(roots[8:10])].reset_index(drop=True)
        model, selected, probabilities = select_qmrct_model(
            fit,
            validation,
            vocab,
            QMRCTConfig(
                parent_contexts=(1,),
                raw_contexts=(1,),
                parent_kappas=(2.0,),
                raw_kappas=(5.0,),
                bootstrap=10,
            ),
        )
        self.assertEqual(probabilities.shape, (len(validation), len(vocab)))
        self.assertEqual(selected["parent_kappa"], 2.0)
        self.assertEqual(selected["raw_kappa"], 5.0)
        self.assertTrue(model.fitted_)


if __name__ == "__main__":
    unittest.main()
