import unittest
from pathlib import Path

import pandas as pd

from project.experiments.gsad.stockpile_dataset import load_stockpile_transitions
from project.experiments.gsad.run_qmrct_stockpile_external import (
    run_stockpile_external,
)


class StockpileDatasetTests(unittest.TestCase):
    def test_preregistered_profiles_yield_direct_adjacent_transitions(self):
        project_root = Path(__file__).resolve().parents[1]
        vocab = tuple(
            pd.read_csv(project_root / "data_v2" / "core" / "rl_label_vocab.csv")
            .sort_values("label_id")["technique_id_parent"]
            .astype(str)
        )
        frame, audit = load_stockpile_transitions(project_root, vocab)
        self.assertEqual(frame["profile"].nunique(), 10)
        self.assertEqual(audit["selected_profiles"], 10)
        self.assertEqual(audit["mapped_steps"], 77)
        self.assertEqual(audit["transition_events_before_vocab_filter"], 66)
        self.assertGreaterEqual(len(frame), 55)
        self.assertTrue(set(frame["target"]).issubset(set(vocab)))
        self.assertTrue((frame["prefix_ids"].map(len) == 1).all())
        self.assertTrue((frame["raw_prefix_ids"].map(len) == 1).all())

    def test_frozen_runner_scores_every_stockpile_transition(self):
        project_root = Path(__file__).resolve().parents[1]
        predictions, metrics = run_stockpile_external(
            project_root=project_root, output_dir=None, bootstrap=10, seed=20260730
        )
        self.assertEqual(len(predictions), 65)
        self.assertEqual(int(metrics["profiles"]), 10)
        self.assertIn("raw_increment_mrr", metrics)


if __name__ == "__main__":
    unittest.main()
