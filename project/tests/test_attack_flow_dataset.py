import unittest
from pathlib import Path

import pandas as pd

from project.experiments.gsad.attack_flow_dataset import load_attack_flow_transitions
from project.experiments.gsad.run_qmrct_attack_flow_external import (
    run_attack_flow_external,
)


class AttackFlowDatasetTests(unittest.TestCase):
    def test_official_corpus_yields_directed_in_vocabulary_transitions(self):
        project_root = Path(__file__).resolve().parents[1]
        vocab = tuple(
            pd.read_csv(project_root / "data_v2" / "core" / "rl_label_vocab.csv")
            .sort_values("label_id")["technique_id_parent"]
            .astype(str)
        )
        frame, audit = load_attack_flow_transitions(
            project_root=project_root,
            vocab=vocab,
            exclude_overlapping_ctid=True,
        )
        self.assertEqual(audit["corpus_files"], 40)
        self.assertEqual(audit["excluded_overlap_files"], 2)
        self.assertGreater(len(frame), 100)
        self.assertGreater(frame["flow"].nunique(), 20)
        self.assertTrue(set(frame["target"]).issubset(set(vocab)))
        self.assertFalse(frame["flow"].str.contains("Turla", case=False).any())
        self.assertTrue((frame["prefix_ids"].map(len) == 1).all())
        self.assertTrue((frame["raw_prefix_ids"].map(len) == 1).all())

    def test_frozen_external_runner_scores_every_transition(self):
        project_root = Path(__file__).resolve().parents[1]
        predictions, metrics = run_attack_flow_external(
            project_root=project_root, output_dir=None, bootstrap=10, seed=20260730
        )
        self.assertEqual(len(predictions), 705)
        self.assertEqual(int(metrics["flows"]), 35)
        self.assertIn("raw_increment_mrr", metrics)


if __name__ == "__main__":
    unittest.main()
