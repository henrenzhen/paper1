import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.experiments.gsad.multirelation_graph_residual import (
    TypedGraphEscapeRedistributor,
    build_attack_relation_matrices,
)
from project.experiments.gsad.run_mrct_development import load_multires_development
from project.experiments.gsad.run_tger_development import (
    TGERConfig,
    select_tger_model,
    summarize_tger,
)


class TGERDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_selector_stays_inside_fixed_grid(self):
        frame, vocab, _ = load_multires_development(self.project_root)
        roots = sorted(frame["root"].unique())
        fit = frame.loc[frame["root"].isin(roots[:8])].reset_index(drop=True)
        validation = frame.loc[frame["root"].isin(roots[8:10])].reset_index(drop=True)
        semantic, tactic, _ = build_attack_relation_matrices(
            self.project_root / "data" / "enterprise-attack-18.1.json", vocab, 5
        )
        model = TypedGraphEscapeRedistributor(vocab, semantic, tactic).fit(
            fit["prefix_ids"], fit["target"], fit["root"]
        )
        base = np.full((len(validation), len(vocab)), 1.0 / len(vocab))
        config = TGERConfig(
            relation_weights=((1.0, 0.0, 0.0), (0.5, 0.25, 0.25)),
            graph_fractions=(0.25, 0.5),
            bootstrap=10,
        )

        selected, candidate = select_tger_model(
            model, base, validation, vocab, config
        )

        self.assertIn(tuple(selected["relation_weights"]), config.relation_weights)
        self.assertIn(selected["graph_fraction"], config.graph_fractions)
        self.assertEqual(candidate.shape, base.shape)

    def test_summary_rejects_no_static_increment(self):
        predictions = pd.DataFrame(
            {
                "root": ["g1", "g1", "g2", "g2"],
                "fold": [0, 0, 1, 1],
                "target_seen": [False, False, False, False],
                "is_self": [False, False, False, False],
                "baseline_correct": [False, False, False, False],
                "candidate_correct": [True, True, True, True],
                "transition_only_correct": [True, True, True, True],
                "baseline_rr": [0.1, 0.1, 0.1, 0.1],
                "candidate_rr": [1.0, 1.0, 1.0, 1.0],
                "transition_only_rr": [1.0, 1.0, 1.0, 1.0],
                "baseline_hit5": [False, False, False, False],
                "candidate_hit5": [True, True, True, True],
                "escape_mass": [0.2, 0.2, 0.2, 0.2],
                "graph_shift_l1": [0.1, 0.1, 0.1, 0.1],
            }
        )

        _, _, gates = summarize_tger(predictions, n_boot=20, seed=9)

        self.assertFalse(gates["static_relation_increment"])
        self.assertFalse(gates["PRIMARY"])


if __name__ == "__main__":
    unittest.main()
