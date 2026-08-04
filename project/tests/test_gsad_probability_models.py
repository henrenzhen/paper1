import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.data_protocol import build_frozen_split  # noqa: E402
from experiments.gsad.probability_models import (  # noqa: E402
    InterpolatedNGram,
    TacticAwareModel,
    UnigramModel,
)


class UnigramTests(unittest.TestCase):
    def test_alpha_is_total_dirichlet_mass_not_per_class_mass(self):
        model = UnigramModel(vocab=("T1", "T2"), alpha=0.5)
        model.fit(targets=["T1"], groups=["G1"])
        np.testing.assert_allclose(model.predict_proba(1), [[5 / 6, 1 / 6]])

    def test_root_balancing_prevents_long_root_from_dominating(self):
        model = UnigramModel(vocab=("T1", "T2"), alpha=0.0)
        model.fit(
            targets=["T1"] * 100 + ["T2"],
            groups=["G1"] * 100 + ["G2"],
        )
        np.testing.assert_allclose(model.predict_proba(1), [[0.5, 0.5]])

    def test_unknown_training_label_is_rejected(self):
        model = UnigramModel(vocab=("T1",), alpha=0.1)
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            model.fit(targets=["T9"], groups=["G1"])


class InterpolatedNGramTests(unittest.TestCase):
    def test_unseen_trigram_backs_off_and_probabilities_sum_to_one(self):
        model = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=3,
            alpha=0.5,
            interpolation=(0.2, 0.3, 0.5),
        )
        model.fit(prefixes=[("T1", "T2")], targets=["T3"], groups=["G1"])
        probs, meta = model.predict_proba_with_meta([["T9", "T2"]])
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertEqual(int(meta.loc[0, "used_order"]), 1)

    def test_root_balanced_transition_counts_do_not_let_long_group_dominate(self):
        prefixes = [("T1",)] * 101
        targets = ["T2"] * 100 + ["T3"]
        groups = ["G1"] * 100 + ["G2"]
        model = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
        )
        model.fit(prefixes, targets, groups=groups)
        probs, _ = model.predict_proba_with_meta([["T1"]])
        self.assertAlmostEqual(float(probs[0, 1]), float(probs[0, 2]), places=12)

    def test_domain_balancing_prevents_many_roots_in_one_domain_from_dominating(self):
        compact = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
        ).fit(
            prefixes=[("T1",), ("T1",)],
            targets=["T2", "T3"],
            groups=["a1", "b1"],
            domains=["source_a", "source_b"],
        )
        expanded = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
        ).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )

        compact_probs, _ = compact.predict_proba_with_meta([["T1"]])
        expanded_probs, _ = expanded.predict_proba_with_meta([["T1"]])

        np.testing.assert_allclose(compact_probs, expanded_probs)

    def test_domain_power_one_reduces_to_root_balancing(self):
        root_model = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
        ).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
        )
        powered = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
            domain_power=1.0,
        ).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )

        root_probs, _ = root_model.predict_proba_with_meta([["T1"]])
        powered_probs, _ = powered.predict_proba_with_meta([["T1"]])

        np.testing.assert_allclose(root_probs, powered_probs)

    def test_domain_partial_pooling_moves_equal_domain_estimate_toward_root_global(self):
        common = dict(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
            domain_power=0.0,
        )
        equal_domain = InterpolatedNGram(**common, domain_kappa=0.0).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )
        pooled = InterpolatedNGram(**common, domain_kappa=3.0).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )
        root = InterpolatedNGram(**common).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
        )

        equal_probs, _ = equal_domain.predict_proba_with_meta([["T1"]])
        pooled_probs, _ = pooled.predict_proba_with_meta([["T1"]])
        root_probs, _ = root.predict_proba_with_meta([["T1"]])

        self.assertLess(
            abs(pooled_probs[0, 1] - root_probs[0, 1]),
            abs(equal_probs[0, 1] - root_probs[0, 1]),
        )

    def test_leave_one_domain_prior_excludes_current_domain_from_shrinkage_target(self):
        model = InterpolatedNGram(
            vocab=("T1", "T2", "T3"),
            order=2,
            alpha=0.0,
            interpolation=(0.0, 1.0),
            domain_power=0.0,
            domain_kappa=1.0,
            leave_one_domain_prior=True,
        ).fit(
            prefixes=[("T1",)] * 4,
            targets=["T2", "T2", "T2", "T3"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )

        probabilities, _ = model.predict_proba_with_meta([["T1"]])

        self.assertAlmostEqual(probabilities[0, 1], 0.625)
        self.assertAlmostEqual(probabilities[0, 2], 0.375)

    def test_interpolation_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            InterpolatedNGram(
                vocab=("T1", "T2"),
                order=3,
                alpha=0.5,
                interpolation=(0.5, 0.5),
            )


class TacticAwareTests(unittest.TestCase):
    def test_shared_source_tactic_transfers_to_unseen_exact_context(self):
        mapping = {
            "T1": {"TA1"},
            "T2": {"TA2"},
            "T3": {"TA3"},
            "T4": {"TA4"},
            "T5": {"TA1"},
        }
        model = TacticAwareModel(
            vocab=("T1", "T2", "T3", "T4", "T5"),
            technique_to_tactics=mapping,
            alpha=0.1,
            technique_weight=0.0,
            tactic_weight=1.0,
        )
        model.fit(
            prefixes=[("T1",), ("T3",)],
            targets=["T2", "T4"],
            groups=["G1", "G2"],
        )
        probs, meta = model.predict_proba_with_meta([["T5"]])
        self.assertGreater(float(probs[0, 1]), float(probs[0, 3]))
        self.assertTrue(bool(meta.loc[0, "tactic_context_seen"]))


class RealDataSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core = PROJECT_ROOT / "data_v2" / "core"
        frames = [
            pd.read_csv(core / "sim_train_parent_min3.csv"),
            pd.read_csv(core / "sim_val_parent_min3.csv"),
            pd.read_csv(core / "sim_test_parent_min3.csv"),
        ]
        cls.split = build_frozen_split(frames)
        cls.vocab = tuple(
            pd.read_csv(core / "rl_label_vocab.csv")
            .sort_values("label_id")["technique_id_parent"]
            .astype(str)
        )

    def test_full_vocabulary_predictions_need_no_validation_targets(self):
        model = InterpolatedNGram(
            vocab=self.vocab,
            order=3,
            alpha=0.5,
            interpolation=(0.2, 0.3, 0.5),
        )
        model.fit(
            prefixes=self.split.fit["prefix_ids"],
            targets=self.split.fit["target"],
            groups=self.split.fit["root"],
        )
        probs, metadata = model.predict_proba_with_meta(
            self.split.validation["prefix_ids"]
        )
        self.assertEqual(probs.shape, (1592, 184))
        self.assertEqual(len(metadata), 1592)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        self.assertNotIn("target", metadata.columns)


if __name__ == "__main__":
    unittest.main()
