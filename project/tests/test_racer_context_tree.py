import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.context_tree import SupportAdaptiveContextTree  # noqa: E402


class SupportAdaptiveContextTreeTests(unittest.TestCase):
    def test_unseen_long_context_recursively_backs_off(self):
        model = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=2, alpha=0.0, kappa=1.0
        ).fit(
            prefixes=(("A",), ("A",), ("B",)),
            targets=("B", "B", "C"),
            groups=("g1", "g2", "g3"),
        )
        probabilities, metadata = model.predict_proba_with_meta((("C", "A"),))
        self.assertEqual(metadata.loc[0, "used_context"], 1)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[0, 1], probabilities[0, 2])

    def test_more_independent_root_support_produces_stronger_shrinkage_weight(self):
        model = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=1, alpha=0.0, kappa=2.0
        ).fit(
            prefixes=(("A",), ("A",), ("A",), ("C",)),
            targets=("B", "B", "B", "A"),
            groups=("g1", "g2", "g3", "g4"),
        )
        _, metadata = model.predict_proba_with_meta((("A",), ("C",)))
        self.assertEqual(metadata.loc[0, "context_root_support"], 3)
        self.assertEqual(metadata.loc[1, "context_root_support"], 1)
        self.assertGreater(metadata.loc[0, "shrinkage_weight"], metadata.loc[1, "shrinkage_weight"])

    def test_duplicating_rows_inside_one_root_does_not_change_predictions(self):
        prefixes = (("A",), ("A",), ("B",))
        targets = ("B", "C", "A")
        groups = ("g1", "g1", "g2")
        base = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=2, alpha=0.1, kappa=1.0
        ).fit(prefixes, targets, groups)
        duplicated = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=2, alpha=0.1, kappa=1.0
        ).fit(
            prefixes + prefixes[:2],
            targets + targets[:2],
            groups + groups[:2],
        )
        query = (("A",), ("B", "A"))
        np.testing.assert_allclose(
            base.predict_proba_with_meta(query)[0],
            duplicated.predict_proba_with_meta(query)[0],
        )

    def test_context_conditionals_average_roots_not_rows(self):
        model = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=1, alpha=0.0, kappa=0.0
        ).fit(
            prefixes=(("A",),) * 100 + (("A",),),
            targets=(("B",) * 100) + ("C",),
            groups=(("long",) * 100) + ("short",),
        )
        probabilities, _ = model.predict_proba_with_meta((("A",),))
        self.assertAlmostEqual(probabilities[0, 1], 0.5)
        self.assertAlmostEqual(probabilities[0, 2], 0.5)

    def test_probabilities_are_finite_normalized_and_vocabulary_complete(self):
        model = SupportAdaptiveContextTree(
            ("A", "B", "C"), max_context=3, alpha=0.2, kappa=0.5
        ).fit(
            prefixes=(("A",), ("A", "B"), ("C",)),
            targets=("B", "C", "A"),
            groups=("g1", "g2", "g3"),
        )
        probabilities, _ = model.predict_proba_with_meta(((), ("A",), ("A", "B")))
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))

    def test_invalid_hyperparameters_and_unknown_target_are_rejected(self):
        with self.assertRaises(ValueError):
            SupportAdaptiveContextTree(("A",), max_context=0)
        with self.assertRaises(ValueError):
            SupportAdaptiveContextTree(("A",), kappa=-1)
        with self.assertRaises(ValueError):
            SupportAdaptiveContextTree(("A", "B")).fit(
                (("A",),), ("C",), ("g",)
            )


if __name__ == "__main__":
    unittest.main()
