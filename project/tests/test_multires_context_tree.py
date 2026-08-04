import unittest

import numpy as np

from project.experiments.gsad.multires_context_tree import (
    MultiResolutionContextTree,
    QuotientMultiResolutionContextTree,
)


class MultiResolutionContextTreeTests(unittest.TestCase):
    def test_raw_context_disambiguates_identical_parent_context(self):
        model = MultiResolutionContextTree(
            vocab=("A", "B"),
            max_parent_context=1,
            max_raw_context=1,
            backoff_strength=1.0,
            raw_backoff_strength=1.0,
        )
        model.fit(
            parent_prefixes=[("P",), ("P",), ("P",), ("P",)],
            raw_prefixes=[("r1",), ("r1",), ("r2",), ("r2",)],
            targets=["A", "A", "B", "B"],
            groups=["g1", "g2", "g3", "g4"],
        )

        probabilities, meta = model.predict_proba_with_meta(
            parent_prefixes=[("P",), ("P",)],
            raw_prefixes=[("r1",), ("r2",)],
        )

        self.assertGreater(probabilities[0, 0], probabilities[0, 1])
        self.assertGreater(probabilities[1, 1], probabilities[1, 0])
        self.assertEqual(meta["raw_used_order"].tolist(), [1, 1])
        self.assertEqual(meta["raw_root_support"].tolist(), [2, 2])

    def test_unseen_raw_context_backs_off_to_parent_distribution(self):
        model = MultiResolutionContextTree(
            vocab=("A", "B"),
            max_parent_context=1,
            max_raw_context=1,
            backoff_strength=1.0,
            raw_backoff_strength=1.0,
        )
        model.fit(
            parent_prefixes=[("P",), ("P",), ("Q",)],
            raw_prefixes=[("r1",), ("r1",), ("r2",)],
            targets=["A", "A", "B"],
            groups=["g1", "g2", "g3"],
        )

        parent_only, _ = model.predict_proba_with_meta(
            parent_prefixes=[("P",)], raw_prefixes=[("unknown",)]
        )
        repeated, meta = model.predict_proba_with_meta(
            parent_prefixes=[("P",)], raw_prefixes=[("unknown",)]
        )

        np.testing.assert_allclose(parent_only, repeated)
        self.assertEqual(int(meta.iloc[0]["raw_used_order"]), 0)

    def test_probabilities_are_normalized_and_lengths_are_validated(self):
        model = MultiResolutionContextTree(
            vocab=("A", "B"), max_parent_context=2, max_raw_context=2
        )
        with self.assertRaises(ValueError):
            model.fit(
                parent_prefixes=[("P",)],
                raw_prefixes=[],
                targets=["A"],
                groups=["g1"],
            )

        model.fit(
            parent_prefixes=[("P",), ("Q",)],
            raw_prefixes=[("r1",), ("r2",)],
            targets=["A", "B"],
            groups=["g1", "g2"],
        )
        probabilities, _ = model.predict_proba_with_meta(
            parent_prefixes=[("P",), ("Q",)],
            raw_prefixes=[("r1",), ("r2",)],
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertTrue(np.isfinite(probabilities).all())


class QuotientMultiResolutionContextTreeTests(unittest.TestCase):
    def test_collapsed_raw_view_exactly_reduces_to_parent_tree(self):
        model = QuotientMultiResolutionContextTree(
            vocab=("A", "B"),
            max_parent_context=1,
            max_raw_context=1,
            parent_kappa=2.0,
            raw_kappa=2.0,
        ).fit(
            parent_prefixes=[("P",), ("Q",)],
            raw_prefixes=[("P",), ("Q",)],
            targets=["A", "B"],
            groups=["g1", "g2"],
        )

        quotient, meta = model.predict_proba_with_meta(
            parent_prefixes=[("P",), ("Q",)],
            raw_prefixes=[("P",), ("Q",)],
        )
        parent, _ = model.parent_tree.predict_proba_with_meta([("P",), ("Q",)])

        np.testing.assert_allclose(quotient, parent)
        self.assertEqual(meta["raw_used_order"].tolist(), [0, 0])

    def test_raw_conditionals_are_root_balanced(self):
        base = QuotientMultiResolutionContextTree(
            vocab=("A", "B"), max_parent_context=1, max_raw_context=1
        ).fit(
            parent_prefixes=[("P",), ("P",), ("P",)],
            raw_prefixes=[("r",), ("r",), ("r",)],
            targets=["A", "A", "B"],
            groups=["g1", "g1", "g2"],
        )
        duplicated = QuotientMultiResolutionContextTree(
            vocab=("A", "B"), max_parent_context=1, max_raw_context=1
        ).fit(
            parent_prefixes=[("P",), ("P",), ("P",), ("P",)],
            raw_prefixes=[("r",), ("r",), ("r",), ("r",)],
            targets=["A", "A", "A", "B"],
            groups=["g1", "g1", "g1", "g2"],
        )

        first, _ = base.predict_proba_with_meta([("P",)], [("r",)])
        second, _ = duplicated.predict_proba_with_meta([("P",)], [("r",)])
        np.testing.assert_allclose(first, second)

    def test_supported_raw_variant_changes_parent_prediction(self):
        model = QuotientMultiResolutionContextTree(
            vocab=("A", "B"),
            max_parent_context=1,
            max_raw_context=1,
            parent_kappa=2.0,
            raw_kappa=1.0,
        ).fit(
            parent_prefixes=[("P",), ("P",), ("P",), ("P",)],
            raw_prefixes=[("r1",), ("r1",), ("r2",), ("r2",)],
            targets=["A", "A", "B", "B"],
            groups=["g1", "g2", "g3", "g4"],
        )
        probabilities, meta = model.predict_proba_with_meta(
            [("P",), ("P",)], [("r1",), ("r2",)]
        )
        self.assertGreater(probabilities[0, 0], probabilities[0, 1])
        self.assertGreater(probabilities[1, 1], probabilities[1, 0])
        self.assertEqual(meta["raw_root_support"].tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
