import unittest

import numpy as np

from project.experiments.gsad.semimarkov_router import QuotientSemiMarkovRouter


class QuotientSemiMarkovRouterTests(unittest.TestCase):
    def test_duplicate_rows_within_one_root_do_not_change_hazard_tables(self):
        base = QuotientSemiMarkovRouter(("A", "B"), kappa=1.0).fit(
            parent_prefixes=[("A",), ("A",)],
            raw_prefixes=[("r1",), ("r2",)],
            targets=["A", "B"],
            groups=["g1", "g2"],
        )
        duplicated = QuotientSemiMarkovRouter(("A", "B"), kappa=1.0).fit(
            parent_prefixes=[("A",), ("A",), ("A",)],
            raw_prefixes=[("r1",), ("r1",), ("r2",)],
            targets=["A", "A", "B"],
            groups=["g1", "g1", "g2"],
        )

        self.assertEqual(base.hazard_tables_, duplicated.hazard_tables_)

    def test_raw_variant_and_dwell_can_change_exit_hazard(self):
        model = QuotientSemiMarkovRouter(("A", "B"), kappa=0.1).fit(
            parent_prefixes=[("A",), ("A", "A"), ("A",), ("A", "A")],
            raw_prefixes=[("r1",), ("r1", "r1"), ("r2",), ("r2", "r2")],
            targets=["A", "A", "B", "B"],
            groups=["g1", "g2", "g3", "g4"],
        )
        base = np.array([[0.7, 0.3], [0.7, 0.3]])

        _, meta = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",), ("A",)],
            raw_prefixes=[("r1",), ("r2",)],
            destination_fraction=0.0,
        )

        self.assertLess(meta.iloc[0]["exit_hazard"], meta.iloc[1]["exit_hazard"])

    def test_probability_mass_is_hazard_destination_factorization(self):
        model = QuotientSemiMarkovRouter(("A", "B", "C"), kappa=1.0).fit(
            parent_prefixes=[("A",), ("A",), ("B",), ("C",)],
            raw_prefixes=[("r1",), ("r2",), ("b",), ("c",)],
            targets=["B", "C", "A", "A"],
            groups=["g1", "g2", "g3", "g4"],
        )
        base = np.array([[0.6, 0.3, 0.1]])

        candidate, meta = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",)],
            raw_prefixes=[("r1",)],
            destination_fraction=0.5,
        )

        hazard = float(meta.iloc[0]["exit_hazard"])
        self.assertAlmostEqual(candidate[0, 0], 1.0 - hazard)
        self.assertAlmostEqual(candidate[0, 1:].sum(), hazard)
        np.testing.assert_allclose(candidate.sum(axis=1), np.ones(1))

    def test_zero_destination_fraction_preserves_base_nonself_ratios(self):
        model = QuotientSemiMarkovRouter(("A", "B", "C"), kappa=1.0).fit(
            parent_prefixes=[("A",), ("B",), ("C",)],
            raw_prefixes=[("a",), ("b",), ("c",)],
            targets=["B", "C", "A"],
            groups=["g1", "g2", "g3"],
        )
        base = np.array([[0.5, 0.4, 0.1]])

        candidate, _ = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",)],
            raw_prefixes=[("a",)],
            destination_fraction=0.0,
        )

        self.assertAlmostEqual(candidate[0, 1] / candidate[0, 2], 4.0)

    def test_zero_hazard_and_destination_fraction_exactly_reduce_to_base(self):
        model = QuotientSemiMarkovRouter(("A", "B", "C"), kappa=1.0).fit(
            parent_prefixes=[("A",), ("B",), ("C",)],
            raw_prefixes=[("a",), ("b",), ("c",)],
            targets=["B", "C", "A"],
            groups=["g1", "g2", "g3"],
        )
        base = np.array([[0.5, 0.4, 0.1]])

        candidate, _ = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",)],
            raw_prefixes=[("a",)],
            destination_fraction=0.0,
            hazard_fraction=0.0,
        )

        np.testing.assert_allclose(candidate, base)

    def test_raw_and_parent_prefixes_must_align(self):
        model = QuotientSemiMarkovRouter(("A", "B"))
        with self.assertRaises(ValueError):
            model.fit(
                parent_prefixes=[("A",)],
                raw_prefixes=[("r1", "r2")],
                targets=["B"],
                groups=["g1"],
            )

    def test_hazard_levels_must_form_canonical_nested_chain(self):
        with self.assertRaises(ValueError):
            QuotientSemiMarkovRouter(
                ("A", "B"), hazard_levels=("parent_dwell", "parent")
            )

    def test_low_support_destination_is_shrunk_to_nonself_prior(self):
        model = QuotientSemiMarkovRouter(
            ("A", "B", "C"), destination_kappa=5.0
        ).fit(
            parent_prefixes=[("A",), ("B",), ("C",)],
            raw_prefixes=[("a",), ("b",), ("c",)],
            targets=["B", "C", "A"],
            groups=["g1", "g2", "g3"],
        )

        base = np.array([[0.6, 0.2, 0.2]])
        candidate, _ = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",)],
            raw_prefixes=[("a",)],
            destination_fraction=1.0,
            hazard_fraction=0.0,
        )

        self.assertGreater(candidate[0, 2], 0.0)
        self.assertAlmostEqual(candidate[0, 0], base[0, 0])

    def test_last_raw_variant_disambiguates_exit_destination(self):
        model = QuotientSemiMarkovRouter(
            ("A", "B", "C"), kappa=1.0, destination_kappa=0.1
        ).fit(
            parent_prefixes=[("A",), ("A",), ("B",), ("C",)],
            raw_prefixes=[("r1",), ("r2",), ("b",), ("c",)],
            targets=["B", "C", "A", "A"],
            groups=["g1", "g2", "g3", "g4"],
        )
        base = np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])

        candidate, _ = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",), ("A",)],
            raw_prefixes=[("r1",), ("r2",)],
            destination_fraction=1.0,
            hazard_fraction=0.0,
        )

        self.assertGreater(candidate[0, 1], candidate[0, 2])
        self.assertGreater(candidate[1, 2], candidate[1, 1])

    def test_dwell_conditioned_destination_disambiguates_same_raw_variant(self):
        model = QuotientSemiMarkovRouter(
            ("A", "B", "C"),
            kappa=1.0,
            destination_kappa=0.1,
            destination_levels=("parent", "parent_raw", "parent_dwell_raw"),
        ).fit(
            parent_prefixes=[("A",), ("A", "A"), ("B",), ("C",)],
            raw_prefixes=[("r",), ("r", "r"), ("b",), ("c",)],
            targets=["B", "C", "A", "A"],
            groups=["g1", "g2", "g3", "g4"],
        )
        base = np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])

        candidate, _ = model.predict_proba_with_meta(
            base,
            parent_prefixes=[("A",), ("A", "A")],
            raw_prefixes=[("r",), ("r", "r")],
            destination_fraction=1.0,
            hazard_fraction=0.0,
        )

        self.assertGreater(candidate[0, 1], candidate[0, 2])
        self.assertGreater(candidate[1, 2], candidate[1, 1])

    def test_domain_balancing_prevents_one_domain_root_count_domination(self):
        compact = QuotientSemiMarkovRouter(("A", "B"), kappa=0.0).fit(
            parent_prefixes=[("A",), ("A",)],
            raw_prefixes=[("r",), ("r",)],
            targets=["A", "B"],
            groups=["a1", "b1"],
            domains=["source_a", "source_b"],
        )
        expanded = QuotientSemiMarkovRouter(("A", "B"), kappa=0.0).fit(
            parent_prefixes=[("A",), ("A",), ("A",), ("A",)],
            raw_prefixes=[("r",), ("r",), ("r",), ("r",)],
            targets=["A", "A", "A", "B"],
            groups=["a1", "a2", "a3", "b1"],
            domains=["source_a", "source_a", "source_a", "source_b"],
        )

        compact_estimate, compact_support = compact.hazard_tables_["parent"][("A",)]
        expanded_estimate, expanded_support = expanded.hazard_tables_["parent"][("A",)]
        self.assertAlmostEqual(compact_estimate, expanded_estimate)
        self.assertGreater(expanded_support, compact_support)


if __name__ == "__main__":
    unittest.main()
