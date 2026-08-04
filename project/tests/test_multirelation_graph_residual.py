import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.experiments.gsad.multirelation_graph_residual import (
    MultiRelationGraphResidual,
    SelfCensoredMultiRelationGraphResidual,
    TypedGraphEscapeRedistributor,
    build_attack_relation_matrices,
)


class MultiRelationGraphResidualTests(unittest.TestCase):
    def test_root_duplication_does_not_change_transition_estimate(self):
        relation = np.array([[0.0, 1.0], [1.0, 0.0]])
        base = MultiRelationGraphResidual(
            vocab=("A", "B"), relation_matrix=relation, support_kappa=2.0
        ).fit(
            prefixes=[("A",), ("A",), ("B",)],
            targets=["A", "A", "B"],
            groups=["g1", "g1", "g2"],
        )
        duplicated = MultiRelationGraphResidual(
            vocab=("A", "B"), relation_matrix=relation, support_kappa=2.0
        ).fit(
            prefixes=[("A",), ("A",), ("A",), ("B",)],
            targets=["A", "A", "A", "B"],
            groups=["g1", "g1", "g1", "g2"],
        )

        np.testing.assert_allclose(base.local_transition_, duplicated.local_transition_)

    def test_sparse_source_borrows_neighbor_transition_without_self_loop(self):
        relation = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        model = MultiRelationGraphResidual(
            vocab=("A", "B", "C"), relation_matrix=relation, support_kappa=2.0
        ).fit(
            prefixes=[("B",), ("B",), ("B",), ("C",)],
            targets=["C", "C", "C", "A"],
            groups=["g1", "g2", "g3", "g4"],
        )

        base = np.full((1, 3), 1.0 / 3.0)
        probabilities, meta = model.predict_proba_with_meta(base, [("A",)])

        self.assertGreater(probabilities[0, 2], probabilities[0, 0])
        self.assertEqual(int(meta.iloc[0]["source_root_support"]), 0)
        self.assertGreater(float(meta.iloc[0]["graph_weight"]), 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1))

    def test_zero_residual_weight_exactly_reduces_to_base(self):
        model = MultiRelationGraphResidual(
            vocab=("A", "B"),
            relation_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
            residual_weight=0.0,
        ).fit(
            prefixes=[("A",), ("B",)],
            targets=["B", "A"],
            groups=["g1", "g2"],
        )
        base = np.array([[0.8, 0.2], [0.1, 0.9]])

        probabilities, _ = model.predict_proba_with_meta(base, [("A",), ("B",)])

        np.testing.assert_allclose(probabilities, base)

    def test_stix_relations_are_symmetric_normalized_and_have_no_diagonal(self):
        bundle = {
            "objects": [
                {
                    "type": "attack-pattern",
                    "name": "Alpha collect files",
                    "description": "collect files from a local directory",
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T0001"}
                    ],
                    "kill_chain_phases": [
                        {"kill_chain_name": "mitre-attack", "phase_name": "collection"}
                    ],
                },
                {
                    "type": "attack-pattern",
                    "name": "Beta archive files",
                    "description": "archive collected files in a directory",
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T0002"}
                    ],
                    "kill_chain_phases": [
                        {"kill_chain_name": "mitre-attack", "phase_name": "collection"}
                    ],
                },
                {
                    "type": "attack-pattern",
                    "name": "Gamma execute shell",
                    "description": "execute a command shell",
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T0003"}
                    ],
                    "kill_chain_phases": [
                        {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attack.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            semantic, tactic, audit = build_attack_relation_matrices(
                path, ("T0001", "T0002", "T0003"), semantic_neighbors=1
            )

        np.testing.assert_allclose(np.diag(semantic), np.zeros(3))
        np.testing.assert_allclose(np.diag(tactic), np.zeros(3))
        np.testing.assert_allclose(semantic, semantic.T)
        np.testing.assert_allclose(tactic, tactic.T)
        self.assertGreater(semantic[0, 1], 0.0)
        self.assertGreater(tactic[0, 1], 0.0)
        self.assertEqual(audit["mapped_descriptions"], 3)


class SelfCensoredMultiRelationGraphResidualTests(unittest.TestCase):
    def test_transition_is_root_balanced_and_has_zero_diagonal(self):
        model = SelfCensoredMultiRelationGraphResidual(
            vocab=("A", "B", "C"),
            semantic_matrix=np.zeros((3, 3)),
            tactic_matrix=np.zeros((3, 3)),
        ).fit(
            prefixes=[("A",), ("A",), ("A",), ("A",), ("A",)],
            targets=["B", "B", "B", "C", "A"],
            groups=["g1", "g1", "g1", "g2", "g3"],
        )

        np.testing.assert_allclose(np.diag(model.transition_matrix_), np.zeros(3))
        self.assertAlmostEqual(model.transition_matrix_[0, 1], 0.5)
        self.assertAlmostEqual(model.transition_matrix_[0, 2], 0.5)

    def test_root_row_duplication_is_invariant(self):
        kwargs = {
            "vocab": ("A", "B"),
            "semantic_matrix": np.zeros((2, 2)),
            "tactic_matrix": np.zeros((2, 2)),
        }
        base = SelfCensoredMultiRelationGraphResidual(**kwargs).fit(
            [("A",), ("A",)], ["B", "A"], ["g1", "g2"]
        )
        duplicated = SelfCensoredMultiRelationGraphResidual(**kwargs).fit(
            [("A",), ("A",), ("A",)], ["B", "B", "A"], ["g1", "g1", "g2"]
        )

        np.testing.assert_allclose(base.transition_matrix_, duplicated.transition_matrix_)

    def test_static_relations_are_applied_after_transition(self):
        semantic = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        model = SelfCensoredMultiRelationGraphResidual(
            vocab=("A", "B", "C"),
            semantic_matrix=semantic,
            tactic_matrix=np.zeros((3, 3)),
            prior_smoothing=0.0,
        ).fit(
            prefixes=[("A",), ("B",), ("C",)],
            targets=["B", "A", "A"],
            groups=["g1", "g2", "g3"],
        )

        components = model.component_probabilities([("A",)])

        self.assertGreater(components["transition"][0, 1], 0.99)
        self.assertGreater(components["semantic"][0, 2], 0.99)

    def test_current_parent_residual_is_exactly_censored(self):
        relation = np.array([[0.0, 1.0], [1.0, 0.0]])
        model = SelfCensoredMultiRelationGraphResidual(
            vocab=("A", "B"),
            semantic_matrix=relation,
            tactic_matrix=relation,
        ).fit(
            prefixes=[("A",), ("B",)],
            targets=["B", "A"],
            groups=["g1", "g2"],
        )
        base = np.array([[0.8, 0.2]])

        _, meta = model.predict_proba_with_meta(
            base,
            [("A",)],
            relation_weights=(0.5, 0.25, 0.25),
            residual_strength=0.5,
        )

        self.assertEqual(float(meta.iloc[0]["current_parent_residual"]), 0.0)

    def test_zero_strength_exactly_reduces_to_base(self):
        model = SelfCensoredMultiRelationGraphResidual(
            vocab=("A", "B"),
            semantic_matrix=np.zeros((2, 2)),
            tactic_matrix=np.zeros((2, 2)),
        ).fit(
            prefixes=[("A",), ("B",)],
            targets=["B", "A"],
            groups=["g1", "g2"],
        )
        base = np.array([[0.7, 0.3], [0.2, 0.8]])

        candidate, _ = model.predict_proba_with_meta(
            base, [("A",), ("B",)], (1.0, 0.0, 0.0), 0.0
        )

        np.testing.assert_allclose(candidate, base)


class TypedGraphEscapeRedistributorTests(unittest.TestCase):
    def test_seen_probabilities_and_total_escape_mass_are_preserved(self):
        model = TypedGraphEscapeRedistributor(
            vocab=("A", "B", "C"),
            semantic_matrix=np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            tactic_matrix=np.zeros((3, 3)),
            max_context=1,
        ).fit(
            prefixes=[("A",), ("B",), ("C",)],
            targets=["B", "C", "A"],
            groups=["g1", "g2", "g3"],
        )
        base = np.array([[0.6, 0.3, 0.1]])

        candidate, meta = model.predict_proba_with_meta(
            base, [("A",)], (0.5, 0.0, 0.5), graph_fraction=1.0
        )

        self.assertAlmostEqual(candidate[0, 1], base[0, 1])
        self.assertAlmostEqual(candidate[0, [0, 2]].sum(), base[0, [0, 2]].sum())
        self.assertAlmostEqual(float(meta.iloc[0]["escape_mass"]), 0.7)
        np.testing.assert_allclose(candidate.sum(axis=1), np.ones(1))

    def test_semantic_relation_smooths_source_successor_not_target(self):
        semantic = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        model = TypedGraphEscapeRedistributor(
            vocab=("A", "B", "C"),
            semantic_matrix=semantic,
            tactic_matrix=np.zeros((3, 3)),
        ).fit(
            prefixes=[("A",), ("B",), ("C",)],
            targets=["B", "C", "A"],
            groups=["g1", "g2", "g3"],
        )

        components = model.component_probabilities([("A",)])

        self.assertGreater(components["transition"][0, 1], 0.99)
        self.assertGreater(components["semantic_source"][0, 2], 0.99)

    def test_zero_graph_fraction_exactly_reduces_to_base(self):
        model = TypedGraphEscapeRedistributor(
            vocab=("A", "B"),
            semantic_matrix=np.zeros((2, 2)),
            tactic_matrix=np.zeros((2, 2)),
        ).fit(
            prefixes=[("A",), ("B",)],
            targets=["B", "A"],
            groups=["g1", "g2"],
        )
        base = np.array([[0.7, 0.3], [0.2, 0.8]])

        candidate, _ = model.predict_proba_with_meta(
            base, [("A",), ("B",)], (1.0, 0.0, 0.0), graph_fraction=0.0
        )

        np.testing.assert_allclose(candidate, base)

    def test_longest_seen_context_controls_escape_set(self):
        model = TypedGraphEscapeRedistributor(
            vocab=("A", "B", "C"),
            semantic_matrix=np.zeros((3, 3)),
            tactic_matrix=np.zeros((3, 3)),
            max_context=2,
        ).fit(
            prefixes=[("A", "B"), ("C", "B"), ("A",)],
            targets=["C", "A", "B"],
            groups=["g1", "g2", "g3"],
        )
        base = np.full((1, 3), 1.0 / 3.0)

        _, meta = model.predict_proba_with_meta(
            base, [("A", "B")], (1.0, 0.0, 0.0), graph_fraction=0.5
        )

        self.assertEqual(int(meta.iloc[0]["context_order"]), 2)
        self.assertEqual(int(meta.iloc[0]["seen_successor_count"]), 1)


if __name__ == "__main__":
    unittest.main()
