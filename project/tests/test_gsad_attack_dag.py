import itertools
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.attack_dag import AttackDAG, compress_leaf_set  # noqa: E402


def brute_force_best(gamma, dag, lam, max_nodes):
    candidates = list(dag.tactic_ids) + sorted(gamma)
    best = None
    for size in range(max_nodes + 1):
        for selected in itertools.combinations(candidates, size):
            descendants = dag.descendants(selected)
            if not set(gamma).issubset(descendants):
                continue
            key = (
                len(descendants) + lam * len(selected),
                len(descendants),
                len(selected),
                tuple(sorted(selected)),
            )
            if best is None or key < best:
                best = key
    return best


class AttackDAGTests(unittest.TestCase):
    def test_multi_parent_membership_is_preserved(self):
        dag = AttackDAG.from_edges({"TA1": {"T1", "T2"}, "TA2": {"T2", "T3"}})
        self.assertEqual(dag.tactics_for("T2"), frozenset({"TA1", "TA2"}))

    def test_technique_node_descends_to_itself(self):
        dag = AttackDAG.from_edges({"TA1": {"T1", "T2"}})
        self.assertEqual(dag.descendants(["T1"]), frozenset({"T1"}))

    def test_unknown_technique_remains_a_leaf(self):
        dag = AttackDAG.from_edges({"TA1": {"T1"}})
        self.assertEqual(dag.descendants(["T9999"]), frozenset({"T9999"}))


class CompressionTests(unittest.TestCase):
    def test_label_sets_are_materialized_only_for_competitive_candidates(self):
        class CountingDAG(AttackDAG):
            def __init__(self, edges):
                super().__init__(edges)
                self.materializations = 0

            def labels_from_mask(self, mask):
                self.materializations += 1
                return super().labels_from_mask(mask)

        edges = {
            f"TA{index:02d}": {f"T{index:02d}", f"T{(index + 1) % 10:02d}"}
            for index in range(10)
        }
        dag = CountingDAG(edges)
        compress_leaf_set(
            frozenset({"T00", "T02", "T04", "T06", "T08"}),
            dag,
            lam=1.0,
            max_nodes=10,
        )
        self.assertLess(dag.materializations, 100)

    def test_compressor_is_exact_and_never_drops_gamma(self):
        dag = AttackDAG.from_edges({"TA1": {"T1", "T2"}, "TA2": {"T2", "T3"}})
        out = compress_leaf_set(
            frozenset({"T1", "T2"}), dag, lam=1.0, max_nodes=3
        )
        self.assertTrue({"T1", "T2"}.issubset(out.descendants))
        self.assertEqual(out.nodes, frozenset({"TA1"}))
        self.assertEqual(out.objective, 3.0)
        self.assertTrue(out.coverage_preserved)

    def test_enumeration_matches_full_brute_force_on_small_graphs(self):
        graphs = [
            {"TA1": {"T1", "T2"}, "TA2": {"T2", "T3"}},
            {"TA1": {"T1", "T3"}, "TA2": {"T2", "T4"}, "TA3": {"T3", "T4"}},
            {"TA1": {"T1", "T2", "T3"}, "TA2": {"T3", "T4", "T5"}},
        ]
        for edges in graphs:
            dag = AttackDAG.from_edges(edges)
            techniques = sorted(set().union(*edges.values()))
            for gamma_size in range(1, min(4, len(techniques) + 1)):
                for gamma in itertools.combinations(techniques, gamma_size):
                    with self.subTest(edges=edges, gamma=gamma):
                        out = compress_leaf_set(
                            frozenset(gamma), dag, lam=0.7, max_nodes=4
                        )
                        expected = brute_force_best(gamma, dag, lam=0.7, max_nodes=4)
                        actual = (
                            out.objective,
                            out.leaf_equivalent_size,
                            len(out.nodes),
                            tuple(sorted(out.nodes)),
                        )
                        self.assertEqual(actual, expected)

    def test_impossible_node_budget_is_rejected(self):
        dag = AttackDAG.from_edges({"TA1": {"T1"}})
        with self.assertRaisesRegex(ValueError, "no feasible"):
            compress_leaf_set(frozenset({"T1"}), dag, lam=1.0, max_nodes=0)


class RealAttackSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core = PROJECT_ROOT / "data_v2" / "core"
        cls.vocab = tuple(
            pd.read_csv(core / "rl_label_vocab.csv")
            .sort_values("label_id")["technique_id_parent"]
            .astype(str)
        )
        cls.dag = AttackDAG.from_stix(
            PROJECT_ROOT / "data" / "enterprise-attack-18.1.json", cls.vocab
        )

    def test_snapshot_has_tactics_and_multi_parent_techniques(self):
        self.assertGreaterEqual(len(self.dag.tactic_ids), 14)
        multi_parent = [
            label for label in self.vocab if len(self.dag.tactics_for(label)) > 1
        ]
        self.assertGreater(len(multi_parent), 0)

    def test_mapping_audit_accounts_for_entire_vocabulary(self):
        audit = self.dag.mapping_audit
        self.assertEqual(audit["vocab_size"], 184)
        self.assertEqual(audit["mapped_techniques"] + audit["missing_techniques"], 184)


if __name__ == "__main__":
    unittest.main()
