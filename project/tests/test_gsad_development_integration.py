import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.data_protocol import (  # noqa: E402
    FrozenSplit,
    TEST_ROOTS,
    build_frozen_split,
)
from experiments.gsad.run_development import (  # noqa: E402
    DevelopmentConfig,
    assign_balanced_root_folds,
    build_arg_parser,
    confidence_acceptance_at_accuracy,
    development_frame,
    evaluate_outer_fold,
    make_inner_roles,
    permute_targets_for_negative_control,
    run_development,
    summarize_development_predictions,
    threshold_audit,
)
from experiments.gsad.attack_dag import AttackDAG  # noqa: E402
from experiments.gsad.shift_policy import ExactThreshold  # noqa: E402


CORE_DIR = PROJECT_ROOT / "data_v2" / "core"


def load_split():
    frames = [
        pd.read_csv(CORE_DIR / "sim_train_parent_min3.csv"),
        pd.read_csv(CORE_DIR / "sim_val_parent_min3.csv"),
        pd.read_csv(CORE_DIR / "sim_test_parent_min3.csv"),
    ]
    return build_frozen_split(frames)


class DevelopmentAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = load_split()

    def test_development_frame_contains_133_roots_and_no_locked_root(self):
        frame = development_frame(self.split)
        roots = set(frame["root"])
        self.assertEqual(len(roots), 133)
        self.assertTrue(roots.isdisjoint(TEST_ROOTS))
        self.assertEqual(len(frame), 10555)

    def test_balanced_fold_assignment_never_splits_a_root(self):
        frame = development_frame(self.split)
        folds = assign_balanced_root_folds(frame, n_splits=5)
        audit = frame.assign(fold=folds).groupby("root")["fold"].nunique()
        self.assertTrue((audit == 1).all())
        self.assertEqual(set(folds), set(range(5)))
        fold_rows = pd.Series(folds).value_counts()
        self.assertLessEqual(int(fold_rows.max() - fold_rows.min()), 200)

    def test_inner_roles_are_disjoint_and_cover_outer_training_roots(self):
        frame = development_frame(self.split)
        folds = assign_balanced_root_folds(frame, n_splits=5)
        outer_training = frame.loc[folds != 0]
        roles = make_inner_roles(outer_training, validation_root_count=20, calibration_root_count=20)
        sets = [roles.fit_roots, roles.validation_roots, roles.calibration_roots]
        self.assertTrue(
            all(left.isdisjoint(right) for idx, left in enumerate(sets) for right in sets[idx + 1 :])
        )
        self.assertEqual(set().union(*sets), set(outer_training["root"]))
        self.assertEqual(len(roles.validation_roots), 20)
        self.assertEqual(len(roles.calibration_roots), 20)

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            DevelopmentConfig(candidate="tune_until_pass")

    def test_cli_exposes_only_preregistered_candidates(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--candidate", "gsad_shift", "--bootstrap", "25"])
        self.assertEqual(args.candidate, "gsad_shift")
        self.assertEqual(args.bootstrap, 25)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--candidate", "tune_until_pass"])

    def test_disabled_exact_threshold_audit_uses_json_null(self):
        audit = threshold_audit(
            ExactThreshold(
                threshold=float("inf"),
                enabled=False,
                coverage=0.0,
                empirical_risk=1.0,
                upper_risk=1.0,
                accepted_roots=0,
            )
        )
        self.assertIsNone(audit["threshold"])
        self.assertFalse(audit["enabled"])


class OuterFoldSmokeTests(unittest.TestCase):
    @staticmethod
    def synthetic_frame(roots):
        rows = []
        for root_index, root in enumerate(roots):
            for step in range(1, 5):
                first = "T1" if (root_index + step) % 2 == 0 else "T2"
                target = "T2" if first == "T1" else "T1"
                rows.append(
                    {
                        "sequence_id": f"{root}_part001",
                        "prefix_len": 1,
                        "root": root,
                        "prefix_ids": (first,),
                        "target": target,
                    }
                )
        return pd.DataFrame(rows)

    def test_outer_fold_returns_one_prediction_per_outer_row_without_role_overlap(self):
        frame = self.synthetic_frame([f"R{i:02d}" for i in range(12)])
        inner_fit = frame[frame.root.isin({"R00", "R01", "R02", "R03"})]
        validation = frame[frame.root.isin({"R04", "R05", "R06"})]
        calibration = frame[frame.root.isin({"R07", "R08", "R09"})]
        outer = frame[frame.root.isin({"R10", "R11"})]
        dag = AttackDAG({"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}}, vocab=("T1", "T2", "T3"))
        result = evaluate_outer_fold(
            inner_fit,
            validation,
            calibration,
            outer,
            vocab=("T1", "T2", "T3"),
            dag=dag,
            config=DevelopmentConfig(
                candidate="gsad_core",
                bootstrap=10,
                n_splits=3,
                min_cluster_support=1,
                min_calibration_support=1,
                max_nodes=3,
                confidence_z=0.0,
                exact_target_risk=1.0,
            ),
            fold_id=0,
        )
        self.assertEqual(len(result.predictions), len(outer))
        self.assertEqual(set(result.predictions["root"]), {"R10", "R11"})
        self.assertEqual(result.audit["role_overlaps"], {})
        self.assertTrue(
            {"gamma", "structured_descendants", "action_kind", "global_gamma"}.issubset(
                result.predictions.columns
            )
        )

    def test_summary_produces_every_gate_and_negative_control_changes_targets(self):
        frame = pd.DataFrame(
            {
                "root": ["A", "A", "B", "B"],
                "target": ["T1", "T2", "T1", "T2"],
                "top1_pred": ["T1", "T1", "T1", "T1"],
                "top1_probability": [0.9, 0.8, 0.7, 0.6],
                "top5": [("T1", "T2")] * 4,
                "gamma": [frozenset({"T1"}), frozenset({"T1", "T2"})] * 2,
                "global_gamma": [frozenset({"T1", "T2"})] * 4,
                "structured_nodes": [frozenset({"T1"}), frozenset({"TA1"})] * 2,
                "structured_descendants": [frozenset({"T1"}), frozenset({"T1", "T2"})] * 2,
                "action_kind": ["exact", "dag", "exact", "dag"],
                "exact_label": ["T1", "", "T1", ""],
                "descendants": [frozenset({"T1"}), frozenset({"T1", "T2"})] * 2,
                "leaf_equivalent_size": [1, 2, 1, 2],
                "display_node_count": [1, 1, 1, 1],
                "raw_structured_leaf_size": [1, 2, 1, 2],
                "safety_score": [0.9, 0.8, 0.7, 0.6],
                "fit_seen": [True] * 4,
                "vocab_size": [2] * 4,
                "confidence_accept_matched": [True, True, False, False],
                "base_correct": [True, False, True, False],
                "global_leaf_equivalent_size": [2] * 4,
                "global_leaf_hit": [True] * 4,
            }
        )
        summary = summarize_development_predictions(
            frame, candidate="gsad_core", n_boot=30, seed=4
        )
        self.assertEqual(set("ABCDEFG") | {"PRIMARY"}, set(summary.gates))
        self.assertIn("row_coverage", summary.metrics)
        permuted = permute_targets_for_negative_control(frame, seed=4)
        self.assertCountEqual(permuted["target"], frame["target"])
        self.assertFalse(permuted["target"].equals(frame["target"]))

    def test_confidence_comparator_freezes_largest_coverage_at_target_accuracy(self):
        frame = pd.DataFrame(
            {
                "root": ["A", "B", "C", "D"],
                "top1_probability": [0.9, 0.8, 0.7, 0.6],
                "base_correct": [True, True, False, False],
            }
        )
        accepted = confidence_acceptance_at_accuracy(frame, target_accuracy=1.0)
        self.assertEqual(accepted.tolist(), [True, True, False, False])


class DevelopmentRunnerTests(unittest.TestCase):
    def test_end_to_end_oof_covers_all_development_roots_and_keeps_test_unseen(self):
        labels = ("T1", "T2", "T3")

        def rows(prefix, count):
            records = []
            for index in range(count):
                root = f"{prefix}_{index:03d}"
                state = labels[index % len(labels)]
                target = labels[(index + 1) % len(labels)]
                records.append(
                    {
                        "sequence_id": f"{root}_part001",
                        "prefix_len": 1,
                        "root": root,
                        "prefix_ids": (state,),
                        "target": target,
                    }
                )
            return pd.DataFrame(records)

        split = FrozenSplit(
            fit=rows("F", 93),
            validation=rows("V", 20),
            calibration=rows("C", 20),
            test=rows("LOCKED", 20),
            excluded_roots=frozenset(),
            audit={"synthetic": True},
        )
        dag = AttackDAG(
            {"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}}, vocab=labels
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_development(
                DevelopmentConfig(
                    candidate="gsad_core",
                    n_splits=2,
                    bootstrap=10,
                    min_cluster_support=1,
                    min_calibration_support=1,
                    max_nodes=3,
                    confidence_z=0.0,
                    exact_target_risk=1.0,
                ),
                split=split,
                vocab=labels,
                dag=dag,
                output_dir=Path(directory) / "result",
            )
            self.assertEqual(len(result.predictions), 133)
            self.assertEqual(result.predictions["root"].nunique(), 133)
            self.assertTrue(set(result.predictions["root"]).isdisjoint(split.test["root"]))
            self.assertEqual(len(result.fold_audits), 2)
            self.assertFalse(result.negative_control.gates["PRIMARY"].passed)
            expected = {
                "predictions.csv",
                "metrics.csv",
                "bootstrap_intervals.csv",
                "gates.json",
                "negative_control_gates.json",
                "fold_audit.json",
                "model_configs.json",
                "data_audit.json",
                "run_manifest.json",
                "iteration_summary.md",
            }
            self.assertTrue(expected.issubset({item.name for item in result.output_dir.iterdir()}))


if __name__ == "__main__":
    unittest.main()
