import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.data_protocol import (  # noqa: E402
    CALIBRATION_ROOTS,
    EXTERNAL_SIM_ROOTS,
    TEST_ROOTS,
    VALIDATION_ROOTS,
    audit_feature_columns,
    build_frozen_split,
    normalize_ctid_actor,
    sim_root,
)


CORE_DIR = PROJECT_ROOT / "data_v2" / "core"


def load_core_frames() -> list[pd.DataFrame]:
    return [
        pd.read_csv(CORE_DIR / "sim_train_parent_min3.csv"),
        pd.read_csv(CORE_DIR / "sim_val_parent_min3.csv"),
        pd.read_csv(CORE_DIR / "sim_test_parent_min3.csv"),
    ]


class RootParsingTests(unittest.TestCase):
    def test_root_suffix_is_removed(self):
        self.assertEqual(sim_root("SIM_010_part042"), "SIM_010")

    def test_non_part_identifier_is_unchanged(self):
        self.assertEqual(sim_root("SIM_010"), "SIM_010")


class FrozenSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_frozen_split(load_core_frames())

    def test_fixed_root_lists_have_expected_sizes(self):
        self.assertEqual(len(VALIDATION_ROOTS), 20)
        self.assertEqual(len(CALIBRATION_ROOTS), 20)
        self.assertEqual(len(TEST_ROOTS), 20)

    def test_external_actor_roots_are_removed_and_partitions_disjoint(self):
        split = self.split
        self.assertEqual(
            [len(x) for x in (split.fit, split.validation, split.calibration, split.test)],
            [7371, 1592, 1592, 1592],
        )
        self.assertEqual(
            [x["root"].nunique() for x in (split.fit, split.validation, split.calibration, split.test)],
            [93, 20, 20, 20],
        )
        root_sets = [
            set(x["root"])
            for x in (split.fit, split.validation, split.calibration, split.test)
        ]
        self.assertTrue(
            all(
                left.isdisjoint(right)
                for idx, left in enumerate(root_sets)
                for right in root_sets[idx + 1 :]
            )
        )
        self.assertTrue(set(EXTERNAL_SIM_ROOTS).isdisjoint(set().union(*root_sets)))

    def test_split_has_expected_total_and_unique_sample_keys(self):
        combined = pd.concat(
            [self.split.fit, self.split.validation, self.split.calibration, self.split.test],
            ignore_index=True,
        )
        self.assertEqual(len(combined), 12147)
        self.assertFalse(combined.duplicated(["sequence_id", "prefix_len"]).any())

    def test_audit_records_excluded_and_retained_counts(self):
        self.assertEqual(self.split.audit["source_rows"], 14128)
        self.assertEqual(self.split.audit["retained_rows"], 12147)
        self.assertEqual(self.split.audit["eligible_roots"], 153)
        self.assertEqual(self.split.audit["excluded_rows"], 1981)


class FeatureAuditTests(unittest.TestCase):
    def test_allowed_inference_fields_pass(self):
        audit_feature_columns(
            ["sequence_id", "root", "prefix_len", "prefix_ids"],
            target_columns=["target"],
        )

    def test_future_context_is_rejected(self):
        blocked = [
            "next_technique_id_parent",
            "true_label",
            "matched_description",
            "matched_command_summary",
            "matched_technique_name",
        ]
        for name in blocked:
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit_feature_columns(
                    ["sequence_id", "prefix_len", name], target_columns=["target"]
                )

    def test_target_is_not_accepted_as_feature(self):
        with self.assertRaises(ValueError):
            audit_feature_columns(["prefix_len", "target"], target_columns=["target"])


class CtidActorTests(unittest.TestCase):
    def test_turla_plans_share_one_actor(self):
        self.assertEqual(normalize_ctid_actor("turla_carbon"), "turla")
        self.assertEqual(normalize_ctid_actor("Turla Snake"), "turla")

    def test_other_actor_names_are_normalized(self):
        self.assertEqual(normalize_ctid_actor(" Wizard Spider "), "wizard_spider")


if __name__ == "__main__":
    unittest.main()
