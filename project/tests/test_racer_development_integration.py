import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.attack_dag import AttackDAG  # noqa: E402
from experiments.gsad.data_protocol import FrozenSplit  # noqa: E402
from experiments.gsad.run_racer_development import (  # noqa: E402
    RacerConfig,
    evaluate_racer_outer_fold,
    run_racer_development,
    summarize_racer_predictions,
)


def synthetic_frame(roots, labels=("T1", "T2", "T3"), rows_per_root=4):
    records = []
    for root_index, root in enumerate(roots):
        for step in range(rows_per_root):
            first = labels[(root_index + step) % len(labels)]
            second = labels[(root_index + step + 1) % len(labels)]
            target = labels[(root_index + step + 2) % len(labels)]
            records.append(
                {
                    "sequence_id": f"{root}_part001",
                    "prefix_len": 2,
                    "root": root,
                    "prefix_ids": (first, second),
                    "target": target,
                }
            )
    return pd.DataFrame(records)


class RacerOuterFoldTests(unittest.TestCase):
    def test_outer_fold_keeps_roles_disjoint_and_emits_paired_outputs(self):
        frame = synthetic_frame([f"R{i:02d}" for i in range(12)])
        result = evaluate_racer_outer_fold(
            inner_fit=frame[frame.root.isin({"R00", "R01", "R02", "R03"})],
            validation=frame[frame.root.isin({"R04", "R05", "R06"})],
            calibration=frame[frame.root.isin({"R07", "R08", "R09"})],
            outer=frame[frame.root.isin({"R10", "R11"})],
            vocab=("T1", "T2", "T3"),
            dag=AttackDAG(
                {"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}},
                vocab=("T1", "T2", "T3"),
            ),
            config=RacerConfig(
                bootstrap=10,
                n_splits=3,
                sar_max_contexts=(1, 2),
                sar_kappas=(1.0,),
            ),
            fold_id=0,
        )
        self.assertEqual(len(result.predictions), 8)
        self.assertEqual(result.audit["role_overlaps"], {})
        self.assertTrue(
            {
                "baseline_top1",
                "sar_top1",
                "racer_top1",
                "baseline_set",
                "racer_set",
                "baseline_rank",
                "sar_rank",
                "racer_rank",
                "is_self",
            }.issubset(result.predictions.columns)
        )
        self.assertTrue(all(result.predictions["racer_set"].map(bool)))
        self.assertGreaterEqual(result.audit["rank_union"]["threshold"], 1)
        self.assertIn("opinion_pool", result.model_config)

    def test_summary_reports_ranking_set_and_nonself_gates(self):
        frame = pd.DataFrame(
            {
                "root": ["A", "A", "B", "B"],
                "target": ["T1", "T2", "T1", "T2"],
                "baseline_top1": ["T2", "T2", "T2", "T2"],
                "sar_top1": ["T1", "T2", "T1", "T2"],
                "racer_top1": ["T1", "T2", "T1", "T2"],
                "baseline_rank": [2, 1, 2, 1],
                "sar_rank": [1, 1, 1, 1],
                "racer_rank": [1, 1, 1, 1],
                "baseline_hit5": [True] * 4,
                "sar_hit5": [True] * 4,
                "racer_hit5": [True] * 4,
                "baseline_set": [frozenset({"T1", "T2"})] * 4,
                "racer_set": [frozenset({"T1"}), frozenset({"T2"})] * 2,
                "is_self": [False] * 4,
                "tail_label": [True] * 4,
                "vocab_size": [2] * 4,
            }
        )
        summary = summarize_racer_predictions(frame, n_boot=30, seed=3)
        self.assertTrue(
            {"top1_gain_pp", "mrr_gain", "set_reduction_relative"}.issubset(
                summary.metrics
            )
        )
        self.assertTrue({"R1", "R2", "R3", "R4", "S1", "S2", "S3", "PRIMARY"}.issubset(summary.gates))


class RacerRunnerTests(unittest.TestCase):
    def test_runner_never_scores_locked_partition_and_writes_audit_artifacts(self):
        labels = ("T1", "T2", "T3")
        split = FrozenSplit(
            fit=synthetic_frame([f"F_{i:03d}" for i in range(93)], rows_per_root=1),
            validation=synthetic_frame([f"V_{i:03d}" for i in range(20)], rows_per_root=1),
            calibration=synthetic_frame([f"C_{i:03d}" for i in range(20)], rows_per_root=1),
            test=synthetic_frame([f"LOCKED_{i:03d}" for i in range(20)], rows_per_root=1),
            excluded_roots=frozenset(),
            audit={"synthetic": True},
        )
        dag = AttackDAG(
            {"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}}, vocab=labels
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_racer_development(
                RacerConfig(
                    n_splits=2,
                    bootstrap=10,
                    sar_max_contexts=(1,),
                    sar_kappas=(1.0,),
                ),
                split=split,
                vocab=labels,
                dag=dag,
                output_dir=Path(directory) / "racer",
            )
            self.assertEqual(len(result.predictions), 133)
            self.assertTrue(
                set(result.predictions["root"]).isdisjoint(set(split.test["root"]))
            )
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
            self.assertTrue(
                expected.issubset({item.name for item in result.output_dir.iterdir()})
            )


if __name__ == "__main__":
    unittest.main()
