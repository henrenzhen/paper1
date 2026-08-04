import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.artifacts import (  # noqa: E402
    freeze_candidate,
    write_canonical_json,
    write_manifest,
)
from experiments.gsad.attack_dag import AttackDAG  # noqa: E402
from experiments.gsad.data_protocol import FrozenSplit  # noqa: E402
from experiments.gsad.metrics import GateResult  # noqa: E402
from experiments.gsad.run_racer_development import (  # noqa: E402
    RacerConfig,
    frozen_file_hashes,
)
import experiments.gsad.run_racer_locked as locked_module  # noqa: E402
from experiments.gsad.run_racer_locked import (  # noqa: E402
    load_freeze_token,
    run_frozen_locked_evaluation,
    run_locked_evaluation,
    verify_development_bundle,
)
from project.tests.test_racer_development_integration import synthetic_frame  # noqa: E402


class LockedRacerEvaluationTests(unittest.TestCase):
    def test_locked_partition_is_scored_once_and_claimed_before_delivery(self):
        labels = ("T1", "T2", "T3")
        split = FrozenSplit(
            fit=synthetic_frame([f"F_{i:03d}" for i in range(50)], rows_per_root=1),
            validation=synthetic_frame([f"V_{i:03d}" for i in range(8)], rows_per_root=1),
            calibration=synthetic_frame([f"C_{i:03d}" for i in range(8)], rows_per_root=1),
            test=synthetic_frame([f"L_{i:03d}" for i in range(8)], rows_per_root=1),
            excluded_roots=frozenset(),
            audit={"synthetic": True},
        )
        dag = AttackDAG(
            {"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}}, vocab=labels
        )
        config = RacerConfig(
            bootstrap=10,
            sar_max_contexts=(1,),
            sar_kappas=(1.0,),
            opinion_weights=(0.0, 0.5, 1.0),
        )
        passed = {
            "PRIMARY": GateResult(True, None, "synthetic", None, "passed")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_path = root / "freeze.json"
            freeze_candidate({"candidate": "racer"}, passed, freeze_path)
            token = load_freeze_token(freeze_path)
            result = run_locked_evaluation(
                config=config,
                split=split,
                vocab=labels,
                dag=dag,
                token=token,
                results_dir=root / "locked",
            )
            self.assertEqual(len(result.predictions), len(split.test))
            self.assertEqual(set(result.predictions["root"]), set(split.test["root"]))
            self.assertTrue((root / "locked" / "LOCKED_EVALUATION_CLAIMED.json").is_file())
            self.assertTrue((root / "LOCKED_EVALUATION_CLAIMED.json").is_file())
            self.assertTrue((root / "locked" / "locked_metrics.csv").is_file())
            self.assertTrue((root / "locked" / "locked_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                run_locked_evaluation(
                    config=config,
                    split=split,
                    vocab=labels,
                    dag=dag,
                    token=token,
                    results_dir=root / "different-locked-dir",
                )

    def test_tampered_freeze_file_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_path = root / "freeze.json"
            token = freeze_candidate(
                {"candidate": "racer"},
                {"PRIMARY": GateResult(True, None, "ok", None, "ok")},
                freeze_path,
            )
            freeze_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_freeze_token(freeze_path)
            self.assertFalse((root / "locked").exists())
            self.assertTrue(token.digest)

    def test_frozen_bundle_rejects_internally_valid_but_wrong_file_hash(self):
        config = RacerConfig(bootstrap=10)
        gates = {"PRIMARY": {"passed": True}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = frozen_file_hashes()
            first = next(iter(hashes))
            hashes[first] = "0" * 64
            manifest = write_manifest(
                root / "run_manifest.json",
                inputs={"files": hashes},
                config=asdict(config),
                split_audit={"synthetic": True},
            )
            write_canonical_json(root / "gates.json", gates)
            token = freeze_candidate(
                {
                    "candidate": "racer",
                    "development_config": asdict(config),
                    "manifest_digest": manifest["manifest_digest"],
                },
                gates,
                root / "freeze_token.json",
            )
            with self.assertRaisesRegex(ValueError, "source or input"):
                verify_development_bundle(token)

    def test_safe_entry_claims_globally_before_verification_and_loading(self):
        labels = ("T1", "T2", "T3")
        split = FrozenSplit(
            fit=synthetic_frame([f"F_{i:03d}" for i in range(12)], rows_per_root=1),
            validation=synthetic_frame([f"V_{i:03d}" for i in range(4)], rows_per_root=1),
            calibration=synthetic_frame([f"C_{i:03d}" for i in range(4)], rows_per_root=1),
            test=synthetic_frame([f"L_{i:03d}" for i in range(4)], rows_per_root=1),
            excluded_roots=frozenset(),
            audit={"synthetic": True},
        )
        dag = AttackDAG(
            {"TA1": {"T1"}, "TA2": {"T2"}, "TA3": {"T3"}}, vocab=labels
        )
        config = RacerConfig(
            bootstrap=5,
            sar_max_contexts=(1,),
            sar_kappas=(1.0,),
            opinion_weights=(0.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = freeze_candidate(
                {"candidate": "racer"},
                {"PRIMARY": {"passed": True}},
                root / "freeze_token.json",
            )

            def verified(_token):
                self.assertTrue((root / "LOCKED_EVALUATION_CLAIMED.json").is_file())
                return config

            def loaded():
                self.assertTrue((root / "LOCKED_EVALUATION_CLAIMED.json").is_file())
                return split, labels, dag

            with mock.patch.object(
                locked_module, "verify_development_bundle", side_effect=verified
            ), mock.patch.object(
                locked_module, "_load_default_experiment", side_effect=loaded
            ):
                result = run_frozen_locked_evaluation(token, root / "locked")
            self.assertEqual(len(result.predictions), len(split.test))


if __name__ == "__main__":
    unittest.main()
