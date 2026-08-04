from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from project.experiments.gsad.dr_vom_validation import (
    expected_calibration_error,
    prediction_diagnostics,
    root_macro_diagnostics,
)
from project.experiments.gsad.run_dr_vom_validation import (
    clone_domain_roots,
    duplicate_domain_rows,
    duplicate_root_rows,
    evaluate_domains,
    fit_validation_models,
    run_validation,
)


class PredictionDiagnosticsTests(unittest.TestCase):
    def test_computes_hand_derived_proper_scores_and_ranks(self):
        probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=float)

        frame = prediction_diagnostics(
            probabilities,
            targets=["A", "A"],
            vocabulary=["A", "B"],
        )

        np.testing.assert_allclose(
            frame["nll"].to_numpy(),
            [-np.log(0.8), -np.log(0.4)],
        )
        np.testing.assert_allclose(frame["brier"].to_numpy(), [0.08, 0.72])
        self.assertEqual(frame["hit1"].tolist(), [1.0, 0.0])
        self.assertEqual(frame["hit3"].tolist(), [1.0, 1.0])
        self.assertEqual(frame["rr"].tolist(), [1.0, 0.5])
        np.testing.assert_allclose(frame["confidence"].to_numpy(), [0.8, 0.6])

    def test_rejects_unknown_target_and_invalid_probability_rows(self):
        with self.assertRaisesRegex(ValueError, "unknown target"):
            prediction_diagnostics([[0.5, 0.5]], ["C"], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            prediction_diagnostics([[0.5, 0.4]], ["A"], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "finite"):
            prediction_diagnostics([[np.nan, np.nan]], ["A"], ["A", "B"])

    def test_ece_and_root_macro_use_declared_aggregation(self):
        frame = pd.DataFrame(
            {
                "root": ["long", "long", "short"],
                "model_nll": [0.0, 2.0, 3.0],
                "model_brier": [0.0, 1.0, 2.0],
                "model_hit1": [1.0, 0.0, 1.0],
                "model_rr": [1.0, 0.5, 1.0],
                "model_confidence": [0.8, 0.6, 0.6],
                "model_correct": [1.0, 0.0, 1.0],
            }
        )

        metrics = root_macro_diagnostics(frame, "model")

        self.assertAlmostEqual(metrics["nll"], 2.0)
        self.assertAlmostEqual(metrics["brier"], 1.25)
        self.assertAlmostEqual(metrics["top1"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], 0.875)
        ece_frame = pd.DataFrame(
            {
                "model_confidence": [0.8, 0.6],
                "model_correct": [1.0, 0.0],
            }
        )
        self.assertAlmostEqual(
            expected_calibration_error(ece_frame, "model", 2),
            0.2,
        )


class ValidationAblationTests(unittest.TestCase):
    @staticmethod
    def _training_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prefix_ids": [["X"], ["X"], ["X"], ["X"], ["X"]],
                "target": ["A", "A", "A", "B", "B"],
                "root": ["d1-long", "d1-long", "d1-long", "d1-short", "d2"],
                "domain": ["d1", "d1", "d1", "d1", "d2"],
            }
        )

    @staticmethod
    def _probabilities(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        models = fit_validation_models(frame, ["A", "B"])
        return {
            name: model.predict_proba_with_meta([["X"]])[0][0]
            for name, model in models.items()
        }

    def test_ablation_models_encode_distinct_aggregation_targets(self):
        probabilities = self._probabilities(self._training_frame())

        self.assertEqual(
            list(probabilities),
            ["row_pooled", "root_balanced", "domain_row_balanced", "dr_vom"],
        )
        np.testing.assert_allclose(
            probabilities["row_pooled"],
            [0.65 / 1.1, 0.45 / 1.1],
        )
        np.testing.assert_allclose(
            probabilities["root_balanced"],
            [(1.0 / 3.0 + 0.05) / 1.1, (2.0 / 3.0 + 0.05) / 1.1],
        )
        np.testing.assert_allclose(
            probabilities["domain_row_balanced"],
            [0.425 / 1.1, 0.675 / 1.1],
        )
        np.testing.assert_allclose(
            probabilities["dr_vom"],
            [0.30 / 1.1, 0.80 / 1.1],
        )

    def test_single_root_row_duplication_only_moves_models_using_row_frequency(self):
        original = self._training_frame()
        duplicated = duplicate_root_rows(original, "d1-long", factor=5)
        original_probs = self._probabilities(original)
        duplicated_probs = self._probabilities(duplicated)

        self.assertGreater(
            np.abs(original_probs["row_pooled"] - duplicated_probs["row_pooled"]).max(),
            1e-3,
        )
        np.testing.assert_allclose(
            original_probs["root_balanced"],
            duplicated_probs["root_balanced"],
            atol=1e-12,
        )
        self.assertGreater(
            np.abs(
                original_probs["domain_row_balanced"]
                - duplicated_probs["domain_row_balanced"]
            ).max(),
            1e-3,
        )
        np.testing.assert_allclose(
            original_probs["dr_vom"],
            duplicated_probs["dr_vom"],
            atol=1e-12,
        )

    def test_source_root_cloning_only_moves_models_without_domain_balance(self):
        original = self._training_frame()
        cloned = clone_domain_roots(original, "d1", factor=5)
        original_probs = self._probabilities(original)
        cloned_probs = self._probabilities(cloned)

        self.assertGreater(
            np.abs(original_probs["root_balanced"] - cloned_probs["root_balanced"]).max(),
            1e-3,
        )
        np.testing.assert_allclose(
            original_probs["domain_row_balanced"],
            cloned_probs["domain_row_balanced"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            original_probs["dr_vom"],
            cloned_probs["dr_vom"],
            atol=1e-12,
        )

    def test_perturbations_validate_factor_and_preserve_original_rows(self):
        frame = self._training_frame()
        with self.assertRaisesRegex(ValueError, "factor"):
            duplicate_domain_rows(frame, "d1", factor=0)
        with self.assertRaisesRegex(ValueError, "unknown domain"):
            clone_domain_roots(frame, "missing", factor=2)
        with self.assertRaisesRegex(ValueError, "unknown root"):
            duplicate_root_rows(frame, "missing", factor=2)
        self.assertEqual(len(duplicate_domain_rows(frame, "d1", 2)), 9)
        self.assertEqual(len(duplicate_root_rows(frame, "d1-long", 2)), 8)
        self.assertEqual(len(clone_domain_roots(frame, "d1", 2)), 9)
        self.assertEqual(len(frame), 5)


class ValidationRunnerTests(unittest.TestCase):
    @staticmethod
    def _domains() -> dict[str, pd.DataFrame]:
        domains: dict[str, pd.DataFrame] = {}
        for domain, targets in {
            "d1": ["A", "A", "B", "A"],
            "d2": ["B", "B", "A", "B"],
            "d3": ["A", "B", "A", "B"],
        }.items():
            domains[domain] = pd.DataFrame(
                {
                    "prefix_ids": [["X"], ["X"], ["A"], ["B"]],
                    "target": targets,
                    "root": [f"{domain}-r1", f"{domain}-r1", f"{domain}-r2", f"{domain}-r2"],
                    "domain": [domain] * 4,
                }
            )
        return domains

    def test_evaluate_domains_emits_every_ablation_and_paired_metric(self):
        result = evaluate_domains(
            self._domains(),
            ["A", "B"],
            bootstrap_replicates=40,
            stress_factors=(),
            seed=7,
        )

        self.assertEqual(
            set(result.domain_metrics["model"]),
            {"row_pooled", "root_balanced", "domain_row_balanced", "dr_vom"},
        )
        self.assertEqual(
            set(result.domain_metrics["heldout_domain"]),
            {"d1", "d2", "d3"},
        )
        self.assertEqual(
            set(result.comparison_metrics["reference"]),
            {"row_pooled", "root_balanced", "domain_row_balanced"},
        )
        self.assertEqual(
            set(result.comparison_metrics["metric"]),
            {
                "top1_gain_pp",
                "mrr_gain",
                "hit5_gain_pp",
                "nll_improvement",
                "brier_improvement",
            },
        )
        self.assertEqual(result.summary["heldout_domains"], ["d1", "d2", "d3"])
        self.assertEqual(len(result.predictions), 12)

    def test_run_validation_refuses_overwrite_and_writes_complete_artifacts(self):
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "validation"
            result = run_validation(
                project_root=Path(temporary),
                output_dir=destination,
                domains=self._domains(),
                vocabulary=["A", "B"],
                bootstrap_replicates=20,
                stress_factors=(),
                seed=11,
            )

            self.assertEqual(len(result.domain_metrics), 12)
            self.assertTrue((destination / "domain_ablation_metrics.csv").is_file())
            self.assertTrue((destination / "aggregate_ablation_intervals.csv").is_file())
            self.assertTrue((destination / "predictions.csv").is_file())
            self.assertTrue((destination / "summary.json").is_file())
            self.assertTrue((destination / "run_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                run_validation(
                    project_root=Path(temporary),
                    output_dir=destination,
                    domains=self._domains(),
                    vocabulary=["A", "B"],
                    bootstrap_replicates=5,
                    stress_factors=(),
                    seed=11,
                )


if __name__ == "__main__":
    unittest.main()
