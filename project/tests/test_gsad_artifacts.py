import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gsad.artifacts import (  # noqa: E402
    claim_locked_run,
    freeze_candidate,
    sha256_file,
    write_canonical_json,
    write_manifest,
)


class HashTests(unittest.TestCase):
    def test_file_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.txt"
            second = Path(directory) / "second.txt"
            first.write_text("same", encoding="utf-8")
            second.write_text("same", encoding="utf-8")
            self.assertEqual(sha256_file(first), sha256_file(second))

    def test_canonical_json_digest_ignores_dictionary_insertion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_canonical_json(Path(directory) / "a.json", {"b": 2, "a": 1})
            second = write_canonical_json(Path(directory) / "b.json", {"a": 1, "b": 2})
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads((Path(directory) / "a.json").read_text(encoding="utf-8")),
                {"a": 1, "b": 2},
            )


class ManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_manifest(
                root / "one.json",
                inputs={"data": "abc"},
                config={"seed": 7},
                split_audit={"overlap": 0},
            )
            second = write_manifest(
                root / "two.json",
                inputs={"data": "abc"},
                config={"seed": 7},
                split_audit={"overlap": 0},
            )
            self.assertEqual(first["manifest_digest"], second["manifest_digest"])


class LockedRunTests(unittest.TestCase):
    def test_only_passing_primary_gate_can_be_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "PRIMARY"):
                freeze_candidate(
                    config={"candidate": "bad"},
                    development_gates={"PRIMARY": {"passed": False}},
                    path=Path(directory) / "token.json",
                )

    def test_locked_run_cannot_be_claimed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = freeze_candidate(
                config={"candidate": "gsad_core", "hashes": {"data": "abc"}},
                development_gates={"PRIMARY": {"passed": True}},
                path=root / "token.json",
            )
            results = root / "locked"
            claim_locked_run(token, results)
            self.assertTrue((root / "LOCKED_EVALUATION_CLAIMED.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                claim_locked_run(token, root / "different-results-dir")

    def test_tampered_freeze_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = freeze_candidate(
                config={"candidate": "gsad_core"},
                development_gates={"PRIMARY": {"passed": True}},
                path=root / "token.json",
            )
            tampered = token.__class__(
                digest="0" * 64,
                config_digest=token.config_digest,
                gate_digest=token.gate_digest,
                token_path=token.token_path,
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                claim_locked_run(tampered, root / "locked")


if __name__ == "__main__":
    unittest.main()
