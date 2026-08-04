import unittest
from pathlib import Path

from project.experiments.gsad.run_qmrct_external import load_ctid_external


class QMRCTExternalTests(unittest.TestCase):
    def test_ctid_loader_reconstructs_raw_prefixes_and_actor_clusters(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = load_ctid_external(project_root)
        self.assertEqual(len(frame), 281)
        self.assertEqual(frame["actor"].nunique(), 9)
        self.assertTrue(
            (frame["raw_prefix_ids"].map(len) == frame["prefix_len"]).all()
        )
        self.assertTrue(
            (frame["prefix_ids"].map(len) == frame["prefix_len"]).all()
        )
        self.assertEqual(
            set(frame.loc[frame["org_name"].str.startswith("turla"), "actor"]),
            {"turla"},
        )


if __name__ == "__main__":
    unittest.main()
