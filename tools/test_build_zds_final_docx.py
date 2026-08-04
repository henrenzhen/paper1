from __future__ import annotations

import subprocess
import sys
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_zds_final_docx.py"
CHART = ROOT / "deliverables" / "assets" / "dr_vom_lodo_performance.png"
DOCX = ROOT / "deliverables" / "ZDS论文拒稿复盘与DR-VOM实验报告_最终版.docx"


class FinalReportBuilderTest(unittest.TestCase):
    def test_builder_creates_nonempty_png_and_docx_with_bundled_runtime(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )
        self.assertTrue(CHART.is_file())
        self.assertGreater(CHART.stat().st_size, 10_000)
        self.assertTrue(DOCX.is_file())
        self.assertGreater(DOCX.stat().st_size, 50_000)

        with zipfile.ZipFile(DOCX) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
        self.assertNotIn('w:val="Heading3"', document_xml)
        self.assertIn('w:val="Heading2"', document_xml)
        self.assertRegex(document_xml, r'<wp:docPr[^>]*descr="[^"]+"')


if __name__ == "__main__":
    unittest.main()
