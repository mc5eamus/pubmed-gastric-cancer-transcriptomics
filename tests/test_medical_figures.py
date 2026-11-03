"""Dependency-free checks for visualizations of the saved saliva snapshot."""

import contextlib
import copy
import csv
import io
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "build_medical_figures.py"
SPEC = importlib.util.spec_from_file_location("medical_figures", MODULE)
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)
DECK_SPEC = importlib.util.spec_from_file_location("medical_presentation", MODULE.with_name("build_medical_presentation.py"))
deck = importlib.util.module_from_spec(DECK_SPEC)
with patch.dict(sys.modules, {"build_medical_figures": figures}):
    DECK_SPEC.loader.exec_module(deck)


class MedicalFigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ("summary.json", "degs.csv", "filter_sensitivity.csv", "manifest.csv", "heatmap.png", "volcano.png"):
            shutil.copyfile(figures.DIRECTORY / name, self.directory / name)
        self.output = self.directory / "presentation"
        self.summary = json.loads((self.directory / "summary.json").read_text())

    def build(self):
        with patch.object(figures, "DIRECTORY", self.directory), patch.object(figures, "OUTPUT", self.output):
            with contextlib.redirect_stdout(io.StringIO()):
                figures.main()

    def test_counts_effects_geometry_and_provenance(self):
        self.build()
        with (self.directory / "degs.csv").open(newline="") as handle:
            expected = {row["gene"]: float(row["log2FC"]) for row in csv.DictReader(handle)}
        effect_svg = ET.parse(self.output / "gene_effects.svg")
        points = [element for element in effect_svg.iter() if "data-gene" in element.attrib]
        self.assertEqual(len(points), self.summary["degs"])
        self.assertEqual({element.get("data-gene") for element in points}, set(expected))
        recovered = set(self.summary["published_genes_recovered"])
        for index, element in enumerate(points):
            gene = element.get("data-gene")
            self.assertAlmostEqual(float(element.get("data-log2fc")), expected[gene], places=12)
            self.assertEqual(element.get("data-recovered") == "true", gene in recovered)
            right = 574 if index < 19 else 1214
            plotted_x = float(element.get("cx")) if gene in recovered else float(element.get("x")) + 5
            self.assertAlmostEqual(plotted_x, right + expected[gene] / 2.2 * 422, places=5)
        filter_svg = ET.parse(self.output / "filter_sensitivity.svg")
        bars = [element for element in filter_svg.iter() if "data-rule" in element.attrib]
        self.assertEqual(len(bars), 3)
        for element in bars:
            self.assertEqual(float(element.get("x")), 510)
            self.assertAlmostEqual(float(element.get("width")), int(element.get("data-count")) / 22000 * 610, places=5)
        record = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(record["analysis_source_commit"], self.summary["source_commit"])
        self.assertEqual(record["unrun"], self.summary["remaining"])
        self.assertIn("no new", record["analysis_status"])
        for name, value in record["outputs_sha256"].items():
            self.assertEqual(figures.digest(self.output / name), value)

    def test_git_checkout_line_endings_and_determinism(self):
        for name in ("degs.csv", "filter_sensitivity.csv", "manifest.csv"):
            target = self.directory / name
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.build()
        first = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.build()
        self.assertEqual(first, {path.name: path.read_bytes() for path in self.output.iterdir()})
        record = json.loads((self.output / "provenance.json").read_text())
        self.assertTrue(all("CRLF normalized to LF" in record["snapshot_verification"][name] for name in ("degs.csv", "filter_sensitivity.csv", "manifest.csv")))

    def test_modified_csv_or_png_is_rejected_before_output(self):
        for name in ("filter_sensitivity.csv", "volcano.png"):
            target = self.directory / name
            original = target.read_bytes()
            with self.subTest(name=name):
                target.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                    self.build()
                self.assertFalse(self.output.exists())
                target.write_bytes(original)

    def test_cloud_checkpoint_drives_displayed_comparisons(self):
        with (deck.RETRY / "gene_counts.csv").open(newline="", encoding="utf-8") as handle:
            counts = {row["dataset"]: row for row in csv.DictReader(handle)}
        overlap = json.loads((deck.RETRY / "deg_overlap.json").read_text())
        go_rows = json.loads((deck.RETRY / "go_summary.json").read_text())
        values = deck.checkpoint_placeholders(counts, overlap, go_rows)
        self.assertEqual(values["__TISSUE_DEGS__"], "1,616")
        self.assertEqual(values["__PAPER_TISSUE_DEGS__"], "1,302")
        self.assertEqual(values["__TISSUE_RETAINED__"], "15,420")
        self.assertEqual(values["__OVERLAP_GENES__"], "GJB2 and MMP12")
        self.assertEqual(values["__OVERLAP_COUNT__"], "2")
        self.assertEqual(values["__TISSUE_DNA_NES__"], "+2.884")
        self.assertEqual(values["__SALIVA_COPII_NES__"], "-2.318")
        changed = copy.deepcopy(counts)
        changed["GSE66229"]["degs"] = "1617"
        changed["GSE66229"]["up"] = "663"
        self.assertEqual(deck.checkpoint_placeholders(changed, overlap, go_rows)["__TISSUE_DEGS__"], "1,617")
        changed["GSE66229"]["up"] = "662"
        with self.assertRaisesRegex(ValueError, "direction counts"):
            deck.checkpoint_placeholders(changed, overlap, go_rows)
        with self.assertRaisesRegex(ValueError, "displayed GO term"):
            deck.checkpoint_placeholders(counts, overlap, [])
        with self.assertRaisesRegex(ValueError, "Duplicate genes"):
            deck.checkpoint_placeholders(counts, {"genes": ["GJB2", "GJB2"], "published_overlap": 0}, go_rows)

    def test_computed_and_published_auc_geometry_without_inversion(self):
        with (deck.RETRY / "computed_metrics.csv").open(newline="", encoding="utf-8") as handle:
            computed = list(csv.DictReader(handle))
        with (deck.ROOT / "reference" / "publication" / "metric_targets.csv").open(newline="", encoding="utf-8") as handle:
            published = list(csv.DictReader(handle))
        plot = ET.fromstring(deck.saliva_auc_plot(computed, published))
        groups = [element for element in plot.iter() if "data-model" in element.attrib]
        self.assertEqual(len(groups), 8)
        for element in groups:
            if element.get("data-origin") == "computed":
                source = next(row for row in computed if row["scenario"] == "D" and row["model"] == element.get("data-model"))
                self.assertEqual(float(element.get("data-auc")), float(source["auc"]))
                line = list(element)[0]
                self.assertAlmostEqual(float(line.get("x1")), 180 + 880 * float(source["auc_ci_low"]), places=6)
                self.assertAlmostEqual(float(line.get("x2")), 180 + 880 * float(source["auc_ci_high"]), places=6)
                point = list(element)[-1]
                self.assertAlmostEqual(float(point.get("x")) + 6, 180 + 880 * float(source["auc"]), places=6)
        svm = next(row for row in computed if row["scenario"] == "D" and row["model"] == "SVM")
        self.assertLess(float(svm["auc"]), .5)
        self.assertEqual(svm["ppv"], "")
        self.assertEqual(svm["lr_positive"], "")
        self.assertEqual(svm["sensitivity"], "0.0")
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            deck.saliva_auc_plot([], published)
        invalid = copy.deepcopy(computed)
        next(row for row in invalid if row["scenario"] == "D" and row["model"] == "RF")["auc"] = "nan"
        with self.assertRaisesRegex(ValueError, "Invalid AUC"):
            deck.saliva_auc_plot(invalid, published)

    def test_completed_leakage_subcheck_uses_same_holdout_without_inferred_ci(self):
        check = json.loads((deck.RETRY / "tissue-rf-fold-local.json").read_text())
        plot = ET.fromstring(deck.leakage_check_plot(check))
        points = [element for element in plot.iter() if "data-leakage-branch" in element.attrib]
        self.assertEqual(len(points), 2)
        for point in points:
            value = check[point.get("data-leakage-branch")]
            self.assertEqual(float(point.get("data-auc")), value)
            self.assertAlmostEqual(float(point.get("cx")), 340 + 720 * value, places=6)
            self.assertNotIn("data-lower", point.attrib)
        self.assertFalse(check["metadata"]["fully_inductive_end_to_end"])
        self.assertEqual(check["holdout_n"], 120)
        with self.assertRaisesRegex(ValueError, "completed, same-holdout"):
            deck.leakage_check_plot({**check, "same_holdout_verified": False})


if __name__ == "__main__":
    unittest.main()
