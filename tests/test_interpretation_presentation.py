"""Offline checks for the interpretation deck's evidence and annotation geometry."""

import base64
import contextlib
import csv
import importlib.util
import io
import json
import math
import re
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "interpretation_presentation", ROOT / "scripts" / "build_interpretation_presentation.py"
)
deck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deck)


class DeckParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = []
        self.assets = []
        self.notes = 0
        self.quizzes = []
        self.current_quiz = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "section":
            self.slides.append(attrs["id"])
        if tag == "aside" and attrs.get("class") == "notes":
            self.notes += 1
        if tag == "form" and attrs.get("class") == "quiz-form":
            self.current_quiz = {"correct": attrs["data-correct"], "options": [], "explanations": []}
            self.quizzes.append(self.current_quiz)
        if tag == "input" and attrs.get("type") == "radio":
            self.current_quiz["options"].append(attrs)
        if tag == "label" and attrs.get("class") == "quiz-option":
            self.current_quiz["explanations"].append(attrs["data-explanation"])
        if tag in {"img", "script", "link"}:
            source = attrs.get("src") or attrs.get("href")
            if source:
                self.assets.append(source)


class InterpretationDeckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ("summary.json", "degs.csv", "manifest.csv", "heatmap.png", "volcano.png"):
            shutil.copyfile(deck.RESULTS / name, self.directory / name)
        self.output = self.directory / "presentation-interpretation.html"
        with (self.directory / "degs.csv").open(newline="", encoding="utf-8") as handle:
            self.genes = list(csv.DictReader(handle))

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            deck.build(self.output, directory=self.directory)
        return self.output.read_text(encoding="utf-8")

    def test_standalone_twelve_slides_original_assets_and_source_records(self):
        content = self.build()
        parser = DeckParser()
        parser.feed(content)
        self.assertEqual(len(parser.slides), 12)
        self.assertEqual(len(set(parser.slides)), 12)
        self.assertEqual(parser.notes, 12)
        self.assertTrue(all(source.startswith("data:") for source in parser.assets))
        self.assertFalse(re.search(r"__[A-Z][A-Z0-9_]*__", content))
        self.assertIn("Permission is hereby granted", content)
        for name in ("heatmap.png", "volcano.png"):
            encoded = base64.b64encode((self.directory / name).read_bytes()).decode("ascii")
            self.assertIn("data:image/png;base64," + encoded, content)
        record = json.loads(re.search(
            r'<script id="interpretation-source-records" type="application/json">(.*?)</script>', content
        ).group(1))
        self.assertEqual([row["gene"] for row in record["selected_genes"]], list(deck.EXAMPLES))
        self.assertEqual([row["label"] for row in record["sample_examples"]], ["1", "0"])
        self.assertEqual(record["summary"]["degs"], 37)
        self.assertEqual(record["summary"]["down"], 37)
        for row in record["selected_genes"]:
            effect, adjusted_p = float(row["log2FC"]), float(row["padj"])
            self.assertIn(
                f'<p class="metric">log<sub>2</sub>FC {effect:.3f} &middot; '
                f'p<sub>adj</sub> {adjusted_p:.5f} &middot; height {-math.log10(adjusted_p):.2f}</p>',
                content,
            )
            self.assertIn(
                f'About {2**effect:.2f}x control expression, or {100*(1-2**effect):.0f}% lower in cancer.',
                content,
            )

    def test_quizzes_follow_each_section_and_explain_every_option(self):
        parser = DeckParser()
        parser.feed(self.build())
        heatmap_end = parser.slides.index("heatmap-readings")
        volcano_end = parser.slides.index("compare")
        self.assertEqual(parser.slides[heatmap_end+1:heatmap_end+4],
                         ["heatmap-quiz-scale", "heatmap-quiz-validation", "volcano-guide"])
        self.assertEqual(parser.slides[volcano_end+1:volcano_end+4],
                         ["volcano-quiz-evidence", "volcano-quiz-thresholds", "limits"])
        self.assertEqual([quiz["correct"] for quiz in parser.quizzes], ["C", "B", "A", "D"])
        names = []
        for quiz in parser.quizzes:
            self.assertEqual([option["value"] for option in quiz["options"]], list("ABCD"))
            self.assertEqual(len({option["name"] for option in quiz["options"]}), 1)
            self.assertIn("required", quiz["options"][0])
            self.assertEqual(len(set(quiz["explanations"])), 4)
            self.assertTrue(all(len(explanation.split()) >= 35 for explanation in quiz["explanations"]))
            names.append(quiz["options"][0]["name"])
        self.assertEqual(len(set(names)), 4)

    def test_heatmap_outlines_match_ranked_rows_and_sample_cells(self):
        root = ET.fromstring(deck.heatmap_figure("data:image/png;base64,", self.genes))
        groups = [node for node in root if "data-heatmap-gene" in node.attrib]
        self.assertEqual(len(groups), 3)
        expected = {"YOD1": 0, "TMPRSS11B": 8, "SPAG9": 36}
        for node in groups:
            name = node.get("data-heatmap-gene")
            self.assertEqual(int(node.get("data-row-index")), expected[name])
            self.assertAlmostEqual(float(node[0].get("y")), 22 + expected[name] * 769 / 37, places=5)
        cells = [node for node in root if "data-sample" in node.attrib]
        self.assertEqual([node.get("data-sample") for node in cells], ["GSM1584837", "GSM1584927"])
        self.assertAlmostEqual(float(cells[1].get("x")), 121 + 90 * 2622 / 94, places=5)
        for name, index in expected.items():
            strip = ET.fromstring(deck.heatmap_figure("data:image/png;base64,", self.genes, gene=name))
            self.assertAlmostEqual(float(strip.get("viewBox").split()[1]), 22 + index * 769 / 37, places=5)

    def test_volcano_rings_match_exact_statistics_and_calibrated_axis_ticks(self):
        root = ET.fromstring(deck.volcano_figure("data:image/png;base64,", self.genes))
        groups = [node for node in root if "data-volcano-gene" in node.attrib]
        lookup = {row["gene"]: row for row in self.genes}
        self.assertEqual(len(groups), 3)
        self.assertAlmostEqual(deck.VOLCANO_ZERO[0] - deck.VOLCANO_SCALE[0], 369, delta=1)
        self.assertAlmostEqual(deck.VOLCANO_ZERO[0] + deck.VOLCANO_SCALE[0], 874, delta=1)
        self.assertAlmostEqual(deck.VOLCANO_ZERO[1] - deck.VOLCANO_SCALE[1] * -math.log10(.05), 283, delta=1)
        for node in groups:
            row = lookup[node.get("data-volcano-gene")]
            self.assertEqual(float(node.get("data-log2fc")), float(row["log2FC"]))
            self.assertAlmostEqual(float(node.get("data-padj")), float(row["padj"]), places=15)
            self.assertAlmostEqual(float(node[0].get("cx")), 621 + 252.3 * float(row["log2FC"]), places=5)
            self.assertAlmostEqual(float(node[0].get("cy")), 565.5 + 217.5 * math.log10(float(row["padj"])), places=5)

    def test_table_readings_are_computed_from_saved_values(self):
        root = ET.fromstring("<table>" + deck.gene_table(self.genes) + "</table>")
        lookup = {row["gene"]: row for row in self.genes}
        for row in root:
            source = lookup[row.get("data-table-gene")]
            effect, adjusted_p = float(source["log2FC"]), float(source["padj"])
            self.assertEqual(row[1].text, f"{effect:.3f}")
            self.assertEqual(row[2].text, f"{adjusted_p:.5f}")
            self.assertEqual(row[3].text, f"{-math.log10(adjusted_p):.2f}")
            self.assertEqual(row[4].text, f"{2**effect:.2f}x")
            self.assertEqual(row[5].text, f"{100*(1-2**effect):.0f}% lower")

    def test_source_changes_are_rejected_before_output(self):
        for name in ("degs.csv", "manifest.csv", "heatmap.png", "volcano.png"):
            path = self.directory / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                    self.build()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_git_line_endings_and_deterministic_build(self):
        for name in ("degs.csv", "manifest.csv"):
            path = self.directory / name
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        first = self.build()
        self.assertEqual(first, self.build())


if __name__ == "__main__":
    unittest.main()
