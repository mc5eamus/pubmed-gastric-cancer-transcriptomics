"""Bundle an offline, annotated guide to the saved saliva heatmap and volcano."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reference" / "computed" / "GSE64951"
EXAMPLES = ("YOD1", "TMPRSS11B", "SPAG9")

# Pixel calibration of the hash-verified original PNGs, not new scientific axes.
HEATMAP_BOUNDS = (121, 22, 2743, 791)
VOLCANO_ZERO = (621, 565.5)
VOLCANO_SCALE = (252.3, 217.5)


def load_sources(directory: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for name in ("degs.csv", "manifest.csv", "heatmap.png", "volcano.png"):
        raw = (directory / name).read_bytes()
        normalized = raw.replace(b"\r\n", b"\n") if name.endswith(".csv") else raw
        expected = summary["snapshot_files_sha256"][name]
        if expected not in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(normalized).hexdigest()):
            raise ValueError(f"Saved source SHA256 changed: {name}")
    with (directory / "degs.csv").open(newline="", encoding="utf-8") as handle:
        genes = list(csv.DictReader(handle))
    with (directory / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    if len(genes) != summary["degs"] or len({row["gene"] for row in genes}) != len(genes):
        raise ValueError("DEG count or unique gene IDs disagree with the summary.")
    if len(samples) != summary["samples"] or len({row["gsm"] for row in samples}) != len(samples):
        raise ValueError("Sample count or unique sample IDs disagree with the summary.")
    if sum(row["label"] == "1" for row in samples) != summary["gc"]:
        raise ValueError("Cancer sample labels disagree with the summary.")
    if sum(row["label"] == "0" for row in samples) != summary["controls"]:
        raise ValueError("Control sample labels disagree with the summary.")
    if [row["gsm"] for row in samples] != [f"GSM{1584837 + index}" for index in range(94)]:
        raise ValueError("The original heatmap sample-order calibration no longer applies.")
    if summary["up"] != 0 or summary["down"] != len(genes):
        raise ValueError("This guide requires the saved all-downregulated saliva snapshot.")
    for row in genes:
        effect, adjusted_p, statistic = map(float, (row["log2FC"], row["padj"], row["t"]))
        if not all(map(math.isfinite, (effect, adjusted_p, statistic))):
            raise ValueError(f"Nonfinite DEG statistic: {row['gene']}")
        if effect >= -1 or not 0 < adjusted_p < .05 or row["is_deg"] != "True" or row["direction"] != "down":
            raise ValueError(f"Invalid saved DEG: {row['gene']}")
    if not set(EXAMPLES).issubset(row["gene"] for row in genes):
        raise ValueError("A highlighted gene is missing.")
    for name, size in (("heatmap.png", (3035, 959)), ("volcano.png", (920, 672))):
        raw = (directory / name).read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", raw[16:24]) != size:
            raise ValueError(f"Original figure dimensions changed: {name}")
    return summary, genes, samples


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def svg_image(uri: str, width: int, height: int) -> str:
    return f'<image href="{uri}" x="0" y="0" width="{width}" height="{height}"/>'


def heatmap_figure(uri: str, genes: list[dict[str, str]], *, gene: str | None = None) -> str:
    ordered = sorted(genes, key=lambda row: -abs(float(row["t"])))
    indexes = {row["gene"]: index for index, row in enumerate(ordered)}
    left, top, right, bottom = HEATMAP_BOUNDS
    row_height = (bottom - top) / len(ordered)
    if gene is not None:
        y = top + indexes[gene] * row_height
        return (
            f'<svg class="row-strip" viewBox="{left} {y:.6f} {right-left} {row_height:.6f}" '
            f'preserveAspectRatio="none" role="img" aria-label="{gene}: original expression colours across all 94 samples">'
            + svg_image(uri, 3035, 959) + "</svg>"
        )
    parts = [
        '<svg class="heatmap-map" viewBox="0 0 3035 959" role="img" aria-labelledby="heatmap-title heatmap-desc">',
        '<title id="heatmap-title">Three highlighted rows in the original saliva heatmap</title>',
        '<desc id="heatmap-desc">1: YOD1, including GSM1584837 and GSM1584927. '
        '2: TMPRSS11B. 3: SPAG9. Outlines are teaching annotations; original pixels and colour scale are unchanged.</desc>',
        svg_image(uri, 3035, 959),
    ]
    for number, name in enumerate(EXAMPLES, 1):
        y = top + indexes[name] * row_height
        parts.append(
            f'<g data-heatmap-gene="{name}" data-row-index="{indexes[name]}">'
            f'<rect x="{left-3}" y="{y:.6f}" width="{right-left+6}" height="{row_height:.6f}" '
            'fill="none" stroke="var(--cp-accent)" stroke-width="4"/>'
            f'<circle cx="35" cy="{y+row_height/2:.6f}" r="22" fill="var(--cp-accent)"/>'
            f'<text x="35" y="{y+row_height/2+9:.6f}" text-anchor="middle" '
            f'font-size="29" fill="var(--cp-accent-fg)">{number}</text></g>'
        )
    cell_width = (right - left) / 94
    for index in (0, 90):
        parts.append(
            f'<rect data-sample="GSM{1584837+index}" x="{left+index*cell_width:.6f}" y="{top}" '
            f'width="{cell_width:.6f}" height="{row_height:.6f}" fill="none" '
            'stroke="var(--cp-text)" stroke-width="5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def volcano_figure(uri: str, genes: list[dict[str, str]]) -> str:
    selected = {row["gene"]: row for row in genes}
    labels = {"YOD1": (332, 65), "TMPRSS11B": (130, 102), "SPAG9": (175, 350)}
    parts = [
        '<svg class="volcano-map" viewBox="0 0 920 672" role="img" aria-labelledby="volcano-title volcano-desc">',
        '<title id="volcano-title">Three highlighted genes in the original full-universe volcano</title>',
        '<desc id="volcano-desc">1: YOD1, stronger statistical evidence. 2: TMPRSS11B, larger decrease. '
        '3: SPAG9, just above the adjusted-p threshold. Rings use raster-calibrated positions; '
        'readouts use the saved exact statistics.</desc>',
        svg_image(uri, 920, 672),
    ]
    for number, name in enumerate(EXAMPLES, 1):
        row = selected[name]
        effect, adjusted_p = float(row["log2FC"]), float(row["padj"])
        x = VOLCANO_ZERO[0] + VOLCANO_SCALE[0] * effect
        y = VOLCANO_ZERO[1] - VOLCANO_SCALE[1] * -math.log10(adjusted_p)
        lx, ly = labels[name]
        parts.append(
            f'<g data-volcano-gene="{name}" data-log2fc="{effect}" data-padj="{adjusted_p}">'
            f'<circle class="point-ring" cx="{x:.6f}" cy="{y:.6f}" r="11" fill="none" '
            'stroke="var(--cp-accent)" stroke-width="3"/>'
            f'<path d="M {x:.6f} {y:.6f} L {lx} {ly+12}" fill="none" '
            'stroke="var(--cp-accent)" stroke-width="2"/>'
            f'<rect x="{lx-15}" y="{ly-16}" width="{153 if name == "TMPRSS11B" else 115}" height="31" rx="6" '
            'fill="var(--cp-surface)" stroke="var(--cp-accent)" stroke-width="1.5"/>'
            f'<text x="{lx-7}" y="{ly+5}" font-size="17" font-weight="700" '
            f'fill="var(--cp-accent)">{number}  {name}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def gene_table(genes: list[dict[str, str]]) -> str:
    lookup = {row["gene"]: row for row in genes}
    rows = []
    for number, name in enumerate(EXAMPLES, 1):
        row = lookup[name]
        effect, adjusted_p = float(row["log2FC"]), float(row["padj"])
        ratio = 2 ** effect
        rows.append(
            f'<tr data-table-gene="{name}"><th scope="row">{number}. {name}</th>'
            f'<td>{effect:.3f}</td><td>{adjusted_p:.5f}</td><td>{-math.log10(adjusted_p):.2f}</td>'
            f'<td>{ratio:.2f}x</td><td>{100*(1-ratio):.0f}% lower</td></tr>'
        )
    return "".join(rows)


def build(output_path: Path, *, directory: Path = RESULTS) -> None:
    summary, genes, samples = load_sources(directory)
    shell = (ROOT / "presentation.html").read_text(encoding="utf-8")
    template = (ROOT / "scripts" / "interpretation_slides.html").read_text(encoding="utf-8")
    replacements = {}
    for marker, tag, identifier in (
        ("__REVEAL_CORE__", "style", "reveal-core"),
        ("__REVEAL_RUNTIME__", "script", "reveal-runtime"),
    ):
        blocks = re.findall(rf'<{tag} id="{identifier}">(.*?)</{tag}>', shell, re.S)
        if len(blocks) != 1:
            raise ValueError(f"Expected exactly one offline {identifier} in presentation.html.")
        replacements[marker] = blocks[0]
    heatmap = image_uri(directory / "heatmap.png")
    volcano = image_uri(directory / "volcano.png")
    replacements.update({
        "__HEATMAP_IMAGE__": heatmap,
        "__VOLCANO_IMAGE__": volcano,
        "__HEATMAP_OVERVIEW__": heatmap_figure(heatmap, genes),
        "__VOLCANO_ANNOTATED__": volcano_figure(volcano, genes),
        "__GENE_TABLE__": gene_table(genes),
    })
    for name in EXAMPLES:
        replacements[f"__{name}_STRIP__"] = heatmap_figure(heatmap, genes, gene=name)
        row = next(row for row in genes if row["gene"] == name)
        effect, adjusted_p = float(row["log2FC"]), float(row["padj"])
        ratio = 2 ** effect
        replacements[f"__{name}_METRICS__"] = (
            f'log<sub>2</sub>FC {effect:.3f} &middot; p<sub>adj</sub> {adjusted_p:.5f} '
            f'&middot; height {-math.log10(adjusted_p):.2f}'
        )
        replacements[f"__{name}_READING__"] = (
            f'About {ratio:.2f}x control expression, or {100*(1-ratio):.0f}% lower in cancer.'
        )
    sample_lookup = {row["gsm"]: row for row in samples}
    record = {
        "purpose": "Interpretation of saved saliva figures; no new analysis or diagnostic validation.",
        "dataset": summary["dataset"],
        "analysis_source_commit": summary["source_commit"],
        "source_sha256": {name: summary["snapshot_files_sha256"][name]
                          for name in ("degs.csv", "manifest.csv", "heatmap.png", "volcano.png")},
        "selected_genes": [next(row for row in genes if row["gene"] == name) for name in EXAMPLES],
        "sample_examples": [sample_lookup[name] for name in ("GSM1584837", "GSM1584927")],
        "annotation_calibration": {
            "heatmap_bounds": HEATMAP_BOUNDS, "volcano_zero": VOLCANO_ZERO, "volcano_scale": VOLCANO_SCALE,
            "note": "Original raster coordinates, approximate to a pixel; no values inferred from colour.",
        },
        "summary": {key: summary[key] for key in ("samples", "gc", "controls", "degs", "up", "down", "verdict")},
    }
    replacements["__SOURCE_RECORDS__"] = json.dumps(record, sort_keys=True).replace("</", "<\\/")
    for marker, value in replacements.items():
        if marker not in template:
            raise ValueError(f"Missing presentation placeholder: {marker}")
        template = template.replace(marker, value)
    unresolved = re.findall(r"__[A-Z][A-Z0-9_]*__", template)
    if unresolved:
        raise ValueError(f"Unresolved presentation placeholders: {unresolved}")
    output_path.write_text(template, encoding="utf-8")
    print(f"Built {output_path.name}: 12 offline slides, 3 examples and 2 quiz questions per visualization.")


if __name__ == "__main__":
    build(ROOT / "presentation-interpretation.html")
