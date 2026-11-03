"""Build presentation figures from saved saliva results, without rerunning analysis."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reference" / "computed" / "GSE64951"
OUTPUT = DIRECTORY / "presentation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(x: float, y: float, value: str, *, size=22, anchor="start", accent=False) -> str:
    color = "var(--cp-accent)" if accent else "var(--cp-text)"
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}">{html.escape(value)}</text>'
    )


def figure(body: list[str], title: str, description: str, width=1280, height=440) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title description">'
        f"<title id=\"title\">{html.escape(title)}</title>"
        f"<desc id=\"description\">{html.escape(description)}</desc>"
        "<style>:root{--cp-bg:#f7f4ef;--cp-text:#242424;--cp-text-muted:#5c5c5c;"
        "--cp-accent:#b11f4b;--cp-border:#dedede;--cp-border-strong:#919191}</style>"
        '<g font-family="Segoe UI, Aptos, Calibri, sans-serif">'
        + "".join(body) + "</g></svg>\n"
    )


def main() -> None:
    summary = json.loads((DIRECTORY / "summary.json").read_text())
    with (DIRECTORY / "degs.csv").open(newline="") as handle:
        genes = list(csv.DictReader(handle))
    with (DIRECTORY / "filter_sensitivity.csv").open(newline="") as handle:
        filters = list(csv.DictReader(handle))
    recovered = set(summary["published_genes_recovered"])
    if len(genes) != summary["degs"] or {row["gene"] for row in genes} != recovered | set(summary["additional_genes"]):
        raise ValueError("Saved DEG statistics disagree with the summary.")
    for row in genes:
        effect, adjusted_p = float(row["log2FC"]), float(row["padj"])
        if not math.isfinite(effect) or not 0 <= adjusted_p < 0.05 or effect >= -1 or row["direction"] != "down":
            raise ValueError(f"Invalid saved significant DEG: {row['gene']}")
    expected_hashes = summary["snapshot_files_sha256"]
    verification = {}
    for name in ("degs.csv", "filter_sensitivity.csv", "manifest.csv", "heatmap.png", "volcano.png"):
        raw = (DIRECTORY / name).read_bytes()
        exact = hashlib.sha256(raw).hexdigest()
        normalized = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() if name.endswith(".csv") else exact
        if exact != expected_hashes[name] and normalized != expected_hashes[name]:
            raise ValueError(f"Saved output SHA256 changed: {name}")
        verification[name] = "exact bytes" if exact == expected_hashes[name] else "CSV with Git checkout CRLF normalized to LF"
    if {row["rule"] for row in filters} != {"any_above_median", "mean_above_median", "half_above_median"} or len(filters) != 3:
        raise ValueError("The three declared filter rules are required.")
    for row in filters:
        if int(row["input_genes"]) != summary["genes_mapped"] or int(row["retained_genes"]) + int(row["removed_genes"]) != int(row["input_genes"]):
            raise ValueError("Filter counts do not conserve the input gene count.")
        if row["rule"] == summary["filter_rule"] and int(row["retained_genes"]) != summary["genes_retained"]:
            raise ValueError("Default filtering disagrees with the saved summary.")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    labels = {
        "any_above_median": "Any sample above median (default)",
        "mean_above_median": "Gene mean above median",
        "half_above_median": "At least half the samples above median",
    }
    start, length, maximum = 510, 610, 22000
    position = lambda value: start + value / maximum * length
    body = [
        text(0, 30, "Saliva low-expression filtering: declared sensitivity analysis", size=29),
        text(0, 66, f"Input: {summary['genes_mapped']:,} mapped genes; reference: {summary['published_genes_retained']:,} retained", size=22),
    ]
    reference_x = position(summary["published_genes_retained"])
    body.append(f'<line x1="{reference_x:g}" x2="{reference_x:g}" y1="86" y2="298" stroke="var(--cp-text)" stroke-width="2" stroke-dasharray="6 5"/>')
    for index, row in enumerate(filters):
        y = 126 + index * 72
        value = int(row["retained_genes"])
        default = row["rule"] == summary["filter_rule"]
        body.extend([
            text(0, y + 7, labels[row["rule"]], size=24, accent=default),
            f'<rect data-rule="{row["rule"]}" data-count="{value}" x="{start}" y="{y - 20}" width="{value / maximum * length:.6f}" height="34" rx="4" fill="{"var(--cp-accent)" if default else "var(--cp-border-strong)"}"/>',
            text(1140, y + 7, f"{value:,}", size=26, accent=default),
        ])
    body.append(f'<line x1="{start}" x2="{start + length}" y1="309" y2="309" stroke="var(--cp-border-strong)" stroke-width="2"/>')
    for value in (0, 10000, 22000):
        body.append(text(position(value), 339, f"{value:,}", size=20, anchor="middle"))
    body.extend([
        text(0, 390, "Dashed line: published retained count. Alternatives were not selected to force agreement.", size=22),
        text(0, 425, "Derived from saved filter_sensitivity.csv; this is not a new DE or model run.", size=19),
    ])
    (OUTPUT / "filter_sensitivity.svg").write_text(figure(
        body, "Filtering sensitivity from the completed raw saliva analysis",
        "Three declared rules retained 20,527, 11,171 and 10,014 genes. The paper reported 20,407. All bars start at zero.",
    ), encoding="utf-8")

    ordered = sorted(genes, key=lambda row: (float(row["log2FC"]), row["gene"]))
    body = [
        text(0, 30, "Computed saliva DEGs: log2 fold change (GC minus control)", size=28),
        text(0, 62, "Filled circles: recovered published genes. Hollow squares: additional DEGs. No intervals shown.", size=21),
    ]
    for column, rows in enumerate((ordered[:19], ordered[19:])):
        offset = column * 640
        left, right = offset + 152, offset + 574
        to_x = lambda effect: right + effect / 2.2 * (right - left)
        threshold = to_x(-1)
        body.append(f'<line x1="{threshold:g}" x2="{threshold:g}" y1="92" y2="479" stroke="var(--cp-text-muted)" stroke-width="1.5" stroke-dasharray="5 4"/>')
        body.append(f'<line x1="{right:g}" x2="{right:g}" y1="92" y2="479" stroke="var(--cp-border)" stroke-width="1"/>')
        for index, row in enumerate(rows):
            y = 107 + index * 20
            value = float(row["log2FC"])
            x = to_x(value)
            published = row["gene"] in recovered
            body.extend([
                text(offset + 2, y + 6, row["gene"], size=21, accent=published),
                f'<line x1="{x:.6f}" x2="{right:g}" y1="{y}" y2="{y}" stroke="var(--cp-border-strong)" stroke-width="1.5"/>',
            ])
            attrs = f'data-gene="{row["gene"]}" data-log2fc="{value:.16g}" data-recovered="{str(published).lower()}"'
            if published:
                body.append(f'<circle {attrs} cx="{x:.6f}" cy="{y}" r="5" fill="var(--cp-accent)"/>')
            else:
                body.append(f'<rect {attrs} x="{x - 5:.6f}" y="{y - 5}" width="10" height="10" fill="var(--cp-bg)" stroke="var(--cp-text)" stroke-width="1.5"/>')
        body.append(f'<line x1="{left:g}" x2="{right:g}" y1="497" y2="497" stroke="var(--cp-border-strong)" stroke-width="2"/>')
        for value in (-2, -1, 0):
            body.append(text(to_x(value), 527, str(value), size=20, anchor="middle"))
    body.append(text(0, 559, "All 37 satisfy log2FC < -1 and BH-adjusted P < 0.05; dashed line marks log2FC = -1.", size=22))
    (OUTPUT / "gene_effects.svg").write_text(figure(
        body, "Effect sizes of the 37 computed saliva DEGs",
        "Actual log2 fold changes from the saved DEG statistics; 14 published genes recovered and 23 additional genes, all downregulated. These are not published effect-size estimates.",
        height=570,
    ), encoding="utf-8")
    sources = ["summary.json", "degs.csv", "filter_sensitivity.csv", "manifest.csv", "heatmap.png", "volcano.png"]
    record = {
        "analysis_status": "partial reproduction; no new raw processing or model evaluation",
        "figure_operation": "deterministic visualization of saved real-cohort results",
        "analysis_source_commit": summary["source_commit"],
        "generator": "scripts/build_medical_figures.py",
        "generator_sha256": digest(Path(__file__)),
        "inputs_sha256": {name: digest(DIRECTORY / name) for name in sources},
        "snapshot_verification": verification,
        "outputs_sha256": {name: digest(OUTPUT / name) for name in ("filter_sensitivity.svg", "gene_effects.svg")},
        "unrun": summary["remaining"],
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Created two source-verified saliva figures in {OUTPUT}; no new analysis is claimed.")


if __name__ == "__main__":
    main()
