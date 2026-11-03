"""Embed medical slides and real figures in the existing offline Reveal shell."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

from build_medical_figures import main as build_figures


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reference" / "computed" / "GSE64951"
RETRY = ROOT / "reference" / "computed" / "retry-20261102"


def checkpoint_placeholders(counts: dict, overlap: dict, go_rows: list) -> dict[str, str]:
    values = {}
    for dataset, cohort in (("GSE66229", "TISSUE"), ("GSE64951", "SALIVA")):
        row = counts[dataset]
        if int(row["up"]) + int(row["down"]) != int(row["degs"]):
            raise ValueError(f"{dataset} DEG direction counts do not sum to the DEG total.")
        for field in ("degs", "up", "down", "genes_retained"):
            label = "RETAINED" if field == "genes_retained" else field.upper()
            values[f"__{cohort}_{label}__"] = f'{int(row[field]):,}'
            values[f"__PAPER_{cohort}_{label}__"] = f'{int(row["published_" + field]):,}'
    genes = overlap["genes"]
    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate genes in tissue-saliva overlap.")
    values["__OVERLAP_GENES__"] = html.escape(" and ".join(genes))
    values["__OVERLAP_COUNT__"] = str(len(genes))
    values["__PAPER_OVERLAP_COUNT__"] = str(overlap["published_overlap"])
    for dataset, go_id, marker in (
        ("GSE66229", "GO:0006259", "__TISSUE_DNA_NES__"),
        ("GSE66229", "GO:0006261", "__TISSUE_REPLICATION_NES__"),
        ("GSE64951", "GO:0090114", "__SALIVA_COPII_NES__"),
        ("GSE64951", "GO:0006446", "__SALIVA_TRANSLATION_NES__"),
    ):
        terms = [
            term for row in go_rows if row["dataset"] == dataset and row["ontology"] == "BP"
            for term in row["top_abs_nes_terms"] if term["Term"].endswith(f"({go_id})")
        ]
        if len(terms) != 1 or not math.isfinite(float(terms[0]["NES"])):
            raise ValueError(f"Missing, duplicate or invalid displayed GO term: {dataset}/{go_id}")
        values[marker] = f'{float(terms[0]["NES"]):+.3f}'
    return values


def saliva_auc_plot(computed: list[dict], published: list[dict]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 320" role="img" aria-labelledby="auc-title auc-description">',
        '<title id="auc-title">Saliva model AUC: publication versus reproduction</title>',
        '<desc id="auc-description">Paired point estimates and 95% confidence intervals on a full zero-to-one scale. '
        'Publication uses 14 genes; reproduction uses 37 discovered DEGs. Full-cohort DE risks selection leakage. '
        'Computed intervals condition on fitted models. No outcome labels or AUC directions were reversed.</desc>',
        '<text x="25" y="24" font-size="21" fill="var(--cp-text)">Model</text>',
        '<text x="1070" y="24" font-size="20" fill="var(--cp-text-soft)">Paper / computed</text>',
        '<circle cx="330" cy="18" r="6" fill="var(--cp-surface)" stroke="var(--cp-text-muted)" stroke-width="3"/>',
        '<text x="348" y="24" font-size="20" fill="var(--cp-text-muted)">Published</text>',
        '<rect x="510" y="12" width="12" height="12" fill="var(--cp-accent)"/>',
        '<text x="532" y="24" font-size="20" fill="var(--cp-accent)">Computed</text>',
        '<line x1="620" x2="620" y1="38" y2="271" stroke="var(--cp-text-soft)" stroke-dasharray="5 5" stroke-width="2"/>',
    ]
    x = lambda value: 180 + 880 * value
    for index, model in enumerate(("RF", "GBM", "DT", "SVM")):
        rows = [row for row in computed if row["scenario"] == "D" and row["dataset"] == "GSE64951" and row["model"] == model]
        targets = [row for row in published if row["source_table"] == "Tab3" and row["scenario"] == "D" and row["dataset"] == "GSE64951" and row["model"] == model and row["metric"] == "auc"]
        if len(rows) != 1 or len(targets) != 1:
            raise ValueError(f"Missing or duplicate saliva AUC source: {model}")
        source, target = rows[0], targets[0]
        y = 60 + index * 58
        parts.append(f'<text x="25" y="{y + 13}" font-size="25" fill="var(--cp-text)">{model}</text>')
        for origin, estimate, lower, upper, offset in (
            ("published", target["published_estimate"], target["published_lower"], target["published_upper"], 0),
            ("computed", source["auc"], source["auc_ci_low"], source["auc_ci_high"], 19),
        ):
            estimate, lower, upper = map(float, (estimate, lower, upper))
            if not all(math.isfinite(value) for value in (estimate, lower, upper)) or not 0 <= lower <= estimate <= upper <= 1:
                raise ValueError(f"Invalid AUC/interval: {origin}/{model}")
            if origin == "published" and target["published_interval_valid"] != "True":
                raise ValueError(f"Invalid publication AUC interval: {model}")
            color = "var(--cp-accent)" if origin == "computed" else "var(--cp-text-muted)"
            cy = y + offset
            parts.append(
                f'<g data-model="{model}" data-origin="{origin}" data-auc="{estimate}" data-lower="{lower}" data-upper="{upper}">'
                f'<line x1="{x(lower):.6f}" x2="{x(upper):.6f}" y1="{cy}" y2="{cy}" stroke="{color}" stroke-width="3"/>'
                f'<line x1="{x(lower):.6f}" x2="{x(lower):.6f}" y1="{cy-5}" y2="{cy+5}" stroke="{color}" stroke-width="2"/>'
                f'<line x1="{x(upper):.6f}" x2="{x(upper):.6f}" y1="{cy-5}" y2="{cy+5}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<rect x="{x(estimate)-6:.6f}" y="{cy-6}" width="12" height="12" fill="{color}"/>'
                if origin == "computed" else
                f'<circle cx="{x(estimate):.6f}" cy="{cy}" r="6" fill="var(--cp-surface)" stroke="{color}" stroke-width="3"/>'
            )
            parts.append("</g>")
        parts.append(f'<text x="1070" y="{y + 13}" font-size="23" fill="var(--cp-text)">{float(target["published_estimate"]):.3f} / {float(source["auc"]):.3f}</text>')
    parts.append('<line x1="180" x2="1060" y1="271" y2="271" stroke="var(--cp-border-strong)" stroke-width="2"/>')
    parts.extend(f'<text x="{x(value):g}" y="298" text-anchor="middle" font-size="21" fill="var(--cp-text)">{value:g}</text>' for value in (0, .25, .5, .75, 1))
    parts.append('<text x="620" y="319" text-anchor="middle" font-size="18" fill="var(--cp-text-soft)">AUC (95% CI); 0.5 = chance discrimination</text></svg>')
    return "".join(parts)


def leakage_check_plot(check: dict) -> str:
    if not check["same_holdout_verified"] or check["status"] != "completed_single_model_subcheck":
        raise ValueError("Leakage comparison requires a completed, same-holdout subcheck.")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 225" role="img" aria-labelledby="leakage-title leakage-description">',
        '<title id="leakage-title">One tissue RF leakage subcheck on the same 120-sample holdout</title>',
        '<desc id="leakage-description">Full-cohort DE versus fold-local DE and selection. AUC point estimates only; no fold-local confidence interval was computed. Full-cohort RMA remains transductive. Other robustness checks are incomplete.</desc>',
    ]
    for label, field, y in (("Full-cohort DE", "paper_like_auc", 55), ("Fold-local DE / selection", "auc", 115)):
        value = float(check[field])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid AUC in completed leakage subcheck.")
        x = 340 + 720 * value
        parts.append(
            f'<text x="15" y="{y+7}" font-size="24" fill="var(--cp-text)">{label}</text>'
            f'<line x1="340" x2="1060" y1="{y}" y2="{y}" stroke="var(--cp-border)" stroke-width="2"/>'
            f'<circle data-leakage-branch="{field}" data-auc="{value}" cx="{x:.6f}" cy="{y}" r="8" fill="var(--cp-accent)"/>'
            f'<text x="1090" y="{y+7}" font-size="24" fill="var(--cp-text)">{value:.4f}</text>'
        )
    parts.append('<line x1="340" x2="1060" y1="155" y2="155" stroke="var(--cp-border-strong)" stroke-width="2"/>')
    parts.extend(f'<text x="{340+720*value:g}" y="183" text-anchor="middle" font-size="22" fill="var(--cp-text)">{value:g}</text>' for value in (0, .25, .5, .75, 1))
    parts.append('<text x="700" y="215" text-anchor="middle" font-size="21" fill="var(--cp-text-soft)">AUC point estimates only; no CI is inferred</text></svg>')
    return "".join(parts)


def main() -> None:
    build_figures()
    base = (ROOT / "presentation.html").read_text(encoding="utf-8")
    slides = (Path(__file__).parent / "medical_slides.html").read_text(encoding="utf-8")
    transfer = json.loads((RETRY / "transfer_provenance.json").read_text(encoding="utf-8"))
    for filename, expected in transfer["files_sha256"].items():
        raw = (RETRY / filename).read_bytes()
        if filename.endswith((".csv", ".json")):
            raw = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Cloud source hash mismatch: {filename}")
    with (RETRY / "gene_counts.csv").open(newline="", encoding="utf-8") as handle:
        counts = {row["dataset"]: row for row in csv.DictReader(handle)}
    go_rows = json.loads((RETRY / "go_summary.json").read_text(encoding="utf-8"))
    overlap = json.loads((RETRY / "deg_overlap.json").read_text(encoding="utf-8"))
    for marker, value in checkpoint_placeholders(counts, overlap, go_rows).items():
        slides = slides.replace(marker, value)
    for row in go_rows:
        if row["positive_nes"] + row["negative_nes"] != row["significant_fdr_lt_005"] or row["significant_fdr_lt_005"] > row["tested_terms"]:
            raise ValueError("GO term counts do not agree.")
        cohort = "TISSUE" if row["dataset"] == "GSE66229" else "SALIVA"
        marker = f'__{cohort}_{row["ontology"]}_COUNT__'
        if slides.count(marker) != 1:
            raise ValueError(f"Missing GO count placeholder: {marker}")
        slides = slides.replace(marker, str(row["significant_fdr_lt_005"]))
    for marker, filename in (("__FILTER_SVG__", "filter_sensitivity.svg"), ("__EFFECTS_SVG__", "gene_effects.svg")):
        svg = (RESULTS / "presentation" / filename).read_text(encoding="utf-8")
        # Remove standalone defaults so the inline charts inherit the active deck theme.
        svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
        prefix = "filter" if filename.startswith("filter") else "effects"
        svg = svg.replace('id="title"', f'id="{prefix}-title"').replace('id="description"', f'id="{prefix}-description"')
        svg = svg.replace('aria-labelledby="title description"', f'aria-labelledby="{prefix}-title {prefix}-description"')
        slides = slides.replace(marker, svg)
    for marker, filename in (("__VOLCANO_PNG__", "volcano.png"), ("__HEATMAP_PNG__", "heatmap.png")):
        encoded = base64.b64encode((RESULTS / filename).read_bytes()).decode("ascii")
        slides = slides.replace(marker, "data:image/png;base64," + encoded)
    for marker, filename in (
        ("__TISSUE_VOLCANO_PNG__", "tissue-volcano.png"), ("__TISSUE_HEATMAP_PNG__", "tissue-heatmap.png"),
        ("__TISSUE_GO_PNG__", "tissue-go.png"), ("__SALIVA_GO_PNG__", "saliva-go.png"),
        ("__MODEL_AUC_PNG__", "model-auc.png"), ("__MODEL_ROC_PNG__", "model-roc.png"),
    ):
        slides = slides.replace(marker, "data:image/png;base64," + base64.b64encode((RETRY / filename).read_bytes()).decode("ascii"))

    with (ROOT / "reference" / "publication" / "metric_targets.csv").open(newline="", encoding="utf-8") as handle:
        published_metrics = list(csv.DictReader(handle))
    with (RETRY / "computed_metrics.csv").open(newline="", encoding="utf-8") as handle:
        computed_metrics = list(csv.DictReader(handle))
    slides = slides.replace("__SALIVA_AUC_PLOT__", saliva_auc_plot(computed_metrics, published_metrics))
    leakage = json.loads((RETRY / "tissue-rf-fold-local.json").read_text(encoding="utf-8"))
    tissue_rf = next(row for row in computed_metrics if row["scenario"] == "A" and row["model"] == "RF")
    if leakage["paper_like_auc"] != float(tissue_rf["auc"]) or leakage["holdout_n"] != sum(int(tissue_rf[key]) for key in ("tp", "tn", "fp", "fn")):
        raise ValueError("Completed leakage subcheck does not match the baseline model record.")
    slides = slides.replace("__LEAKAGE_CHECK_PLOT__", leakage_check_plot(leakage))
    if "__" in slides:
        raise ValueError("Unresolved medical slide placeholder.")
    pattern = r'<div class="slides">.*?</div>\s*</div>\s*</main>'
    output, replaced = re.subn(pattern, lambda _: slides + "\n</div>\n</main>", base, count=1, flags=re.S)
    if replaced != 1:
        raise ValueError("The reusable Reveal deck shell was not found.")
    output = output.replace("Saliva, signals &amp; stomach cancer", "Gastric cancer transcriptomics / medical research")
    output = output.replace("Ten plain-language slides on Lopes et al. (2025) and a partial reproduction of gastric cancer saliva transcriptomics.", "Ten medical-research slides on Lopes et al. (2025), actual tissue/saliva DE, GO and model visualizations; robustness remains in progress.")
    output = output.replace("Each dot is a gene. Left = lower activity in cancer; higher = stronger statistical evidence.", "GC-minus-control log2FC; original axes retained. Statistical implementations and vertical quantities are not assumed identical.")
    output = output.replace("Paper / 14 differing genes", "Publication / 14 saliva DEGs")
    output = output.replace("Reproduction / 37 differing genes", "Reproduction / 37 saliva DEGs")
    output = output.replace('svg text { font-family: inherit; fill: var(--cp-text); }', 'svg text { font-family: inherit; }')
    source_panel = """
  <dialog id="sources-dialog" aria-labelledby="sources-heading">
    <div class="panel-top"><div><h2 id="sources-heading">Medical presentation / sources and evidence</h2><div class="panel-caption">Original plots and source records are embedded; no internet is needed to read the deck.</div></div><button class="close-panel" type="button" data-close="sources-dialog" aria-label="Close sources">Close</button></div>
    <h3>Current slide</h3><p id="current-source"></p>
    <h3>Publication</h3><p>Lopes et al., <i>Scientific Reports</i> 15, 18491 (2025).
    <a href="https://doi.org/10.1038/s41598-025-96864-0" target="_blank" rel="noopener noreferrer">DOI 10.1038/s41598-025-96864-0</a>;
    <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12116728/" target="_blank" rel="noopener noreferrer">PMC full article</a>, CC BY 4.0.
    Published model estimates are explicitly distinguished from measured reproduction results; saliva panels differ (14 versus 37 genes).</p>
    <h3>Completed execution checkpoint</h3><p>Cloud retry <code>results/retry-20261102/</code>, execution commit
    <code>c72855d6729a7c08bf61cccbe1656ce1607e068a</code>: four raw cohorts (695 samples), tissue/saliva DE, GO/GSEA and model scenarios A-D completed.
    Tissue: 1,616 DEGs, saliva: 37, overlap: GJB2 and MMP12. Saliva significant GO BP/CC/MF counts: 88/53/39.
    Saliva held-out SVM AUC: 0.311, compared with published 0.868. Robustness remains incomplete; this is not a completed full-run presentation.</p>
    <p>The last execution report confirmed active robustness at 09:56 UTC. Subsequent continuation messages failed remote priority admission;
    live process state is now unverified. Session idling is not evidence that the scientific process completed or exited. No duplicate run was started.</p>
    <p>Source records: <code>reference/computed/retry-20261102/gene_counts.csv</code>, <code>deg_overlap.json</code>,
    <code>go_summary.json</code>, <code>computed_metrics.csv</code> and <code>transfer_provenance.json</code>. Their exact execution SHA256s are verified locally and recorded in the HTML's embedded medical-source-records data.
    Large raw tables, manifests and original logs remain in the coordinated execution session.</p>
    <p>Original saliva summary and effect-size data: <code>reference/computed/GSE64951/</code>, analysis code
    <code>c11b4d0d370feb9efa1e82681dc25c2819697134</code>. The new run's saliva results and original plot hashes agree with that snapshot.
    <code>REPORT.md</code> is the earlier saliva-only audit, not the current cloud-run report.</p>
    <h3>Interpretation limits</h3><p>Same public cohorts, not new patients. No established affy/limma numerical parity.
    Four median-polish nonconvergence warnings do not identify specific affected cohorts or probe sets.
    gseapy per-ontology permutation FDR is not clusterProfiler adjusted P; zero empirical FDR is not mathematical zero.
    Full-cohort DE risks selection leakage; cohort-wide RMA remains transductive. Independent clinical saliva validation is not established.</p>
    <p>Computed AUC CIs use 2,000 stratified bootstrap replicates, conditional on fitted models, not the whole pipeline.
    SVM saliva outcomes or AUC direction are not reversed. Undefined PPV and LR+ remain undefined; infinite LR+ in source records remains infinite.</p>
    <p>One completed tissue RF same-holdout fold-local-DE subcheck gives AUC 0.9948 versus 0.9970 in the full-cohort-DE branch.
    <code>tissue-rf-fold-local.json</code> records the verified 120-sample holdout, selected genes and source/prediction hashes.
    This does not complete split resampling, permutation, stability or shift checks. Full-cohort RMA remains transductive.</p>
    <h3>Actual figures</h3><p>Original reproduction tissue/saliva volcano and heatmap PNG bytes are preserved.
    New GO summary PNGs are computed from the completed GO tables; original model AUC/ROC PNGs derive from actual stored predictions. Their scientific colors and axes are unchanged.
    Saliva gene-effect and filter-sensitivity SVGs derive from measured saved statistics, not simulated data.
    The original-plot viewer uses publication Figure 3D (CC BY 4.0), cropped only for display; axes are not statistically matched.</p>
    <p>Reveal.js 5.2.1 is embedded under MIT; its full license remains in this file. Arrows navigate; N opens notes, S sources, Esc closes panels. Theme and print controls work offline.</p>
  </dialog>
"""
    output, source_count = re.subn(r'<dialog id="sources-dialog".*?</dialog>', lambda _: source_panel, output, count=1, flags=re.S)
    if source_count != 1:
        raise ValueError("Reusable source dialog was not found.")
    records = {
        "transfer_provenance": transfer,
        "gene_counts": counts,
        "deg_overlap": overlap,
        "go_summary": go_rows,
        "computed_metrics": computed_metrics,
        "tissue_rf_fold_local": leakage,
    }
    data_script = '<script id="medical-source-records" type="application/json">' + json.dumps(records).replace("</", "<\\/") + "</script>"
    output = output.replace("</body>", data_script + "\n</body>")
    adjustments = {
        'const dialogs = [notesDialog, sourcesDialog, plotsDialog];':
            'const heatmapDialog = document.getElementById("heatmap-dialog");\n'
            '        const heatmapButton = document.getElementById("heatmap-button");\n'
            '        const tissueDialog = document.getElementById("tissue-dialog");\n'
            '        const tissueButton = document.getElementById("tissue-button");\n'
            '        const goDialog = document.getElementById("go-dialog");\n'
            '        const goButton = document.getElementById("go-button");\n'
            '        const modelDialog = document.getElementById("model-dialog");\n'
            '        const modelButton = document.getElementById("model-button");\n'
            '        const dialogs = [notesDialog, sourcesDialog, plotsDialog, heatmapDialog, tissueDialog, goDialog, modelDialog];',
        '[plotsDialog, plotsButton]]);': '[plotsDialog, plotsButton], [heatmapDialog, heatmapButton], [tissueDialog, tissueButton], [goDialog, goButton], [modelDialog, modelButton]]);',
        'plotsButton.setAttribute("aria-expanded", "false");':
            'plotsButton.setAttribute("aria-expanded", "false");\n'
            '        heatmapButton.setAttribute("aria-expanded", "false");\n'
            '        tissueButton.setAttribute("aria-expanded", "false");\n'
            '        goButton.setAttribute("aria-expanded", "false");\n'
            '        modelButton.setAttribute("aria-expanded", "false");',
        'plotsButton.addEventListener("click", () => openPanel(plotsDialog));':
            'plotsButton.addEventListener("click", () => openPanel(plotsDialog));\n'
            '        heatmapButton.addEventListener("click", () => openPanel(heatmapDialog));\n'
            '        tissueButton.addEventListener("click", () => openPanel(tissueDialog));\n'
            '        goButton.addEventListener("click", () => openPanel(goDialog));\n'
            '        modelButton.addEventListener("click", () => openPanel(modelDialog));',
    }
    for previous, replacement in adjustments.items():
        if output.count(previous) != 1:
            raise ValueError(f"Reusable dialog integration changed: {previous}")
        output = output.replace(previous, replacement)
    target = ROOT / "presentation-medical.html"
    target.write_text(output, encoding="utf-8", newline="\n")
    record = {
        "status": "medical research draft; completed raw processing, DE, GO and ML scenarios A-D; robustness incomplete; remote live process status unverified",
        "inputs_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "presentation.html", Path(__file__), Path(__file__).parent / "medical_slides.html",
                ROOT / "reference" / "publication" / "metric_targets.csv",
                RESULTS / "presentation" / "provenance.json",
                RESULTS / "presentation" / "full_run_attempts.json",
                RETRY / "transfer_provenance.json",
                *(RETRY / filename for filename in transfer["files_sha256"]),
            )
        },
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (RESULTS / "presentation" / "deck_provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Created {target.name}: {target.stat().st_size:,} bytes; original presentation unchanged.")


if __name__ == "__main__":
    main()
