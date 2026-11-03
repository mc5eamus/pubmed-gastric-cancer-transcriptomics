# Gastric cancer transcriptomics: independent Python reproduction

Rebuilds the analysis of [Lopes et al., Scientific Reports 15, 18491 (2025)](https://doi.org/10.1038/s41598-025-96864-0).
Python **3.14+**, with pinned dependencies and a hash-bearing `uv.lock`.

**Status:** partial reproduction, not a completed numerical replication.
Actual raw saliva processing recovered all 14 published DEGs among 37
downregulated genes; mapped/filter counts differ (20,953 / 20,527).
The cloud retry completed raw processing for all four cohorts, tissue/saliva
DE and GO enrichment. Tissue yielded 1,616 DEGs, with two tissue-saliva
overlapping genes; significant saliva BP terms were found, unlike the paper.
Model scenarios A-D completed; saliva held-out SVM AUC was 0.311 rather
than published 0.868, using 37 discovered rather than 14 fixed genes.
Cohort-scale robustness remains incomplete.
The new source-verified checkpoint is under `reference/computed/retry-20261102/`;
`REPORT.md` preserves the earlier saliva-only audit and its limitations.
No published number is presented as computed.
This is research software, not a validated diagnostic.

## Plain-language presentation

Open **[presentation.html](presentation.html)** directly in a modern browser.
The ten-slide Reveal.js presentation is a single, fully offline file: its
runtime, styles, diagrams, presenter notes and source citations are embedded.
Only following an external citation link requires internet access.

Use the arrow keys or on-screen arrows to navigate, **N** for the current
slide's notes, **S** for sources, **Esc** to close a panel or toggle the overview,
and **F** for full screen. The toolbar also offers theme switching and printing
of all ten slides without notes.

The results slides include common-scale paper-versus-reproduction gene-count
and gene-overlap charts, paired preprocessing counts, and explicit placeholders
for unassessed prediction results. On slide 8, **Compare original plots** opens
the paper's saliva volcano plot beside the computed plot, with their original
axes and attribution preserved. The figures are embedded and work offline.

The presentation distinguishes published findings from computed results:
the reproduction recovered all 14 published saliva genes among 37 differing
genes, but has not yet evaluated prediction models or established screening
performance. Its evidence snapshot is commit
`c72855d6729a7c08bf61cccbe1656ce1607e068a`; it is not a live results dashboard.

## Medical-research presentation

Open **[presentation-medical.html](presentation-medical.html)** for the separate
ten-slide, fully offline medical-audience draft. The first slide sets the
clinical context; subsequent slides use the publication's RMA, moderated DE,
BH/FDR, GO/GSEA, validation and diagnostic-performance terminology. Each slide
has a takeaway headline, presenter notes and source references. The
plain-language presentation is preserved.

The deck embeds **actual tissue and saliva volcano/heatmap figures**, completed
GO enrichment figures, and DEG effect-size/filtering-sensitivity figures
derived from the saved saliva results. Select the saliva heatmap for
full-resolution inspection; the tissue heatmap and GO plots have separate
inspection panels. The volcano slide also offers the original
paper-versus-reproduction viewer.
Published and computed saliva AUCs/95% CIs are compared on a common scale.
Original model ROC and AUC figures for all scenarios are available in the
inspection panel. Predictions were not inverted to improve the poor SVM result.
Computed CIs use 2,000 stratified bootstraps conditional on the fitted model;
these full-cohort-DE results risk selection leakage and are not
robustness-corrected. No robustness result is invented.

**This is a partial-reproduction draft, not a completed full-run presentation.**
The original local attempt failed dependency TLS and memory prerequisites.
An isolated cloud attempt restored exact dependencies but initially failed
GEO downloads. A subsequent retry recovered original raw-data access and
completed raw RMA for 695 samples across all four cohorts, DE and GO/GSEA,
without fallback or reduced defaults. Model scenarios A-D have since completed;
cohort-scale robustness remains incomplete, so the full-run prerequisite is not met.
One tissue RF fold-local-selection subcheck completed on the verified same
120-sample holdout (AUC 0.9948 versus 0.9970). No completed resampling block or
final robustness result has been received. Later continuation messages failed
remote priority admission; live process state cannot currently be verified.
Session idling is not evidence of scientific completion or failure.
Prior failure history is preserved in
`reference/computed/GSE64951/presentation/full_run_attempts.json`.

The completed checkpoint in `reference/computed/retry-20261102/` contains
hash-verified small execution records, original figures and transfer
provenance; full raw tables and logs remain in the coordinated cloud session.
Four median-polish nonconvergence warnings were emitted. Their count does not
identify affected cohorts or probe sets; numerical affy parity is unestablished.
GO per-ontology permutation FDR is not clusterProfiler adjusted P.

Regenerate the derived figures and bundled medical deck using Python's
standard library, without installing scientific dependencies:

```powershell
python .\scripts\build_medical_presentation.py
```

The builder reads `presentation.html` as the offline Reveal shell and
`scripts/medical_slides.html` as the medical content. Figure provenance,
source hashes and deck provenance are saved under
`reference/computed/GSE64951/presentation/`. CSV source hashes are checked
against the saved snapshot, allowing only Git's Windows CRLF-to-LF checkout
conversion; original PNG bytes are verified unchanged.

## Heatmap and volcano interpretation presentation

Open **[presentation-interpretation.html](presentation-interpretation.html)** for a
separate twelve-slide Reveal.js lesson on reading the saved saliva visualizations.
Three numbered examples follow **YOD1, TMPRSS11B and SPAG9** across the original
heatmap and full-universe volcano. Enlarged heatmap row strips, marked volcano
points, readings and interpretations distinguish relative expression, effect
size, statistical evidence and diagnostic limitations.

Each section ends with two deliberately tricky single-answer multiple-choice
questions. Select an option and choose **Check answer** for an option-specific
explanation, the correct reasoning and a takeaway. **Try again** resets only that
question; responses stay in place while navigating the open deck. Questions cover
row standardization, selection bias, effect size versus evidence, and strict
DEG thresholds. Synthetic examples are explicitly labelled. Printed quiz slides
include their solutions.

The deck is a single fully offline HTML file, with original images, Reveal.js,
presenter notes and source records embedded. Source hashes are checked before
building; example statistics come from the saved DEG table, and the two example
samples' cancer/control identities come from the manifest, not from colour.
It interprets the saved saliva snapshot rather than reporting the current status
of other cohorts or model runs. Existing presentations are preserved.

Arrow keys navigate; **N** opens notes, **Esc** closes a panel or opens the
overview, and **F** enters full screen. The toolbar provides theme switching and
printing. Figure inspection buttons open the unannotated originals with a
full-resolution option.
Within quiz options, arrow keys change the selected radio button rather than
advancing slides; use Tab to reach **Check answer**, or the on-screen arrows to
continue the presentation.

Regenerate without installing dependencies:

```powershell
python .\scripts\build_interpretation_presentation.py
```

The builder reuses the embedded Reveal.js runtime/styles in `presentation.html`
and the content in `scripts/interpretation_slides.html`; it does not rerun
expression analysis.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
REPO="$(pwd)"
uv python install 3.14
uv sync --frozen
uv run --frozen gastric-repro --help
uv run --frozen pytest -q
```

CI runs the offline unit tests on Python 3.14. Tests use small fixtures, not
the public multi-GB CEL data.

## Article-ordered notebook

Open the executed, reference-mode presentation with Jupyter:

```bash
uv run --frozen --group notebook jupyter notebook notebooks/lopes2025_reproduction.ipynb
```

The parameter cell defaults to **reference** mode: it is offline and reads only
checked-in `reference/` files. Unrun stages appear as explicit placeholders; a
published target is never presented as a computed result. **Reduced** mode runs
the README's exploratory settings and downloads public data; its results are
not paper-faithful. **Full** mode reads an existing completed `results/` output
from `gastric-repro run` and never launches that long run from the notebook.
Change `RUN_MODE` at the top to select a mode. CI executes the notebook in
reference mode without network access.

For a notebook that implements the core filtering, differential-expression,
heatmap, and volcano calculations directly in its cells (without importing the
package or invoking the CLI), open
`notebooks/lopes2025_reproduction_standalone.ipynb`. It runs offline from the
checked-in saliva result and automatically uses local full-cohort expression
matrices when they are available.
The heatmap always shows individual samples, never two group means: offline
mode displays the hash-verified saved 37-DEG by 94-sample raw-CEL figure and
labels it as saved, not recomputed. With a local saliva expression matrix,
the notebook requires every included sample, excludes other columns, and
recomputes row z-scores across the cohort with control/cancer sample labels.
Heatmap source, dimensions, group counts, and the output hash are recorded in
`notebook_results/provenance.json`.

The saliva model stage supports both the published 14-gene panel and discovered
DEGs: run `gastric-repro ml --saliva-gene-set published` or
`gastric-repro ml --saliva-gene-set discovered` using separate output directories
for side-by-side results.

## One-command analysis

```bash
uv run --frozen gastric-repro run --cache "$REPO/data" --output "$REPO/results"
```

This is a **large, long-running analysis**, not a smoke test: downloads,
independent raw RMA for four cohorts, annotation, differential expression,
GO enrichment, feature selection, scenarios A–D, bootstrap intervals and all
five robustness checks. Plan for tens of GB of disk, substantial RAM for
full-cohort arrays, and many CPU-hours or more for resampling.
Do not use a reduced run to claim the default analysis completed.

Raw processing requires an original HG-U133_Plus_2 ASCII or XDA v1 CDF. Supply a local CDF with
`--cdf /absolute/path/HG-U133_Plus_2.cdf` if automatic acquisition is unavailable.
The default original CDF was successfully downloaded and checksummed during
implementation. A gene-redefined BrainArray CDF is not an equivalent substitute.
GO defaults download named GO 2023 snapshots and record SHA256; alternatively
supply **all three** `--go-bp`, `--go-cc`, `--go-mf` local GMT paths. Network
access to GEO/NCBI, HGNC, the CDF source, Europe PMC, and Enrichr is required.
An inaccessible resource is a reported failure, not permission to invent data.

### Stages and resumability

```bash
# Start with a lightweight metadata/sample audit:
uv run --frozen gastric-repro download --metadata-only --cache "$REPO/data" --output "$REPO/results"
# Download raw CELs without normalization:
uv run --frozen gastric-repro download --cache "$REPO/data" --output "$REPO/results"
# Each later stage is also individually runnable:
uv run --frozen gastric-repro normalise --cache "$REPO/data" --output "$REPO/results"
uv run --frozen gastric-repro evidence --cache "$REPO/data" --output "$REPO/results"
uv run --frozen gastric-repro de --output "$REPO/results"
uv run --frozen gastric-repro enrichment --cache "$REPO/data" --output "$REPO/results"
uv run --frozen gastric-repro ml --output "$REPO/results"
uv run --frozen gastric-repro robustness --output "$REPO/results"
uv run --frozen gastric-repro report --output "$REPO/results"
```

`normalise` uses verified cached downloads; rerunning does not trust a modified
cache. `--dataset GSE64951` restricts download/normalization to one cohort.
Raw CELs are fetched individually from each GSM's metadata rather than unpacking
the whole series TAR; this permits per-sample checksums, exact inclusion, and
resuming without downloading unrelated samples.
`ml` includes direct validation on both external tissue cohorts and all saliva
samples. `validation` reruns the same deterministic source-training analysis;
it never trains on target labels. No untrusted model pickle is loaded.

Re-use the **same settings** across individual stages. DE input hashes and filter
settings are checked before reuse. A stage status JSON records completion or
failure; partial outputs are not proof that a run completed. Use a fresh output
directory for a different protocol/seed to avoid confusing artifacts.

### Processed-data fallback (not faithful RMA)

Only when raw processing is impossible, explicitly opt in:

```bash
uv run --frozen gastric-repro normalise --allow-series-matrix \
  --cache "$REPO/data" --output "$REPO/results-fallback"
```

Each dataset's provenance records the deviation and expression-scale conversion.
In particular, log2 MAS5 values are **not** raw-CEL RMA. Do not mix fallback and
raw outputs without noting that difference.

### Resource controls

Defaults: seed 2025, stratified 70/30 split, CV 10 folds × 5 repeats, up to 10
candidates/model, 200 trees, 2,000 bootstrap replicates, 100 splits, 100 label
permutations/model, 1,000 GSEA permutations, one worker.
Use `--jobs` for grid-search workers, subject to memory limits. An intentionally
reduced exploratory run can use:

```bash
uv run --frozen gastric-repro run --cache "$REPO/data" --output "$REPO/results-exploratory" \
  --folds 2 --repeats 1 --candidates 1 --estimators 20 \
  --bootstrap 20 --splits 2 --permutations 2 --gsea-permutations 20
```

These settings are recorded and are **not** a paper-faithful experiment.
Use `--filter-rule` to run any of the three predeclared median rules; never
choose it based on test AUC.

## Outputs

`data/` and `results/` are git-ignored; never commit CELs or raw archives.

* Per-cohort `manifest.csv`: GSM, inclusion/exclusion, class, source metadata.
* `expression.csv`: unique genes × GSM, normalized log2 values;
  preprocessing provenance includes download/annotation checksums.
* `gene_counts.csv`, DEG lists, filter sensitivity, overlap, volcano/heatmap PNGs.
* GO rankings, enrichment results and published-term comparisons.
* Selected genes, importance, split IDs, chosen parameters and prediction tables.
* Tables 1–3 point comparisons plus Scenario C; all 108 source point/interval
  targets from Tables 1–4 compared when article evidence is available.
* Stratified bootstrap and DeLong intervals, including undefined/infinite ratios.
* Robustness distributions, rankings, null runs, signature frequencies and
  separately labeled target-standardization results.
* `REPORT.md` and stage/run provenance inside the output directory.

Small checked-in publication transcriptions and source-access records live
under `reference/publication/`; these are **published references**, not model
results. Verified GSM manifests and metadata provenance for all four cohorts
are under `reference/acquisition/`.
Small actual raw-saliva results, including DEG statistics, plots and CEL
checksums, are under `reference/computed/GSE64951/`. These are computed
results, not the published reference targets.

The repository-level `REPORT.md` remains the human-reviewed audit. Generated
reports expose measurements and missing stages, not automatic proof of clinical
validity. Supplementary Table S1 versions are unverified until its DOCX download
succeeds; the evidence stage records availability and transcribes its tables.