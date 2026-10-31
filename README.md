# Gastric cancer transcriptomics: independent Python reproduction

Rebuilds the analysis of [Lopes et al., Scientific Reports 15, 18491 (2025)](https://doi.org/10.1038/s41598-025-96864-0).
Python **3.14+**, with pinned dependencies and a hash-bearing `uv.lock`.

**Status:** partial reproduction, not a completed numerical replication.
Actual raw saliva processing recovered all 14 published DEGs among 37
downregulated genes; mapped/filter counts differ (20,953 / 20,527).
Tissue processing, GSEA, model tables and cohort-scale robustness remain unrun.
Read `REPORT.md` for claim-by-claim status, assumptions, source discrepancies
and data-access limitations. No published number is presented as computed.
This is research software, not a validated diagnostic.

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