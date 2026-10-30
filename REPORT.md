# Reproduction status: Lopes et al. (2025)

**This is an implementation and audit, not a completed numerical reproduction.**
No raw-cohort RMA, Tables 1–3 replication, or large robustness experiment has
been completed for this report. Published values are reference targets, never
substitutes for measurements. A successful unit test is not a reproduced
biological claim.

Source: [Scientific Reports, DOI 10.1038/s41598-025-96864-0](https://doi.org/10.1038/s41598-025-96864-0),
PMID 40425785, PMCID PMC12116728, CC BY 4.0.
The [Europe PMC article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12116728/fullTextXML)
was retrieved (122,137 bytes; SHA256
`e87f4a520bfed5d61f74a41d91d1138bd7020cabb81fe217b1c22e8070ac31d0`).

## Claim-by-claim verdict

“Not reproduced” below means **not yet assessed on raw cohort data**, not
evidence that the publication is incorrect.

| Published claim | Verdict | What is required for a verdict based on data |
|---|---|---|
| 400 tissue training, 94 saliva, 132 Chinese tissue, 69 Italian tissue samples | Partially reproduced: metadata counts match exactly | Raw CEL parsing/normalization remains separate from the metadata audit |
| 21,367 annotated unique genes | Not reproduced | Raw normalization and declared annotation/alias release |
| 14,782 tissue / 20,407 saliva genes after filtering | Not reproduced | Compare all three declared median-filter interpretations |
| Tissue 1,302 DEGs: 579 up, 723 down | Not reproduced | Moderated DE on raw-derived log2 expression |
| Saliva 14 DEGs, all down; no tissue overlap | Not reproduced | Exact list comparison and overlap artifact |
| Tissue cell-cycle GSEA and saliva CC/MF terms; no significant saliva BP | Not reproduced | All-ontology GSEA, GO snapshot hashes, NES/FDR comparisons |
| CDH3, DKC1, ESM1, MUSTN1, RCC2, TMT1A, VEGFD tissue signature | Not reproduced | Discovered signature and importance ranks, not a forced panel |
| Table 1: internal tissue performance | Not reproduced | Fixed split, selected panel, predictions and confidence intervals |
| Table 2: external tissue performance | Not reproduced | Direct source-model predictions without target retraining |
| Scenario C: tissue-model saliva AUC approximately 0.5 | Not reproduced | Direct predictions on the complete saliva cohort |
| Table 3: saliva SVM AUC 0.868, CI 0.721–0.974 | Not reproduced | Saliva DEG models on the held-out split |
| Claims survive selection leakage, split variance, permutation, shift, signature instability | Not reproduced | Complete seeded robustness runs with failure accounting |

The CLI writes a separate computed report, provenance, comparisons and predictions
under its output directory. Review stage status files before interpreting any
artifacts from a partial or failed run.

### Completed cohort metadata audit

The public series/GSM metadata were retrieved and checksummed. All manifest
hashes were reviewed against their provenance records. The exact GSMs, labels,
subseries membership and individual CEL URLs are retained under
`reference/acquisition/<GSE>/`; no raw CELs are committed.

| Dataset | GC | Control | Ambiguous / excluded |
|---|---:|---:|---:|
| GSE66229 | 300 | 100 | 0 / 0 |
| GSE64951 | 63 | 31 | 0 / 0 |
| GSE54129 | 111 | 21 | 0 / 0 |
| GSE13911 | 38 | 31 | 0 / 0 |

## Source discrepancies and unresolved evidence

* **Table 1 SVM NPV:** the source says `0.848 (0.852–0.750)`. The interval is
  reversed and does not contain its estimate. The transcription preserves it;
  comparisons flag it as invalid rather than “repairing” the publication.
* **RF internal AUC interval:** Table 1 says `0.966–1.000`, whereas Table 4
  says `0.973–1.000`. Both are retained as separate source targets.
* **Table 2 GSE13911 SVM NPV:** missing in the source; it is not assigned zero.
* **Supplementary Table S1:** the XML identifies
  `41598_2025_96864_MOESM1_ESM.docx` (451,494 bytes; published MD5
  `602c371a4d026cff0705a18ef6eebc19`). Publisher DNS failed in this sandbox;
  Europe PMC's supplementary endpoint timed out and PMC presented a download
  challenge. **Package versions in S1 remain unverified.** Only R 4.3.2 and
  RStudio 2024.04.2.764 are confirmed from the article. The `evidence` stage
  retries the publisher download and transcribes all DOCX tables if accessible.
* **Raw-processing prerequisite verified:** the original CDF was downloaded
  from `aroma-project.org` using the downloader (6,886,951 bytes; SHA256
  `c241e617b8f3ee30a9640d77d885d77c1845c6a17c214f93b75c32f822538ddb`).
  An earlier `urllib` request returned HTTP 403, but this was not a persistent
  access blocker. A local `--cdf` option is also supported. Do not substitute
  a gene-redefined BrainArray CDF and call it the same preprocessing.

## Assumptions and deviations

1. **RMA:** independent Python implementation, not `affy` called through R.
   Synthetic CEL/CDF and numerical tests do not establish bit-for-bit parity
   with `affy::rma`. Raw processing must succeed by default; processed matrices
   require an explicit fallback flag and remain a major deviation. CEL support
   supports v3 text, XDA v4 and Calvin v1 intensity CELs (including gzip); CDF
   supports ASCII and XDA v1 expression definitions. The real saliva CELs are
   Calvin, so text/XDA-only parsing is insufficient. Unsupported formats fail
   visibly.
2. **Data scale matters:** GSE66229's processed matrix is log10 RMA, requiring
   multiplication by `log2(10)`. GSE64951 and GSE13911 deposit MAS5 signals;
   taking their log2 does **not** make them RMA. GSE54129 deposits RMA values.
   The fallback is never an equivalent raw-data reproduction.
3. **Cohort identity:** GSE66229 is a SuperSeries: GSE62254 tumors and GSE66222
   normals must not be ingested twice. Labels come from sample characteristics,
   not accession order, filename suffixes, or blanket keyword searches.
   Tissue controls include patient-matched/adjacent normals, not uniformly
   healthy volunteers. GSE54129's series-level control description conflicts
   with some sample source text; tissue characteristics determine class.
4. **Probe mapping:** highest-mean-expression probe per unambiguous gene;
   deterministic ties. Ambiguous multi-symbol probes are excluded. HGNC alias
   resolution is recorded, including METTL7A→TMT1A, SEPT10→SEPTIN10,
   FIGF→VEGFD. Annotation updates can change the tested universe.
5. **Low-expression rule:** default retains genes with any sample above the
   global matrix median. Alternatives retain genes with mean above that median,
   or at least half the samples above it. Counts are compared to the paper;
   no rule is selected using AUC, DEG identity or desired outcomes.
6. **DE:** independent-sample two-group OLS with empirical-Bayes variance
   moderation and BH adjustment. No limma trend/robust extension or patient
   blocking; matched specimens may violate independence. Significance uses
   strict `abs(log2FC) > 1` and `padj < 0.05`.
7. **GO:** all filtered genes ranked by signed moderated t; versioned GO 2023
   GMT snapshots (or user-supplied snapshots), fixed permutation seed.
   `gseapy` permutation FDR is not `clusterProfiler`'s adjusted P value.
   With 1,000 permutations, extremely small published adjusted P values cannot
   be reproduced by ordinary permutation counts. Ontologies are tested
   separately; no `rrvgo` semantic-term reduction is applied.
8. **ML:** median imputation; deterministic greedy correlation pruning is an
   approximation to `recipes::step_corr`; random undersampling occurs only
   inside training fits. RF/GBM impurity-rank consensus operationalises an
   unspecified visual/importance selection rule; DT is secondary. Random seeds,
   tuning grids and fixed tree counts are ours, not recovered author settings.
   SVM scaling is training-fitted and RBF is the declared kernel.
   Saliva retains its discovered DEGs, not a forced copy of the published 14.
9. **Leakage:** the paper-like branch uses full-cohort DE before the 70/30 split
   and is explicitly leaky. Panel ranking and tuning use training data only.
   The same-holdout control moves DEG filtering and ranking inside CV fits.
   Repeated splits and null runs use fresh training-only DE and model selection;
   they are a different protocol from full-cohort-DE reproduction. Cohort-wide
   RMA remains transductive even with fold-local DE; patient-grouped splitting
   and frozen inductive normalization are not claimed.
10. **Nulls and stability:** permutations assume independent/exchangeable samples.
    No training DEGs gives a declared intercept-only predictor (AUC 0.5), not
    relaxed significance thresholds. Other errors abort by default. Per-model
    and maximum-over-four null exceedances use threshold 0.87 and plus-one
    empirical probabilities. Split rankings share ties; signature frequencies
    use all requested runs as denominator.
11. **Shift:** separate gene-wise standardization fits source statistics on
    source training samples and target statistics on the whole unlabeled target.
    Source models are refit on transformed source training samples with fixed
    parameters. This is explicitly **transductive target adaptation**, not the
    paper's direct validation. ComBat is not used.
12. **Evaluation:** cancer is positive; probability threshold 0.5; stratified
    percentile bootstrap (2,000 replicates), plus a separate clipped-normal
    DeLong AUC interval. CIs condition on the trained model and observed class
    counts, not selection uncertainty. Undefined ratios remain missing;
    positive divided by zero remains infinity. PPV/NPV depend on cohort
    prevalence, which is not screening prevalence.

## Interpretation limits

Differences can arise from annotation, symbol updates, CEL processing,
underspecified filtering/selection, stochastic splits, engine defaults and
cross-cohort expression shifts. A highly accurate internal tissue classifier
does not establish useful saliva screening, and there is no independent
external saliva cohort in this study. Until raw-cohort results and robustness
outputs are computed, neither numerical replication nor clinical utility has
been established here.
