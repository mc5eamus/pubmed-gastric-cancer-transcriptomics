"""File-based stage orchestration; published targets never substitute for data."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import pandas as pd

from gastric_repro.published import (
    COHORTS, SALIVA_GENES, TISSUE_GENES, article_metric_targets,
    compare_article_metrics, compare_metrics, published_go_targets, published_metrics,
)


@dataclass(frozen=True)
class Settings:
    cache: Path
    output: Path
    seed: int = 2025
    filter_rule: str = "any_above_median"
    folds: int = 10
    repeats: int = 5
    candidates: int = 10
    bootstrap: int = 2000
    splits: int = 100
    permutations: int = 100
    gsea_permutations: int = 1000
    estimators: int = 200
    jobs: int = 1
    allow_series_matrix: bool = False
    cdf: Path | None = None
    saliva_gene_set: str = "discovered"

    def __post_init__(self):
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if self.saliva_gene_set not in {"discovered", "published"}:
            raise ValueError("saliva_gene_set must be 'discovered' or 'published'")
        if self.folds < 2 or not 1 <= self.candidates <= 10:
            raise ValueError("folds must be >= 2; candidates must be in [1, 10]")
        if min(self.repeats, self.bootstrap, self.splits, self.permutations,
               self.gsea_permutations, self.estimators) < 1:
            raise ValueError("repeats and resampling counts must be positive")
        if self.jobs == 0:
            raise ValueError("jobs cannot be zero")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_preprocessing(directory: Path) -> None:
    provenance = directory / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()).get("status") != "complete":
        raise ValueError(f"{directory.name}: preprocessing did not complete successfully")


def load_cohort(output: Path, accession: str) -> tuple[pd.DataFrame, pd.Series]:
    directory = Path(output) / accession
    check_preprocessing(directory)
    expression = pd.read_csv(directory / "expression.csv", index_col=0)
    manifest = pd.read_csv(directory / "manifest.csv")
    if not {"gsm", "label", "included"}.issubset(manifest):
        raise ValueError(f"{accession}: manifest requires gsm, label, included")
    included = manifest["included"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if included.isna().any():
        raise ValueError(f"{accession}: unrecognized manifest inclusion flag")
    selected = manifest.loc[included]
    if selected.gsm.duplicated().any():
        raise ValueError(f"{accession}: duplicate manifest GSM")
    labels = selected.set_index("gsm")["label"]
    if set(labels.index) != set(expression.columns):
        raise ValueError(f"{accession}: expression/manifest sample mismatch")
    labels = pd.to_numeric(labels.reindex(expression.columns), errors="raise")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{accession}: labels must include exactly classes 0 and 1")
    if expression.empty or not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError(f"{accession}: expression identifiers must be nonempty and unique")
    return expression, labels.astype(int)


def de_stage(settings: Settings) -> None:
    from gastric_repro.de import (
        differential_expression, filter_expression, filter_sensitivity_counts,
        plot_heatmap, plot_volcano,
    )

    deg_sets = {}
    summaries = []
    for accession, expected, published_degs, up, down in (
        ("GSE66229", 14782, 1302, 579, 723),
        ("GSE64951", 20407, 14, 0, 14),
    ):
        expression, labels = load_cohort(settings.output, accession)
        directory = settings.output / accession / "de"
        directory.mkdir(parents=True, exist_ok=True)
        sensitivity = filter_sensitivity_counts(expression)
        sensitivity["published_retained"] = expected
        sensitivity["absolute_difference"] = abs(sensitivity.retained_genes - expected)
        sensitivity["closest_by_gene_count_only"] = (
            sensitivity.absolute_difference == sensitivity.absolute_difference.min()
        )
        sensitivity.to_csv(directory / "filter_sensitivity.csv", index=False)
        filtered = filter_expression(expression, settings.filter_rule)
        table = differential_expression(filtered, labels)
        table.to_csv(directory / "differential_expression.csv", index_label="gene")
        degs = table.loc[table.is_deg]
        degs.to_csv(directory / "degs.csv", index_label="gene")
        deg_sets[accession] = set(degs.index)
        plot_volcano(table, directory / "volcano.png")
        if len(degs):
            plot_heatmap(filtered, directory / "heatmap.png", results=table)
        else:
            (directory / "heatmap.png").unlink(missing_ok=True)
        summaries.append({
            "dataset": accession, "genes_before_filter": len(expression),
            "published_genes_before_filter": 21367, "genes_retained": len(filtered),
            "published_genes_retained": expected, "degs": len(degs),
            "published_degs": published_degs,
            "up": int((degs.direction == "up").sum()), "published_up": up,
            "down": int((degs.direction == "down").sum()), "published_down": down,
        })
        write_json(directory / "provenance.json", {
            "filter": settings.filter_rule, "expression_sha256": sha256(
                settings.output / accession / "expression.csv"),
            "manifest_sha256": sha256(settings.output / accession / "manifest.csv"),
            "feature_scope": "full_cohort_before_split; selection leakage intentional",
            "de_method": table.attrs,
        })
    pd.DataFrame(summaries).to_csv(settings.output / "gene_counts.csv", index=False)
    write_json(settings.output / "deg_overlap.json", {
        "genes": sorted(deg_sets["GSE66229"] & deg_sets["GSE64951"]),
        "published_overlap": 0,
        "saliva_published_missing": sorted(set(SALIVA_GENES) - deg_sets["GSE64951"]),
        "saliva_unreported_found": sorted(deg_sets["GSE64951"] - set(SALIVA_GENES)),
    })


def read_de(settings: Settings, accession: str) -> pd.DataFrame:
    directory = settings.output / accession / "de"
    check_preprocessing(directory.parent)
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["filter"] != settings.filter_rule:
        raise ValueError("DE filter settings differ; rerun the de stage")
    for filename, key in (("expression.csv", "expression_sha256"),
                          ("manifest.csv", "manifest_sha256")):
        if sha256(settings.output / accession / filename) != provenance[key]:
            raise ValueError(f"Stale DE results for {accession}; rerun de")
    table = pd.read_csv(directory / "differential_expression.csv", index_col=0)
    if table.is_deg.dtype != bool:
        raise ValueError("DE is_deg column must contain boolean values")
    return table


def enrichment_stage(settings: Settings, gene_sets=None) -> None:
    from gastric_repro.enrichment import run_go_prerank
    from gastric_repro.resources import GO_LIBRARIES, go_snapshots

    automatic = gene_sets is None
    gene_sets = go_snapshots(settings.cache) if automatic else gene_sets
    comparisons = []
    for accession in ("GSE66229", "GSE64951"):
        table = run_go_prerank(
            read_de(settings, accession), gene_sets,
            settings.output / accession / "enrichment", seed=settings.seed,
            permutations=settings.gsea_permutations, threads=max(1, settings.jobs),
            source_versions=GO_LIBRARIES if automatic else None,
        )
        table["go_id"] = table["Term"].str.extract(r"(GO:\d+)", expand=False)
        targets = published_go_targets()
        targets = targets.loc[targets.dataset == accession]
        comparison = targets.merge(
            table, on=["ontology", "go_id"], how="left", validate="one_to_one",
        )
        comparison["nes_difference"] = (
            pd.to_numeric(comparison["NES"], errors="raise") - comparison.published_nes
        )
        comparison["p_adjustment_caveat"] = (
            "gseapy permutation FDR vs clusterProfiler p.adjust; not equivalent"
        )
        comparisons.append(comparison)
    pd.concat(comparisons, ignore_index=True).to_csv(
        settings.output / "go_published_comparison.csv", index=False,
    )


def model_options(settings: Settings) -> dict:
    return dict(seed=settings.seed, folds=settings.folds, repeats=settings.repeats,
                candidates=settings.candidates, n_estimators=settings.estimators,
                n_jobs=settings.jobs)


def saliva_model_features(expression: pd.DataFrame, de: pd.DataFrame, gene_set: str) -> list[str]:
    if gene_set == "discovered":
        features = de.index[de.is_deg].tolist()
    elif gene_set == "published":
        features = list(SALIVA_GENES)
    else:
        raise ValueError("saliva gene set must be 'discovered' or 'published'")
    missing = sorted(set(features) - set(expression.index))
    if missing:
        raise ValueError(f"Saliva gene set is missing from expression: {missing}")
    return features


def evaluate_scenario(settings, predictions, labels, scenario, accession, subdir="validation"):
    from gastric_repro.metrics import evaluate_predictions

    directory = settings.output / subdir / scenario / accession
    directory.mkdir(parents=True, exist_ok=True)
    labels = labels.reindex(predictions.index)
    if labels.isna().any():
        raise ValueError("Predictions lack aligned truth labels")
    predictions.assign(truth=labels).to_csv(directory / "predictions.csv", index_label="gsm")
    rows = []
    for model in predictions:
        if model == "truth":
            continue
        result = evaluate_predictions(
            labels.to_numpy(), predictions[model].to_numpy(),
            seed=settings.seed, n_bootstrap=settings.bootstrap,
        )
        result.bootstrap.to_csv(directory / f"{model}_intervals.csv")
        write_json(directory / f"{model}_delong.json", asdict(result.delong))
        rows.append(dict(
            scenario=scenario, dataset=accession, model=model, **result.point,
            auc_ci_low=result.bootstrap.loc["auc", "lower"],
            auc_ci_high=result.bootstrap.loc["auc", "upper"],
        ))
    return rows


def save_model_record(settings, accession, result):
    directory = settings.output / accession / "models"
    directory.mkdir(parents=True, exist_ok=True)
    result.importance.to_csv(directory / "importance.csv")
    result.holdout_predictions.to_csv(directory / "holdout_predictions.csv", index_label="gsm")
    published = set(TISSUE_GENES if accession == "GSE66229" else SALIVA_GENES)
    pd.DataFrame([
        dict(gene=gene, selected=gene in result.features, published=gene in published)
        for gene in sorted(set(result.features) | published)
    ]).to_csv(directory / "signature_comparison.csv", index=False)
    write_json(directory / "selection.json", {
        "selected_genes": result.features,
        "requested_gene_count": 7 if accession == "GSE66229" else len(result.candidate_features),
        "achieved_gene_count": len(result.features),
        "published_genes": TISSUE_GENES if accession == "GSE66229" else SALIVA_GENES,
        "split": asdict(result.split), "assumptions": result.assumptions,
        "best_parameters": {
            stage: {name: search.best_params_ for name, search in searches.items()}
            for stage, searches in result.searches.items()
        },
        "cv_best_auc": {
            stage: {name: search.best_score_ for name, search in searches.items()}
            for stage, searches in result.searches.items()
        },
    })


def ml_stage(settings: Settings) -> dict:
    from gastric_repro.ml import fit_models

    results = {}
    cohorts = {accession: load_cohort(settings.output, accession) for accession in COHORTS}
    rows = []
    for accession, scenario, top_n in (("GSE66229", "A", 7), ("GSE64951", "D", None)):
        expression, labels = cohorts[accession]
        de = read_de(settings, accession)
        if accession == "GSE64951":
            features = saliva_model_features(expression, de, settings.saliva_gene_set)
            top_n = len(features)
        else:
            features = de.index[de.is_deg].tolist()
        if not features:
            raise ValueError(f"{accession}: no DEGs; cannot fit paper-like models")
        result = fit_models(
            expression, labels, features, top_n=top_n or len(features),
            feature_scope="full_cohort_de_faithful", **model_options(settings),
        )
        results[accession] = result
        save_model_record(settings, accession, result)
        predictions = result.holdout_predictions.drop(columns="truth")
        rows.extend(evaluate_scenario(settings, predictions, labels, scenario, accession))
    for accession, scenario in (("GSE54129", "B"), ("GSE13911", "B"), ("GSE64951", "C")):
        expression, labels = cohorts[accession]
        predictions = results["GSE66229"].predict_external(expression)
        rows.extend(evaluate_scenario(settings, predictions, labels, scenario, accession))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(settings.output / "computed_metrics.csv", index=False)
    compare_metrics(metrics).to_csv(settings.output / "tables_1_3_comparison.csv", index=False)
    cells_path = settings.output / "evidence" / "article_tables.csv"
    if cells_path.exists():
        intervals = []
        for row in rows:
            path = (settings.output / "validation" / row["scenario"] / row["dataset"]
                    / f"{row['model']}_intervals.csv")
            frame = pd.read_csv(path).rename(columns={
                "estimate": "computed_estimate", "lower": "computed_lower",
                "upper": "computed_upper",
            })
            for key in ("scenario", "dataset", "model"):
                frame[key] = row[key]
            intervals.append(frame)
        computed = pd.concat(intervals, ignore_index=True)
        computed.to_csv(settings.output / "computed_metric_intervals.csv", index=False)
        compare_article_metrics(
            article_metric_targets(pd.read_csv(cells_path)), computed,
        ).to_csv(settings.output / "all_published_metric_comparison.csv", index=False)
    return results


def robustness_stage(settings: Settings, fitted=None) -> None:
    """Run all five checks without mixing direct and transductive validation."""
    from sklearn.base import clone

    from gastric_repro.metrics import binary_metrics
    from gastric_repro.ml import make_split
    from gastric_repro.robustness import (
        de_holdout_run, fit_fold_local_holdout, permutation_pipeline_test,
        repeated_pipeline_runs, transductive_standardize,
    )

    directory = settings.output / "robustness"
    directory.mkdir(parents=True, exist_ok=True)
    summary, repeated_rows, null_rows = [], [], []
    for accession in ("GSE66229", "GSE64951"):
        expression, labels = load_cohort(settings.output, accession)
        # Saliva retains every fold's discovered DEG rather than truncating to seven.
        top_n = 7 if accession == "GSE66229" else len(expression)
        split = make_split(expression, labels, seed=settings.seed)
        for model in ("RF", "GBM", "DT", "SVM"):
            target = directory / accession / model
            target.mkdir(parents=True, exist_ok=True)
            local = fit_fold_local_holdout(
                expression, labels, final_models=(model,), split=split, top_n=top_n,
                selector_kwargs={"filter_rule": settings.filter_rule},
                **model_options(settings),
            )
            local.holdout_predictions.to_csv(
                target / "fold_local_predictions.csv", index_label="gsm")
            local_auc = binary_metrics(
                local.holdout_predictions.truth, local.holdout_predictions[model],
            )["auc"]
            write_json(target / "fold_local.json", {
                "auc": local_auc, "signature": local.signatures[model],
                "metadata": local.assumptions, "split": asdict(local.split),
            })
            summary.append(dict(
                check="selection_leakage", dataset=accession, model=model,
                auc=local_auc, protocol="same 30% holdout; fold-local DEG and ranking",
            ))
            runner = partial(
                de_holdout_run, model=model, top_n=top_n,
                folds=settings.folds, repeats=settings.repeats,
                candidates=settings.candidates, n_estimators=settings.estimators,
                n_jobs=settings.jobs, filter_rule=settings.filter_rule,
            )
            repeated = repeated_pipeline_runs(
                runner, expression, labels, n_runs=settings.splits, seed=settings.seed,
            )
            repeated.runs.to_csv(target / "split_runs.csv", index=False)
            repeated.runs.to_json(target / "split_runs.json", orient="records", indent=2)
            stability = repeated.signature_frequency
            published = TISSUE_GENES if accession == "GSE66229" else SALIVA_GENES
            stability = stability.reindex(stability.index.union(pd.Index(published))).fillna(0)
            stability["published"] = stability.index.isin(published)
            stability.to_csv(target / "signature_stability.csv", index_label="gene")
            write_json(target / "split_provenance.json", repeated.assumptions)
            repeated_rows.append(repeated.runs.assign(dataset=accession, model=model))
            summary.append(dict(
                check="split_variance", dataset=accession, model=model,
                auc_mean=repeated.runs.auc.mean(),
                auc_sd=repeated.runs.auc.std(),
                auc_q025=repeated.runs.auc.quantile(.025),
                auc_q975=repeated.runs.auc.quantile(.975),
                runs=settings.splits,
            ))
            if accession == "GSE64951":
                null = permutation_pipeline_test(
                    runner, expression, labels, n_permutations=settings.permutations,
                    seed=settings.seed, threshold=.87,
                )
                null.runs.to_csv(target / "permutation_runs.csv", index=False)
                null.runs.to_json(target / "permutation_runs.json", orient="records", indent=2)
                write_json(target / "permutation_provenance.json", null.assumptions)
                null_rows.append(null.runs.assign(model=model))
                summary.append(dict(
                    check="permutation_null_train_only", dataset=accession, model=model,
                    exceedances=null.exceedances, probability=null.probability,
                    empirical_p=null.empirical_p, runs=settings.permutations,
                ))
    split_runs = pd.concat(repeated_rows, ignore_index=True)
    # All four models share split seeds. Ties share one best-model vote.
    maximum = split_runs.groupby(["dataset", "run"]).auc.transform("max")
    split_runs["best"] = split_runs.auc == maximum
    ties = split_runs.groupby(["dataset", "run"]).best.transform("sum")
    split_runs["best_fraction"] = split_runs.best / ties
    split_runs.to_csv(directory / "split_auc_distributions.csv", index=False)
    split_runs.groupby(["dataset", "model"]).best_fraction.mean().rename(
        "fraction_ranked_best_ties_shared",
    ).to_csv(directory / "model_rank_frequencies.csv")
    null_runs = pd.concat(null_rows, ignore_index=True)
    null_runs.to_csv(directory / "saliva_permutation_all_models.csv", index=False)
    familywise = null_runs.groupby("run").auc.max()
    exceedances = int((familywise >= .87).sum())
    summary.append(dict(
        check="permutation_null_train_only_max_over_models", dataset="GSE64951",
        model="max_of_four", exceedances=exceedances,
        probability=exceedances / settings.permutations,
        empirical_p=(1 + exceedances) / (settings.permutations + 1),
        runs=settings.permutations,
    ))
    # Refit on transformed SOURCE TRAINING samples only, retaining the chosen
    # genes and parameters. External labels are never used in this adaptation.
    fitted = ml_stage(settings) if fitted is None else fitted
    tissue = fitted["GSE66229"]
    source, _ = load_cohort(settings.output, "GSE66229")
    source = source.loc[list(tissue.features), list(tissue.train_ids)]
    source_labels = load_cohort(settings.output, "GSE66229")[1].loc[list(tissue.train_ids)]
    shift_rows = []
    for accession in ("GSE54129", "GSE13911", "GSE64951"):
        target_expression, labels = load_cohort(settings.output, accession)
        transformed = transductive_standardize(
            source, target_expression.loc[list(tissue.features)],
        )
        predictions = pd.DataFrame(index=target_expression.columns)
        for model, pipeline in tissue.models.items():
            refit = clone(pipeline).fit(transformed.source.T, source_labels)
            predictions[model] = refit.predict_proba(transformed.target.T)[:, 1]
        shift_rows.extend(evaluate_scenario(
            settings, predictions, labels, "standardized_target", accession,
            subdir="robustness/shift",
        ))
        write_json(directory / "shift" / "standardized_target" / accession / "provenance.json",
                   transformed.assumptions)
    pd.DataFrame(shift_rows).to_csv(directory / "shift_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False)
    write_json(directory / "limitations.json", {
        "normalization": "cohort-wide RMA remains transductive; DEG/selection are fold-local",
        "paired_samples": "no reliable subject block supplied; splits/permutations assume independence",
        "permutation_statistic": "train-only-DE pipeline, threshold fixed at 0.87",
        "split_repeats": "train-only-DE outer holdouts; tuning selection not fold-local in this check",
        "shift": "target-cohort standardization is transductive, not direct validation",
        "signature_frequency": "selected genes; also records DEG sets in per-run metadata",
    })


def run_metadata(settings: Settings) -> dict:
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    sources = Path(__file__).resolve().parent
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "settings": asdict(settings),
        "versions": {
            name: importlib.metadata.version(name) for name in (
                "gastric-repro", "numpy", "scipy", "pandas", "scikit-learn",
                "xgboost", "imbalanced-learn", "gseapy", "matplotlib", "requests",
            )
        },
        "published_targets_are_not_measurements": True,
        "source_lock_sha256": sha256(lock) if lock.is_file() else None,
        "implementation_sha256": {
            path.name: sha256(path) for path in sorted(sources.glob("*.py"))
        },
        "completed_stages": [],
        "status": "running",
    }


def report_stage(settings: Settings) -> None:
    """Write a data-driven report; no numerical claim passes without a measurement."""
    output = settings.output
    published_metrics().to_csv(output / "published_metrics.csv", index=False)
    lines = [
        "# Computed reproduction report", "",
        "Source: Lopes et al., DOI 10.1038/s41598-025-96864-0.",
        "Read the repository REPORT.md for assumptions and interpretation limits.", "",
    ]
    for filename, title in (
        ("gene_counts.csv", "Gene and DEG counts"),
        ("tables_1_3_comparison.csv", "Tables 1–3 and Scenario C"),
        ("robustness/summary.csv", "Robustness"),
    ):
        path = output / filename
        lines += [f"## {title}", ""]
        if path.exists():
            lines += [f"Computed artifact: `{filename}`.", "", "```csv",
                      path.read_text().strip(), "```", ""]
        else:
            lines += ["**Not assessed:** this stage has not produced a result.", ""]
    lines += [
        "## Verdict", "",
        "Numerical proximity is not proof of methodological equivalence. Compare the",
        "signed differences and intervals, cohort manifests, annotation hashes, filter",
        "sensitivity, and leakage-controlled estimates before assigning a claim verdict.",
        "Scenario C has only an approximate published target (AUC ≈ 0.5).",
        "No tolerance-based pass/fail criterion is inferred from rounded paper values.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
