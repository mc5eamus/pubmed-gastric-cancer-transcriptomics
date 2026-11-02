"""Reproducible, explicit CLI stages; imports do not download data or fit models."""

import argparse
from pathlib import Path
import sys

from gastric_repro.pipeline import (
    Settings, de_stage, enrichment_stage, ml_stage, report_stage,
    robustness_stage, run_metadata, write_json,
)
from gastric_repro.published import COHORTS
from gastric_repro.resources import publication_evidence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent Lopes et al. (2025) reproduction on Python 3.14.",
    )
    result.add_argument("stage", choices=[
        "run", "evidence", "download", "normalise", "de", "enrichment",
        "ml", "validation", "robustness", "report",
    ])
    result.add_argument("--cache", type=Path, default=Path.cwd() / "data")
    result.add_argument("--output", type=Path, default=Path.cwd() / "results")
    result.add_argument("--dataset", choices=list(COHORTS), action="append",
                        help="Download/normalise subset; repeat this option for multiple cohorts")
    result.add_argument("--metadata-only", action="store_true",
                        help="Download metadata and manifests, not multi-GB raw CELs")
    result.add_argument("--cdf", type=Path, help="Local original HG-U133_Plus_2 ASCII/XDA CDF")
    result.add_argument("--allow-series-matrix", action="store_true",
                        help="Permit documented non-RMA processed-data fallback if raw processing fails")
    result.add_argument("--seed", type=int, default=2025)
    result.add_argument("--filter-rule", choices=[
        "any_above_median", "mean_above_median", "half_above_median",
    ], default="any_above_median")
    result.add_argument("--folds", type=int, default=10)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--candidates", type=int, default=10)
    result.add_argument("--bootstrap", type=int, default=2000)
    result.add_argument("--splits", type=int, default=100)
    result.add_argument("--permutations", type=int, default=100)
    result.add_argument("--gsea-permutations", type=int, default=1000)
    result.add_argument("--estimators", type=int, default=200)
    result.add_argument("--jobs", type=int, default=1)
    result.add_argument("--saliva-gene-set", choices=("discovered", "published"),
                        default="discovered",
                        help="Saliva model candidates: discovered DEGs or the published 14-gene panel")
    for ontology in ("bp", "cc", "mf"):
        result.add_argument(f"--go-{ontology}", type=Path,
                            help="Local GMT snapshot; supply all three or use cached GO 2023 downloads")
    return result


def dispatch(stage: str, settings: Settings, args) -> None:
    datasets = args.dataset or list(COHORTS)
    if stage == "evidence":
        publication_evidence(settings.cache, settings.output / "evidence")
    elif stage == "download":
        from gastric_repro.data import download_dataset
        for accession in datasets:
            download_dataset(
                accession, settings.cache, settings.output / accession,
                metadata_only=args.metadata_only,
            )
    elif stage == "normalise":
        from gastric_repro.data import prepare_dataset
        for accession in datasets:
            prepare_dataset(
                accession, settings.cache, settings.output / accession,
                cdf=settings.cdf, allow_series_matrix=settings.allow_series_matrix,
            )
    elif stage == "de":
        de_stage(settings)
    elif stage == "enrichment":
        paths = {name: getattr(args, f"go_{name.lower()}") for name in ("BP", "CC", "MF")}
        enrichment_stage(
            settings, None if all(value is None for value in paths.values()) else paths,
        )
    elif stage in ("ml", "validation"):
        # Validation fits source-training models deterministically, never target models.
        ml_stage(settings)
    elif stage == "robustness":
        robustness_stage(settings)
    elif stage == "report":
        report_stage(settings)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    paths = [getattr(args, f"go_{name}") for name in ("bp", "cc", "mf")]
    if any(path is not None for path in paths) and not all(path is not None for path in paths):
        argument_parser.error("--go-bp, --go-cc, and --go-mf must be supplied together")
    if args.stage == "run" and (args.dataset or args.metadata_only):
        argument_parser.error("run requires all cohorts and full data; use download for subsets/metadata")
    for name in ("cache", "output", "cdf", "go_bp", "go_cc", "go_mf"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    try:
        settings = Settings(**{
            name: getattr(args, name) for name in Settings.__dataclass_fields__
        })
    except ValueError as error:
        argument_parser.error(str(error))
    settings.output.mkdir(parents=True, exist_ok=True)
    record = run_metadata(settings)
    log = settings.output / f"{args.stage}_run.json"
    write_json(log, record)
    try:
        if args.stage == "run":
            # Keep source models in memory; no unsafe model pickle loading.
            for stage in ("evidence", "normalise", "de", "enrichment"):
                print(f"Starting {stage}", flush=True)
                dispatch(stage, settings, args)
                record["completed_stages"].append(stage)
                write_json(log, record)
            print("Starting ml (scenarios A–D)", flush=True)
            fitted = ml_stage(settings)
            record["completed_stages"].append("ml")
            write_json(log, record)
            print("Starting robustness", flush=True)
            robustness_stage(settings, fitted=fitted)
            record["completed_stages"].append("robustness")
            report_stage(settings)
            record["completed_stages"].append("report")
        else:
            dispatch(args.stage, settings, args)
            record["completed_stages"].append(args.stage)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(log, record)
        print(f"{args.stage} failed: {error}", file=sys.stderr)
        return 1
    record["status"] = "completed"
    write_json(log, record)
    print(f"Completed {args.stage}; artifacts: {settings.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
