"""Published targets, not computed results (DOI: 10.1038/s41598-025-96864-0)."""

import re

import pandas as pd


COHORTS = {
    "GSE66229": {1: 300, 0: 100},
    "GSE64951": {1: 63, 0: 31},
    "GSE54129": {1: 111, 0: 21},
    "GSE13911": {1: 38, 0: 31},
}
TISSUE_GENES = ("CDH3", "DKC1", "ESM1", "MUSTN1", "RCC2", "TMT1A", "VEGFD")
SALIVA_GENES = (
    "ANKRD37", "BAG3", "DSP", "FRMD4B", "JMJD1C", "NAA20", "RND3",
    "SDR16C5", "SEPTIN10", "TMPRSS11B", "TUFT1", "ZBED2", "ZFAND2A", "ZHX1",
)


def published_go_targets() -> pd.DataFrame:
    return pd.DataFrame([
        ("GSE66229", "BP", "GO:0007049", "cell cycle", 4.54, 8.35e-28),
        ("GSE66229", "BP", "GO:1903047", "mitotic cell cycle process", 4.54, 1.79e-23),
        ("GSE64951", "CC", "GO:0016020", "membrane", -2.05, .0029),
        ("GSE64951", "CC", "GO:0071944", "cell periphery", -2.05, .0029),
        ("GSE64951", "MF", "GO:0016787", "hydrolase activity", -2.05, .0125),
    ], columns=["dataset", "ontology", "go_id", "published_term",
                "published_nes", "published_padj"])


def published_metrics() -> pd.DataFrame:
    rows = [
        ("A", "GSE66229", "RF", .988, .944, .967, .950),
        ("A", "GSE66229", "SVM", .982, .944, .933, .942),
        ("A", "GSE66229", "GBM", .984, .922, .967, .933),
        ("A", "GSE66229", "DT", .929, .833, .967, .867),
        ("B", "GSE54129", "RF", .843, .405, .905, .485),
        ("B", "GSE54129", "SVM", .857, .910, .762, .886),
        ("B", "GSE54129", "GBM", .856, .451, .905, .523),
        ("B", "GSE54129", "DT", .802, .090, 1., .235),
        ("B", "GSE13911", "RF", .913, .737, .968, .841),
        ("B", "GSE13911", "SVM", .770, 1., 0., .551),
        ("B", "GSE13911", "GBM", .882, .711, .936, .812),
        ("B", "GSE13911", "DT", .526, .053, 1., .478),
        ("D", "GSE64951", "RF", .842, .684, .800, .724),
        ("D", "GSE64951", "SVM", .868, .790, .700, .759),
        ("D", "GSE64951", "GBM", .821, .579, .800, .655),
        ("D", "GSE64951", "DT", .713, .526, .900, .655),
    ]
    frame = pd.DataFrame(rows, columns=[
        "scenario", "dataset", "model", "auc", "sensitivity", "specificity", "accuracy",
    ])
    # These are the only LR/CI targets transcribed here; absent values stay absent.
    for column in ("lr_positive", "lr_negative", "auc_ci_low", "auc_ci_high"):
        frame[column] = float("nan")
    frame.loc[(frame.scenario == "A") & (frame.model == "RF"),
              ["lr_positive", "lr_negative"]] = [28.36, .06]
    frame.loc[(frame.scenario == "D") & (frame.model == "SVM"),
              ["lr_positive", "lr_negative", "auc_ci_low", "auc_ci_high"]] = [
                  2.63, .30, .721, .974,
              ]
    return frame


def compare_metrics(computed: pd.DataFrame) -> pd.DataFrame:
    """Outer join preserves missing experiments rather than inventing measurements."""
    keys = ["scenario", "dataset", "model"]
    if computed.duplicated(keys).any():
        raise ValueError("Duplicate scenario/dataset/model measurements")
    merged = published_metrics().merge(
        computed, on=keys, how="outer", suffixes=("_published", "_computed"),
        indicator=True,
    )
    for metric in ("auc", "sensitivity", "specificity", "accuracy",
                   "lr_positive", "lr_negative", "auc_ci_low", "auc_ci_high"):
        if f"{metric}_computed" in merged:
            merged[f"{metric}_difference"] = (
                merged[f"{metric}_computed"] - merged[f"{metric}_published"]
            )
    return merged


def article_metric_targets(cells: pd.DataFrame) -> pd.DataFrame:
    """Parse the source's full Tables 1–4; preserve apparent source errors."""
    records = []
    table_map = {
        "Tab1": ("A", "GSE66229"), "Tab2": ("B", None),
        "Tab3": ("D", "GSE64951"), "Tab4": (None, None),
    }
    metrics = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "auc"]
    for table_id, (scenario, dataset) in table_map.items():
        table = cells.loc[cells.table == table_id].pivot(
            index="row", columns="column", values="text")
        for _, row in table.iterrows():
            values = row.dropna().astype(str).tolist()
            if not values:
                continue
            if table_id == "Tab2" and values[0].startswith("GSE"):
                dataset = values[0].split()[0]
                continue
            if table_id == "Tab4":
                if len(values) < 5 or not values[0].startswith(("Tissue", "Saliva")):
                    continue
                dataset = re.search(r"GSE\d+", values[0]).group()
                scenario = "D" if dataset == "GSE64951" else (
                    "A" if dataset == "GSE66229" else "B")
                model, raw_metrics = values[1].removeprefix("Saliva "), values[2:]
                names = ["sensitivity", "specificity", "auc"]
            else:
                if values[0] not in {"RF", "GBM", "DT", "SVM"}:
                    continue
                model = values[0]
                # Missing NPV is meaningful: retain the original table columns.
                raw_metrics = [row.get(column, "") for column in range(2, 8)]
                names = metrics
            for name, text in zip(names, raw_metrics, strict=True):
                text = "" if pd.isna(text) else str(text)
                numbers = [float(x) for x in re.findall(r"\d+\.\d+", text)]
                estimate, lower, upper = numbers if len(numbers) == 3 else (
                    float("nan"), float("nan"), float("nan"))
                records.append({
                    "source_table": table_id, "scenario": scenario, "dataset": dataset,
                    "model": model, "metric": name, "published_estimate": estimate,
                    "published_lower": lower, "published_upper": upper,
                    "published_interval_valid": lower <= estimate <= upper,
                    "published_text": text,
                })
    return pd.DataFrame(records)


def compare_article_metrics(targets: pd.DataFrame, computed: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "dataset", "model", "metric"]
    if computed.duplicated(keys).any():
        raise ValueError("Duplicate computed metric intervals")
    merged = targets.merge(computed, on=keys, how="outer", indicator=True)
    for part in ("estimate", "lower", "upper"):
        merged[f"{part}_difference"] = merged[f"computed_{part}"] - merged[f"published_{part}"]
    return merged
