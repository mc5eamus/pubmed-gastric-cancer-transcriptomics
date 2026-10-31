import pandas as pd
import pytest

from gastric_repro.published import (
    article_metric_targets, compare_article_metrics, compare_metrics, published_metrics,
)


def test_published_targets_not_measurements():
    targets = published_metrics()
    assert len(targets) == 16
    assert not targets.duplicated(["scenario", "dataset", "model"]).any()
    saliva = targets[(targets.scenario == "D") & (targets.model == "SVM")].iloc[0]
    assert saliva.auc == .868
    assert saliva.auc_ci_low == .721


def test_comparison_preserves_unrun_experiments_and_scenario_c():
    computed = pd.DataFrame([
        dict(scenario="A", dataset="GSE66229", model="RF", auc=.9),
        dict(scenario="C", dataset="GSE64951", model="RF", auc=.51),
    ])
    comparison = compare_metrics(computed)
    assert len(comparison) == 17
    rf = comparison[(comparison.scenario == "A") & (comparison.model == "RF")].iloc[0]
    assert rf.auc_difference == pytest.approx(-.088)
    assert comparison.auc_computed.isna().sum() == 15
    assert comparison.loc[comparison.scenario == "C", "auc_published"].isna().all()


def test_duplicate_experiments_rejected():
    row = dict(scenario="A", dataset="GSE66229", model="RF", auc=.9)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_metrics(pd.DataFrame([row, row]))


def test_source_error_and_missing_npv_are_not_silently_corrected():
    cells = pd.DataFrame([
        ["Tab1", 1, 1, "SVM"], ["Tab1", 1, 5, "0.848 (0.852–0.750)"],
        ["Tab2", 1, 1, "GSE13911 (Caucasian)"],
        ["Tab2", 2, 1, "SVM"], ["Tab2", 2, 7, "0.770 (0.646–0.884)"],
    ], columns=["table", "row", "column", "text"])
    targets = article_metric_targets(cells)
    npv = targets[(targets.source_table == "Tab1") & (targets.metric == "npv")].iloc[0]
    assert npv.published_lower == .852
    assert npv.published_upper == .750
    assert not npv.published_interval_valid
    external = targets[(targets.source_table == "Tab2") & (targets.metric == "npv")]
    assert external.published_estimate.isna().all()
    computed = pd.DataFrame([dict(
        scenario="B", dataset="GSE13911", model="SVM", metric="auc",
        computed_estimate=.8, computed_lower=.6, computed_upper=.9,
    )])
    comparison = compare_article_metrics(targets, computed)
    row = comparison[(comparison.source_table == "Tab2") & (comparison.metric == "auc")].iloc[0]
    assert row.estimate_difference == pytest.approx(.03)
