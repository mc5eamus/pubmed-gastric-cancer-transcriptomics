import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from gastric_repro.de import (
    DEFAULT_FILTER_RULE,
    benjamini_hochberg,
    differential_expression,
    filter_expression,
    filter_sensitivity_counts,
    plot_heatmap,
    plot_saliva_deg_volcano,
    plot_volcano,
    squeeze_var,
)


@pytest.fixture
def expression():
    return pd.DataFrame(
        [[1, 2, 1, 5, 6, 5], [5, 6, 5, 1, 2, 1], [3, 4, 3, 3, 4, 3]],
        index=["up", "down", "null"],
        columns=list("abcdef"),
        dtype=float,
    )


@pytest.fixture
def labels(expression):
    return pd.Series([0, 0, 0, 1, 1, 1], index=expression.columns)


def test_ols_contrast_and_label_alignment(expression, labels):
    actual = differential_expression(expression, labels.iloc[::-1])
    assert actual.index.equals(expression.index)
    np.testing.assert_allclose(actual.log2FC, [4, -4, 0])
    np.testing.assert_allclose(actual.var_residual, [1 / 3] * 3)
    np.testing.assert_allclose(actual.mean_group1 - actual.mean_group0, actual.log2FC)
    np.testing.assert_allclose(actual.t, actual.log2FC / actual.standard_error)
    np.testing.assert_allclose(actual.pvalue, 2 * stats.t.sf(abs(actual.t), actual.df_total))
    assert actual.loc["null", "pvalue"] == 1
    assert actual.loc["up", "direction"] == "up"
    assert actual.loc["down", "direction"] == "down"
    assert actual.loc["null", "direction"] == "not_deg"
    assert (actual.df_total == 12).all()


def test_squeeze_var_matches_corrected_log_moments():
    df, df_prior, var_prior = 8.0, 10.0, 2.5
    target_variance = special.polygamma(1, df / 2) + special.polygamma(1, df_prior / 2)
    logmean = np.log(var_prior) - special.digamma(df_prior / 2) + np.log(df_prior / 2)
    center = logmean + special.digamma(df / 2) - np.log(df / 2)
    spread = np.sqrt(target_variance)
    variances = np.exp(center + np.array([-spread, 0, spread]))
    result = squeeze_var(variances, df)
    assert result["df_prior"] == pytest.approx(df_prior)
    assert result["var_prior"] == pytest.approx(var_prior)
    np.testing.assert_allclose(
        result["var_post"], (df * variances + df_prior * var_prior) / (df + df_prior)
    )


@pytest.mark.parametrize("variances", [[0, 0, 0], [1, 1, 1], [0, 1, 100], [3]])
def test_degenerate_priors_remain_positive(variances):
    result = squeeze_var(variances, 4)
    assert result["var_prior"] > 0
    assert result["df_prior"] > 0
    assert np.isfinite(result["var_post"]).all()
    assert (result["var_post"] > 0).all()


def test_constant_genes_and_single_gene(labels):
    expression = pd.DataFrame(
        [[2] * 6, [1, 1, 1, 3, 3, 3]], index=["constant", "shift"], columns=labels.index
    )
    result = differential_expression(expression, labels)
    assert np.isfinite(result[["t", "pvalue", "padj", "var_post"]]).all().all()
    assert result.loc["constant", "pvalue"] == 1
    assert result.loc["constant", "t"] == 0
    assert result.loc["shift", "is_deg"]
    assert differential_expression(expression.iloc[:1], labels).iloc[0].pvalue == 1


def test_de_strict_fold_change_boundary(labels):
    expression = pd.DataFrame(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1.01, 1.01, 1.01]],
        index=["boundary", "above"], columns=labels.index,
    )
    result = differential_expression(expression, labels)
    assert (result.padj < 0.05).all()
    assert not result.loc["boundary", "is_deg"]
    assert result.loc["above", "is_deg"]


def test_de_strict_adjusted_p_boundary(expression, labels, monkeypatch):
    monkeypatch.setattr(
        "gastric_repro.de.benjamini_hochberg", lambda p: np.full_like(p, 0.05)
    )
    assert not differential_expression(expression, labels).is_deg.any()


def test_label_reversal_flips_contrast_not_pvalues(expression, labels):
    forward = differential_expression(expression, labels)
    reverse = differential_expression(expression, 1 - labels)
    np.testing.assert_allclose(forward.log2FC, -reverse.log2FC)
    np.testing.assert_allclose(forward.t, -reverse.t)
    np.testing.assert_allclose(forward.pvalue, reverse.pvalue)
    np.testing.assert_allclose(forward.padj, reverse.padj)


def test_unbalanced_three_sample_design(expression, labels):
    columns = ["a", "b", "d"]
    actual = differential_expression(expression[columns], labels.loc[columns])
    assert (actual.df_residual == 1).all()
    np.testing.assert_allclose(actual.log2FC, [3.5, -4.5, -0.5])
    assert np.isfinite(actual.t).all()


def test_bh_reference_and_ties():
    np.testing.assert_allclose(
        benjamini_hochberg([0.01, 0.04, 0.03, 0.002, 1]),
        [0.025, 0.05, 0.05, 0.01, 1],
    )
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.01, 0.9]), [0.015, 0.015, 0.9])
    assert benjamini_hochberg([]).size == 0


@pytest.mark.parametrize("pvalues", [[np.nan], [-0.1], [1.01], [[0.1]]])
def test_bh_rejects_invalid(pvalues):
    with pytest.raises(ValueError):
        benjamini_hochberg(pvalues)


def test_declared_filters_and_sensitivity_counts():
    expression = pd.DataFrame(
        [[0, 0, 0, 10], [0, 0, 0, 0], [3, 3, 3, 3], [2, 2, 2, 2]],
        index=["spike", "low", "high", "at_median"],
    )
    assert DEFAULT_FILTER_RULE == "any_above_median"
    assert filter_expression(expression).index.tolist() == ["spike", "high"]
    assert filter_expression(expression, "mean_above_median").index.tolist() == ["spike", "high"]
    assert filter_expression(expression, "half_above_median").index.tolist() == ["high"]
    counts = filter_sensitivity_counts(expression).set_index("rule")
    assert counts.retained_genes.tolist() == [2, 2, 1]
    assert (counts.global_median == 2).all()
    assert (counts.input_genes == counts.retained_genes + counts.removed_genes).all()
    assert filter_expression(pd.DataFrame(np.ones((3, 4)))).empty
    with pytest.raises(ValueError, match="unknown rule"):
        filter_expression(expression, "tuned")


@pytest.mark.parametrize("mutation", ["nan", "inf", "duplicate_genes", "duplicate_samples", "empty"])
def test_expression_validation(expression, labels, mutation):
    if mutation == "nan":
        expression.iloc[0, 0] = np.nan
    elif mutation == "inf":
        expression.iloc[0, 0] = np.inf
    elif mutation == "duplicate_genes":
        expression.index = ["x"] * len(expression)
    elif mutation == "duplicate_samples":
        expression.columns = ["x"] * expression.shape[1]
    else:
        expression = expression.iloc[:0]
    with pytest.raises(ValueError):
        differential_expression(expression, labels)
    with pytest.raises(ValueError):
        filter_expression(expression)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "nan", "invalid", "one_group"])
def test_label_validation(expression, labels, mutation):
    if mutation == "missing":
        labels = labels.iloc[:-1]
    elif mutation == "extra":
        labels.loc["extra"] = 0
    elif mutation == "duplicate":
        labels.index = ["a"] * len(labels)
    elif mutation == "nan":
        labels.iloc[0] = np.nan
    elif mutation == "invalid":
        labels.iloc[0] = 2
    else:
        labels[:] = 0
    with pytest.raises(ValueError):
        differential_expression(expression, labels)


def test_residual_df_required(expression, labels):
    with pytest.raises(ValueError, match="degrees of freedom"):
        differential_expression(expression.iloc[:, [0, 3]], labels.iloc[[0, 3]])


@pytest.mark.parametrize("variance,df", [([-1], 2), ([np.nan], 2), ([], 2), ([1], 0), ([1], np.inf)])
def test_squeeze_var_validation(variance, df):
    with pytest.raises(ValueError):
        squeeze_var(variance, df)


def test_headless_plots(expression, labels, tmp_path):
    results = differential_expression(expression, labels)
    volcano = plot_volcano(results, tmp_path / "plots" / "volcano.png")
    heatmap = plot_heatmap(expression, tmp_path / "heatmap.png", results=results)
    for path in (volcano, heatmap):
        assert path.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="nonempty"):
        plot_heatmap(expression, tmp_path / "empty.png", genes=[])
    with pytest.raises(ValueError, match="nonempty"):
        plot_heatmap(expression, tmp_path / "nodeg.png", results=results.assign(is_deg=False))


def test_saliva_volcano_labels_published_and_additional_degs(tmp_path):
    results = pd.DataFrame(
        {"log2FC": [-2.0, -1.5, 0.1], "padj": [0.01, 0.02, 0.5],
         "is_deg": [True, True, False]},
        index=["PUBLISHED", "ADDITIONAL", "BACKGROUND"],
    )
    path = plot_saliva_deg_volcano(
        results, tmp_path / "saliva.png", published_genes=["PUBLISHED"],
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def test_saliva_volcano_rejects_duplicate_genes(tmp_path):
    results = pd.DataFrame(
        {"log2FC": [1.0, 1.0], "padj": [0.01, 0.01], "is_deg": [True, True]},
        index=["DUPLICATE", "DUPLICATE"],
    )
    with pytest.raises(ValueError, match="unique gene IDs"):
        plot_saliva_deg_volcano(results, tmp_path / "saliva.png", published_genes=[])
