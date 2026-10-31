from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from gastric_repro.ml import (
    CorrelationFilter, SplitIDs, consensus_importance, fit_models, make_split,
    model_pipeline,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(42)
    labels = pd.Series(
        [0] * 44 + [1] * 28, index=[f"GSM{i}" for i in range(72)]
    )
    values = rng.normal(size=(11, 72))
    values[:3] += labels.to_numpy() * 2
    values[3] = values[0]
    expression = pd.DataFrame(
        values, index=[f"g{i:02}" for i in range(11)], columns=labels.index
    )
    expression.iloc[5, 0] = np.nan
    return expression, labels


def small_fit(expression, labels, **kwargs):
    options = dict(
        folds=2, repeats=1, candidates=1, n_estimators=6, top_n=4,
        selection_models=("RF", "DT"), final_models=("RF", "DT", "SVM"),
    )
    options.update(kwargs)
    return fit_models(expression, labels, list(expression.index), **options)


def test_shared_stratified_split_and_fitted_training_steps(cohort):
    expression, labels = cohort
    split = make_split(expression, labels)
    result = small_fit(expression, labels, split=split)
    assert result.split == split
    assert len(result.test_ids) == 22
    assert not set(result.train_ids) & set(result.test_ids)
    assert set(result.train_ids) | set(result.test_ids) == set(labels.index)
    assert len(result.features) == 4
    assert result.importance.index.is_unique
    assert result.holdout_predictions.index.tolist() == list(result.test_ids)
    assert result.holdout_predictions["truth"].equals(labels.loc[list(result.test_ids)])
    for pipeline in result.models.values():
        classifier = pipeline.named_steps["classifier"]
        assert classifier.classes_.tolist() == [0, 1]
        sample = pipeline.named_steps["sample"]
        train_y = labels.loc[list(result.train_ids)].to_numpy()
        counts = np.bincount(train_y[sample.sample_indices_])
        assert counts[0] == counts[1] == np.bincount(train_y).min()
        expected = np.nanmedian(
            expression.loc[list(result.features), list(result.train_ids)].T, axis=0
        )
        np.testing.assert_allclose(pipeline.named_steps["impute"].statistics_, expected)
    assert isinstance(result.models["SVM"].named_steps["scale"], StandardScaler)
    assert result.models["SVM"].named_steps["classifier"].kernel == "rbf"
    assert result.assumptions["holdout_used_for_tuning"] is False


def test_holdout_values_do_not_change_selection_or_fit(cohort):
    expression, labels = cohort
    split = make_split(expression, labels, seed=13)
    original = small_fit(expression, labels, split=split, seed=13)
    changed = expression.copy()
    changed.loc[:, list(split.test_ids)] = 100000
    altered = small_fit(changed, labels, split=split, seed=13)
    assert original.features == altered.features
    pd.testing.assert_frame_equal(original.importance, altered.importance)
    for name in original.models:
        assert (
            original.searches["final"][name].best_params_
            == altered.searches["final"][name].best_params_
        )
        train = expression.loc[:, list(split.train_ids)]
        np.testing.assert_array_equal(
            original.predict_external(train)[name], altered.predict_external(train)[name]
        )


def test_external_direct_predict_reorders_genes_and_never_fits(cohort, monkeypatch):
    expression, labels = cohort
    result = small_fit(expression, labels)
    target = expression.loc[:, list(result.test_ids)]
    expected = result.predict_external(target)
    for pipeline in result.models.values():
        monkeypatch.setattr(
            pipeline, "fit", lambda *args, **kwargs: pytest.fail("External refit")
        )
    actual = result.predict_external(target.iloc[::-1])
    pd.testing.assert_frame_equal(expected, actual)
    with pytest.raises(ValueError, match="Missing required genes"):
        result.predict_external(target.drop(index=result.features[0]))


def test_all_models_and_grid_limit(cohort):
    expression, labels = cohort
    result = fit_models(
        expression, labels, expression.index, folds=2, repeats=1, candidates=1,
        n_estimators=4, top_n=3, feature_scope="full_cohort_de_faithful",
    )
    assert set(result.models) == {"RF", "GBM", "DT", "SVM"}
    assert result.assumptions["full_cohort_de_leakage"]
    for phase in result.searches.values():
        assert all(len(search.cv_results_["params"]) == 1 for search in phase.values())
    with pytest.raises(ValueError, match="between 1 and 10"):
        small_fit(expression, labels, candidates=11)


def test_correlation_filter_is_train_fitted_and_keeps_constants():
    X = np.array([[0, 0, 2, 1], [1, -1, 2, -1], [2, -2, 2, 1]], dtype=float)
    corr = CorrelationFilter().fit(X)
    assert corr.get_support().tolist() == [True, False, True, True]
    assert corr.transform(X[::-1]).shape == (3, 3)
    assert corr.get_feature_names_out(["a", "b", "c", "d"]).tolist() == ["a", "c", "d"]
    with pytest.raises(ValueError, match="Feature count"):
        corr.transform(X[:, :2])
    with pytest.raises(ValueError, match="threshold"):
        CorrelationFilter(2).fit(X)


def test_all_missing_gene_does_not_destroy_feature_mapping(cohort):
    expression, labels = cohort
    expression.loc["g00"] = np.nan
    result = small_fit(expression, labels)
    assert len(result.importance) == len(expression)
    assert np.isfinite(result.holdout_predictions["RF"]).all()


def test_consensus_prioritizes_rf_gbm_and_dt_breaks_ties_only():
    def fitted(values):
        return SimpleNamespace(named_steps={
            "corr": SimpleNamespace(get_support=lambda: np.ones(3, dtype=bool)),
            "classifier": SimpleNamespace(feature_importances_=np.asarray(values)),
        })

    ranking = consensus_importance(
        {"RF": fitted([3, 2, 1]), "GBM": fitted([2, 3, 1]), "DT": fitted([0, 9, 1])},
        ["a", "b", "c"],
    )
    assert ranking.index.tolist() == ["b", "a", "c"]
    assert ranking.loc["a", "consensus_rank"] == ranking.loc["b", "consensus_rank"]
    assert ranking.loc["c", "consensus_rank"] > ranking.loc["a", "consensus_rank"]


def test_invalid_labels_splits_and_insufficient_folds(cohort):
    expression, labels = cohort
    with pytest.raises(ValueError, match="Labels"):
        small_fit(expression, labels.iloc[:-1])
    split = make_split(expression, labels)
    invalid = SplitIDs(split.train_ids + (split.test_ids[0],), split.test_ids)
    with pytest.raises(ValueError, match="partition"):
        small_fit(expression, labels, split=invalid)
    with pytest.raises(ValueError, match="minority class"):
        small_fit(expression, labels, folds=40)
    with pytest.raises(ValueError, match="Unknown model"):
        model_pipeline("invalid")
