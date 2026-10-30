from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from gastric_repro.ml import make_split
from gastric_repro.robustness import (
    FoldLocalDESelector, NoDifferentialGenesError, PipelineRun, nested_evaluate,
    de_holdout_run, permutation_pipeline_test, repeated_pipeline_runs,
    fit_fold_local_holdout, fold_local_holdout_run, nested_pipeline_run,
    transductive_standardize,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(19)
    labels = pd.Series([0] * 24 + [1] * 24, index=[f"GSM{i}" for i in range(48)])
    values = 5 + rng.normal(0, .35, size=(10, 48))
    values[:4] += 4 * labels.to_numpy()
    expression = pd.DataFrame(
        values, index=[f"gene{i}" for i in range(10)], columns=labels.index
    )
    return expression, labels


def test_selector_clone_training_ids_and_validation_independence(cohort):
    expression, labels = cohort
    ids = list(labels.index[:16]) + list(labels.index[24:40])
    X = expression.loc[:, ids].T
    selector = FoldLocalDESelector(
        top_n=2, selection_models=("RF", "DT"), n_estimators=5
    )
    fitted = clone(selector).fit(X, labels.loc[ids])
    assert fitted.fit_sample_ids_ == tuple(ids)
    assert len(fitted.selected_features_) == 2
    assert set(fitted.selected_features_) <= set(expression.index[:4])
    validation = expression.drop(columns=ids).T.copy()
    original_signature = fitted.selected_features_
    validation.iloc[:, :] = 9999
    transformed = fitted.transform(validation)
    assert transformed.shape == (16, 2)
    assert original_signature == fitted.selected_features_
    assert fitted.get_feature_names_out().tolist() == list(original_signature)
    with pytest.raises(ValueError, match="Missing selected"):
        fitted.transform(validation.drop(columns=original_signature[0]))


def test_no_deg_is_an_explicit_failure(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    actual = de.differential_expression

    def no_degs(*args, **kwargs):
        result = actual(*args, **kwargs)
        result["is_deg"] = False
        return result

    monkeypatch.setattr(de, "differential_expression", no_degs)
    with pytest.raises(NoDifferentialGenesError, match="DEG rule"):
        FoldLocalDESelector(selection_models=("DT",)).fit(expression.T, labels)


def test_nested_cv_recomputes_de_inside_inner_and_outer_training(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    calls = []
    actual = de.differential_expression

    def record(matrix, groups):
        calls.append(set(matrix.columns))
        return actual(matrix, groups)

    monkeypatch.setattr(de, "differential_expression", record)
    result = nested_evaluate(
        expression, labels, model="DT", outer_folds=2, inner_folds=2,
        repeats=1, candidates=1, n_estimators=4,
        selector_kwargs={"selection_models": ("DT",), "top_n": 2},
    )
    assert len(calls) == 6  # Two inner fits and one refit for each outer fold.
    for fold in result.folds.itertuples():
        train, test = set(fold.train_ids), set(fold.test_ids)
        fitted = result.fitted_models[fold.fold].named_steps["select"]
        assert set(fitted.fit_sample_ids_) == train
        assert not train & test
        assert any(call < train for call in calls)
        assert train in calls
    assert result.predictions["score"].notna().all()
    assert result.auc >= .9
    assert result.assumptions["scenario"] == "nested_fold_local_de_and_selection"
    assert (result.signature_frequency["frequency"] <= 1).all()


def test_shared_holdout_fold_local_all_models_and_external_prediction(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    split = make_split(expression, labels)
    calls = []
    actual = de.differential_expression

    def record(matrix, groups):
        calls.append(set(matrix.columns))
        return actual(matrix, groups)

    monkeypatch.setattr(de, "differential_expression", record)
    result = fit_fold_local_holdout(
        expression, labels, split=split, folds=2, repeats=1, candidates=1,
        top_n=2, n_estimators=3, selector_kwargs={"selection_models": ("DT",)},
    )
    assert result.split == split
    assert set(result.models) == {"RF", "GBM", "DT", "SVM"}
    assert len(result.features) == 2
    assert result.selected_features == result.features
    assert not result.importance.empty
    assert len(calls) == 12  # 2 CV fits + refit for each of four models.
    for call in calls:
        assert call <= set(split.train_ids)
        assert not call & set(split.test_ids)
    for name, fitted in result.models.items():
        selector = fitted.named_steps["select"]
        assert selector.fit_sample_ids_ == split.train_ids
        assert result.signatures[name] == selector.selected_features_
        assert result.searches["final"][name].best_params_
    target = expression.loc[:, list(split.test_ids)].iloc[::-1]
    predicted = result.predict_external(target)
    pd.testing.assert_frame_equal(
        predicted, result.holdout_predictions.drop(columns="truth")
    )
    with pytest.raises(ValueError, match="Missing required genes"):
        result.predict_external(target.drop(index=result.features[0]))
    assert result.assumptions["cv_feature_selection_is_fold_local"]


def test_shared_holdout_fold_local_selection_ignores_holdout_values(cohort):
    expression, labels = cohort
    split = make_split(expression, labels, seed=7)
    options = dict(
        split=split, seed=7, folds=2, repeats=1, candidates=1, top_n=2,
        n_estimators=3, final_models=("DT",),
        selector_kwargs={"selection_models": ("DT",)},
    )
    original = fit_fold_local_holdout(expression, labels, **options)
    altered_expression = expression.copy()
    altered_expression.loc[:, list(split.test_ids)] = 1000000
    altered = fit_fold_local_holdout(altered_expression, labels, **options)
    assert original.signatures == altered.signatures
    pd.testing.assert_frame_equal(original.importance, altered.importance)
    assert (
        original.searches["final"]["DT"].best_params_
        == altered.searches["final"]["DT"].best_params_
    )


def test_fold_local_holdout_wrapper_catches_only_no_degs(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    actual = de.differential_expression

    def no_degs(*args):
        result = actual(*args)
        result["is_deg"] = False
        return result

    monkeypatch.setattr(de, "differential_expression", no_degs)
    pipeline = partial(
        fold_local_holdout_run, model="DT", folds=2, repeats=1,
        candidates=1, n_estimators=3,
        selector_kwargs={"selection_models": ("DT",)},
    )
    result = permutation_pipeline_test(pipeline, expression, labels, n_permutations=3)
    assert result.runs["auc"].tolist() == [.5] * 3
    assert result.empirical_p == .25
    assert (result.runs["status"] == "ok").all()
    for metadata in result.runs["metadata"]:
        assert metadata["no_deg_policy"] == "constant_full_run"

    def broken(*args):
        raise RuntimeError("Unrelated numerical error")

    monkeypatch.setattr(de, "differential_expression", broken)
    with pytest.raises(RuntimeError, match="Unrelated numerical error"):
        pipeline(expression, labels, seed=1)


def test_repeated_fresh_pipeline_and_signature_frequency(cohort):
    expression, labels = cohort
    calls = []

    def fresh(matrix, groups, *, seed):
        calls.append((seed, groups.copy()))
        # A tiny deterministic stand-in exercises the runner contract, not science.
        score = .9 if seed % 2 else .8
        return PipelineRun(score, ("gene0", f"gene{seed % 3 + 1}"))

    result = repeated_pipeline_runs(fresh, expression, labels, n_runs=8, seed=3)
    assert len(calls) == 8
    assert len({seed for seed, _ in calls}) == 8
    assert result.signature_frequency.loc["gene0", "frequency"] == 1
    assert result.probability == pytest.approx(result.exceedances / 8)
    assert np.isnan(result.empirical_p)
    again = repeated_pipeline_runs(fresh, expression, labels, n_runs=8, seed=3)
    pd.testing.assert_frame_equal(result.runs, again.runs)
    for _, groups in calls:
        pd.testing.assert_series_equal(groups, labels)


def test_permutations_preserve_counts_and_plus_one_tail(cohort):
    expression, labels = cohort
    seen = []

    def full_pipeline(matrix, groups, *, seed):
        seen.append(groups.copy())
        assert groups.value_counts().to_dict() == labels.value_counts().to_dict()
        assert groups.index.equals(labels.index)
        return PipelineRun(.87, ("gene0",))

    result = permutation_pipeline_test(
        full_pipeline, expression, labels, n_permutations=7, seed=12
    )
    assert len(seen) == 7
    assert any(not item.equals(labels) for item in seen)
    assert result.exceedances == 7
    assert result.probability == result.empirical_p == 1
    low = permutation_pipeline_test(
        lambda *args, **kwargs: PipelineRun(.5, ()),
        expression, labels, n_permutations=7,
    )
    assert low.empirical_p == 1 / 8
    assert low.signature_frequency.empty


def test_failed_permutations_do_not_create_favorable_denominator(cohort):
    expression, labels = cohort

    def failure(*args, **kwargs):
        raise NoDifferentialGenesError("No genes")

    with pytest.raises(NoDifferentialGenesError):
        permutation_pipeline_test(failure, expression, labels, n_permutations=2)
    recorded = permutation_pipeline_test(
        failure, expression, labels, n_permutations=2, on_error="record"
    )
    assert recorded.assumptions["failed_runs"] == 2
    assert np.isnan(recorded.probability)
    assert np.isnan(recorded.empirical_p)
    assert recorded.runs["error"].str.contains("NoDifferentialGenesError").all()


def test_ready_full_pipeline_repeats_and_permutations_refit_de(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    seen = []
    actual = de.differential_expression

    def record(matrix, groups):
        seen.append((set(matrix.columns), groups.copy()))
        return actual(matrix, groups)

    monkeypatch.setattr(de, "differential_expression", record)
    pipeline = partial(
        de_holdout_run, model="DT", folds=2, repeats=1, candidates=1,
        n_estimators=3, top_n=2, selection_models=("DT",),
    )
    repeated = repeated_pipeline_runs(pipeline, expression, labels, n_runs=2)
    assert len(seen) == 2
    for metadata in repeated.runs["metadata"]:
        assert metadata["scenario"] == "train_only_de_holdout"
        assert metadata["predictor"] == "tuned_model"
        assert metadata["deg_count"] == len(metadata["deg_genes"])
        assert set(metadata["train_ids"]) in [item[0] for item in seen]
        assert not set(metadata["train_ids"]) & set(metadata["test_ids"])
    permuted = permutation_pipeline_test(
        pipeline, expression, labels, n_permutations=3, seed=10
    )
    assert len(seen) == 5
    assert permuted.assumptions["completed_runs"] == 3
    assert not np.isnan(permuted.empirical_p)
    assert any(not groups.equals(labels.loc[groups.index]) for _, groups in seen[2:])


def test_ready_pipeline_no_degs_has_explicit_constant_policy(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    actual = de.differential_expression

    def no_degs(*args):
        result = actual(*args)
        result["is_deg"] = False
        return result

    monkeypatch.setattr(de, "differential_expression", no_degs)
    outcome = de_holdout_run(expression, labels)
    assert outcome.auc == .5
    assert outcome.signature == ()
    assert outcome.metadata["predictor"] == "intercept_only"
    assert outcome.metadata["no_deg_policy"] == "constant"
    assert outcome.metadata["deg_genes"] == ()
    with pytest.raises(NoDifferentialGenesError):
        de_holdout_run(expression, labels, no_deg_policy="raise")


def test_nested_callable_defines_no_deg_baseline_for_every_permutation(cohort, monkeypatch):
    from gastric_repro import de

    expression, labels = cohort
    actual = de.differential_expression

    def no_degs(*args):
        result = actual(*args)
        result["is_deg"] = False
        return result

    monkeypatch.setattr(de, "differential_expression", no_degs)
    pipeline = partial(
        nested_pipeline_run, model="DT", outer_folds=2, inner_folds=2,
        repeats=1, candidates=1, n_estimators=3,
        selector_kwargs={"selection_models": ("DT",), "top_n": 2},
    )
    result = permutation_pipeline_test(pipeline, expression, labels, n_permutations=3)
    assert result.runs["auc"].tolist() == [.5] * 3
    assert result.runs["signature"].tolist() == [()] * 3
    assert result.empirical_p == .25
    assert result.assumptions["completed_runs"] == 3
    for metadata in result.runs["metadata"]:
        assert metadata["no_deg_policy"] == "constant"
        assert metadata["fully_inductive_end_to_end"] is False
        assert "transductive" in metadata["upstream_normalization"]


def test_transductive_scenario_is_separate_and_explicit(cohort):
    expression, _ = cohort
    expression.loc["gene9"] = 1
    target = expression.iloc[::-1, :10] * 3 + 40
    result = transductive_standardize(expression, target)
    np.testing.assert_allclose(result.source.mean(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(result.target.mean(axis=1), 0, atol=1e-13)
    assert (result.target.loc["gene9"] == 0).all()
    assert result.assumptions["target_cohort_statistics_used"]
    assert not result.assumptions["direct_external_validation"]
    assert not result.assumptions["no_leakage_claim"]
    assert result.target.index.equals(expression.index)
    with pytest.raises(ValueError, match="same genes"):
        transductive_standardize(expression, target.iloc[:-1])
