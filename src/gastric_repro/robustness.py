"""Fold-local selection and explicitly separate robustness scenarios.

No experiment is run at import time. Large repeated/permutation experiments
require an explicitly supplied fresh full-pipeline callable; results are never
invented or reused from a previously selected signature.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .ml import (
    _candidate_grid,
    _check_split,
    _validate_data,
    consensus_importance,
    fit_models,
    make_split,
    model_pipeline,
)


class NoDifferentialGenesError(ValueError):
    """The predeclared fold-local DEG rule retained no genes."""


class FoldLocalDESelector(TransformerMixin, BaseEstimator):
    """Fit filtering, DE and consensus tree ranking on this fit's training rows.

    Place this transformer *inside* the estimator passed to GridSearchCV.
    Input is a samples x genes DataFrame. Median imputation for DE is also
    training-fitted. DE uses the shared moderated two-group implementation.
    No-DEG folds fail by default. An explicit ``no_deg_policy="constant"``
    instead emits a zero-valued intercept feature and an empty gene signature,
    giving the downstream classifier constant predictions. There is no covert
    p-value relaxation. Fixed ranking trees are not separately tuned.
    """

    def __init__(
        self, top_n=7, seed=2025, filter_rule="any_above_median",
        selection_models=("RF", "GBM", "DT"), n_estimators=200,
        corr_threshold=0.85, no_deg_policy="raise",
    ):
        self.top_n = top_n
        self.seed = seed
        self.filter_rule = filter_rule
        self.selection_models = selection_models
        self.n_estimators = n_estimators
        self.corr_threshold = corr_threshold
        self.no_deg_policy = no_deg_policy

    def fit(self, X, y):
        from .de import differential_expression, filter_expression

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldLocalDESelector requires a samples x genes DataFrame")
        if not X.index.is_unique or not X.columns.is_unique or self.top_n < 1:
            raise ValueError("Require unique identifiers and top_n >= 1")
        if not self.selection_models or set(self.selection_models) - {"RF", "GBM", "DT"}:
            raise ValueError("Ranking models must be RF, GBM or DT")
        if self.no_deg_policy not in {"raise", "constant"}:
            raise ValueError("no_deg_policy must be 'raise' or 'constant'")
        if isinstance(y, pd.Series):
            if not y.index.is_unique or set(y.index) != set(X.index):
                raise ValueError("Selector labels must exactly match training samples")
            labels = y.reindex(X.index)
        else:
            labels = pd.Series(np.asarray(y), index=X.index)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.fit_sample_ids_ = tuple(X.index)
        self.de_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        complete = pd.DataFrame(
            self.de_imputer_.fit_transform(X), index=X.index, columns=X.columns
        )
        filtered = filter_expression(complete.T, rule=self.filter_rule)
        self.filtered_features_ = tuple(filtered.index)
        if filtered.empty:
            return self._no_genes("Training expression filter retained no genes")
        self.de_results_ = differential_expression(filtered, labels)
        degs = tuple(sorted(self.de_results_.index[self.de_results_["is_deg"]]))
        if not degs:
            return self._no_genes("Training DEG rule retained no genes")
        self.deg_features_ = degs
        fitted = {}
        for name in self.selection_models:
            fitted[name] = model_pipeline(
                name, seed=self.seed, n_estimators=self.n_estimators,
                corr_threshold=self.corr_threshold,
            ).fit(X.loc[:, list(degs)], labels)
        self.importance_ = consensus_importance(fitted, degs)
        self.selected_features_ = tuple(self.importance_.index[: self.top_n])
        return self

    def _no_genes(self, message):
        if self.no_deg_policy == "raise":
            raise NoDifferentialGenesError(message)
        self.deg_features_ = ()
        self.selected_features_ = ()
        self.importance_ = pd.DataFrame()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldLocalDESelector requires a DataFrame")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing selected genes: {sorted(missing)}")
        if not self.selected_features_:
            return pd.DataFrame({"__no_deg_intercept__": np.zeros(len(X))}, index=X.index)
        return X.loc[:, list(self.selected_features_)].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.selected_features_ or ("__no_deg_intercept__",), dtype=object
        )


def _frequency(signatures, n_runs):
    counts = Counter(gene for signature in signatures for gene in set(signature))
    result = pd.DataFrame(
        [
            {"gene": gene, "count": count, "frequency": count / n_runs}
            for gene, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        columns=["gene", "count", "frequency"],
    )
    return result.set_index("gene")


@dataclass
class FoldLocalHoldoutResult:
    """Fitted per-model selectors and predictions on one shared holdout."""

    models: dict
    split: object
    signatures: dict
    holdout_predictions: pd.DataFrame
    searches: dict = field(repr=False)
    importances: dict = field(repr=False)
    assumptions: dict = field(default_factory=dict)

    @property
    def train_ids(self):
        return self.split.train_ids

    @property
    def test_ids(self):
        return self.split.test_ids

    @property
    def features(self):
        signatures = list(self.signatures.values())
        if all(signature == signatures[0] for signature in signatures):
            return signatures[0]
        return tuple(sorted({gene for signature in signatures for gene in signature}))

    @property
    def selected_features(self):
        return self.features

    @property
    def importance(self):
        return next(iter(self.importances.values()))

    def predict_external(self, expression):
        """Predict directly using fitted fold-local selectors; no external refit."""
        if not isinstance(expression, pd.DataFrame):
            raise TypeError("External expression must be a genes x samples DataFrame")
        if not expression.index.is_unique or not expression.columns.is_unique:
            raise ValueError("External gene and sample identifiers must be unique")
        if np.isinf(expression.to_numpy(dtype=float)).any():
            raise ValueError("External expression cannot contain infinity")
        scores = {}
        for name, fitted in self.models.items():
            missing = set(self.signatures[name]) - set(expression.index)
            if missing:
                raise ValueError(f"Missing required genes: {sorted(missing)}")
            scores[name] = (
                fitted.predict_proba(expression.T)[:, 1]
                if self.signatures[name] else np.full(expression.shape[1], 0.5)
            )
        return pd.DataFrame(scores, index=expression.columns)


def fit_fold_local_holdout(
    expression, labels, *, split=None, seed=2025, folds=10, repeats=5,
    candidates=10, top_n=7, n_estimators=200, n_jobs=1,
    final_models=("RF", "GBM", "DT", "SVM"), selector_kwargs=None,
    scale=False, scale_svm=True, no_deg_policy="constant",
):
    """Fold-local DE/selection/tuning on the same 30% holdout as fit_models.

    Pass the faithful branch's ``split`` for an apples-to-apples comparison.
    The selector is INSIDE every classifier's GridSearchCV pipeline, so its
    imputation, median expression filter, DE and fixed-tree consensus ranking
    are refitted using only each CV-training fold. After tuning the complete
    pipeline is refitted on the full training partition, never on the holdout.

    Ranking trees use fixed parameters (unlike the faithful branch's separately
    tuned ranking trees). All four final classifiers are tuned by default.
    The predeclared constant no-DEG policy keeps every fold defined; "raise"
    instead propagates NoDifferentialGenesError. Upstream full-cohort RMA is
    still transductive: fold-local DE alone is not fully inductive evaluation.
    """
    y = _validate_data(expression, labels)
    if not final_models or set(final_models) - {"RF", "GBM", "DT", "SVM"}:
        raise ValueError("Final models must be RF, GBM, DT or SVM")
    if not isinstance(candidates, int) or not 1 <= candidates <= 10:
        raise ValueError("candidates must be an integer between 1 and 10")
    if folds < 2 or repeats < 1 or top_n < 1 or n_estimators < 1:
        raise ValueError("Require folds >= 2 and positive repeats, top_n and estimators")
    split = make_split(expression, labels, seed=seed) if split is None else split
    _check_split(split, expression.columns, y)
    train_ids, test_ids = list(split.train_ids), list(split.test_ids)
    X_train, y_train = expression.loc[:, train_ids].T, y.loc[train_ids]
    if y_train.value_counts().min() < folds:
        raise ValueError("Training minority class is smaller than the CV fold count")
    options = dict(selector_kwargs or {})
    options.setdefault("seed", seed)
    options.setdefault("top_n", top_n)
    options.setdefault("n_estimators", n_estimators)
    options.setdefault("no_deg_policy", no_deg_policy)
    models, searches, signatures, importances = {}, {}, {}, {}
    predictions = pd.DataFrame({"truth": y.loc[test_ids]})
    for model in final_models:
        estimator = Pipeline([
            ("select", FoldLocalDESelector(**options)),
            ("model", model_pipeline(
                model, seed=seed, n_estimators=n_estimators,
                scale=scale_svm if model == "SVM" else scale,
            )),
        ])
        grid = [
            {f"model__{key}": values for key, values in candidate.items()}
            for candidate in _candidate_grid(model, candidates, seed)
        ]
        search = GridSearchCV(
            estimator, grid, scoring="roc_auc", refit=True, error_score="raise",
            cv=RepeatedStratifiedKFold(
                n_splits=folds, n_repeats=repeats, random_state=seed
            ),
            n_jobs=n_jobs,
        ).fit(X_train, y_train)
        fitted = search.best_estimator_
        selector = fitted.named_steps["select"]
        models[model], searches[model] = fitted, search
        signatures[model] = selector.selected_features_
        importances[model] = selector.importance_
        predictions[model] = (
            fitted.predict_proba(expression.loc[:, test_ids].T)[:, 1]
            if signatures[model] else np.full(len(test_ids), 0.5)
        )
    return FoldLocalHoldoutResult(
        models, split, signatures, predictions, {"final": searches}, importances,
        {
            "scenario": "shared_holdout_fold_local_de_selection",
            "holdout_used_for_tuning": False,
            "cv_feature_selection_is_fold_local": True,
            "ranking_estimators": "fixed parameters, not separately tuned",
            "no_deg_policy": options["no_deg_policy"],
            "consensus": "RF/GBM primary; DT tie-break; gene ID last",
            "svm_scaling": bool(scale_svm), "svm_kernel": "rbf",
            "upstream_normalization": "provided; full-cohort RMA is transductive",
            "fully_inductive_end_to_end": False, "seed": seed,
        },
    )


@dataclass
class NestedResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    auc: float
    signature_frequency: pd.DataFrame
    fitted_models: dict = field(repr=False)
    assumptions: dict = field(default_factory=dict)


def nested_evaluate(
    expression, labels, *, model="RF", seed=2025, outer_folds=5,
    inner_folds=10, repeats=5, candidates=10, n_estimators=200,
    n_jobs=1, selector_kwargs=None, scale=None,
):
    """Nested stratified CV with fold-local filtering, DE, ranking and tuning.

    Outer out-of-fold AUC is reported separately from each fold AUC. Selected
    signatures can differ across folds; this does not produce a single final
    externally deployable signature. No-DEG folds raise, rather than silently
    disappearing from a favorable performance average.
    """
    y = _validate_data(expression, labels)
    if model not in {"RF", "GBM", "DT", "SVM"}:
        raise ValueError("Unknown final model")
    if not isinstance(candidates, int) or not 1 <= candidates <= 10:
        raise ValueError("candidates must be an integer between 1 and 10")
    if outer_folds < 2 or inner_folds < 2 or repeats < 1:
        raise ValueError("Require at least two folds and positive repeats")
    if y.value_counts().min() < outer_folds:
        raise ValueError("Minority class is smaller than the outer fold count")
    X = expression.T
    predictions = pd.DataFrame(index=X.index, data={"truth": y, "score": np.nan})
    rows, models, signatures = [], {}, []
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(X, y)):
        X_train, y_train = X.iloc[train], y.iloc[train]
        if y_train.value_counts().min() < inner_folds:
            raise ValueError("Outer-training minority class is smaller than inner folds")
        options = dict(selector_kwargs or {})
        options.setdefault("seed", seed + fold)
        options.setdefault("n_estimators", n_estimators)
        estimator = Pipeline([
            ("select", FoldLocalDESelector(**options)),
            ("model", model_pipeline(
                model, seed=seed + fold, n_estimators=n_estimators,
                scale=(model == "SVM") if scale is None else scale,
            )),
        ])
        grid = [
            {f"model__{key}": values for key, values in candidate.items()}
            for candidate in _candidate_grid(model, candidates, seed + fold)
        ]
        search = GridSearchCV(
            estimator, grid, scoring="roc_auc", refit=True, error_score="raise",
            cv=RepeatedStratifiedKFold(
                n_splits=inner_folds, n_repeats=repeats, random_state=seed + fold
            ),
            n_jobs=n_jobs,
        ).fit(X_train, y_train)
        fitted = search.best_estimator_
        signature = fitted.named_steps["select"].selected_features_
        score = (
            fitted.predict_proba(X.iloc[test])[:, 1]
            if signature else np.full(len(test), 0.5)
        )
        predictions.loc[X.index[test], "score"] = score
        signatures.append(signature)
        models[fold] = fitted
        rows.append({
            "fold": fold, "auc": roc_auc_score(y.iloc[test], score),
            "train_ids": tuple(X.index[train]), "test_ids": tuple(X.index[test]),
            "signature": signature, "best_params": search.best_params_,
        })
    return NestedResult(
        predictions=predictions, folds=pd.DataFrame(rows),
        auc=float(roc_auc_score(y, predictions["score"])),
        signature_frequency=_frequency(signatures, outer_folds),
        fitted_models=models,
        assumptions={
            "scenario": "nested_fold_local_de_and_selection",
            "outer_folds": outer_folds, "inner_folds": inner_folds,
            "seed": seed,
            "no_deg_policy": (selector_kwargs or {}).get("no_deg_policy", "raise"),
            "upstream_normalization": (
                "provided expression; full-cohort RMA is transductive and not refit in folds"
            ),
            "fully_inductive_end_to_end": False,
        },
    )


@dataclass(frozen=True)
class PipelineRun:
    """Minimal return contract for a fresh full-pipeline experiment."""

    auc: float
    signature: tuple
    metadata: dict = field(default_factory=dict)


def fold_local_holdout_run(expression, labels, *, seed=2025, model="RF", **kwargs):
    """Fresh shared-holdout fold-local callable for robustness drivers.

    By default, a NoDifferentialGenesError in *any* CV training fold invokes a
    predeclared intercept-only full-run baseline: AUC 0.5 and no signature.
    Other errors propagate. Set no_deg_policy="constant" to instead use the
    selector's per-fold intercept feature policy without abandoning the run.
    """
    options = dict(kwargs)
    split = options.pop("split", None)
    split = make_split(expression, labels, seed=seed) if split is None else split
    options.setdefault("no_deg_policy", "raise")
    try:
        fitted = fit_fold_local_holdout(
            expression, labels, seed=seed, split=split, final_models=(model,),
            **options,
        )
    except NoDifferentialGenesError:
        return PipelineRun(
            0.5, (),
            {
                "scenario": "shared_holdout_fold_local_de_selection",
                "predictor": "intercept_only", "no_deg_policy": "constant_full_run",
                "fallback_trigger": "NoDifferentialGenesError in a training fold",
                "train_ids": split.train_ids, "test_ids": split.test_ids,
                "upstream_normalization": "provided; full-cohort RMA is transductive",
                "fully_inductive_end_to_end": False,
            },
        )
    prediction = fitted.holdout_predictions
    return PipelineRun(
        float(roc_auc_score(prediction["truth"], prediction[model])),
        fitted.signatures[model],
        {
            **fitted.assumptions, "train_ids": split.train_ids, "test_ids": split.test_ids,
            "best_params": fitted.searches["final"][model].best_params_,
        },
    )


def nested_pipeline_run(expression, labels, *, seed=2025, selector_kwargs=None, **kwargs):
    """Fresh nested workflow compatible with repeated/permutation drivers.

    An explicit, predeclared constant-feature no-DEG baseline is enabled by
    default so null permutations remain defined. The reported signature is the
    union of outer-fold gene signatures, not a single deployed model signature.
    Full-cohort upstream RMA remains transductive despite fold-local DE.
    """
    options = dict(selector_kwargs or {})
    options.setdefault("no_deg_policy", "constant")
    result = nested_evaluate(
        expression, labels, seed=seed, selector_kwargs=options, **kwargs
    )
    signature = tuple(sorted({
        gene for fold_signature in result.folds["signature"] for gene in fold_signature
    }))
    return PipelineRun(
        result.auc, signature,
        {
            **result.assumptions,
            "signature_definition": "union of outer-fold signatures; not a deployed model",
            "fold_signatures": result.folds["signature"].tolist(),
            "fold_aucs": result.folds["auc"].tolist(),
        },
    )


def de_holdout_run(
    expression, labels, *, seed=2025, model="RF", folds=10, repeats=5,
    candidates=10, top_n=7, n_estimators=200, n_jobs=1,
    filter_rule="any_above_median", selection_models=("RF", "GBM", "DT"),
    no_deg_policy="constant",
):
    """A ready-to-use fresh full-pipeline callable for robustness drivers.

    Split first, then training-only median imputation, expression filtering,
    DE, tuned consensus ranking, final model tuning and holdout evaluation.
    Pass resource settings with functools.partial for smoke tests.

    The predeclared default for no training DEGs is an intercept-only predictor
    (AUC 0.5, empty signature), making the statistic defined under permutations.
    This is recorded explicitly, not a relaxed DEG threshold. Choose
    ``no_deg_policy="raise"`` to reject such fits instead. This scenario uses
    train-only DE, not the paper-faithful full-cohort DE branch.
    """
    from .de import differential_expression, filter_expression

    if no_deg_policy not in {"constant", "raise"}:
        raise ValueError("no_deg_policy must be 'constant' or 'raise'")
    if model not in {"RF", "GBM", "DT", "SVM"}:
        raise ValueError("Unknown final model")
    y = _validate_data(expression, labels)
    split = make_split(expression, labels, seed=seed)
    train_ids = list(split.train_ids)
    training = expression.loc[:, train_ids].T
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    complete = pd.DataFrame(
        imputer.fit_transform(training), index=training.index, columns=training.columns
    ).T
    filtered = filter_expression(complete, rule=filter_rule)
    degs = ()
    if not filtered.empty:
        de_result = differential_expression(filtered, y.loc[train_ids])
        degs = tuple(de_result.index[de_result["is_deg"]])
    metadata = {
        "scenario": "train_only_de_holdout",
        "train_ids": split.train_ids, "test_ids": split.test_ids,
        "deg_count": len(degs), "deg_genes": degs, "filter_rule": filter_rule,
        "no_deg_policy": no_deg_policy, "model": model,
        "upstream_normalization": "provided; full-cohort RMA is transductive",
        "fully_inductive_end_to_end": False,
    }
    if not degs:
        if no_deg_policy == "raise":
            raise NoDifferentialGenesError("Training DEG rule retained no genes")
        # A constant training-prevalence predictor has tied ranks: AUC exactly 0.5.
        score = np.full(len(split.test_ids), y.loc[train_ids].mean())
        auc = float(roc_auc_score(y.loc[list(split.test_ids)], score))
        return PipelineRun(auc, (), {**metadata, "predictor": "intercept_only"})
    fitted = fit_models(
        expression, labels, degs, seed=seed, folds=folds, repeats=repeats,
        candidates=candidates, top_n=top_n, split=split, n_jobs=n_jobs,
        n_estimators=n_estimators, selection_models=selection_models,
        final_models=(model,), feature_scope="training_only_de",
    )
    predictions = fitted.holdout_predictions
    return PipelineRun(
        float(roc_auc_score(predictions["truth"], predictions[model])),
        fitted.features, {**metadata, "predictor": "tuned_model"},
    )


@dataclass
class RobustnessResult:
    runs: pd.DataFrame
    signature_frequency: pd.DataFrame
    exceedances: int
    probability: float
    empirical_p: float
    assumptions: dict


def _run_summary(
    pipeline, expression, labels, *, n_runs, seed, threshold, permutation,
    on_error,
):
    _validate_data(expression, labels)
    if not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    if not 0 <= threshold <= 1 or on_error not in {"raise", "record"}:
        raise ValueError("Require threshold in [0,1] and on_error='raise' or 'record'")
    rows, signatures = [], []
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for run, sequence in enumerate(seeds):
        permutation_seed, fit_seed = sequence.generate_state(2)
        run_labels = labels.copy()
        if permutation:
            run_labels.iloc[:] = np.random.default_rng(
                int(permutation_seed)
            ).permutation(run_labels.to_numpy())
        try:
            outcome = pipeline(expression.copy(), run_labels, seed=int(fit_seed))
            if not isinstance(outcome, PipelineRun):
                raise TypeError("Full-pipeline callable must return PipelineRun")
            if not np.isfinite(outcome.auc) or not 0 <= outcome.auc <= 1:
                raise ValueError("Pipeline AUC must be finite and lie in [0,1]")
            signature = tuple(outcome.signature)
            if len(set(signature)) != len(signature):
                raise ValueError("Pipeline signature contains duplicate genes")
            signatures.append(signature)
            rows.append({
                "run": run, "seed": int(fit_seed), "auc": outcome.auc,
                "signature": signature, "status": "ok", "error": None,
                "metadata": outcome.metadata,
            })
        except Exception as exc:
            if on_error == "raise":
                raise
            rows.append({
                "run": run, "seed": int(fit_seed), "auc": np.nan,
                "signature": (), "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "metadata": {},
            })
    runs = pd.DataFrame(rows)
    exceedances = int((runs["auc"] >= threshold).sum())
    completed = int(runs["status"].eq("ok").sum())
    all_complete = completed == n_runs
    return RobustnessResult(
        runs=runs, signature_frequency=_frequency(signatures, n_runs),
        exceedances=exceedances,
        probability=exceedances / n_runs if all_complete else np.nan,
        empirical_p=(1 + exceedances) / (n_runs + 1)
        if permutation and all_complete else np.nan,
        assumptions={
            "scenario": "full_pipeline_label_permutation" if permutation
            else "repeated_fresh_full_pipeline",
            "threshold": threshold, "requested_runs": n_runs,
            "completed_runs": completed, "failed_runs": n_runs - completed,
            "seed": seed,
            "signature_frequency_denominator": "all requested runs",
            "failed_runs_policy": "probability unavailable if any run fails",
            "callable_contract": "fresh filtering + DE + selection + tuning + evaluation",
            "label_permutation": "unrestricted; requires exchangeable independent samples",
            "permutation_tail": "P(null AUC >= fixed threshold); not observed-statistic p-value"
            if permutation else None,
        },
    )


def repeated_pipeline_runs(
    pipeline, expression, labels, *, n_runs=100, seed=2025, threshold=0.87,
    on_error="raise",
):
    """Repeat an entire callable pipeline for 100 independently seeded fits.

    The callable receives (expression, labels, seed=...) and must construct new
    models, splits, DE and selected genes on *every* call, returning PipelineRun.
    Use a fixed test cohort only if its labels never guide fitting decisions.
    """
    return _run_summary(
        pipeline, expression, labels, n_runs=n_runs, seed=seed,
        threshold=threshold, permutation=False, on_error=on_error,
    )


def permutation_pipeline_test(
    pipeline, expression, labels, *, n_permutations=100, seed=2025,
    threshold=0.87, on_error="raise",
):
    """Estimate P(null AUC >= 0.87) including fresh DEG and feature selection.

    The same full-pipeline contract as repeated_pipeline_runs applies. Never
    pass a callable that only refits classifiers on an observed-label signature.
    Report both the exceedance fraction and (1 + exceedances)/(B + 1).
    Failed runs are never silently dropped or counted as AUC=0.5.
    """
    return _run_summary(
        pipeline, expression, labels, n_runs=n_permutations, seed=seed,
        threshold=threshold, permutation=True, on_error=on_error,
    )


@dataclass
class TransductiveStandardization:
    source: pd.DataFrame
    target: pd.DataFrame
    assumptions: dict


def transductive_standardize(source, target):
    """Separately z-score each gene within source and entire target cohorts.

    This is a separate, explicitly transductive scenario, NOT direct external
    validation and NOT a no-leakage preprocessing claim: each target sample uses
    information from every other target sample. Do not mix it with the direct
    prediction branch, or apply it across a train/test split before fitting.
    Constant genes are centered to zero. Missing values are rejected.
    """
    for frame in (source, target):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Require nonempty genes x samples DataFrames")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Gene and sample identifiers must be unique")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Standardization requires finite expression")
    if set(source.index) != set(target.index):
        raise ValueError("Source and target must contain the same genes")

    def standardize(frame):
        mean = frame.mean(axis=1)
        std = frame.std(axis=1, ddof=0).replace(0, 1)
        return frame.sub(mean, axis=0).div(std, axis=0)

    return TransductiveStandardization(
        source=standardize(source), target=standardize(target.reindex(source.index)),
        assumptions={
            "scenario": "independent_cohort_standardization_transductive_target",
            "target_cohort_statistics_used": True,
            "direct_external_validation": False,
            "uses_target_labels": False,
            "no_leakage_claim": False,
            "standard_deviation_ddof": 0,
        },
    )
