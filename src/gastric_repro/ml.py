"""Training-only preprocessing, feature ranking and held-out model evaluation.

Expression matrices are genes x samples; sklearn sees samples x genes. The
correlation filter is an explicitly approximate ``step_corr``: in supplied
feature order it retains a gene only if its absolute training correlation with
every retained gene is <= the threshold. This is not caret's mean-correlation
removal algorithm. Candidate genes are lexicographically ordered; the final
signature uses consensus order. All preprocessing and undersampling are fitted anew
inside each CV fit. SVMs use training-fitted standard scaling by default.

Supplied features may themselves leak outcome information. In particular, the
paper-faithful full-cohort DE branch is not leakage-free; declare its provenance
with ``feature_scope="full_cohort_de_faithful"``. For unbiased outer evaluation,
use the fold-local selector in :mod:`gastric_repro.robustness`.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    ParameterSampler,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Deterministic, order-prioritized approximation to step_corr."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("CorrelationFilter requires a nonempty 2D matrix")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if not np.isfinite(X).all():
            raise ValueError("Impute missing values before correlation filtering")
        self.n_features_in_ = X.shape[1]
        centered = X - X.mean(axis=0)
        norm = np.linalg.norm(centered, axis=0)
        # Constant genes have no defined correlation and are retained.
        normalized = np.divide(
            centered, norm, out=np.zeros_like(centered), where=norm != 0
        )
        kept = []
        for column in range(X.shape[1]):
            if not kept or np.all(
                np.abs(normalized[:, kept].T @ normalized[:, column])
                <= self.threshold
            ):
                kept.append(column)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[kept] = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from the training matrix")
        return X[:, self.support_]

    def get_support(self):
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)[self.support_]


@dataclass(frozen=True)
class SplitIDs:
    train_ids: tuple
    test_ids: tuple


def _validate_data(expression, labels):
    if not isinstance(expression, pd.DataFrame) or not isinstance(labels, pd.Series):
        raise TypeError("expression must be a DataFrame and labels a Series")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Gene and sample identifiers must be unique")
    if not labels.index.is_unique or set(labels.index) != set(expression.columns):
        raise ValueError("Labels must uniquely cover exactly the expression samples")
    y = labels.reindex(expression.columns)
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Labels must contain both binary classes 0 and 1")
    if not all(isinstance(gene, str) for gene in expression.index):
        raise ValueError("Gene identifiers must be strings")
    values = expression.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Expression may contain NaN, but not infinity")
    return y.astype(int)


def make_split(expression, labels, *, seed=2025, test_size=0.30):
    """Create the shared 70/30 stratified split before any model selection."""
    y = _validate_data(expression, labels)
    train, test = train_test_split(
        expression.columns.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=y.to_numpy(),
    )
    return SplitIDs(tuple(train), tuple(test))


def _check_split(split, samples, labels):
    train, test = list(split.train_ids), list(split.test_ids)
    if (
        len(set(train)) != len(train)
        or len(set(test)) != len(test)
        or set(train) & set(test)
        or set(train) | set(test) != set(samples)
    ):
        raise ValueError("Split must partition all samples without overlap or duplicates")
    if any(set(labels.loc[ids]) != {0, 1} for ids in (train, test)):
        raise ValueError("Both split partitions must contain both classes")


def model_pipeline(
    model, *, seed=2025, corr_threshold=0.85, scale=False, n_estimators=200
):
    """Construct an unfitted pipeline; resampling is never applied at prediction."""
    if model == "RF":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=1
        )
    elif model == "GBM":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
    elif model == "DT":
        classifier = DecisionTreeClassifier(random_state=seed)
    elif model == "SVM":
        classifier = SVC(probability=True, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("corr", CorrelationFilter(corr_threshold)),
            ("scale", StandardScaler() if scale else "passthrough"),
            ("sample", RandomUnderSampler(random_state=seed)),
            ("classifier", classifier),
        ]
    )


def _candidate_grid(model, candidates, seed):
    spaces = {
        "RF": {
            "classifier__max_features": ["sqrt", 0.5, 1.0],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_depth": [None, 3, 6],
        },
        "GBM": {
            "classifier__max_depth": [1, 2, 3],
            "classifier__learning_rate": [0.03, 0.1, 0.2],
            "classifier__subsample": [0.7, 1.0],
        },
        "DT": {
            "classifier__max_depth": [None, 2, 3, 5],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__criterion": ["gini", "entropy"],
        },
        "SVM": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__gamma": ["scale", 0.01, 0.1],
            "classifier__kernel": ["rbf"],
        },
    }
    return [
        {key: [value] for key, value in candidate.items()}
        for candidate in ParameterSampler(
            spaces[model], n_iter=candidates, random_state=seed
        )
    ]


def _search(model, X, y, *, seed, folds, repeats, candidates, n_jobs,
            corr_threshold, scale, n_estimators):
    search = GridSearchCV(
        model_pipeline(
            model, seed=seed, corr_threshold=corr_threshold,
            scale=scale, n_estimators=n_estimators,
        ),
        _candidate_grid(model, candidates, seed),
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=repeats, random_state=seed
        ),
        n_jobs=n_jobs,
        error_score="raise",
        refit=True,
    )
    search.fit(X, y)
    return search


def consensus_importance(models, features):
    """Approximate paper consensus: RF/GBM primary, DT secondary, gene ID last.

    Correlation-filtered genes receive zero importance. Tree impurity importance
    is not interpreted as causal importance, and ranks are not effect sizes.
    Mean RF/GBM ranks are primary when available; DT only breaks their ties.
    With DT alone, DT rank is primary. This deterministic operationalization
    does not claim to recover an unspecified original ranking implementation.
    """
    genes = sorted(features)
    table = pd.DataFrame(index=pd.Index(genes, name="gene"))
    for name in sorted(models):
        pipeline = models[name]
        support = pipeline.named_steps["corr"].get_support()
        values = np.zeros(len(features))
        values[support] = pipeline.named_steps["classifier"].feature_importances_
        table[f"{name}_importance"] = pd.Series(values, index=features).reindex(genes)
        # Equal importances share a rank; final consensus ties use gene IDs.
        table[f"{name}_rank"] = table[f"{name}_importance"].rank(
            ascending=False, method="average"
        )
    ranks = [column for column in table if column.endswith("_rank")]
    if not ranks:
        raise ValueError("At least one tree model is required for consensus ranking")
    primary = [column for column in ("RF_rank", "GBM_rank") if column in ranks]
    table["consensus_rank"] = table[primary or ranks].mean(axis=1)
    sort_by = ["consensus_rank"]
    if primary and "DT_rank" in ranks:
        sort_by.append("DT_rank")
    return table.sort_values(sort_by, kind="stable")


@dataclass
class ModelResult:
    models: dict
    selection_models: dict
    split: SplitIDs
    features: tuple
    candidate_features: tuple
    importance: pd.DataFrame
    holdout_predictions: pd.DataFrame
    searches: dict = field(repr=False)
    assumptions: dict = field(default_factory=dict)

    @property
    def train_ids(self):
        return self.split.train_ids

    @property
    def test_ids(self):
        return self.split.test_ids

    @property
    def selected_features(self):
        return self.features

    def predict_external(self, expression):
        return predict_external(self, expression)


def predict_external(result, expression):
    """Direct external probabilities, reindexed to training genes; never refit.

    Extra genes are ignored. Missing required genes are an error, while NaNs in
    present genes are handled by the already training-fitted imputer.
    """
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("External expression must be a genes x samples DataFrame")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("External gene and sample identifiers must be unique")
    missing = set(result.features) - set(expression.index)
    if missing:
        raise ValueError(f"Missing required genes: {sorted(missing)}")
    X = expression.reindex(result.features).T
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("External expression cannot contain infinity")
    return pd.DataFrame(
        {name: pipeline.predict_proba(X)[:, 1]
         for name, pipeline in result.models.items()},
        index=X.index,
    )


def fit_models(
    expression, labels, features, *, seed=2025, folds=10, repeats=5,
    candidates=10, top_n=7, split=None, n_jobs=1, n_estimators=200,
    corr_threshold=0.85, scale=False, scale_svm=True,
    selection_models=("RF", "GBM", "DT"),
    final_models=("RF", "GBM", "DT", "SVM"),
    feature_scope="provided_unverified",
):
    """Select a consensus signature and tune final models on the same train IDs.

    Defaults use repeated 10-fold/5-repeat ROC-AUC search, at most 10 candidate
    configurations per model, and a top-seven signature. ``n_estimators`` is a
    fixed resource parameter, not a held-out-data-driven tuning choice. A
    supplied ``SplitIDs`` allows explicit split reuse across analysis branches.
    Training CV scores after feature ranking are *not* unbiased performance
    estimates: ranking uses the entire training partition. Only the untouched
    holdout (conditional on feature provenance) or nested CV estimates generalization.
    """
    y = _validate_data(expression, labels)
    features = tuple(sorted(set(features)))
    if not features or set(features) - set(expression.index):
        raise ValueError("features must be a nonempty list of present gene IDs")
    if not isinstance(candidates, int) or not 1 <= candidates <= 10:
        raise ValueError("candidates must be an integer between 1 and 10")
    if folds < 2 or repeats < 1 or top_n < 1 or n_estimators < 1:
        raise ValueError("Require folds >= 2 and positive repeats, top_n and estimators")
    if not selection_models or set(selection_models) - {"RF", "GBM", "DT"}:
        raise ValueError("Selection requires one or more of RF, GBM, DT")
    if not final_models or set(final_models) - {"RF", "GBM", "DT", "SVM"}:
        raise ValueError("Final models must be RF, GBM, DT or SVM")
    split = make_split(expression, labels, seed=seed) if split is None else split
    _check_split(split, expression.columns, y)
    train_ids, test_ids = list(split.train_ids), list(split.test_ids)
    y_train = y.loc[train_ids]
    if y_train.value_counts().min() < folds:
        raise ValueError("Training minority class is smaller than the CV fold count")
    X_train = expression.loc[list(features), train_ids].T
    search_options = dict(
        seed=seed, folds=folds, repeats=repeats, candidates=candidates,
        n_jobs=n_jobs, corr_threshold=corr_threshold, n_estimators=n_estimators,
    )
    preliminary = {
        name: _search(name, X_train, y_train, scale=scale, **search_options)
        for name in selection_models
    }
    selection = {name: search.best_estimator_ for name, search in preliminary.items()}
    importance = consensus_importance(selection, features)
    selected = tuple(importance.index[:top_n])
    X_train = expression.loc[list(selected), train_ids].T
    final_searches = {
        name: _search(
            name, X_train, y_train,
            scale=(scale_svm if name == "SVM" else scale), **search_options
        )
        for name in final_models
    }
    models = {name: search.best_estimator_ for name, search in final_searches.items()}
    predictions = pd.DataFrame({"truth": y.loc[test_ids]})
    for name, pipeline in models.items():
        predictions[name] = pipeline.predict_proba(
            expression.loc[list(selected), test_ids].T
        )[:, 1]
    return ModelResult(
        models=models, selection_models=selection, split=split, features=selected,
        candidate_features=features, importance=importance,
        holdout_predictions=predictions,
        searches={"selection": preliminary, "final": final_searches},
        assumptions={
            "feature_scope": feature_scope,
            "holdout_used_for_tuning": False,
            "full_cohort_de_leakage": feature_scope == "full_cohort_de_faithful",
            "correlation_filter": "ordered greedy approximate step_corr",
            "resampling": "training-only random undersampling within each fit",
            "svm_scaling": bool(scale_svm),
            "svm_kernel": "rbf",
            "consensus": "mean RF/GBM ranks primary; DT tie-break; gene ID last",
            "selection_cv_is_nested": False,
            "upstream_normalization": "provided; full-cohort RMA is transductive",
            "upstream_normalization_refitted": False,
            "seed": seed,
        },
    )
