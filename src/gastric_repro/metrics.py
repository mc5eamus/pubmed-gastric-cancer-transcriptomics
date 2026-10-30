"""Binary diagnostic metrics with explicit undefined and infinite quantities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _inputs(y_true, scores):
    y = np.asarray(y_true)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and scores must be nonempty aligned 1D arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Require binary labels and finite scores")
    return y.astype(int), p


def _ratio(numerator, denominator):
    if denominator:
        return float(numerator / denominator)
    return float("inf") if numerator > 0 else float("nan")


def binary_metrics(y_true, scores, *, threshold=0.5):
    """Compute metrics with label 1 positive and score >= threshold positive.

    AUC accepts continuous scores. Diagnostic ratios assume the supplied
    threshold is meaningful (0.5 by default for probabilities). 0/0 is NaN;
    positive/0 is +infinity. Predictive values are cohort-prevalence dependent.
    """
    y, p = _inputs(y_true, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    prediction = p >= threshold
    tp = int(np.sum((y == 1) & prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(y),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "lr_positive": _ratio(sensitivity, 1 - specificity),
        "lr_negative": _ratio(1 - sensitivity, specificity),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@dataclass(frozen=True)
class AUCInterval:
    estimate: float
    lower: float
    upper: float
    variance: float
    confidence: float
    method: str = "DeLong normal (clipped to [0, 1])"


def delong_auc_ci(y_true, scores, *, confidence=0.95):
    """Unpaired one-curve DeLong AUC interval, accounting for tied scores.

    At least two observations per class are needed for variance estimation.
    A zero estimated variance yields a point interval, which can occur with
    perfect separation and should not be mistaken for certainty.
    """
    y, p = _inputs(y_true, scores)
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
    if min(len(positive), len(negative)) < 2:
        raise ValueError("DeLong variance requires at least two samples per class")
    v_positive = (
        np.searchsorted(negative, positive, side="left")
        + np.searchsorted(negative, positive, side="right")
    ) / (2 * len(negative))
    v_negative = 1 - (
        np.searchsorted(positive, negative, side="left")
        + np.searchsorted(positive, negative, side="right")
    ) / (2 * len(positive))
    auc = float(v_positive.mean())
    variance = float(
        np.var(v_positive, ddof=1) / len(positive)
        + np.var(v_negative, ddof=1) / len(negative)
    )
    margin = norm.ppf((1 + confidence) / 2) * np.sqrt(variance)
    return AUCInterval(
        auc, float(max(0, auc - margin)), float(min(1, auc + margin)),
        variance, confidence,
    )


def bootstrap_metrics(
    y_true, scores, *, n_bootstrap=2000, confidence=0.95, seed=2025, threshold=0.5
):
    """Stratified percentile intervals conditional on observed class counts.

    Undefined (NaN) replicates are excluded and explicitly counted. Infinite
    likelihood ratios are retained in the empirical quantiles, using discrete
    inverse-CDF quantiles to avoid undefined interpolation with infinity.
    These intervals resample predictions, not the fitting pipeline, and do not
    capture model-selection uncertainty.
    """
    y, p = _inputs(y_true, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Stratified bootstrap requires both classes")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    point = binary_metrics(y, p, threshold=threshold)
    names = [name for name in point if name not in {"tp", "tn", "fp", "fn"}]
    draws = {name: [] for name in names}
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in classes
        ])
        values = binary_metrics(y[indices], p[indices], threshold=threshold)
        for name in names:
            draws[name].append(values[name])
    alpha = (1 - confidence) / 2
    rows = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        valid = values[~np.isnan(values)]
        lower, upper = (
            np.quantile(valid, [alpha, 1 - alpha], method="inverted_cdf")
            if len(valid) else (np.nan, np.nan)
        )
        rows[name] = {
            "estimate": point[name], "lower": lower, "upper": upper,
            "valid_replicates": len(valid),
            "undefined_replicates": int(np.isnan(values).sum()),
            "infinite_replicates": int(np.isinf(values).sum()),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "metric"
    result.attrs.update(
        seed=seed, confidence=confidence, n_bootstrap=n_bootstrap,
        method="stratified percentile; NaNs excluded, infinities retained",
    )
    return result


@dataclass
class EvaluationResult:
    point: dict
    bootstrap: pd.DataFrame
    delong: AUCInterval


def evaluate_predictions(y_true, scores, **kwargs):
    """Return point estimates, bootstrap intervals and a separate DeLong CI."""
    return EvaluationResult(
        binary_metrics(y_true, scores, threshold=kwargs.get("threshold", 0.5)),
        bootstrap_metrics(y_true, scores, **kwargs),
        delong_auc_ci(y_true, scores, confidence=kwargs.get("confidence", 0.95)),
    )
