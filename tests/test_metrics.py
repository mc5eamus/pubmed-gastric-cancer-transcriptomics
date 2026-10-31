import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from gastric_repro.metrics import (
    binary_metrics, bootstrap_metrics, delong_auc_ci, evaluate_predictions,
)


def test_known_confusion_and_diagnostic_ratios():
    y = [1, 1, 1, 0, 0, 0, 0]
    score = [.9, .8, .2, .7, .4, .3, .1]
    result = binary_metrics(y, score)
    assert [result[k] for k in ("tp", "tn", "fp", "fn")] == [2, 3, 1, 1]
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["specificity"] == .75
    assert result["accuracy"] == pytest.approx(5 / 7)
    assert result["ppv"] == pytest.approx(2 / 3)
    assert result["npv"] == .75
    assert result["lr_positive"] == pytest.approx((2 / 3) / .25)
    assert result["lr_negative"] == pytest.approx((1 / 3) / .75)
    assert result["auc"] == roc_auc_score(y, score)


def test_undefined_and_infinite_metrics_are_not_coerced():
    perfect = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert np.isinf(perfect["lr_positive"])
    assert perfect["lr_negative"] == 0
    all_negative = binary_metrics([0, 0, 1, 1], [.1] * 4)
    assert np.isnan(all_negative["ppv"])
    assert np.isnan(all_negative["lr_positive"])
    single = binary_metrics([1, 1], [0, 1])
    assert np.isnan(single["auc"])
    assert np.isnan(single["specificity"])


def test_delong_matches_direct_pairwise_variance_with_ties():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([.1, .9, .4, .5, .5, .5, .8, .7])
    pair = (
        (p[y == 1, None] > p[y == 0]).astype(float)
        + .5 * (p[y == 1, None] == p[y == 0])
    )
    expected_variance = pair.mean(1).var(ddof=1) / 4 + pair.mean(0).var(ddof=1) / 4
    interval = delong_auc_ci(y, p)
    assert interval.estimate == pytest.approx(roc_auc_score(y, p))
    assert interval.variance == pytest.approx(expected_variance)
    margin = norm.ppf(.975) * np.sqrt(expected_variance)
    assert interval.lower == pytest.approx(max(0, pair.mean() - margin))
    assert interval.upper == pytest.approx(min(1, pair.mean() + margin))
    tied = delong_auc_ci([0, 0, 1, 1], [.5] * 4)
    assert tied.lower == tied.upper == .5
    with pytest.raises(ValueError, match="at least two"):
        delong_auc_ci([0, 1], [.1, .9])


def test_bootstrap_seed_stratification_and_infinity_intervals():
    y, p = [0] * 4 + [1] * 4, [0] * 4 + [1] * 4
    one = bootstrap_metrics(y, p, n_bootstrap=50, seed=8)
    two = bootstrap_metrics(y, p, n_bootstrap=50, seed=8)
    pd.testing.assert_frame_equal(one, two)
    assert one.loc["auc", "lower"] == one.loc["auc", "upper"] == 1
    assert one.loc["lr_positive", "infinite_replicates"] == 50
    assert np.isinf(one.loc["lr_positive", "lower"])
    assert one["undefined_replicates"].sum() == 0
    undefined = bootstrap_metrics(y, [.1] * 8, n_bootstrap=15)
    assert undefined.loc["ppv", "undefined_replicates"] == 15
    assert undefined.loc["ppv", "valid_replicates"] == 0
    assert np.isnan(undefined.loc["ppv", "lower"])
    evaluation = evaluate_predictions(y, p, n_bootstrap=10)
    assert evaluation.delong.estimate == evaluation.point["auc"] == 1


@pytest.mark.parametrize(
    "y,p", [([], []), ([0, 1], [.2]), ([0, 2], [.1, .9]), ([0, 1], [np.nan, 1])]
)
def test_metrics_reject_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        binary_metrics(y, p)


def test_bootstrap_rejects_single_class_or_invalid_count():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_metrics([0, 0], [.1, .2])
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_metrics([0, 1], [.1, .2], n_bootstrap=0)
