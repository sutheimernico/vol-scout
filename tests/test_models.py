"""Tests for src/vol_scout/models.py: the MODEL_REGISTRY contenders.

No test in this module makes a network call. GARCH tests simulate their own
GARCH(1,1) process (known omega/alpha/beta) rather than depending on real
market data, so the ground truth is independently computable.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vol_scout.data import multi_day_rv
from vol_scout.models import (
    ForecastResult,
    forecast_har_rv,
    forecast_random_walk,
    har_features,
    har_rv_coefficients,
)

# 30 hand-picked daily RV values (all realized RV proxies are non-negative,
# so a mild upward-trending sequence is a realistic-shaped fixture).
_RV_30 = pd.Series(
    [0.0001 + 0.00002 * i for i in range(30)],
    index=pd.date_range("2020-01-01", periods=30, freq="D"),
    name="rv",
)


def _train_df(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"rv": rv})


def test_forecast_random_walk_returns_forecast_result():
    result = forecast_random_walk(_train_df(_RV_30))

    assert isinstance(result, ForecastResult)
    assert isinstance(result.fit_predict_seconds, float)
    assert result.fit_predict_seconds >= 0.0


def test_forecast_random_walk_point_1_is_last_rv_exactly():
    result = forecast_random_walk(_train_df(_RV_30))

    assert result.point[1] == pytest.approx(_RV_30.iloc[-1])


def test_forecast_random_walk_point_22_is_22x_last_rv_exactly():
    result = forecast_random_walk(_train_df(_RV_30))

    assert result.point[22] == pytest.approx(22.0 * _RV_30.iloc[-1])


# --- HAR-RV -------------------------------------------------------------


def test_har_features_hand_computed_constant_trailing_window():
    # First 22 values constant -> weekly/monthly means equal the daily
    # value exactly, regardless of window size.
    rv = pd.Series([0.0004] * 22)

    daily, weekly, monthly = har_features(rv, 21)

    expected = math.log(0.0004)
    assert daily == pytest.approx(expected)
    assert weekly == pytest.approx(expected)
    assert monthly == pytest.approx(expected)


def test_har_features_hand_computed_trending_series_lags_differ():
    rv = pd.Series([0.0001 * (i + 1) for i in range(25)])  # strictly increasing
    origin_idx = 24

    daily, weekly, monthly = har_features(rv, origin_idx)

    assert daily == pytest.approx(math.log(rv.iloc[24]))
    assert weekly == pytest.approx(math.log(rv.iloc[20:25].mean()))
    assert monthly == pytest.approx(math.log(rv.iloc[3:25].mean()))
    # More history averaged in -> lower mean on a strictly increasing
    # series -> the three lags are visibly distinct.
    assert daily > weekly > monthly


def test_har_features_raises_below_monthly_window():
    rv = pd.Series([0.0004] * 10)

    with pytest.raises(ValueError):
        har_features(rv, 5)


def _simulate_har_process(
    n: int,
    intercept: float,
    beta_d: float,
    beta_w: float,
    beta_m: float,
    noise_std: float,
    seed: int,
) -> pd.Series:
    """Recursively generate a series whose h=1 HAR-RV relationship is exactly
    known: `log(RV_t) = intercept + beta_d*log(RV_{t-1}) +
    beta_w*log(mean(RV_{t-5..t-1})) + beta_m*log(mean(RV_{t-22..t-1})) +
    noise`. Fitting `har_rv_coefficients(rv, horizon=1)` on the result should
    recover `[intercept, beta_d, beta_w, beta_m]` -- this is the ground truth
    the OLS-recovery test checks against.
    """
    rng = np.random.default_rng(seed)
    log_rv = np.empty(n)
    log_rv[:22] = math.log(1e-4) + rng.normal(0, noise_std, 22)
    for t in range(22, n):
        log_d = log_rv[t - 1]
        log_w = math.log(np.exp(log_rv[t - 5 : t]).mean())
        log_m = math.log(np.exp(log_rv[t - 22 : t]).mean())
        log_rv[t] = (
            intercept + beta_d * log_d + beta_w * log_w + beta_m * log_m + rng.normal(0, noise_std)
        )
    return pd.Series(np.exp(log_rv))


_HAR_GENERATING_COEFFICIENTS = (-1.0, 0.5, 0.3, 0.1)  # intercept, beta_d, beta_w, beta_m


def test_har_rv_coefficients_recovers_known_generating_process():
    intercept, beta_d, beta_w, beta_m = _HAR_GENERATING_COEFFICIENTS
    rv = _simulate_har_process(
        n=1000, intercept=intercept, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        noise_std=0.02, seed=3,
    )

    coefficients = har_rv_coefficients(rv, horizon=1)

    assert coefficients[0] == pytest.approx(intercept, abs=0.1)
    assert coefficients[1] == pytest.approx(beta_d, abs=0.1)
    assert coefficients[2] == pytest.approx(beta_w, abs=0.1)
    assert coefficients[3] == pytest.approx(beta_m, abs=0.1)


def test_forecast_har_rv_returns_finite_positive_both_horizons():
    intercept, beta_d, beta_w, beta_m = _HAR_GENERATING_COEFFICIENTS
    rv = _simulate_har_process(
        n=300, intercept=intercept, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        noise_std=0.02, seed=3,
    )

    result = forecast_har_rv(_train_df(rv))

    assert isinstance(result, ForecastResult)
    assert math.isfinite(result.point[1])
    assert math.isfinite(result.point[22])
    assert result.point[1] > 0.0
    assert result.point[22] > 0.0


def test_har_features_and_targets_poison_value_boundaries():
    n = 100
    poison_day = 60
    clean_rv = pd.Series([0.0001 + 0.000001 * i for i in range(n)])
    rv = clean_rv.copy()
    rv.iloc[poison_day] = 1e9

    target_h1 = np.log(rv.shift(-1))
    target_h22 = np.log(multi_day_rv(rv, 22).shift(-22))

    def assert_features_exclude_poison(o: int) -> None:
        daily, weekly, monthly = har_features(rv, o)
        assert daily == pytest.approx(math.log(clean_rv.iloc[o]))
        assert weekly == pytest.approx(math.log(clean_rv.iloc[o - 4 : o + 1].mean()))
        assert monthly == pytest.approx(math.log(clean_rv.iloc[o - 21 : o + 1].mean()))

    # o = poison_day - 22: the h=22 target's far edge reaches exactly
    # poison_day (sum(RV_{o+1..o+22}) includes it at the boundary) --
    # legitimate, not a leak. Features at this SAME origin look only
    # backward and must not see it.
    o = poison_day - 22
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 22].sum() + 1e9))
    assert_features_exclude_poison(o)

    # o = poison_day - 1: the h=1 target legitimately equals the poison
    # (tomorrow, relative to o, is poison_day). Features must not.
    o = poison_day - 1
    assert target_h1.iloc[o] == pytest.approx(math.log(1e9))
    assert_features_exclude_poison(o)

    # o well beyond both the 22-day feature and target windows: neither
    # target reaches the poison at all.
    o = poison_day - 30
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))
    assert_features_exclude_poison(o)
