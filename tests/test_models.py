"""Tests for src/vol_scout/models.py: the MODEL_REGISTRY contenders.

No test in this module makes a network call. GARCH tests simulate their own
GARCH(1,1) process (known omega/alpha/beta) rather than depending on real
market data, so the ground truth is independently computable. The Chronos-2
tests are the one exception to "no model calls": most make a REAL forward
pass against locally cached weights (`~/.cache/huggingface`) on tiny
synthetic inputs -- a 200-400 point context predicting 22 steps takes well
under a second on CPU (verified during M4.1's feasibility check) -- mocking
the model would buy nothing but false confidence in the median/context-cap
logic these tests exist to catch. Only `_capped_context`'s own unit tests
avoid the model entirely (pure array slicing).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from arch.utility.exceptions import DataScaleWarning

from vol_scout.data import multi_day_rv
from vol_scout.models import (
    MODEL_REGISTRY,
    ForecastResult,
    _capped_context,
    _get_chronos2_pipeline,
    _har_feature_frame,
    forecast_chronos2,
    forecast_garch,
    forecast_har_rv,
    forecast_lgbm,
    forecast_random_walk,
    har_features,
    har_rv_coefficients,
    lgbm_features,
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


def _ohlcv_rv_df(rv: pd.Series) -> pd.DataFrame:
    """Synthetic OHLCV+rv frame for LightGBM tests: `rv` carries the actual
    signal under test (hand-picked or simulated by the caller); open/high/
    low/close are simple, deterministic placeholders (a mild trend with a
    small fixed intraday range and no relationship to `rv` at all) -- just
    enough to make `range_pct`/`overnight_gap` well-defined and non-constant
    without them dominating whatever generating relationship a given test
    puts into `rv`. DatetimeIndex (calendar features need real timestamps).
    """
    n = len(rv)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "rv": rv.to_numpy(),
        },
        index=idx,
    )


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


def test_har_rv_coefficients_raises_below_four_origins():
    # n=23: the monthly-window floor (origin_idx>=21) and the h=1 target's
    # own realized-ness (needs one more row after the origin) together
    # leave exactly ONE fully-realized origin (idx=21) -- below the >=4
    # floor OLS needs for its 4 parameters (fail-loud guard, untested until
    # now).
    rv = pd.Series([0.0001 + 0.000001 * i for i in range(23)])

    with pytest.raises(ValueError):
        har_rv_coefficients(rv, horizon=1)


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
    # target reaches the poison at all -- coarse control.
    o = poison_day - 30
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))
    assert_features_exclude_poison(o)

    # Sharpest "neither" control: o+23 is exactly one day PAST the h=22
    # target's far edge (o+22) -- poison_day sits exactly one day OUTSIDE
    # the h=22 target window here, the tightest margin at which the poison
    # must still be absent from everything (tighter than the -30 coarse
    # control above).
    o = poison_day - 23
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))
    assert_features_exclude_poison(o)


def test_har_features_matches_har_feature_frame_across_many_origins():
    # Parity pin (M2 hardening finding): har_features (scalar, used at
    # actual predict time in forecast_har_rv) and _har_feature_frame
    # (vectorized, used to build the OLS training design matrix) must
    # agree position-for-position -- a future refactor of either one that
    # silently lets them diverge would desync what the model trains on
    # from what it predicts with, without any single existing test
    # noticing (the OLS-recovery test only exercises _har_feature_frame;
    # the poison test above only exercises har_features).
    rv = pd.Series([0.0001 * (1.0 + 0.05 * math.sin(i / 3.0)) for i in range(150)])
    frame = _har_feature_frame(rv)

    for o in range(21, len(rv)):
        daily, weekly, monthly = har_features(rv, o)
        row = frame.iloc[o]
        assert daily == pytest.approx(row["log_d"])
        assert weekly == pytest.approx(row["log_w"])
        assert monthly == pytest.approx(row["log_m"])


# --- GARCH(1,1) ----------------------------------------------------------


def _simulate_garch11(
    n: int, omega: float, alpha: float, beta: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a real GARCH(1,1) return path with known omega/alpha/beta,
    seeded at its own unconditional variance. Returns `(returns, sigma2)` --
    `sigma2` is the ground-truth conditional variance path (never seen by
    the model under test, which only gets `returns`).
    """
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    returns[0] = rng.normal(0.0, math.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0.0, math.sqrt(sigma2[t]))
    return returns, sigma2


def _close_train_df(returns: np.ndarray) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close})


_GARCH_OMEGA, _GARCH_ALPHA, _GARCH_BETA = 1e-6, 0.05, 0.90  # persistence 0.95, realistic equity vol


def test_forecast_garch_returns_finite_positive_both_horizons():
    returns, _ = _simulate_garch11(1500, _GARCH_OMEGA, _GARCH_ALPHA, _GARCH_BETA, seed=0)

    result = forecast_garch(_close_train_df(returns))

    assert isinstance(result, ForecastResult)
    assert math.isfinite(result.point[1])
    assert math.isfinite(result.point[22])
    assert result.point[1] > 0.0
    assert result.point[22] > 0.0


def test_forecast_garch_raises_below_thirty_return_observations():
    # 20 close prices -> 19 non-NaN log-returns after dropna -- below the
    # >=30 floor forecast_garch needs to fit (fail-loud guard, untested
    # until now).
    returns = np.random.default_rng(0).normal(0.0, 0.01, 20)

    with pytest.raises(ValueError):
        forecast_garch(_close_train_df(returns))


def test_forecast_garch_mean_reverts_down_after_a_large_shock():
    # A large shock immediately before the cutoff pushes the model's OWN
    # filtered conditional variance at the origin well ABOVE its estimated
    # long-run level -- the textbook GARCH forecast path then mean-reverts
    # DOWNWARD over the horizon. This direction is a consequence of the
    # forced shock's magnitude (8 std devs), not a hardcoded seed-dependent
    # guess -- see the companion "tiny shock" test below for the mirror case.
    returns, sigma2 = _simulate_garch11(1500, _GARCH_OMEGA, _GARCH_ALPHA, _GARCH_BETA, seed=0)
    returns[-1] = 8.0 * math.sqrt(sigma2[-1])

    result = forecast_garch(_close_train_df(returns))

    avg_variance_over_horizon = result.point[22] / 22.0
    assert avg_variance_over_horizon < result.point[1]


def test_forecast_garch_mean_reverts_up_after_a_tiny_shock():
    # Mirror case: a near-zero return immediately before the cutoff pushes
    # the filtered conditional variance well BELOW the long-run level ->
    # the forecast path mean-reverts UPWARD.
    returns, sigma2 = _simulate_garch11(1500, _GARCH_OMEGA, _GARCH_ALPHA, _GARCH_BETA, seed=0)
    returns[-1] = 0.001 * math.sqrt(sigma2[-1])

    result = forecast_garch(_close_train_df(returns))

    avg_variance_over_horizon = result.point[22] / 22.0
    assert avg_variance_over_horizon > result.point[1]


def test_forecast_garch_percent_scaling_avoids_data_scale_warning():
    # `arch`'s own optimizer warns (DataScaleWarning) when fit on raw ~1%
    # daily-vol returns -- the well-known gotcha this project's ``* 100``
    # scaling (module docstring) exists to avoid. Pin that the scaling is
    # actually applied: no DataScaleWarning on a realistic-scale series.
    returns, _ = _simulate_garch11(1500, _GARCH_OMEGA, _GARCH_ALPHA, _GARCH_BETA, seed=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecast_garch(_close_train_df(returns))

    assert not any(issubclass(w.category, DataScaleWarning) for w in caught)


# --- LightGBM feature builder --------------------------------------------
#
# `lgbm_features(df, o)` reuses `_har_feature_frame` verbatim for its three
# RV lags (module docstring: "LightGBM sees the same information HAR-RV
# does, plus these extras") -- no parallel reimplementation of that window
# logic. Unlike `har_features`, it does NOT raise on insufficient history:
# it is a thin positional accessor into a fully vectorized frame, NaN
# wherever a lag/gap window isn't realized yet (same graceful convention as
# `_har_feature_frame`/`data.multi_day_rv`) -- this is what makes it
# possible to hand-verify on a tiny 10-row frame at all (a 10-row frame can
# never satisfy `har_features`'s own `origin_idx >= 21` floor).


def test_lgbm_features_hand_computed_on_tiny_ten_row_frame():
    # Explicit OHLC values, hand-picked so range_pct/overnight_gap are
    # exactly computable; explicit independent rv values (same convention
    # as `_RV_30` -- rv is a column already computed, not derived from
    # OHLC). Dates chosen to land exactly on a real month-end (2021-01-31)
    # for the calendar-feature check below.
    idx = pd.date_range("2021-01-27", periods=10, freq="D")
    open_ = pd.Series([100, 100, 101, 102, 103, 110, 111, 112, 113, 114], index=idx, dtype=float)
    high = pd.Series([101, 102, 103, 104, 105, 112, 113, 114, 115, 116], index=idx, dtype=float)
    low = pd.Series([99, 100, 100, 101, 102, 108, 109, 110, 111, 112], index=idx, dtype=float)
    close = pd.Series([100, 101, 102, 103, 104, 111, 112, 113, 114, 115], index=idx, dtype=float)
    rv = pd.Series([0.0001 * (i + 1) for i in range(10)], index=idx)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "rv": rv})

    # o=5 (2021-02-01): lag_1/lag_5 fully realized, lag_22 NOT (only 6 rows
    # of history exist) -- NaN, not a raise.
    o = 5
    features = lgbm_features(df, o)
    assert features["lag_1"] == pytest.approx(math.log(rv.iloc[5]))
    assert features["lag_5"] == pytest.approx(math.log(rv.iloc[1:6].mean()))
    assert math.isnan(features["lag_22"])
    assert features["range_pct"] == pytest.approx((112.0 - 108.0) / 111.0)
    assert features["overnight_gap"] == pytest.approx(abs(110.0 - 104.0) / 104.0)
    assert features["day_of_week"] == float(idx[5].dayofweek)
    assert features["month"] == float(idx[5].month)
    assert features["is_month_end"] == 0.0

    # o=4 (2021-01-31) IS a real calendar month-end.
    assert lgbm_features(df, 4)["is_month_end"] == 1.0

    # o=0: no prior close at all -> overnight_gap is NaN, not a crash.
    assert math.isnan(lgbm_features(df, 0)["overnight_gap"])
    # range_pct never needs history -- defined even at o=0.
    assert lgbm_features(df, 0)["range_pct"] == pytest.approx((101.0 - 99.0) / 100.0)


def test_lgbm_features_rv_lags_match_har_feature_frame_exactly():
    # Locks the reuse contract: lgbm_features's three RV lags are IDENTICAL
    # to _har_feature_frame's output, not independently recomputed --
    # guards a future refactor from silently letting the two diverge.
    rv = pd.Series([0.0001 * (1.0 + 0.05 * math.sin(i / 3.0)) for i in range(60)])
    df = _ohlcv_rv_df(rv)
    har_frame = _har_feature_frame(rv)

    for o in (21, 22, 40, 59):
        features = lgbm_features(df, o)
        assert features["lag_1"] == pytest.approx(har_frame["log_d"].iloc[o])
        assert features["lag_5"] == pytest.approx(har_frame["log_w"].iloc[o])
        assert features["lag_22"] == pytest.approx(har_frame["log_m"].iloc[o])


def test_lgbm_calendar_features_depend_only_on_origin_timestamp():
    # Structurally leak-proof -- computed purely from the origin's own
    # timestamp, nothing to poison (plan M3.3's non-poison calendar check).
    rv = pd.Series([0.0001 + 0.000001 * i for i in range(60)])
    df = _ohlcv_rv_df(rv)
    o = 40
    ts = df.index[o]

    features = lgbm_features(df, o)

    assert features["day_of_week"] == float(ts.dayofweek)
    assert features["month"] == float(ts.month)
    assert features["is_month_end"] == float(ts.is_month_end)


# --- forecast_lgbm --------------------------------------------------------


def test_forecast_lgbm_returns_finite_positive_both_horizons():
    intercept, beta_d, beta_w, beta_m = _HAR_GENERATING_COEFFICIENTS
    rv = _simulate_har_process(
        n=300, intercept=intercept, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        noise_std=0.02, seed=3,
    )
    df = _ohlcv_rv_df(rv)

    result = forecast_lgbm(df)

    assert isinstance(result, ForecastResult)
    assert math.isfinite(result.point[1])
    assert math.isfinite(result.point[22])
    assert result.point[1] > 0.0
    assert result.point[22] > 0.0


def test_forecast_lgbm_smoke_correlates_with_known_har_generating_signal():
    # Same generating process M2.2's HAR-RV OLS-recovery test uses. The
    # noiseless one-step-ahead expectation at the cutoff is exactly known
    # from the generating coefficients themselves (no need to peek at
    # forecast_har_rv's own fitted output -- that would just be comparing
    # two learned estimates to each other). LightGBM sees the identical
    # lag_1/5/22 information HAR-RV's OLS does (plus extras); it should
    # land in the same ballpark as this noiseless expectation, not an exact
    # match (GBDT on one noisy draw != closed-form OLS) -- a generous
    # factor-of-3 band both ways is loose enough not to be flaky while
    # still catching a genuinely broken pipeline (wrong target column,
    # forgetting to exponentiate the log-scale output, etc.).
    intercept, beta_d, beta_w, beta_m = _HAR_GENERATING_COEFFICIENTS
    rv = _simulate_har_process(
        n=300, intercept=intercept, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        noise_std=0.02, seed=3,
    )
    df = _ohlcv_rv_df(rv)
    cutoff_idx = len(rv) - 1
    log_d, log_w, log_m = har_features(rv, cutoff_idx)
    expected_rv_h1 = math.exp(intercept + beta_d * log_d + beta_w * log_w + beta_m * log_m)

    result = forecast_lgbm(df)

    assert expected_rv_h1 / 3.0 < result.point[1] < expected_rv_h1 * 3.0


def test_lgbm_registered_in_model_registry():
    assert MODEL_REGISTRY["lgbm"] is forecast_lgbm


def test_forecast_lgbm_raises_below_min_training_rows():
    # n=40: h=1's fully-realized origin count (21<=o<=cutoff-1 -> 18 rows)
    # is already below the >=30 floor forecast_lgbm needs before fitting
    # (fail-loud guard, untested until now).
    rv = pd.Series([0.0001 + 0.000001 * i for i in range(40)])
    df = _ohlcv_rv_df(rv)

    with pytest.raises(ValueError):
        forecast_lgbm(df)


# --- Anti-leakage sentinel: lgbm_features / targets ----------------------
#
# House sentinel style (timeseries_showdown/tests/test_baselines.py's
# test_training_frame_features_never_touch_the_targets_own_or_future_value):
# poison one specific day, hand-verify the EXACT boundary at which the
# poison legitimately enters a target vs. must stay out of the features,
# rather than a vague "no leak somewhere" assertion.


def test_lgbm_features_and_targets_poison_value_boundaries():
    n = 100
    poison_day = 60
    clean_rv = pd.Series([0.0001 + 0.000001 * i for i in range(n)])
    rv = clean_rv.copy()
    rv.iloc[poison_day] = 1e9  # sentinel: leaks if any feature picks it up early
    df = _ohlcv_rv_df(rv)

    target_h1 = np.log(rv.shift(-1))
    target_h22 = np.log(multi_day_rv(rv, 22).shift(-22))

    def assert_features_exclude_poison(o: int) -> None:
        features = lgbm_features(df, o)
        assert features["lag_1"] == pytest.approx(math.log(clean_rv.iloc[o]))
        assert features["lag_5"] == pytest.approx(math.log(clean_rv.iloc[o - 4 : o + 1].mean()))
        assert features["lag_22"] == pytest.approx(math.log(clean_rv.iloc[o - 21 : o + 1].mean()))
        # Literal-passthrough check, on top of the exact-window checks above
        # (which already close the "diluted leak through a rolling mean"
        # blind spot a literal check alone would miss).
        assert 1e9 not in features.values()
        assert math.log(1e9) not in features.values()

    # o well beyond both the 22-day feature and target windows: neither
    # target reaches the poison at all -- coarse control.
    o = poison_day - 30
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))
    assert_features_exclude_poison(o)

    # Sharpest "neither" control: o+23 is exactly one day PAST the h=22
    # target's far edge (o+22) -- poison_day sits exactly one day outside
    # the h=22 target window here, the tightest margin at which the poison
    # must still be absent from everything (features AND both targets).
    o = poison_day - 23
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))
    assert_features_exclude_poison(o)

    # o = poison_day - 22: the h=22 target's far edge reaches EXACTLY
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

    # o = poison_day itself: lag_1 is DEFINED as the origin's own freshest
    # value -- legitimately equals the poison here. The origin's own h=1/
    # h=22 targets (strictly forward-looking) must NOT contain it yet.
    o = poison_day
    features = lgbm_features(df, o)
    assert features["lag_1"] == pytest.approx(math.log(1e9))
    assert target_h1.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1]))
    assert target_h22.iloc[o] == pytest.approx(math.log(clean_rv.iloc[o + 1 : o + 23].sum()))


def test_lgbm_ohlc_derived_features_poison_value_boundaries():
    # The RV poison suite above poisons only the `rv` series -- it is
    # structurally BLIND to a leak confined to range_pct/overnight_gap,
    # which are derived purely from OHLC and never touch `rv` at all (e.g.
    # a `range_pct = (high.shift(-1) - low) / close` bug would sail through
    # every assertion above uncaught). This poisons the OHLC columns
    # themselves and hand-verifies the same boundary precision for each of
    # the three terms feeding the two OHLC-derived features: `high`
    # (range_pct's numerator, zero temporal offset), `open` (overnight_gap's
    # O_o term, zero temporal offset), and `close` -- the one column with
    # TWO distinct legitimate touchpoints: range_pct's OWN-origin
    # denominator, and overnight_gap's C_{o-1} term one day LATER.
    n = 100
    poison_day = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    clean_close = 100.0 + np.arange(n) * 0.01
    clean_open = clean_close - 0.05
    clean_high = clean_close + 0.1
    clean_low = clean_close - 0.1
    rv = np.full(n, 0.0001)

    def make_df(*, poison_high=False, poison_open=False, poison_close=False) -> pd.DataFrame:
        high, open_, close = clean_high.copy(), clean_open.copy(), clean_close.copy()
        if poison_high:
            high[poison_day] = 1e9
        if poison_open:
            open_[poison_day] = 1e9
        if poison_close:
            close[poison_day] = 1e9
        return pd.DataFrame(
            {"open": open_, "high": high, "low": clean_low, "close": close, "rv": rv}, index=idx
        )

    def clean_range_pct(o: int) -> float:
        return (clean_high[o] - clean_low[o]) / clean_close[o]

    def clean_overnight_gap(o: int) -> float:
        return abs(clean_open[o] - clean_close[o - 1]) / clean_close[o - 1]

    control_origins = (poison_day - 30, poison_day - 1, poison_day + 1, poison_day + 30)

    # --- high poisoned: range_pct's H_o term, no shift at all ------------
    df_high = make_df(poison_high=True)
    expected = (1e9 - clean_low[poison_day]) / clean_close[poison_day]
    assert lgbm_features(df_high, poison_day)["range_pct"] == pytest.approx(expected)
    for o in control_origins:
        assert lgbm_features(df_high, o)["range_pct"] == pytest.approx(clean_range_pct(o))
        assert lgbm_features(df_high, o)["overnight_gap"] == pytest.approx(clean_overnight_gap(o))

    # --- open poisoned: overnight_gap's O_o term, no shift at all --------
    df_open = make_df(poison_open=True)
    expected = abs(1e9 - clean_close[poison_day - 1]) / clean_close[poison_day - 1]
    assert lgbm_features(df_open, poison_day)["overnight_gap"] == pytest.approx(expected)
    for o in control_origins:
        assert lgbm_features(df_open, o)["overnight_gap"] == pytest.approx(clean_overnight_gap(o))
        assert lgbm_features(df_open, o)["range_pct"] == pytest.approx(clean_range_pct(o))

    # --- close poisoned: two distinct legitimate touchpoints -------------
    df_close = make_df(poison_close=True)
    expected_range_pct = (clean_high[poison_day] - clean_low[poison_day]) / 1e9
    assert lgbm_features(df_close, poison_day)["range_pct"] == pytest.approx(expected_range_pct)
    expected_gap_next_day = abs(clean_open[poison_day + 1] - 1e9) / 1e9
    assert lgbm_features(df_close, poison_day + 1)["overnight_gap"] == pytest.approx(
        expected_gap_next_day
    )
    # overnight_gap AT poison_day itself uses C_{poison_day-1} (clean,
    # unrelated to the poison) -- must be unaffected by it.
    assert lgbm_features(df_close, poison_day)["overnight_gap"] == pytest.approx(
        clean_overnight_gap(poison_day)
    )
    for o in (poison_day - 30, poison_day - 1, poison_day + 2, poison_day + 30):
        assert lgbm_features(df_close, o)["range_pct"] == pytest.approx(clean_range_pct(o))
        assert lgbm_features(df_close, o)["overnight_gap"] == pytest.approx(clean_overnight_gap(o))


def test_lgbm_calendar_features_are_structurally_leak_proof():
    # Not a poison test -- there is nothing to poison. day_of_week/month/
    # is_month_end are computed purely from the origin's own timestamp
    # (plan M3.3's "one-line assertion, not a poison test" item).
    rv = pd.Series([0.0001 + 0.000001 * i for i in range(80)])
    df = _ohlcv_rv_df(rv)

    for o in (21, 50, 79):
        ts = df.index[o]
        features = lgbm_features(df, o)
        assert features["day_of_week"] == float(ts.dayofweek)
        assert features["month"] == float(ts.month)
        assert features["is_month_end"] == float(ts.is_month_end)


# --- Chronos-2 zero-shot (M4) ---------------------------------------------
#
# Zero-shot: no fitting, so there is no "known generating process" recovery
# test like HAR-RV/LightGBM have -- correctness here means "the plumbing
# (context capping, log-RV in, median out, h=1/h=22 derivation) is right,"
# not "the forecast is accurate" (that question is M5/M6's backtest job).


def _synthetic_log_rv_series(n: int = 250, seed: int = 7) -> pd.Series:
    """A HAR-like, always-positive RV path in LOG space -- reuses
    `_simulate_har_process` (already available in this file) purely because
    it is a convenient realistic-shaped fixture, not because Chronos-2's
    zero-shot forecast has any notion of "the correct model": unlike
    HAR-RV/LightGBM, there is nothing to recover here.
    """
    intercept, beta_d, beta_w, beta_m = _HAR_GENERATING_COEFFICIENTS
    rv = _simulate_har_process(
        n=n, intercept=intercept, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        noise_std=0.02, seed=seed,
    )
    return np.log(rv)


def test_capped_context_truncates_to_last_n_points():
    series = pd.Series(np.arange(3000, dtype=float))
    capped = _capped_context(series, context_cap=2048)

    assert len(capped) == 2048
    np.testing.assert_allclose(capped, series.to_numpy()[-2048:])


def test_capped_context_is_noop_when_series_shorter_than_cap():
    series = pd.Series(np.arange(100, dtype=float))
    capped = _capped_context(series, context_cap=2048)

    assert len(capped) == 100
    np.testing.assert_allclose(capped, series.to_numpy())


def test_capped_context_tail_boundary_is_exact():
    # Poison-style boundary check (house style, adapted -- pure array
    # slicing, no model call): the value exactly ONE position before the
    # cap's start must be excluded; the value exactly AT the cap's start
    # must be the capped context's first element. Pins "last N", not some
    # other off-by-one slice.
    n, cap = 50, 20
    series = pd.Series(np.arange(n, dtype=float))
    series.iloc[n - cap - 1] = 1e9  # one before the boundary -> must be excluded
    series.iloc[n - cap] = -1e9  # exactly at the boundary -> included, first element

    capped = _capped_context(series, context_cap=cap)

    assert len(capped) == cap
    assert capped[0] == -1e9
    assert 1e9 not in capped


def test_forecast_chronos2_returns_median_array_of_correct_length():
    result = forecast_chronos2(_synthetic_log_rv_series(), horizon=22)

    assert isinstance(result, np.ndarray)
    assert result.shape == (22,)
    assert np.isfinite(result).all()


def test_forecast_chronos2_respects_context_cap(monkeypatch):
    # Same interception pattern as
    # `timeseries_showdown.foundation`'s own context-cap test: spy on the
    # pipeline's `predict_quantiles` and check the actual length of the
    # context array it receives, rather than trusting the cap was applied.
    pipe = _get_chronos2_pipeline()
    captured: dict[str, int] = {}
    real_predict_quantiles = pipe.predict_quantiles

    def spy(inputs, prediction_length, quantile_levels):
        captured["len"] = len(inputs[0])
        return real_predict_quantiles(
            inputs=inputs, prediction_length=prediction_length, quantile_levels=quantile_levels
        )

    monkeypatch.setattr(pipe, "predict_quantiles", spy)

    long_series = np.log(pd.Series([0.0001] * 400))
    forecast_chronos2(long_series, horizon=22, context_cap=300)

    assert captured["len"] == 300


def test_forecast_chronos2_is_deterministic_across_repeated_calls():
    # Ground rule: verify determinism, don't assume it. Chronos-2's quantile
    # head is not sampling-based, so two calls on the identical input should
    # be bit-for-bit identical with no seed of our own to set.
    series = _synthetic_log_rv_series()

    first = forecast_chronos2(series, horizon=22)
    second = forecast_chronos2(series, horizon=22)

    np.testing.assert_array_equal(first, second)
