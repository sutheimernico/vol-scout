"""Volatility-forecasting contenders: `MODEL_REGISTRY`, one callable per model.

**Contract.** Every model function has the signature `(train_df: pd.DataFrame)
-> ForecastResult`. `train_df` is the OHLCV frame (`data.load_ohlcv`'s columns
-- `open, high, low, close, volume`, `pd.DatetimeIndex`) with one additional
column, `rv` -- this project's chosen daily realized-variance proxy (Parkinson,
primary; see `data.py`) -- already computed and sliced to everything known as
of the forecast origin (`df.loc[:origin]`, mirroring
`timeseries_showdown.backtest.BacktestWindow.train_slice`). The origin itself
is `train_df`'s last row. One call returns **both** horizons -- see
`ForecastResult` below -- this is simpler than a per-horizon call because no
model here needs quantiles or covariates (plan §6).

**Scale (binding, do not drift):** every proxy, every model's `point` output,
and every downstream metric operates on the **daily-variance scale**
(squared-return units) -- never volatility (std-dev) and never annualized.

**log-RV vs. GARCH's structural exception (plan §4):** HAR-RV, LightGBM, and
Chronos-2 all fit/forecast `log(RV)` and exponentiate back to variance scale
before returning `ForecastResult.point` -- "log-HAR", a standard variant
(Corsi 2009's own follow-on literature): RV is heavy right-tailed, log-space
is better-behaved for an OLS fit, a GBDT target, and a foundation model's
input distribution alike. **GARCH is the one exception, structurally, not by
choice**: it is fit directly on daily log-returns (`data.log_return`, not the
`rv` column at all) and its native output *is already* a conditional-variance
forecast -- there is no "log-GARCH-RV" equivalent to reach for. Random-walk
stays on the raw scale too, deliberately -- it is the simplest possible
baseline, no transform of any kind.

**h=22 target definition (binding, plan §5):** the h=22 ground truth is
`sum(RV_{o+1 .. o+22})` -- the sum of the 22 daily RV-proxy values strictly
after the origin `o` -- never a proxy formula re-applied to a 22-day OHLC bar.
Every model's h=22 forecast targets this same sum, but each model gets there
differently:

- **random-walk** -- `22 * RV_o`: today's daily-variance level held constant
  and summed over the horizon (the simplest non-trivial extension of
  "yesterday's value" to a multi-day sum).
- **HAR-RV / LightGBM** -- direct: two separately-fit models per origin (one
  targeting `log(RV_{o+1})`, one targeting `log(sum(RV_{o+1..o+22}))`) --
  avoids compounding a 1-day model's own forecast error 22 times.
- **GARCH** -- iterated aggregation: one `arch_model(...).forecast(horizon=22)`
  call gives the analytic 22-step-ahead variance path in a single shot (`arch`
  runs the GARCH(1,1) recursion internally); h=1 is that path's first step,
  h=22 is the sum of all 22 steps.
- **Chronos-2** (M4) -- direct, one `predict_quantiles(prediction_length=22)`
  call; h=1 is the first day's median, h=22 is the sum of all 22 days'
  medians (documented as an approximation in that adapter's own docstring).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from vol_scout.data import log_return, multi_day_rv


@dataclass
class ForecastResult:
    """One model's forecast at one origin, both horizons, variance scale.

    `point`: `{1: rv_hat_h1, 22: rv_hat_h22}` -- daily-variance-scale point
    forecasts (module docstring's scale convention; never volatility,
    never annualized).
    `fit_predict_seconds`: wall-clock time for this one call (fit + predict
    together -- every model here refits from scratch per origin, so there is
    no separate "fit" cost to report independently).
    """

    point: dict[int, float]
    fit_predict_seconds: float


def forecast_random_walk(train_df: pd.DataFrame) -> ForecastResult:
    """Naive baseline: the origin's own daily RV, held constant.

    `point[1] = RV_o`; `point[22] = 22 * RV_o` -- see module docstring's h=22
    section for why this (not a re-derived proxy) is the correct random-walk
    extension to a 22-day sum. No fitting at all -- the floor every other
    model in `MODEL_REGISTRY` must beat.
    """
    start = time.perf_counter()
    rv_o = float(train_df["rv"].iloc[-1])
    elapsed = time.perf_counter() - start
    return ForecastResult(point={1: rv_o, 22: 22.0 * rv_o}, fit_predict_seconds=elapsed)


# --- HAR-RV ------------------------------------------------------------
#
# Hand-implemented on purpose (plan: "the learning core") -- OLS via
# `numpy.linalg.lstsq`, no `statsmodels`. Corsi (2009)'s three lags: daily
# (the origin's own RV), weekly (trailing 5-day mean including the origin),
# monthly (trailing 22-day mean including the origin) -- all backward-
# looking, fit on `log(RV)` (module docstring's log-RV convention),
# exponentiated back to variance scale in `forecast_har_rv`.


def har_features(rv: pd.Series, origin_idx: int) -> tuple[float, float, float]:
    """`(log(RV_o), log(mean(RV_{o-4..o})), log(mean(RV_{o-21..o})))` at
    position `origin_idx` (0-based, `rv.iloc[origin_idx]`).

    Note this is `log(mean(RV))`, not `mean(log(RV))` -- the mean is taken
    on the raw RV window, then logged once.

    Raises if `origin_idx < 21`: a full monthly window is required, rather
    than silently averaging a partial one and returning a number that means
    something different from every other origin's monthly lag (same
    fail-loud-on-insufficient-history style as
    `timeseries_showdown.baselines._require_rolling_history`).
    """
    if origin_idx < 21:
        raise ValueError(
            f"har_features needs origin_idx >= 21 for a full monthly window, got {origin_idx}"
        )
    daily = rv.iloc[origin_idx]
    weekly = rv.iloc[origin_idx - 4 : origin_idx + 1].mean()
    monthly = rv.iloc[origin_idx - 21 : origin_idx + 1].mean()
    return math.log(daily), math.log(weekly), math.log(monthly)


def _har_feature_frame(rv: pd.Series) -> pd.DataFrame:
    """Vectorized `log(RV_d), log(RV_w), log(RV_m)` for every position in
    `rv`, position-for-position equivalent to `har_features` (see the poison
    boundary test) -- NaN wherever the weekly/monthly trailing window isn't
    full yet, same "full window or NaN" convention as `data.multi_day_rv`.
    Used to build the OLS training design matrix across many origins at
    once, rather than looping `har_features` in Python per origin.
    """
    return pd.DataFrame(
        {
            "log_d": np.log(rv),
            "log_w": np.log(rv.rolling(5).mean()),
            "log_m": np.log(rv.rolling(22).mean()),
        }
    )


def har_rv_coefficients(rv: pd.Series, horizon: int) -> np.ndarray:
    """OLS fit (`numpy.linalg.lstsq`) of `log(sum(RV_{o+1..o+horizon}))` on
    `[1, log(RV_o), log(mean(RV_{o-4..o})), log(mean(RV_{o-21..o}))]`, over
    every origin `o` in `rv` where both sides are fully realized (features
    need a full monthly window; the target needs `horizon` more days of
    history after `o`) -- `multi_day_rv(rv, horizon).shift(-horizon)` is
    `sum(RV_{o+1..o+horizon})` read AT `o` for any `horizon>=1` (h=1 falls
    out of the same expression, since a rolling sum over 1 day is the day
    itself), so h=1 and h=22 share this one implementation.

    Returns `[intercept, beta_daily, beta_weekly, beta_monthly]`. Exposed
    directly (not only via `forecast_har_rv`) so a test can verify the OLS
    recovers known coefficients from a synthetic data-generating process,
    independently of the final point-forecast wiring.
    """
    features = _har_feature_frame(rv)
    target = np.log(multi_day_rv(rv, horizon).shift(-horizon))
    design = features.assign(target=target).dropna()
    if len(design) < 4:
        raise ValueError(
            f"har_rv_coefficients needs >=4 fully-realized origins to fit its 4 "
            f"parameters, got {len(design)}"
        )

    x = np.column_stack([np.ones(len(design)), design["log_d"], design["log_w"], design["log_m"]])
    y = design["target"].to_numpy()
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def forecast_har_rv(train_df: pd.DataFrame) -> ForecastResult:
    """Hand-implemented HAR-RV (Corsi 2009): two separately-fit OLS models
    (h=1, h=22 -- module docstring's "direct" convention) on daily/weekly/
    monthly `log(RV)` lags, applied once to the cutoff origin (`train_df`'s
    last row) and exponentiated back to variance scale.
    """
    start = time.perf_counter()
    rv = train_df["rv"].reset_index(drop=True)
    cutoff_idx = len(rv) - 1

    x_cutoff = np.array([1.0, *har_features(rv, cutoff_idx)])

    point: dict[int, float] = {}
    for horizon in (1, 22):
        coefficients = har_rv_coefficients(rv, horizon)
        point[horizon] = math.exp(float(x_cutoff @ coefficients))

    elapsed = time.perf_counter() - start
    return ForecastResult(point=point, fit_predict_seconds=elapsed)


# --- GARCH(1,1) ----------------------------------------------------------
#
# Via the `arch` package -- the classical benchmark being compared AGAINST,
# not "the learning core" (that's HAR-RV, hand-rolled deliberately, see
# above); `arch` is the standard, actively-maintained Python GARCH
# implementation, and hand-rolling a GARCH MLE would reimplement well-tested
# numerical optimization for no benefit here.
_GARCH_FORECAST_HORIZON = 22


def forecast_garch(train_df: pd.DataFrame) -> ForecastResult:
    """GARCH(1,1), fit on daily log-returns (`data.log_return`, NOT the `rv`
    column -- module docstring's structural exception), converted back to
    daily-variance units.

    **Percent-scaling convention (pin this exactly, it is a classic silent
    factor-100 trap):** `arch`'s own optimizer is numerically unstable on
    raw returns of ~1% daily magnitude and emits a `DataScaleWarning`
    recommending values roughly in `[1, 1000]` -- so we fit on
    `returns * 100` (percent-return units; variance in that space is
    `100**2 = 10_000` times the daily-variance-scale value). `res.forecast
    (horizon=22)` returns its variance path in that SAME percent-return**2
    scale, so before returning anything we divide by `100.0 ** 2` to land
    back in the daily-variance units every other model/proxy in this
    project uses (module docstring's binding scale convention). Getting
    either factor wrong (or only doing one of the two) is exactly the
    "silent factor-100" bug this docstring exists to prevent -- see
    `tests/test_models.py::test_forecast_garch_percent_scaling_avoids_data
    _scale_warning` for the round-trip pin.

    `fc.variance.iloc[-1]` is read positionally (`.to_numpy()`), not by
    column name (`h.01`, `h.02`, ... in the installed `arch` 8.0 -- the
    plan flagged this naming as version-sensitive; position is guaranteed
    order regardless of the exact zero-padding a given version uses).

    h=22 uses the iterated aggregation from the module docstring: one
    `.forecast(horizon=22)` call gives the analytic 22-step-ahead variance
    path in a single shot (`arch` runs the GARCH(1,1) recursion internally,
    no manual recursion needed); `point[1]` is that path's first step,
    `point[22]` is the sum of all 22 steps.
    """
    start = time.perf_counter()
    returns = log_return(train_df).dropna()
    if len(returns) < 30:
        raise ValueError(
            f"forecast_garch needs >=30 return observations to fit, got {len(returns)}"
        )

    am = arch_model(
        returns.to_numpy() * 100.0, mean="Constant", vol="Garch", p=1, q=1, dist="normal"
    )
    res = am.fit(disp="off")
    fc = res.forecast(horizon=_GARCH_FORECAST_HORIZON, reindex=False)
    variance = fc.variance.iloc[-1].to_numpy() / 100.0**2  # percent^2 -> daily-variance scale

    elapsed = time.perf_counter() - start
    return ForecastResult(
        point={1: float(variance[0]), 22: float(variance.sum())},
        fit_predict_seconds=elapsed,
    )


# --- LightGBM feature builder ---------------------------------------------
#
# Reuses `_har_feature_frame` verbatim for its three RV lags -- lets the
# README fairly claim "LightGBM sees the same information HAR-RV does, plus
# these extras" (plan M3.1), not a different, parallel-reimplemented
# information set. `range_pct`/`overnight_gap` are the "extras": distinct
# signals from the RV proxies, raw (not logged). Calendar features
# (`day_of_week`/`month`/`is_month_end`) come purely from the origin's own
# timestamp -- always known, structurally leak-proof, nothing to poison.
_LGBM_FEATURE_COLUMNS = [
    "lag_1",
    "lag_5",
    "lag_22",
    "range_pct",
    "overnight_gap",
    "day_of_week",
    "month",
    "is_month_end",
]


def _lgbm_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature frame for every position in `df`, one row per
    origin, column order matching `_LGBM_FEATURE_COLUMNS`.

    `lag_1`/`lag_5`/`lag_22` are `_har_feature_frame(df["rv"])`'s
    `log_d`/`log_w`/`log_m` columns, reused unchanged (see section header) --
    NaN wherever the weekly/monthly window isn't full yet. `range_pct` needs
    no history (defined from row `o` alone); `overnight_gap` needs a prior
    close (NaN at the first row). Calendar columns are always defined.
    """
    har = _har_feature_frame(df["rv"])
    close = df["close"]
    prev_close = close.shift(1)
    idx = df.index
    return pd.DataFrame(
        {
            "lag_1": har["log_d"],
            "lag_5": har["log_w"],
            "lag_22": har["log_m"],
            "range_pct": (df["high"] - df["low"]) / close,
            "overnight_gap": (df["open"] - prev_close).abs() / prev_close,
            "day_of_week": idx.dayofweek.astype(float),
            "month": idx.month.astype(float),
            "is_month_end": idx.is_month_end.astype(float),
        },
        index=idx,
    )


def lgbm_features(df: pd.DataFrame, origin_idx: int) -> dict[str, float]:
    """Per-origin LightGBM feature vector at position `origin_idx` (0-based):
    a thin positional accessor into `_lgbm_feature_frame` (`.iloc
    [origin_idx].to_dict()`).

    Deliberately does NOT raise on insufficient history like `har_features`
    does -- it is a read of an already-NaN-graceful vectorized frame (same
    convention as `_har_feature_frame`/`data.multi_day_rv`), which is what
    makes it hand-verifiable on a frame far shorter than 22 rows (`lag_22`
    is simply `NaN` there, not an error). `forecast_lgbm`'s training-frame
    `dropna()` -- not this accessor -- is where insufficient-history rows
    get excluded before fitting.
    """
    return _lgbm_feature_frame(df).iloc[origin_idx].to_dict()


MODEL_REGISTRY: dict[str, Callable[[pd.DataFrame], ForecastResult]] = {
    "random_walk": forecast_random_walk,
    "har_rv": forecast_har_rv,
    "garch": forecast_garch,
}
