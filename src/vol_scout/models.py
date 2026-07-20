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

import time
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


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


MODEL_REGISTRY: dict[str, Callable[[pd.DataFrame], ForecastResult]] = {
    "random_walk": forecast_random_walk,
}
