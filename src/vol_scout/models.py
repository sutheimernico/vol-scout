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
import torch
from arch import arch_model
from chronos import BaseChronosPipeline
from lightgbm import LGBMRegressor

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


def _log_direct_target(rv: pd.Series, horizon: int) -> pd.Series:
    """`log(sum(RV_{o+1..o+horizon}))` read AT `o`, for any `horizon>=1` --
    `multi_day_rv(rv, horizon).shift(-horizon)` is that sum (h=1 falls out
    of the same expression, since a rolling sum over 1 day is the day
    itself). Module docstring's h=22 target definition, ONE implementation
    shared by HAR-RV's OLS design matrix (`har_rv_coefficients`) and
    LightGBM's training target (`_lgbm_design_matrix`) -- both "direct"
    models must provably predict the identical thing, not two independently
    written expressions that could silently drift apart.
    """
    return np.log(multi_day_rv(rv, horizon).shift(-horizon))


def har_rv_coefficients(rv: pd.Series, horizon: int) -> np.ndarray:
    """OLS fit (`numpy.linalg.lstsq`) of `log(sum(RV_{o+1..o+horizon}))` on
    `[1, log(RV_o), log(mean(RV_{o-4..o})), log(mean(RV_{o-21..o}))]`, over
    every origin `o` in `rv` where both sides are fully realized (features
    need a full monthly window; the target needs `horizon` more days of
    history after `o`).

    Returns `[intercept, beta_daily, beta_weekly, beta_monthly]`. Exposed
    directly (not only via `forecast_har_rv`) so a test can verify the OLS
    recovers known coefficients from a synthetic data-generating process,
    independently of the final point-forecast wiring.
    """
    features = _har_feature_frame(rv)
    target = _log_direct_target(rv, horizon)
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


# --- LightGBM contender ----------------------------------------------------
#
# Feature-engineered ML contender: two separately-fit `LGBMRegressor`
# models per origin (module docstring's "direct" convention, same
# eligibility rule as HAR-RV -- `_log_direct_target` is the one shared
# target definition both models fit against). Fixed, modest hyperparameters
# (no tuning, no per-origin search -- ground rule); `random_state`,
# `n_jobs=1`, and `deterministic=True` together make one call reproducible
# bit-for-bit under a fixed seed.
_LGBM_PARAMS: dict = {
    "n_estimators": 50,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 5,
    "random_state": 0,
    "n_jobs": 1,
    "deterministic": True,
    "verbosity": -1,
}
# Arbitrary but generous floor: LightGBM's own `min_child_samples=5` already
# needs several times that many rows to grow a non-trivial tree at all --
# this guards against silently fitting on a near-empty design matrix after
# `dropna()`, same fail-loud-on-insufficient-history spirit as HAR-RV's
# `>=4` OLS-parameter check.
_LGBM_MIN_TRAINING_ROWS = 30


def _lgbm_design_matrix(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Feature/target frame for one horizon's `LGBMRegressor` fit:
    `_lgbm_feature_frame` columns plus `_log_direct_target(rv, horizon)` --
    the same target expression `har_rv_coefficients` fits against, so
    HAR-RV and LightGBM are provably predicting the identical thing. Rows
    with any NaN (insufficient lag/gap/target history) are dropped, same
    convention as `har_rv_coefficients`'s design matrix.
    """
    features = _lgbm_feature_frame(train_df)
    target = _log_direct_target(train_df["rv"], horizon)
    return features.assign(target=target).dropna()


def forecast_lgbm(train_df: pd.DataFrame) -> ForecastResult:
    """Feature-engineered LightGBM contender: two separately-fit
    `LGBMRegressor` models (h=1, h=22 -- module docstring's "direct"
    convention), applied once to the cutoff origin (`train_df`'s last row)
    and exponentiated back to variance scale.
    """
    start = time.perf_counter()
    cutoff_features = _lgbm_feature_frame(train_df)[_LGBM_FEATURE_COLUMNS].iloc[[-1]]

    point: dict[int, float] = {}
    for horizon in (1, 22):
        design = _lgbm_design_matrix(train_df, horizon)
        if len(design) < _LGBM_MIN_TRAINING_ROWS:
            raise ValueError(
                f"forecast_lgbm needs >={_LGBM_MIN_TRAINING_ROWS} fully-realized "
                f"training origins for horizon={horizon}, got {len(design)}"
            )
        model = LGBMRegressor(**_LGBM_PARAMS)
        model.fit(design[_LGBM_FEATURE_COLUMNS], design["target"])
        pred_log = model.predict(cutoff_features)[0]
        point[horizon] = math.exp(float(pred_log))

    elapsed = time.perf_counter() - start
    return ForecastResult(point=point, fit_predict_seconds=elapsed)


# --- Chronos-2 zero-shot ---------------------------------------------------
#
# Zero-shot foundation-model contender (plan M4): no fitting at all, just a
# forward pass through a pretrained model. Fed `log(RV)` (module docstring's
# log-RV convention) -- consistent with HAR-RV/LightGBM, unlike GARCH's
# structural exception.
#
# CPU feasibility (verified during M4.1, on this machine): model load
# ~1s (weights already cached locally, no download), one
# `predict_quantiles(prediction_length=22)` call on a ~2000-point context
# ~0.2-0.4s -- comfortably under the plan's ~2s flag threshold, so no
# compute valve (e.g. coarsening `STEP_TRADING_DAYS` for this model) is
# needed for the M5 backtest's ~4000+ calls.
#
# Determinism (verified, not assumed): two calls on the identical context
# return bit-for-bit identical quantiles -- Chronos-2's quantile head is not
# sampling-based, so there is no seed of our own to set.
CHRONOS2_MODEL_ID = "amazon/chronos-2"

# ~2048 trading days (~8 years) of daily history -- not a hard model limit,
# just a documented cap that keeps each forward pass fast and gives the
# model several volatility regimes of history without materializing a
# longer tensor than needed. Same constant value
# `timeseries_showdown.foundation.CONTEXT_CAP` uses, for the same reason.
CONTEXT_CAP = 2048

_chronos2_pipeline: BaseChronosPipeline | None = None


def _get_chronos2_pipeline() -> BaseChronosPipeline:
    """Lazily load and cache the Chronos-2 pipeline (module-level singleton).

    Lazy so importing this module never forces a model load -- tests that
    only exercise pure helpers (e.g. `_capped_context`) stay fast, and the
    M5 backtest script pays the ~1s load cost once per process, not once
    per origin. Weights are pre-cached locally (`~/.cache/huggingface`); no
    network call at test/run time.
    """
    global _chronos2_pipeline
    if _chronos2_pipeline is None:
        _chronos2_pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS2_MODEL_ID, device_map="cpu", torch_dtype=torch.float32
        )
    return _chronos2_pipeline


def _capped_context(series: pd.Series, context_cap: int) -> np.ndarray:
    """Last `context_cap` observations of `series` (or all of it, if shorter).

    Generic array-slicing helper, no vol-scout-specific logic -- same
    contract as `timeseries_showdown.foundation._capped_context`.
    """
    values = np.asarray(series, dtype=float)
    if len(values) > context_cap:
        return values[-context_cap:]
    return values


def forecast_chronos2(
    log_rv_series: pd.Series, horizon: int = 22, context_cap: int = CONTEXT_CAP
) -> np.ndarray:
    """Zero-shot Chronos-2 median (q50) forecast, on the LOG-RV scale.

    `log_rv_series` is already `log(RV)` -- the caller's responsibility
    (module docstring's log-RV convention: Chronos-2 receives log-RV, not
    raw RV, same as HAR-RV/LightGBM). Returns the `horizon`-length q50
    (median) path, still in LOG space -- callers (`_forecast_chronos2_
    adapter`) exponentiate back to variance scale.

    One `predict_quantiles(..., prediction_length=horizon)` call: Chronos-2's
    own `model_prediction_length` is 1024 (verified in the sibling
    `timeseries_showdown` project), comfortably above `horizon=22` -- no
    library-internal autoregressive chunking, unlike Chronos-Bolt-small.
    """
    pipe = _get_chronos2_pipeline()
    context = _capped_context(log_rv_series, context_cap).astype(np.float32)

    quantiles, _mean = pipe.predict_quantiles(
        inputs=[context], prediction_length=horizon, quantile_levels=[0.5]
    )
    # quantiles[0]: the single series in the batch, shape
    # (n_variates=1, horizon, n_quantiles=1); [0][0] drops the (always-1,
    # univariate) variate dim -> (horizon, 1); [:, 0] drops the
    # single-quantile dim -> (horizon,).
    return quantiles[0][0].numpy()[:, 0].astype(float)


MODEL_REGISTRY: dict[str, Callable[[pd.DataFrame], ForecastResult]] = {
    "random_walk": forecast_random_walk,
    "har_rv": forecast_har_rv,
    "garch": forecast_garch,
    "lgbm": forecast_lgbm,
}
