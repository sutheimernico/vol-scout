"""Rolling-origin backtest: window mechanics (`make_origins`) and per-origin
model orchestration (`run_all_models`).

**Window config (design-decision §2, plan).** `OOS_START`/`STEP_TRADING_DAYS`/
`HORIZONS` are the ONE place `scripts/run_backtest.py`, `scripts/
run_chronos2.py`, and every test that checks the real production config read
from -- so the compute scripts and the tests that pin them can never drift
apart. Origins tile FORWARD from `OOS_START` (`OOS_START`, `OOS_START + 5
trading days`, `OOS_START + 10 trading days`, ...) -- unlike
`timeseries_showdown.make_windows`'s backward-from-the-end tiling. Forward
tiling is deliberate: new trading days arrive continuously, and tiling
forward from a fixed start means every previously-computed origin's position
in the index is unaffected by a later data refresh -- new origins only ever
append at the tail. That is what makes the compute scripts' append-only,
skip-already-computed-`(model, ticker, origin)` resumability actually correct
across repeated `scripts/fetch_data.py` refreshes, not just "usually fine".

**Origin semantics (binding, M3 review ruling).** Every origin `o` returned
by `make_origins` IS itself a trading day present in `df.index`. A forecast
at `o` is made using `train_df = df.loc[:o]` -- INCLUSIVE of `o`'s own row
(the forecast happens after day `o`'s close, so `o`'s own OHLC/RV is fair
game). This is exactly `train_df["rv"].iloc[-1]` in every `models.py`
contender. Getting this one day off in either direction either leaks the
future (train_df extends past `o`) or silently discards the freshest,
highest-information observation (train_df stops one day short) -- see
`test_run_all_models_train_df_includes_the_origin_day_itself` and the
poison-sentinel tests in `tests/test_backtest.py` for the boundary proof.

**Target definition (binding, plan §5), same for every model.** Realized
h=1 is `RV_{o+1}`; realized h=22 is `sum(RV_{o+1..o+22})` -- `target_rv`
below, always on the RAW variance scale (never `log`, unlike several
models' own training targets in `models.py` -- this is ground truth, not a
model's fitting target). An origin is excluded by `make_origins` entirely
(not given a partial target) whenever `o + max(HORIZONS)` trading days would
reach past the end of `df` -- origins are shared across both horizons, so
the exclusion uses the LARGEST horizon even for the h=1 target, which is
always the less demanding one.

**Per-model failure tolerance (binding, plan's M5 ground rules).**
`run_all_models` must never let one model's failure at one origin (e.g.
GARCH non-convergence on some regime) kill the whole backtest. A failing
`(model, origin)` call still produces both horizons' rows -- `rv_pred`/
`fit_predict_seconds` as NaN, a non-empty `error` message -- rather than
raising out of the loop. This is a deliberate, documented one-column
extension of the plan §7 persistence contract's exact column list
(`model, ticker, origin, horizon, rv_true, rv_pred, fit_predict_seconds`):
without recording failures as rows, a model that fails deterministically at
the same origin on every re-run would be retried forever by the resumable
compute scripts' skip-already-computed-origin logic -- a failure IS a
computed result and must be recorded as one to keep resumability lossless.
`error` is `""` on success.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from vol_scout.data import multi_day_rv
from vol_scout.models import MODEL_REGISTRY, ForecastResult

# Production backtest config (design-decision §2). OOS_START is the first
# trading day of 2018 -- the backtest spans Feb 2018's vol spike, the Dec
# 2018 selloff, the 2020 COVID crash, the 2022 hiking cycle, and 2023+.
# STEP_TRADING_DAYS=5 is a weekly re-forecast cadence. HORIZONS is the pair
# of forecast horizons every model in MODEL_REGISTRY is evaluated on.
OOS_START = "2018-01-02"
STEP_TRADING_DAYS = 5
HORIZONS = (1, 22)


def make_origins(
    df: pd.DataFrame,
    oos_start: str | pd.Timestamp = OOS_START,
    step_trading_days: int = STEP_TRADING_DAYS,
) -> list[pd.Timestamp]:
    """Forward-tiled backtest origins over `df`'s trading-day index (module
    docstring: window config + origin semantics).

    The first origin is the first index date at or after `oos_start`. Each
    subsequent origin steps forward by exactly `step_trading_days` trading
    days (index positions, not calendar days). The LAST origin returned is
    the latest whose `origin + max(HORIZONS)` trading days still exists in
    `df` -- an origin needing target data beyond the end of `df` is excluded
    entirely, for either horizon (no partial targets, module docstring).

    Raises `ValueError` if `oos_start` is at or after the end of `df`, or if
    not even the first candidate origin has `max(HORIZONS)` trading days of
    room left before the end of `df`.
    """
    idx = df.index
    oos_start_ts = pd.Timestamp(oos_start)
    start_pos = idx.searchsorted(oos_start_ts)
    if start_pos >= len(idx):
        raise ValueError(
            f"oos_start {oos_start_ts.date()} is at or after the end of the data "
            f"({idx[-1].date() if len(idx) else 'empty'})"
        )

    max_horizon = max(HORIZONS)
    last_valid_pos = len(idx) - 1 - max_horizon
    if last_valid_pos < start_pos:
        raise ValueError(
            f"no origin from {oos_start_ts.date()} has {max_horizon} trading days of "
            "target data still inside the frame -- df ends too soon after oos_start"
        )

    positions = range(start_pos, last_valid_pos + 1, step_trading_days)
    return [idx[p] for p in positions]


def target_rv(rv: pd.Series, horizon: int) -> pd.Series:
    """Realized ground truth at each origin `o`, RAW variance scale:
    `sum(RV_{o+1..o+horizon})` (module docstring's target definition).

    `multi_day_rv(rv, horizon)` (`data.py`) is the low-level trailing
    rolling sum; `.shift(-horizon)` is the forward alignment this module
    owns, mirroring `models._log_direct_target`'s identical shift on the log
    scale -- but this is the never-logged ground truth, not a model's own
    fitting target. h=1 falls out of the same expression (a 1-day rolling
    sum is the day itself). NaN past the end of `rv` (not enough future
    data) -- `make_origins` is what guarantees callers never read a NaN
    here for an origin it actually returned.
    """
    return multi_day_rv(rv, horizon).shift(-horizon)


def run_all_models(
    df: pd.DataFrame,
    origins: list[pd.Timestamp],
    models: dict[str, Callable[[pd.DataFrame], ForecastResult]] = MODEL_REGISTRY,
) -> pd.DataFrame:
    """Run every model in `models` at every origin in `origins`, both
    `HORIZONS`, one long-format row per `(model, origin, horizon)`.

    Matches `results/forecasts.csv`'s persistence contract (plan §7) minus
    the `ticker` column (this function is ticker-agnostic -- the caller
    adds it) plus one `error` column (module docstring's failure-tolerance
    extension).

    `df` must already carry an `rv` column (the caller's job -- e.g.
    `data.parkinson_rv`; see `models.py`'s module docstring for the contract
    every registered model function assumes). For each origin `o`,
    `train_df = df.loc[:o]` (INCLUSIVE, module docstring) is what every
    model function actually receives; `rv_true` is read from `target_rv`
    applied to the FULL `df["rv"]` series, never from `train_df` -- the
    ground truth must see real future rows that `train_df` deliberately
    does not.
    """
    rv = df["rv"]
    targets = {horizon: target_rv(rv, horizon) for horizon in HORIZONS}

    rows: list[dict] = []
    for origin in origins:
        train_df = df.loc[:origin]

        y_true: dict[int, float] = {}
        for horizon in HORIZONS:
            value = targets[horizon].loc[origin]
            if pd.isna(value):
                raise ValueError(
                    f"origin {origin} has no realized h={horizon} target -- "
                    "make_origins should have excluded it; this is a bug, not "
                    "an expected 'missing data' case"
                )
            y_true[horizon] = float(value)

        for model_name, model_fn in models.items():
            try:
                result: ForecastResult | None = model_fn(train_df)
                error = ""
            except Exception as exc:  # noqa: BLE001 - per-model failure tolerance, see module docstring
                result = None
                error = str(exc)

            for horizon in HORIZONS:
                rows.append(
                    {
                        "model": model_name,
                        "origin": origin,
                        "horizon": horizon,
                        "rv_true": y_true[horizon],
                        "rv_pred": result.point[horizon] if result is not None else float("nan"),
                        "fit_predict_seconds": (
                            result.fit_predict_seconds if result is not None else float("nan")
                        ),
                        "error": error,
                    }
                )

    return pd.DataFrame(rows)
