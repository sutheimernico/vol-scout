"""Run Chronos-2 alone over the full rolling-origin backtest, appending
resumable rows to `results/forecasts.csv` (plan M4.2's deferred item --
`_forecast_chronos2_adapter` was implemented and registered in
`MODEL_REGISTRY` in M4.2 itself, but this script was blocked on Task M5.1's
`backtest.make_origins`/`run_all_models` not existing yet; both exist now).

Split from `scripts/run_backtest.py` into its own script -- a CPU-bound
foundation-model run over ~4000+ (ticker, origin) calls benefits from being
launched/monitored/resumed independently of the four much cheaper classical/
ML models, matching the sibling `timeseries_showdown` project's own split
for its slower model.

Same resumability contract as `run_backtest.py`: `backtest.run_and_append_
resumable` skips any `(model, ticker, origin)` triple already present in
`results/forecasts.csv`, so this is safe to kill and re-run, and safe to run
before or after `run_backtest.py` -- both append to the same file, and the
`(model, ticker, origin)` skip key means neither run can ever clobber or
duplicate the other's rows regardless of interleaving.
"""

from __future__ import annotations

from pathlib import Path

from vol_scout.backtest import (
    OOS_START,
    STEP_TRADING_DAYS,
    make_origins,
    run_and_append_resumable,
)
from vol_scout.data import TICKERS, load_ohlcv, parkinson_rv
from vol_scout.models import MODEL_REGISTRY

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FORECASTS_PATH = Path(__file__).resolve().parent.parent / "results" / "forecasts.csv"
MODELS_TO_RUN = {"chronos2": MODEL_REGISTRY["chronos2"]}


def main() -> None:
    total_rows = 0
    total_failures = 0

    for ticker in TICKERS:
        df = load_ohlcv(ticker, DATA_DIR)
        df["rv"] = parkinson_rv(df)
        origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)

        summary = run_and_append_resumable(df, ticker, origins, MODELS_TO_RUN, FORECASTS_PATH)
        total_rows += summary["rows_written"]
        total_failures += summary["failures"]
        print(
            f"{ticker}: {len(origins)} origins, "
            f"{summary['rows_written']} new rows, {summary['failures']} failures"
        )

    print(f"done: {total_rows} new rows written, {total_failures} total failures")


if __name__ == "__main__":
    main()
