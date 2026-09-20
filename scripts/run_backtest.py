"""Run the rolling-origin backtest for every model EXCEPT Chronos-2 (random
walk, HAR-RV, GARCH, LightGBM -- the four cheap-to-refit models), over all
ten tickers, appending resumable rows to `results/forecasts.csv`.

Chronos-2 has its own script, `scripts/run_chronos2.py` (plan M4.2's
deferred item, unblocked once `backtest.make_origins`/`run_all_models`
existed) -- splitting the foundation-model run out matches the sibling
`timeseries_showdown` project's own precedent for its slower model.

Safe to re-run: `backtest.run_and_append_resumable` skips any
`(model, ticker, origin)` triple already present in `results/forecasts.csv`,
so a killed run resumes losslessly (plan Ground rule 8) instead of
recomputing everything from scratch.
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
MODELS_TO_RUN = {name: fn for name, fn in MODEL_REGISTRY.items() if name != "chronos2"}


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
