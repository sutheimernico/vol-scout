# vol-scout

Does modern ML / foundation-model forecasting beat **HAR-RV** (the econometric
standard) at forecasting equity volatility? A rolling-origin backtest with proper
statistical testing. No trading strategy, no PnL claim, no alpha claim.

> **Status: work in progress.** Data pipeline, benchmark and ML models, backtest
> mechanics and the evaluation statistics are implemented and tested. The full backtest
> run, figures and a Streamlit demo are still open (see [PROJECT.md](PROJECT.md)), so
> this repository reports **no results yet**.

## Approach

- **Universe**: SPY (main series) plus nine robustness tickers spanning different
  volatility drivers (AAPL, NVDA, JPM, XOM, JNJ, PG, AMZN, CAT, VGK).
- **Target**: daily realized-volatility *proxy* computed from OHLC bars (Parkinson;
  Garman-Klass and squared returns as sensitivity checks). True realized volatility
  needs intraday data, which this project does not use.
- **Models**: random walk, HAR-RV (hand-implemented OLS), GARCH(1,1) (`arch`),
  LightGBM, and Chronos-2 (zero-shot foundation model).
- **Evaluation**: rolling-origin backtest at two horizons; QLIKE (primary), RMSE and
  Mincer-Zarnowitz regressions; Diebold-Mariano test with the Harvey-Leybourne-Newbold
  small-sample correction against HAR-RV, with an explicit "tie" verdict.
- **Leakage discipline**: anti-leakage tests (poisoned future values must not change
  forecasts) cover the feature builders and models.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q            # tests never touch the network
```

## Data

Price data comes from Yahoo Finance via `yfinance` and is **not included** in this
repository (it should not be redistributed). Fetch it once; the script is append-only
and safe to re-run:

```bash
uv run python scripts/fetch_data.py     # writes data/<TICKER>.csv
```

Prices are fetched with `auto_adjust=True`, i.e. back-adjusted for later splits and
dividends, which is a subtle lookahead. The code documents this; it is accepted for a
free, simple data source.

## Running the backtest

All runners append to `results/forecasts.csv` and are resumable (a killed run picks up
where it stopped):

```bash
uv run python scripts/run_backtest.py     # random walk, HAR-RV, GARCH, LightGBM
uv run python scripts/run_chronos2.py     # Chronos-2 (CPU; downloads weights from Hugging Face on first use)
uv run python scripts/compute_metrics.py  # metrics and DM/HLN verdicts from forecasts.csv
```

## Layout

- `src/vol_scout/data.py`: OHLCV fetch and cache, volatility proxies
- `src/vol_scout/models.py`: model registry and forecasters
- `src/vol_scout/backtest.py`: origins, windows, resumable runner
- `src/vol_scout/stats.py`: QLIKE / RMSE / Mincer-Zarnowitz, DM + HLN tests
- `docs/superpowers/`: design spec and implementation plan

## Stack

Python, pandas, NumPy, SciPy, `arch`, LightGBM, scikit-learn, Chronos (PyTorch, CPU),
`yfinance`, pytest, ruff, uv.

## License

MIT, see [LICENSE](LICENSE).
