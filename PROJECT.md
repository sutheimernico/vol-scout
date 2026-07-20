# vol-scout — Project (Source of Truth)

Does modern ML and foundation-model forecasting beat HAR-RV (the econometric
standard) at forecasting equity volatility? Honest rolling-origin backtest,
no trading strategy, no PnL claim — the portfolio's econometrics piece.

Design spec (binding): `docs/superpowers/specs/2026-07-20-vol-scout-design.md`.
Implementation plan: `docs/superpowers/plans/2026-07-20-vol-scout-v1.md`.

## Constraints (from the design spec, non-negotiable)
- No trading strategy, no backtested PnL, no alpha claim.
- No dashboard/service — a small read-only Streamlit demo only.
- No intraday data, no paid data sources, no API keys anywhere.
- Tests never make live network calls (yfinance or HF Hub).
- Every headline number traces to a committed, script-regenerable artifact plus
  a DM/HLN verdict.

## Data policy
OHLCV CSVs fetched via yfinance are cached **and committed** to `data/` (like
`timeseries-showdown` commits `data/hour.csv`) — free, small (10 tickers ×
~16 years of daily bars is a few MB total), and this is what makes the whole
backtest reproducible from a fresh clone without needing network access at all
past the initial fetch.

## Status
- [x] Task 0 — Scaffold
- [x] M1 — Data + volatility proxies (`data.py`: fetch+cache, Parkinson/
  Garman-Klass/squared-return proxies, `multi_day_rv`; real OHLCV committed
  for all 10 tickers)
- [x] M2 — Benchmarks (`models.py`: `ForecastResult`/`MODEL_REGISTRY`,
  random-walk, hand-implemented HAR-RV via OLS, GARCH(1,1) via `arch`)
- [x] M3 — LightGBM contender (`models.py`: `lgbm_features`/`_lgbm_feature_frame`
  reusing HAR's RV lags + range/overnight-gap/calendar extras,
  `forecast_lgbm` — two direct `LGBMRegressor` fits per origin, registered as
  `"lgbm"`; full anti-leakage poison-boundary suite)
- [ ] M4 — Chronos-2 zero-shot
- [ ] M5 — Rolling-origin backtest + statistics
- [ ] M6 — Figures, README, Streamlit app

## Needs Nico
- Git remote / public-visibility decision before any first push.
