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
- Every headline number traces to a script-regenerable artifact plus
  a DM/HLN verdict.

## Data policy
OHLCV bars are fetched via yfinance by `scripts/fetch_data.py` into `data/`
(10 tickers x ~16 years of daily bars, a few MB). The CSVs are **not
committed** (git-ignored) because Yahoo Finance data should not be
redistributed; a fresh clone runs `uv run python scripts/fetch_data.py` once
and everything downstream (backtest, metrics) works from that local cache.
Note that the fetch returns the data as Yahoo currently serves it, so a later
fetch can differ slightly from an earlier one.

## Status
- [x] Task 0 — Scaffold
- [x] M1 — Data + volatility proxies (`data.py`: fetch+cache, Parkinson/
  Garman-Klass/squared-return proxies, `multi_day_rv`; real OHLCV cached
  locally for all 10 tickers)
- [x] M2 — Benchmarks (`models.py`: `ForecastResult`/`MODEL_REGISTRY`,
  random-walk, hand-implemented HAR-RV via OLS, GARCH(1,1) via `arch`)
- [x] M3 — LightGBM contender (`models.py`: `lgbm_features`/`_lgbm_feature_frame`
  reusing HAR's RV lags + range/overnight-gap/calendar extras,
  `forecast_lgbm` — two direct `LGBMRegressor` fits per origin, registered as
  `"lgbm"`; full anti-leakage poison-boundary suite)
- [ ] M4 — Chronos-2 zero-shot (`models.py`: `forecast_chronos2`/
  `_forecast_chronos2_adapter` on log(RV), registered as `"chronos2"`;
  CPU feasibility verified, ~0.2-0.4s/call; full backtest run deferred,
  needs M5.1's window config first)
- [ ] M5 — Rolling-origin backtest + statistics
- [ ] M6 — Figures, README, Streamlit app

