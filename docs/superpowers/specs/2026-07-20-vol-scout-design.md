# vol-scout — Design Spec

Date: 2026-07-20 · Status: approved design, pre-plan

## Research question

**Do modern ML and foundation models beat the econometric standard (HAR-RV) at
forecasting equity volatility?** Framed by a one-page contrast chapter: the same
models fail on daily *returns* (near random walk), which motivates volatility as
the right forecastable target. This is the portfolio's econometrics piece —
signal-trader-demo asks "is there retail alpha?" (no), equity-scout asks "which
process is robust?", vol-scout asks "what *is* forecastable, and with what?".

## Non-goals

- No trading strategy, no backtested PnL, no alpha claim of any kind.
- No dashboard/service; a small read-only Streamlit demo app only.
- The possible consumer (equity-scout's vol-targeting strategy) is documented as
  an outlook paragraph, **not built**.
- No intraday data, no paid data sources, no API keys required for tests.

## Data

- Free daily OHLCV via yfinance, cached to `data/` (CSV/parquet); tests never
  make live calls (same policy as signal-trader-demo/equity-scout).
- Main series: SPY (literature standard). Robustness set: ~8 liquid single
  names/ETFs across sectors plus one Europe ETF (exact list fixed in the plan).
- Volatility proxy from OHLC: **Parkinson** (primary), **Garman-Klass** and
  squared close-to-close returns as sensitivity checks. Documented prominently
  as caveat #1: this is a proxy, true realized volatility needs intraday data
  which is not freely available.

## Contenders

| Model | Role | Implementation |
|---|---|---|
| Random-walk vol (yesterday's RV) | naive benchmark | trivial |
| **HAR-RV** | the standard to beat | hand-implemented OLS on daily/weekly/monthly RV lags (deliberate: the learning core) |
| GARCH(1,1) | classical econometrics | `arch` package (one new, established dependency — justified in commit body) |
| LightGBM | feature-engineered ML | vol/calendar/range features; anti-leakage tests in the house sentinel style |
| Chronos-2 zero-shot | foundation model | univariate on the RV series; adapter pattern from timeseries-showdown |

Horizons: **h=1** and **h=22** trading days.

## Evaluation

- Rolling-origin backtest over multiple OOS years (exact windows fixed in the
  plan; window config as shared constants, tests against the real data).
- Primary metric **QLIKE** (robust under proxy noise), RMSE as second opinion,
  Mincer-Zarnowitz regression for bias.
- **Diebold-Mariano with HLN correction** for every headline pair (pattern:
  timeseries-showdown `stats.py`). Overlapping h=22 forecasts → HAC/Newey-West
  handling documented.
- Editorial rule (binding, same as the ml-lab SOTA plan): uncertainty in the
  same breath as every comparative claim; differences within noise are ties.

## Honest limitations (in README from day one)

1. OHLC proxy instead of intraday realized volatility.
2. One dominant market regime in the OOS window; robustness set mitigates, not
   solves.
3. Volatility forecastability ≠ trading edge; the vol-targeting connection is
   an outlook, not a result.
4. Chronos-2 sees only the univariate vol series — fair vs. HAR (same
   information set), but stated explicitly.
5. Survivorship: robustness names picked today; noted, and the piece's claims
   are about SPY primarily.

## Structure & process

- Own repo `~/private/vol-scout`, house conventions via project-bootstrap
  (PROJECT.md, uv + ruff + pytest gate, main + autopilot/work, MIT).
- Modules, one responsibility each: `data.py` (cache + vol proxies),
  `models.py` (contenders), `backtest.py` (rolling origin), `stats.py`
  (QLIKE/MZ/DM-HLN), `viz.py` (house chart style), `app.py` (Streamlit,
  AppTest-covered), `scripts/` (thin runners, resumable long computations).
- Milestones M1–M6 analog timeseries-showdown: data+proxies → benchmarks
  (RW/HAR/GARCH) → LightGBM → Chronos-2 → stats+figures → README/app.
  Implementation via subagent-driven-development with two-stage review per
  milestone; compute runs detached (setsid) with resume-safe artifacts.
- Deliverables: README with the money chart (QLIKE by horizon per model),
  returns-contrast chapter, "Where this sits in 2026" section with citations.

## Success criteria

- Every headline number backed by a committed artifact and a DM/HLN verdict.
- The returns-contrast chapter reproduces the expected null result cleanly.
- Full gate green offline (no keys, no live calls) — `uv run pytest -q && uv
  run ruff check .`.
- A reader can reproduce every figure from `scripts/` on free data.

## Open points for the implementation plan

- Exact robustness-set tickers and OOS window boundaries.
- GARCH forecast aggregation to h=22 (iterated vs. direct — decide in plan).
- Whether Chronos-2 gets the raw RV series or log-RV (decide in plan, document).
