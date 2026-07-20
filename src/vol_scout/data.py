"""OHLCV fetch + local CSV cache.

Provider seam: a thin `PriceProvider` Protocol decouples the local cache
from yfinance, the only vendor this project uses (pattern from
`signal-trader-demo/src/signal_trader/market_data/provider.py`). CAVEAT:
`YFinanceProvider` calls `yfinance.download(..., auto_adjust=True)`, which
returns back-adjusted OHLC -- values restated for later splits/dividends,
i.e. a subtle lookahead. We keep it (free, simple) and document it;
downstream code must not pretend these were the prices known on the bar's
own date.

Cache contract: `data/<TICKER>.csv` holds columns `date, open, high, low,
close, volume`, sorted by date, no duplicates. `fetch_and_cache_ohlcv` is
append-only: a call whose requested range is already fully covered by the
cache never touches the provider; a call for a wider range only asks the
provider for the missing tail and merges it in, so previously-cached rows
are never re-fetched or lost. Tests must never make live network calls --
see `FakePriceProvider` in `tests/test_data.py`, which actively raises if
asked to re-fetch an already-cached range.

Volatility proxies, computed directly from OHLC bars (honest limitation #1
from the design spec: these are proxies for realized volatility, not true
realized volatility, which needs intraday data this project does not have).
`parkinson_rv` is the primary proxy used downstream; `garman_klass_rv` and
`squared_return_rv` are sensitivity checks -- each captures a different
slice of the day's price action (intraday range vs. close-to-close), so
they can and do disagree sharply on any single day (see
`test_parkinson_and_garman_klass_sensitivity_vs_squared_return`). These are
low-level, per-row primitives only; `multi_day_rv` is a dumb trailing
rolling sum with no forecast-alignment logic -- callers (M2/M3/M5's target
builders) own shifting it correctly relative to a forecast origin, mirroring
the primitive/caller split `timeseries_showdown.baselines
._rolling_as_of_origin` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import yfinance as yf

_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# The ten-ticker universe (design-decisions table, plan §1): SPY is the main
# series every headline number is about; the other nine are a robustness set
# spanning distinct volatility drivers (sector, geography) plus one Europe
# ETF. Defined once here, reused by `scripts/fetch_data.py` and later by
# `viz.py`/`app.py` so the universe lives in exactly one place.
TICKERS: dict[str, str] = {
    "SPY": "Broad index (main series) -- literature-standard proxy for "
    "'the market'; every headline number in the README is about SPY.",
    "AAPL": "Technology (mega-cap) -- most liquid single name globally; "
    "'normal' large-cap vol regime.",
    "NVDA": "Technology (semiconductors) -- high-beta growth name; "
    "stress-tests every model at the high-vol end of the range.",
    "JPM": "Financials -- systemically important bank; vol regime tied to "
    "macro/credit/rate cycles, distinct driver from broad-market beta.",
    "XOM": "Energy -- integrated oil major; vol driven by commodity shocks, "
    "a third distinct driver.",
    "JNJ": "Health Care -- diversified pharma/consumer-health; defensive, "
    "historically low and stable vol.",
    "PG": "Consumer Staples -- household/personal-care staple; the "
    "lowest-vol contrast in the set.",
    "AMZN": "Consumer Discretionary -- e-commerce/cloud mega-cap; liquid, "
    "moderate-to-high vol.",
    "CAT": "Industrials -- heavy machinery; economically cyclical, "
    "macro-cycle-driven vol.",
    "VGK": "Europe (developed markets) -- Vanguard FTSE Europe ETF, the "
    "most direct Europe analog to SPY's own role; avoids one "
    "single-country ETF's idiosyncrasies.",
}


@runtime_checkable
class PriceProvider(Protocol):
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return daily OHLCV bars with columns: date, open, high, low, close, volume."""
        ...


class YFinanceProvider:
    """yfinance-backed provider returning normalized OHLCV bars for one ticker."""

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
        out = raw.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                     "Volume": "volume"}
        ).reset_index().rename(columns={"Date": "date"})
        out["date"] = pd.to_datetime(out["date"])
        return out[_OHLCV_COLUMNS].sort_values("date").reset_index(drop=True)


def _cache_path(ticker: str, data_dir: Path) -> Path:
    return Path(data_dir) / f"{ticker}.csv"


def fetch_and_cache_ohlcv(
    ticker: str,
    start: str,
    end: str,
    data_dir: Path,
    provider: PriceProvider | None = None,
) -> pd.DataFrame:
    """Fetch [start, end] for `ticker`, using and extending the local CSV cache.

    - Empty cache: fetches the full range from `provider` and writes it.
    - Range fully covered by the cache: returns the cache, never calls
      `provider.fetch`.
    - Wider range: fetches only the missing tail (day after the cache's last
      date through `end`), merges it into the existing rows, dedupes, and
      re-sorts -- append-only, previously-cached rows are never dropped.

    Only forward (tail) extension is handled, matching this project's actual
    use (periodic re-runs of `scripts/fetch_data.py` appending newer bars);
    a request starting before the cache's earliest date is not a case this
    project needs, since `DATA_START` is fixed and fetched once.
    """
    provider = provider or YFinanceProvider()
    path = _cache_path(ticker, data_dir)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        cached_end = cached["date"].max()
        if end_ts <= cached_end and start_ts >= cached["date"].min():
            return load_ohlcv(ticker, data_dir)
        tail_start = cached_end + pd.Timedelta(days=1)
        new_rows = provider.fetch(ticker, tail_start.strftime("%Y-%m-%d"), end)
        combined = pd.concat([cached, new_rows], ignore_index=True)
    else:
        combined = provider.fetch(ticker, start, end)

    combined = (
        combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    )
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return load_ohlcv(ticker, data_dir)


def load_ohlcv(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load `data/<ticker>.csv` into a DataFrame indexed by a DatetimeIndex."""
    path = _cache_path(ticker, data_dir)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def log_return(df: pd.DataFrame) -> pd.Series:
    """ln(C_t / C_{t-1}) per row; first value is NaN (no prior close)."""
    return np.log(df["close"] / df["close"].shift(1))


def squared_return_rv(df: pd.DataFrame) -> pd.Series:
    """Squared close-to-close log return: literally `log_return(df) ** 2`.

    Not a second, independent formula -- one definition, reused, so the two
    can never drift apart.
    """
    return log_return(df) ** 2


def parkinson_rv(df: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) range-based variance proxy: (1/(4*ln2)) * ln(H/L)**2."""
    return (1.0 / (4.0 * np.log(2.0))) * np.log(df["high"] / df["low"]) ** 2


def garman_klass_rv(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980) variance proxy: 0.5*ln(H/L)**2 - (2*ln2-1)*ln(C/O)**2."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    return 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2


def multi_day_rv(daily_rv: pd.Series, days: int) -> pd.Series:
    """Trailing rolling sum of `days` daily RV values.

    Deliberately dumb -- no forward/backward shifting relative to any
    forecast origin. Callers own alignment (see module docstring).
    """
    return daily_rv.rolling(days).sum()
