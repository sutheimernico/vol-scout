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
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import yfinance as yf

_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


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
