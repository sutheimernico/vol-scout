"""Fetch and cache OHLCV history for the vol-scout ten-ticker universe.

The one legitimate live network call in this project: everything else
(tests, models, backtest) runs against the committed CSVs this script
produces in `data/`. Safe to re-run -- `fetch_and_cache_ohlcv` only fetches
the missing tail on top of whatever is already cached.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from vol_scout.data import TICKERS, fetch_and_cache_ohlcv

DATA_START = "2010-01-04"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    end = date.today().isoformat()
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = fetch_and_cache_ohlcv(ticker, DATA_START, end, DATA_DIR)
        print(f"  {ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()
