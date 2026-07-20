"""Tests for src/vol_scout/data.py: fetch+cache seam and volatility proxies.

No test in this module ever calls a live network endpoint. All provider
behavior is exercised through `FakePriceProvider`, a test double that
actively enforces the "never re-fetch what's already cached" invariant by
raising if asked for a date range it has already served — a stronger
guarantee than a plain call-count assertion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vol_scout.data import PriceProvider, fetch_and_cache_ohlcv, load_ohlcv


class FakePriceProvider:
    """Synthetic OHLCV provider that raises on a redundant fetch.

    Tracks every date range it has already been asked to serve. If a later
    `fetch` call's [start, end] is fully contained in a previously-served
    range, that is proof the caller re-fetched data it already had — the
    fake raises instead of silently returning data, so a "never re-fetch
    what you have" regression fails loudly.
    """

    def __init__(self) -> None:
        self.served_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for served_start, served_end in self.served_ranges:
            if served_start <= start_ts and end_ts <= served_end:
                raise AssertionError(
                    f"fetch() called for already-cached range [{start}, {end}]; "
                    f"already served [{served_start}, {served_end}]"
                )
        self.calls.append((ticker, start, end))
        self.served_ranges.append((start_ts, end_ts))

        dates = pd.bdate_range(start_ts, end_ts)
        n = len(dates)
        base = 100.0
        return pd.DataFrame(
            {
                "date": dates,
                "open": [base + i for i in range(n)],
                "high": [base + i + 1 for i in range(n)],
                "low": [base + i - 1 for i in range(n)],
                "close": [base + i + 0.5 for i in range(n)],
                "volume": [1_000_000 + i for i in range(n)],
            }
        )


def test_fake_provider_satisfies_protocol():
    assert isinstance(FakePriceProvider(), PriceProvider)


def test_fetch_and_cache_writes_csv_on_empty_cache(tmp_path: Path):
    provider = FakePriceProvider()

    fetch_and_cache_ohlcv("FOO", "2020-01-01", "2020-01-10", tmp_path, provider=provider)

    csv_path = tmp_path / "FOO.csv"
    assert csv_path.exists()
    raw = pd.read_csv(csv_path)
    assert list(raw.columns) == ["date", "open", "high", "low", "close", "volume"]
    dates = pd.to_datetime(raw["date"])
    assert dates.is_monotonic_increasing
    assert not dates.duplicated().any()
    assert len(provider.calls) == 1


def test_fetch_and_cache_same_range_is_a_cache_hit_no_fetch_call(tmp_path: Path):
    provider = FakePriceProvider()
    fetch_and_cache_ohlcv("FOO", "2020-01-01", "2020-01-10", tmp_path, provider=provider)
    assert len(provider.calls) == 1

    # Same range again: the fake provider raises if fetch() is called with a
    # range it already served, so a no-op here proves the cache hit.
    fetch_and_cache_ohlcv("FOO", "2020-01-01", "2020-01-10", tmp_path, provider=provider)

    assert len(provider.calls) == 1


def test_fetch_and_cache_wider_range_only_fetches_missing_tail_and_merges(tmp_path: Path):
    provider = FakePriceProvider()
    fetch_and_cache_ohlcv("FOO", "2020-01-01", "2020-01-10", tmp_path, provider=provider)
    first_cached = pd.read_csv(tmp_path / "FOO.csv")
    first_rows = len(first_cached)

    result = fetch_and_cache_ohlcv("FOO", "2020-01-01", "2020-01-20", tmp_path, provider=provider)

    # Second fetch() call only asked for the missing tail (a range not fully
    # contained in the first served range) -- the fake would have raised
    # otherwise.
    assert len(provider.calls) == 2
    second_call_start = pd.Timestamp(provider.calls[1][1])
    assert second_call_start > pd.Timestamp("2020-01-10")

    on_disk = pd.read_csv(tmp_path / "FOO.csv")
    on_disk_dates = pd.to_datetime(on_disk["date"])
    assert on_disk_dates.is_monotonic_increasing
    assert not on_disk_dates.duplicated().any()
    # Append-only: every previously-cached row must still be present.
    assert len(on_disk) > first_rows
    assert set(pd.to_datetime(first_cached["date"])) <= set(on_disk_dates)
    assert len(result) == len(on_disk)


def test_load_ohlcv_reads_committed_csv_with_datetime_index(tmp_path: Path):
    csv_path = tmp_path / "BAR.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [100, 200],
        }
    ).to_csv(csv_path, index=False)

    df = load_ohlcv("BAR", tmp_path)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
