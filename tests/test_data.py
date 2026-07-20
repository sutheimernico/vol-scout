"""Tests for src/vol_scout/data.py: fetch+cache seam and volatility proxies.

No test in this module ever calls a live network endpoint. All provider
behavior is exercised through `FakePriceProvider`, a test double that
actively enforces the "never re-fetch what's already cached" invariant by
raising if asked for a date range it has already served — a stronger
guarantee than a plain call-count assertion.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from vol_scout.data import (
    PriceProvider,
    fetch_and_cache_ohlcv,
    garman_klass_rv,
    load_ohlcv,
    log_return,
    multi_day_rv,
    parkinson_rv,
    squared_return_rv,
)


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


# --- volatility proxies ------------------------------------------------------
#
# Fixture rows (hand-picked so the expected numbers can be derived independently
# of the implementation, via math.log directly, not by calling the functions
# under test):
#   row0: O=100, H=105, L=95,  C=100   (no prior close -> log_return NaN)
#   row1: O=100, H=110, L=100, C=105   (prior close 100)
#   row2: O=105, H=108, L=102, C=104   (prior close 105)
#   row3: O=104, H=106, L=103, C=107   (prior close 104)

_PROXY_FIXTURE = pd.DataFrame(
    {
        "open": [100.0, 100.0, 105.0, 104.0],
        "high": [105.0, 110.0, 108.0, 106.0],
        "low": [95.0, 100.0, 102.0, 103.0],
        "close": [100.0, 105.0, 104.0, 107.0],
    }
)


def test_log_return_hand_computed_and_first_row_is_nan():
    result = log_return(_PROXY_FIXTURE)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(math.log(105.0 / 100.0))
    assert result.iloc[2] == pytest.approx(math.log(104.0 / 105.0))
    assert result.iloc[3] == pytest.approx(math.log(107.0 / 104.0))


def test_squared_return_rv_hand_computed_and_first_row_is_nan():
    result = squared_return_rv(_PROXY_FIXTURE)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(math.log(105.0 / 100.0) ** 2)
    assert result.iloc[2] == pytest.approx(math.log(104.0 / 105.0) ** 2)
    assert result.iloc[3] == pytest.approx(math.log(107.0 / 104.0) ** 2)


def test_squared_return_rv_is_literally_log_return_squared():
    # One definition, not two -- squared_return_rv must be exactly log_return**2.
    lr = log_return(_PROXY_FIXTURE)
    sq = squared_return_rv(_PROXY_FIXTURE)

    pd.testing.assert_series_equal((lr**2).iloc[1:], sq.iloc[1:], check_names=False)


def test_parkinson_rv_hand_computed():
    ln2 = math.log(2.0)
    expected = [
        (1.0 / (4.0 * ln2)) * math.log(105.0 / 95.0) ** 2,
        (1.0 / (4.0 * ln2)) * math.log(110.0 / 100.0) ** 2,
        (1.0 / (4.0 * ln2)) * math.log(108.0 / 102.0) ** 2,
        (1.0 / (4.0 * ln2)) * math.log(106.0 / 103.0) ** 2,
    ]

    result = parkinson_rv(_PROXY_FIXTURE)

    for i, exp in enumerate(expected):
        assert result.iloc[i] == pytest.approx(exp)


def test_garman_klass_rv_hand_computed():
    ln2 = math.log(2.0)
    c = 2.0 * ln2 - 1.0
    expected = [
        0.5 * math.log(105.0 / 95.0) ** 2 - c * math.log(100.0 / 100.0) ** 2,
        0.5 * math.log(110.0 / 100.0) ** 2 - c * math.log(105.0 / 100.0) ** 2,
        0.5 * math.log(108.0 / 102.0) ** 2 - c * math.log(104.0 / 105.0) ** 2,
        0.5 * math.log(106.0 / 103.0) ** 2 - c * math.log(107.0 / 104.0) ** 2,
    ]

    result = garman_klass_rv(_PROXY_FIXTURE)

    for i, exp in enumerate(expected):
        assert result.iloc[i] == pytest.approx(exp)


def test_parkinson_and_garman_klass_sensitivity_vs_squared_return():
    # row0: seed close only.
    # row1: big intraday range (H/L = 120/80 = 1.5), nearly flat close-to-close
    #       (100 -> 100.5) -- Parkinson/GK should dwarf the squared return.
    # row2: narrow intraday range (H/L = 115.5/114.5), big close-to-close move
    #       (100.5 -> 115.2) -- squared return should dwarf Parkinson/GK.
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 115.0],
            "high": [101.0, 120.0, 115.5],
            "low": [99.0, 80.0, 114.5],
            "close": [100.0, 100.5, 115.2],
        }
    )

    parkinson = parkinson_rv(df)
    gk = garman_klass_rv(df)
    sq = squared_return_rv(df)

    # Row 1: big range, flat close -> Parkinson/GK >> squared return.
    assert parkinson.iloc[1] > 100 * sq.iloc[1]
    assert gk.iloc[1] > 100 * sq.iloc[1]

    # Row 2: narrow range, big close move -> squared return >> Parkinson/GK.
    assert sq.iloc[2] > 100 * parkinson.iloc[2]
    assert sq.iloc[2] > 100 * gk.iloc[2]


def test_multi_day_rv_trailing_rolling_sum_hand_verified():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = multi_day_rv(s, 3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(s.iloc[0] + s.iloc[1] + s.iloc[2])
    assert result.iloc[3] == pytest.approx(s.iloc[1] + s.iloc[2] + s.iloc[3])
    assert result.iloc[4] == pytest.approx(s.iloc[2] + s.iloc[3] + s.iloc[4])
