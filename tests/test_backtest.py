"""Tests for `vol_scout.backtest`: rolling-origin window mechanics
(`make_origins`) and per-origin model orchestration (`run_all_models`).

Fixture style matches `tests/test_models.py`'s synthetic-series-with-known-
values approach: every hand-computed assertion below is arithmetic a reader
can verify by re-reading the fixture, not an opaque end-to-end number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_scout.backtest import (
    HORIZONS,
    OOS_START,
    STEP_TRADING_DAYS,
    make_origins,
    run_all_models,
    run_and_append_resumable,
    target_rv,
)
from vol_scout.data import load_ohlcv, parkinson_rv
from vol_scout.models import ForecastResult


def _synthetic_df(n_rows: int, start: str = "2017-01-02") -> pd.DataFrame:
    """`n_rows` trading days (business-day index, no holiday calendar --
    fine for these tests, which only ever check position arithmetic, not
    real NYSE holidays) with a deterministic `rv` column (`rv[i] = i + 1`,
    so every target/feature sum below is hand-computable by construction).
    """
    idx = pd.bdate_range(start=start, periods=n_rows)
    rv = pd.Series(np.arange(1, n_rows + 1, dtype=float), index=idx)
    return pd.DataFrame({"rv": rv}, index=idx)


# --- make_origins: synthetic ------------------------------------------------


def test_make_origins_synthetic_starts_exactly_at_oos_start():
    df = _synthetic_df(500)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    assert origins[0] == oos_start


def test_make_origins_synthetic_steps_by_step_trading_days():
    df = _synthetic_df(500)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    positions = [df.index.get_loc(o) for o in origins]
    assert all(b - a == 5 for a, b in zip(positions, positions[1:], strict=False))


def test_make_origins_synthetic_last_origin_has_full_22_day_horizon():
    df = _synthetic_df(500)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    last_pos = df.index.get_loc(origins[-1])
    # the last origin must have exactly 22 trading days still inside df...
    assert last_pos + 22 <= len(df.index) - 1
    # ...and the NEXT candidate origin (one step further) would not.
    next_pos = last_pos + 5
    assert next_pos + 22 > len(df.index) - 1


def test_make_origins_synthetic_every_origin_is_a_real_index_date():
    df = _synthetic_df(500)
    oos_start = df.index[50]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    assert all(o in df.index for o in origins)


def test_make_origins_raises_when_oos_start_past_end_of_data():
    df = _synthetic_df(50)
    with pytest.raises(ValueError, match="oos_start"):
        make_origins(df, oos_start=df.index[-1] + pd.Timedelta(days=1), step_trading_days=5)


def test_make_origins_raises_when_no_origin_has_full_horizon():
    df = _synthetic_df(50)
    # oos_start is the very last row -- not even 1 day of target data left,
    # let alone 22.
    with pytest.raises(ValueError, match="22"):
        make_origins(df, oos_start=df.index[-1], step_trading_days=5)


# --- make_origins: real SPY data --------------------------------------------


def test_make_origins_real_data_uses_production_constants():
    df = load_ohlcv("SPY", "data")
    origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)
    assert len(origins) > 20  # comfortably above the DM/HLN n>=20 floor
    assert origins[0] == pd.Timestamp(OOS_START)


def test_make_origins_real_data_origins_are_real_trading_days():
    df = load_ohlcv("SPY", "data")
    origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)
    assert all(o in df.index for o in origins)


def test_make_origins_real_data_h1_windows_never_overlap():
    df = load_ohlcv("SPY", "data")
    origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)
    positions = [df.index.get_loc(o) for o in origins]
    # the h=1 test "window" is the single day at position+1; consecutive
    # origins step by >=5 positions, so these single days can never coincide.
    h1_days = [p + 1 for p in positions]
    assert len(h1_days) == len(set(h1_days))


def test_make_origins_real_data_h22_windows_overlap_by_22_minus_step():
    df = load_ohlcv("SPY", "data")
    origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)
    positions = [df.index.get_loc(o) for o in origins[:2]]
    first_window = set(range(positions[0] + 1, positions[0] + 23))  # o+1 .. o+22
    second_window = set(range(positions[1] + 1, positions[1] + 23))
    overlap = first_window & second_window
    assert len(overlap) == 22 - STEP_TRADING_DAYS


# --- target_rv ---------------------------------------------------------------


def test_target_rv_h1_is_next_days_rv_hand_computed():
    df = _synthetic_df(30)
    # rv[i] = i+1 (1-indexed values), so at position 9 (rv=10), h=1 target
    # is rv at position 10, i.e. 11.
    target = target_rv(df["rv"], horizon=1)
    assert target.iloc[9] == 11.0


def test_target_rv_h22_is_sum_of_next_22_days_hand_computed():
    df = _synthetic_df(60)
    # at position 9 (0-indexed), h=22 target = sum(rv[10..31]) = sum(11..32)
    target = target_rv(df["rv"], horizon=22)
    expected = sum(range(11, 33))
    assert target.iloc[9] == expected


def test_target_rv_is_nan_past_the_end_of_the_frame():
    df = _synthetic_df(30)
    target = target_rv(df["rv"], horizon=22)
    assert pd.isna(target.iloc[-1])


# --- run_all_models: basic contract -----------------------------------------


def test_run_all_models_one_row_per_model_origin_horizon():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    result = run_all_models(df, origins, models={"random_walk": _dummy_random_walk})
    assert len(result) == len(origins) * len(HORIZONS)


def test_run_all_models_columns_match_persistence_contract_plus_error():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    result = run_all_models(df, origins, models={"random_walk": _dummy_random_walk})
    expected = {"model", "origin", "horizon", "rv_true", "rv_pred", "fit_predict_seconds", "error"}
    assert set(result.columns) == expected


def test_run_all_models_multiple_models_multiplies_row_count():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    models = {"a": _dummy_random_walk, "b": _dummy_random_walk}
    result = run_all_models(df, origins, models=models)
    assert len(result) == len(origins) * len(HORIZONS) * 2
    assert set(result["model"].unique()) == {"a", "b"}


# --- run_all_models: origin-inclusive semantics (M3 review ruling) --------


def test_run_all_models_train_df_includes_the_origin_day_itself():
    """`train_df = df.loc[:origin]` -- INCLUSIVE of the origin day (binding,
    M3 review ruling): the model call must see the origin's own row as its
    freshest observation, not stop one day short of it.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)

    seen_last_rv = {}

    def spy_model(train_df: pd.DataFrame) -> ForecastResult:
        seen_last_rv["value"] = float(train_df["rv"].iloc[-1])
        return ForecastResult(point={1: 1.0, 22: 22.0}, fit_predict_seconds=0.0)

    run_all_models(df, origins[:1], models={"spy": spy_model})
    expected_rv_at_origin = float(df.loc[origins[0], "rv"])
    assert seen_last_rv["value"] == expected_rv_at_origin


def test_run_all_models_never_passes_future_rows_to_the_model():
    """Sentinel: poison every row strictly AFTER each origin with a huge
    value and confirm no model call ever sees it in its `train_df`.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    origin = origins[len(origins) // 2]
    origin_pos = df.index.get_loc(origin)

    poisoned = df.copy()
    poisoned.loc[poisoned.index[origin_pos + 1 :], "rv"] = 1e9

    def spy_model(train_df: pd.DataFrame) -> ForecastResult:
        assert not (train_df["rv"] == 1e9).any()
        return ForecastResult(point={1: 1.0, 22: 22.0}, fit_predict_seconds=0.0)

    run_all_models(poisoned, [origin], models={"spy": spy_model})


# --- run_all_models: target alignment poison test ---------------------------


def test_run_all_models_rv_true_h1_matches_poisoned_next_day():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    origin = origins[10]
    origin_pos = df.index.get_loc(origin)

    poisoned = df.copy()
    poisoned.iloc[origin_pos + 1, poisoned.columns.get_loc("rv")] = 1e9

    result = run_all_models(poisoned, [origin], models={"random_walk": _dummy_random_walk})
    h1_row = result[result["horizon"] == 1].iloc[0]
    assert h1_row["rv_true"] == 1e9


def test_run_all_models_rv_true_h22_sums_poisoned_day_22_days_out():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    origin = origins[10]
    origin_pos = df.index.get_loc(origin)

    poisoned = df.copy()
    poisoned.iloc[origin_pos + 22, poisoned.columns.get_loc("rv")] = 1e9

    result = run_all_models(poisoned, [origin], models={"random_walk": _dummy_random_walk})
    h22_row = result[result["horizon"] == 22].iloc[0]
    # the h=22 window is exactly [o+1, o+22] -- position o+22 IS the poisoned
    # row, so it legitimately belongs in this sum (the far boundary case).
    expected = poisoned.iloc[origin_pos + 1 : origin_pos + 23]["rv"].sum()
    assert h22_row["rv_true"] == expected
    assert h22_row["rv_true"] > 1e8  # sanity: the poison actually landed in the sum


def test_run_all_models_rv_true_excludes_poison_one_day_beyond_h22():
    """Poison the day exactly one step beyond the h=22 window (`o+23`) and
    confirm it never contaminates `rv_true` for either horizon.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    origin = origins[10]
    origin_pos = df.index.get_loc(origin)

    poisoned = df.copy()
    poisoned.iloc[origin_pos + 23, poisoned.columns.get_loc("rv")] = 1e9

    result = run_all_models(poisoned, [origin], models={"random_walk": _dummy_random_walk})
    assert (result["rv_true"] < 1e8).all()


# --- run_all_models: per-model failure tolerance ----------------------------


def test_run_all_models_records_failure_and_continues_to_next_origin():
    """A model that raises at one origin must not kill the whole run: the
    failing origin gets a recorded failure row (NaN rv_pred, non-empty
    `error`), and every OTHER origin still gets a normal row.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    failing_origin = origins[3]

    def flaky_model(train_df: pd.DataFrame) -> ForecastResult:
        if train_df.index[-1] == failing_origin:
            raise RuntimeError("synthetic non-convergence")
        return ForecastResult(point={1: 1.0, 22: 22.0}, fit_predict_seconds=0.001)

    result = run_all_models(df, origins, models={"flaky": flaky_model})

    # every origin produced rows -- the failure did not shrink the frame.
    assert len(result) == len(origins) * len(HORIZONS)

    failed_rows = result[result["origin"] == failing_origin]
    assert (failed_rows["error"] != "").all()
    assert failed_rows["rv_pred"].isna().all()
    assert failed_rows["fit_predict_seconds"].isna().all()
    # rv_true is model-independent (real future data) -- still populated
    # even though the model itself failed.
    assert failed_rows["rv_true"].notna().all()

    ok_rows = result[result["origin"] != failing_origin]
    assert (ok_rows["error"] == "").all()
    assert ok_rows["rv_pred"].notna().all()


def test_run_all_models_failure_in_one_model_does_not_affect_another():
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)

    def always_fails(train_df: pd.DataFrame) -> ForecastResult:
        raise RuntimeError("boom")

    result = run_all_models(
        df, origins, models={"broken": always_fails, "fine": _dummy_random_walk}
    )
    assert (result[result["model"] == "broken"]["error"] != "").all()
    assert (result[result["model"] == "fine"]["error"] == "").all()


# --- run_all_models: real end-to-end smoke on real SPY data ----------------


def test_run_all_models_smoke_on_real_spy_data_random_walk_only():
    df = load_ohlcv("SPY", "data")
    df["rv"] = parkinson_rv(df)
    origins = make_origins(df, oos_start=OOS_START, step_trading_days=STEP_TRADING_DAYS)
    # keep the smoke test cheap: just the last 5 origins.
    result = run_all_models(df, origins[-5:], models={"random_walk": _dummy_random_walk})
    assert len(result) == 5 * len(HORIZONS)
    assert (result["error"] == "").all()
    assert (result["rv_true"] > 0).all()
    assert (result["rv_pred"] > 0).all()


def _dummy_random_walk(train_df: pd.DataFrame) -> ForecastResult:
    rv_o = float(train_df["rv"].iloc[-1])
    return ForecastResult(point={1: rv_o, 22: 22.0 * rv_o}, fit_predict_seconds=0.0)


# --- run_and_append_resumable: the house resumable-CSV pattern -------------


def test_run_and_append_resumable_creates_file_with_expected_rows(tmp_path):
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    out_path = tmp_path / "forecasts.csv"

    summary = run_and_append_resumable(
        df, "TICK", origins[:3], {"random_walk": _dummy_random_walk}, out_path
    )

    assert summary == {"rows_written": 3 * len(HORIZONS), "failures": 0}
    written = pd.read_csv(out_path)
    assert len(written) == 3 * len(HORIZONS)
    assert set(written.columns) == {
        "model", "ticker", "origin", "horizon", "rv_true", "rv_pred",
        "fit_predict_seconds", "error",
    }
    assert (written["ticker"] == "TICK").all()
    assert (written["error"].fillna("") == "").all()


def test_run_and_append_resumable_second_call_never_recomputes_done_triples(tmp_path):
    """The core resumability guarantee -- pattern from `tests/test_data.py`'s
    `FakePriceProvider`, which raises if asked to re-fetch an already-cached
    range: here, a spy model raises if it is ever called for an origin
    that a PRIOR call already recorded in the CSV.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    out_path = tmp_path / "forecasts.csv"

    run_and_append_resumable(
        df, "TICK", origins[:3], {"random_walk": _dummy_random_walk}, out_path
    )

    done_origins = set(origins[:3])

    def raise_if_already_done(train_df: pd.DataFrame) -> ForecastResult:
        if train_df.index[-1] in done_origins:
            raise AssertionError("model called again for an origin already on disk")
        return _dummy_random_walk(train_df)

    summary = run_and_append_resumable(
        df, "TICK", origins[:5], {"random_walk": raise_if_already_done}, out_path
    )
    # only origins[3] and origins[4] are new.
    assert summary == {"rows_written": 2 * len(HORIZONS), "failures": 0}
    written = pd.read_csv(out_path)
    assert len(written) == 5 * len(HORIZONS)


def test_run_and_append_resumable_resumes_per_model_not_just_per_origin(tmp_path):
    """A run killed halfway through one origin's model loop must resume at
    (model, origin) granularity: if model A finished for origin `o` but
    model B did not, a re-run only redoes B for `o`, never re-running A.
    """
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    out_path = tmp_path / "forecasts.csv"

    # first call: only model "a" runs for these origins (simulating a run
    # that got killed after finishing "a" but before starting "b").
    run_and_append_resumable(df, "TICK", origins[:2], {"a": _dummy_random_walk}, out_path)

    calls: list[str] = []

    def spy_a(train_df: pd.DataFrame) -> ForecastResult:
        calls.append("a")
        return _dummy_random_walk(train_df)

    def spy_b(train_df: pd.DataFrame) -> ForecastResult:
        calls.append("b")
        return _dummy_random_walk(train_df)

    summary = run_and_append_resumable(
        df, "TICK", origins[:2], {"a": spy_a, "b": spy_b}, out_path
    )
    assert calls == ["b", "b"]  # "a" never re-run, "b" runs for both origins
    assert summary == {"rows_written": 2 * len(HORIZONS), "failures": 0}


def test_run_and_append_resumable_counts_failures_in_summary(tmp_path):
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    out_path = tmp_path / "forecasts.csv"

    def always_fails(train_df: pd.DataFrame) -> ForecastResult:
        raise RuntimeError("boom")

    summary = run_and_append_resumable(df, "TICK", origins[:2], {"broken": always_fails}, out_path)
    assert summary == {"rows_written": 2 * len(HORIZONS), "failures": 2 * len(HORIZONS)}


def test_run_and_append_resumable_different_tickers_do_not_collide(tmp_path):
    df = _synthetic_df(200)
    oos_start = df.index[100]
    origins = make_origins(df, oos_start=oos_start, step_trading_days=5)
    out_path = tmp_path / "forecasts.csv"

    run_and_append_resumable(df, "AAA", origins[:2], {"random_walk": _dummy_random_walk}, out_path)
    summary = run_and_append_resumable(
        df, "BBB", origins[:2], {"random_walk": _dummy_random_walk}, out_path
    )
    # same origins, different ticker -- must NOT be skipped as already-done.
    assert summary == {"rows_written": 2 * len(HORIZONS), "failures": 0}
    written = pd.read_csv(out_path)
    assert set(written["ticker"].unique()) == {"AAA", "BBB"}
