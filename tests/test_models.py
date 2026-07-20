"""Tests for src/vol_scout/models.py: the MODEL_REGISTRY contenders.

No test in this module makes a network call. GARCH tests simulate their own
GARCH(1,1) process (known omega/alpha/beta) rather than depending on real
market data, so the ground truth is independently computable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vol_scout.models import ForecastResult, forecast_random_walk

# 30 hand-picked daily RV values (all realized RV proxies are non-negative,
# so a mild upward-trending sequence is a realistic-shaped fixture).
_RV_30 = pd.Series(
    [0.0001 + 0.00002 * i for i in range(30)],
    index=pd.date_range("2020-01-01", periods=30, freq="D"),
    name="rv",
)


def _train_df(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"rv": rv})


def test_forecast_random_walk_returns_forecast_result():
    result = forecast_random_walk(_train_df(_RV_30))

    assert isinstance(result, ForecastResult)
    assert isinstance(result.fit_predict_seconds, float)
    assert result.fit_predict_seconds >= 0.0


def test_forecast_random_walk_point_1_is_last_rv_exactly():
    result = forecast_random_walk(_train_df(_RV_30))

    assert result.point[1] == pytest.approx(_RV_30.iloc[-1])


def test_forecast_random_walk_point_22_is_22x_last_rv_exactly():
    result = forecast_random_walk(_train_df(_RV_30))

    assert result.point[22] == pytest.approx(22.0 * _RV_30.iloc[-1])
