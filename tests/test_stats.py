"""Tests for `vol_scout.stats`: QLIKE/RMSE/Mincer-Zarnowitz (M5.2), then
HLN-corrected Diebold-Mariano, per-window spread, and the tie-verdict rule
(M5.3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vol_scout.stats import mincer_zarnowitz, qlike, rmse

# --- qlike -------------------------------------------------------------------


def test_qlike_hand_computed_single_pair():
    # formula (module docstring, plan M5.2): ln(f) + y/f, per observation.
    y, f = 4.0, 2.0
    expected = math.log(f) + y / f
    assert qlike([y], [f]) == pytest.approx(expected)


def test_qlike_hand_computed_averaged_over_several_observations():
    y = np.array([4.0, 9.0, 1.0])
    f = np.array([2.0, 3.0, 2.0])
    expected = np.mean(np.log(f) + y / f)
    assert qlike(y, f) == pytest.approx(expected)


def test_qlike_perfect_forecast_hand_computed():
    y = f = np.array([2.0, 5.0, 10.0])
    expected = np.mean(np.log(f) + y / f)
    assert qlike(y, f) == pytest.approx(expected)
    # sanity: a perfect forecast is NOT zero under this convention (unlike
    # the y/f - ln(y/f) - 1 variant) -- documented in the module docstring.
    assert qlike(y, f) != pytest.approx(0.0)


def test_qlike_raises_on_non_positive_prediction():
    with pytest.raises(ValueError, match="rv_pred"):
        qlike([1.0], [0.0])
    with pytest.raises(ValueError, match="rv_pred"):
        qlike([1.0], [-1.0])


def test_qlike_raises_on_negative_true_value():
    with pytest.raises(ValueError, match="rv_true"):
        qlike([-0.1], [1.0])


def test_qlike_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        qlike([1.0, 2.0], [1.0])


# --- rmse ----------------------------------------------------------------


def test_rmse_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    f = np.array([2.0, 2.0, 2.0])
    # errors: -1, 0, 1 -> squared: 1, 0, 1 -> mean 2/3 -> sqrt
    expected = math.sqrt(2.0 / 3.0)
    assert rmse(y, f) == pytest.approx(expected)


def test_rmse_zero_for_perfect_forecast():
    y = f = np.array([1.0, 5.0, 9.0])
    assert rmse(y, f) == pytest.approx(0.0)


# --- mincer_zarnowitz --------------------------------------------------------


def test_mincer_zarnowitz_perfect_forecast_is_identity_regression():
    y = f = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    result = mincer_zarnowitz(y, f)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert result["slope"] == pytest.approx(1.0, abs=1e-9)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_mincer_zarnowitz_constant_bias_recovered_in_intercept():
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    bias = 0.7
    y = f + bias
    result = mincer_zarnowitz(y, f)
    assert result["intercept"] == pytest.approx(bias, abs=1e-9)
    assert result["slope"] == pytest.approx(1.0, abs=1e-9)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_mincer_zarnowitz_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        mincer_zarnowitz([1.0], [1.0])
