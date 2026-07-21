"""Tests for `vol_scout.stats`: QLIKE/RMSE/Mincer-Zarnowitz (M5.2), then
HLN-corrected Diebold-Mariano, per-window spread, and the tie-verdict rule
(M5.3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vol_scout.stats import (
    HORIZON_DM_H,
    DMResult,
    dm_test,
    dm_verdict,
    headline_pair_stats,
    mincer_zarnowitz,
    per_window_spread,
    qlike,
    rmse,
)

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


# --- dm_test / DMResult ------------------------------------------------------


def test_dm_test_hand_computed_h1_no_autocovariance_correction():
    # h=1: v_hat is just the population variance of the loss differential
    # (gamma_0), no autocovariance terms -- the simplest case to hand-check.
    losses_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    losses_b = [2.0, 2.0, 2.0, 2.0, 2.0]
    d = np.array(losses_a) - np.array(losses_b)
    n = len(d)
    dbar = float(np.mean(d))
    gamma_0 = float(np.mean((d - dbar) ** 2))
    v_hat = gamma_0 / n
    expected_dm = dbar / math.sqrt(v_hat)
    # HLN correction at h=1: (n + 1 - 2 + 0) / n = (n - 1) / n
    expected_hln = expected_dm * math.sqrt((n - 1) / n)

    result = dm_test(losses_a, losses_b, h=1)
    assert result.n == n
    assert result.mean_loss_diff == pytest.approx(dbar)
    assert result.dm_stat == pytest.approx(expected_dm)
    assert result.dm_stat_hln == pytest.approx(expected_hln)
    assert result.p_value is not None


def test_dm_test_hln_at_h1_reduces_to_the_paired_t_statistic():
    # Algebraic identity (see module docstring): at h=1, HLN's correction is
    # exactly sqrt((n-1)/n), which turns the DM statistic into the ordinary
    # paired t-statistic (Welch/Student, ddof=1 sample variance).
    rng = np.random.default_rng(0)
    losses_a = rng.normal(loc=1.0, scale=0.3, size=15)
    losses_b = rng.normal(loc=1.2, scale=0.3, size=15)

    d = losses_a - losses_b
    n = len(d)
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    t_stat = mean_diff / (sd / math.sqrt(n))

    result = dm_test(losses_a.tolist(), losses_b.tolist(), h=1)
    assert result.dm_stat_hln == pytest.approx(t_stat)


def test_dm_test_degenerate_zero_variance_returns_none_stats():
    # identical loss series -> zero-variance differential -> no test possible.
    losses_a = [1.0, 2.0, 3.0, 4.0]
    losses_b = [1.0, 2.0, 3.0, 4.0]
    result = dm_test(losses_a, losses_b, h=1)
    assert result.dm_stat is None
    assert result.dm_stat_hln is None
    assert result.p_value is None
    assert result.mean_loss_diff == pytest.approx(0.0)


def test_dm_test_overlapping_h22_uses_autocovariance_terms():
    # h=22: with n=25 losses, autocovariance lags 1..21 all contribute --
    # hand-verify v_hat differs from the naive h=1 variance for genuinely
    # autocorrelated data (constructed via a smooth trend so lags matter).
    rng = np.random.default_rng(1)
    trend = np.linspace(0, 2, 25)
    losses_a = (trend + rng.normal(scale=0.05, size=25)).tolist()
    losses_b = (0.5 * trend + rng.normal(scale=0.05, size=25)).tolist()

    result_h1 = dm_test(losses_a, losses_b, h=1)
    result_h22 = dm_test(losses_a, losses_b, h=22)
    assert result_h1.dm_stat != pytest.approx(result_h22.dm_stat)


def test_dm_test_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        dm_test([1.0, 2.0], [1.0], h=1)


def test_dm_test_raises_below_two_windows():
    with pytest.raises(ValueError, match="at least 2"):
        dm_test([1.0], [1.0], h=1)


def test_dm_test_raises_on_h_below_one():
    with pytest.raises(ValueError, match="h must be"):
        dm_test([1.0, 2.0], [2.0, 1.0], h=0)


# --- dm_verdict ---------------------------------------------------------


def test_dm_verdict_tie_when_p_value_none():
    result = DMResult(n=4, dm_stat=None, dm_stat_hln=None, p_value=None, mean_loss_diff=0.0)
    assert dm_verdict(result) == "tie"


def test_dm_verdict_tie_when_p_value_above_alpha():
    result = DMResult(n=30, dm_stat=1.0, dm_stat_hln=1.0, p_value=0.2, mean_loss_diff=-0.5)
    assert dm_verdict(result, alpha=0.05) == "tie"


def test_dm_verdict_a_wins_when_significant_and_mean_diff_negative():
    # mean_loss_diff = mean(losses_a - losses_b) < 0 -> a's losses are lower.
    result = DMResult(n=30, dm_stat=-3.0, dm_stat_hln=-3.0, p_value=0.001, mean_loss_diff=-0.5)
    assert dm_verdict(result, alpha=0.05) == "a"


def test_dm_verdict_b_wins_when_significant_and_mean_diff_positive():
    result = DMResult(n=30, dm_stat=3.0, dm_stat_hln=3.0, p_value=0.001, mean_loss_diff=0.5)
    assert dm_verdict(result, alpha=0.05) == "b"


# --- per_window_spread ----------------------------------------------------


def test_per_window_spread_hand_computed():
    losses = [1.0, 2.0, 3.0, 4.0]
    spread = per_window_spread(losses)
    assert spread["mean"] == pytest.approx(2.5)
    assert spread["min"] == pytest.approx(1.0)
    assert spread["max"] == pytest.approx(4.0)
    assert spread["std"] == pytest.approx(float(np.std(losses, ddof=1)))


# --- HORIZON_DM_H / headline_pair_stats -------------------------------------


def test_horizon_dm_h_matches_design_decision_3():
    # plan design-decision §3, binding: h=1 pair uses h=1 (non-overlapping
    # windows), h=22 pair uses h=22 (the forecast horizon itself -- the
    # standard DM(1995)/HLN(1997) prescription for genuinely overlapping
    # h-step-ahead forecasts).
    assert HORIZON_DM_H == {1: 1, 22: 22}


def test_headline_pair_stats_uses_correct_h_per_horizon():
    rng = np.random.default_rng(2)
    losses_h1_a = rng.normal(1.0, 0.2, size=25).tolist()
    losses_h1_b = rng.normal(1.3, 0.2, size=25).tolist()
    losses_h22_a = rng.normal(5.0, 0.5, size=25).tolist()
    losses_h22_b = rng.normal(6.0, 0.5, size=25).tolist()

    per_model_losses = {
        "model_a": {1: losses_h1_a, 22: losses_h22_a},
        "model_b": {1: losses_h1_b, 22: losses_h22_b},
    }
    result = headline_pair_stats(per_model_losses, pairs=[("model_a", "model_b")])

    expected_h1 = dm_test(losses_h1_a, losses_h1_b, h=HORIZON_DM_H[1])
    expected_h22 = dm_test(losses_h22_a, losses_h22_b, h=HORIZON_DM_H[22])

    pair_result = result["model_a_vs_model_b"]
    assert pair_result["h1"]["dm"]["dm_stat_hln"] == pytest.approx(expected_h1.dm_stat_hln)
    assert pair_result["h22"]["dm"]["dm_stat_hln"] == pytest.approx(expected_h22.dm_stat_hln)
    # h=22's overlap means a DIFFERENT h was actually applied than h=1's --
    # this is the "not left to the caller to get right by accident" check.
    assert pair_result["h1"]["dm"]["dm_stat_hln"] != pytest.approx(
        dm_test(losses_h1_a, losses_h1_b, h=22).dm_stat_hln
    )


def test_headline_pair_stats_includes_verdict_and_spread():
    per_model_losses = {
        "model_a": {1: [1.0, 1.0, 1.0, 1.0, 1.0] * 6, 22: [5.0] * 30},
        "model_b": {1: [2.0, 2.0, 2.0, 2.0, 2.0] * 6, 22: [5.0] * 30},
    }
    result = headline_pair_stats(per_model_losses, pairs=[("model_a", "model_b")])
    pair_result = result["model_a_vs_model_b"]

    # model_a strictly, deterministically beats model_b at h=1 (1.0 < 2.0
    # every single window, zero variance in the differential is NOT the
    # case here since a-b is a constant -1.0 -- deterministic zero-variance
    # differential -> degenerate DM, but the mean is unambiguous).
    assert pair_result["h1"]["spread_a"]["mean"] == pytest.approx(1.0)
    assert pair_result["h1"]["spread_b"]["mean"] == pytest.approx(2.0)
    assert pair_result["h22"]["verdict"] == "tie"  # identical losses -> tie


def test_headline_pair_stats_raises_on_horizon_length_mismatch():
    per_model_losses = {
        "model_a": {1: [1.0, 2.0, 3.0], 22: [1.0, 2.0, 3.0]},
        "model_b": {1: [1.0, 2.0], 22: [1.0, 2.0, 3.0]},
    }
    with pytest.raises(ValueError, match="same length"):
        headline_pair_stats(per_model_losses, pairs=[("model_a", "model_b")])
