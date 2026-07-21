"""Evaluation metrics for the rolling-origin backtest (`backtest.run_all_
models`'s output): QLIKE (primary), RMSE (second opinion), and a
Mincer-Zarnowitz regression for bias -- every one of them on the same
variance scale `models.py`'s module docstring pins throughout this project.

**QLIKE convention (pin this, it differs across papers -- plan M5.2, binding):**
`qlike(y, f) = mean(ln(f) + y/f)`, `y` = realized RV (truth), `f` = the
model's point forecast. Patton (2011)'s own paper also states an equivalent
form, `y/f - ln(y/f) - 1`; the two differ ONLY by `-ln(y) - 1`, a term that
depends on `y` alone (never on `f`, i.e. never on which model produced the
forecast) -- so for any two models' losses at the SAME observation, the two
conventions' loss DIFFERENCE is identical, and every DM/paired-comparison
result downstream is unaffected by which one is used. The plan's simpler
form is used here; do not "fix" this to the other paper's form expecting a
different verdict, there isn't one. Both forms need `f > 0` (the `ln`) --
RV proxies are non-negative by construction (squared or squared-log-range
terms), so `y < 0` should never occur upstream either; guarded here anyway,
for the same "surface a bug loudly, don't paper over it" reason `f <= 0` is.

**DM/HLN's `h` parameter -- NOT the forecast horizon in general (adapted
from `timeseries_showdown/stats.py`'s own module docstring, same
distinction, different resolution here).** `h` is the Diebold-Mariano test's
own parameter for how many lags of autocovariance the loss differential
series needs -- it matters whenever the per-window losses being compared
come from OVERLAPPING forecast windows of the same series and are therefore
autocorrelated across origins. `HORIZON_DM_H` below is the one place
vol-scout's own two headline horizons' `h` choice lives (design-decision §3
of the plan, binding):

- **h=1 pair -> `h=1`.** Origins step by `STEP_TRADING_DAYS` (>=5) trading
  days apart, and the h=1 test "window" is a single day (`o+1`) -- these
  single days across different origins never coincide, so the loss
  differential series is NOT autocorrelated by construction. This is the
  textbook non-overlapping case, same choice `timeseries_showdown` made for
  its own disjoint windows.
- **h=22 pair -> `h=22`.** Each origin's 22-trading-day test window overlaps
  its neighbor's by `22 - STEP_TRADING_DAYS` days -- genuine, heavy overlap
  (the classic Diebold-Mariano motivating case for h-step-ahead forecast
  comparison). Per Diebold & Mariano (1995)'s own prescription for
  comparing h-step-ahead forecasts, `h` is set to the forecast horizon
  itself here -- unlike the h=1 pair, where "the DM `h`" and "the forecast
  horizon" happen to both be 1 only by coincidence, for the h=22 pair they
  are the same NUMBER for a real reason: a k-step-ahead forecast error's
  own autocorrelation structure is theoretically nonzero up to lag k-1
  (Diebold & Mariano 1995; the small-sample correction on top of it is
  Harvey, Leybourne & Newbold 1997). A future reader must not assume
  "h in dm_test always equals 1" just because that was the h=1 pair's
  answer here, or the sibling project's ONLY answer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as scipy_stats


def _as_arrays(rv_true, rv_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(rv_true, dtype=float)
    f = np.asarray(rv_pred, dtype=float)
    if y.shape != f.shape:
        raise ValueError(
            f"rv_true and rv_pred must have the same length, got {y.shape} vs {f.shape}"
        )
    return y, f


def qlike(rv_true, rv_pred) -> float:
    """QLIKE, averaged over observations: `mean(ln(f) + y/f)` (module
    docstring's convention -- Patton 2011, up to an additive `f`-independent
    constant vs. that paper's own alternate form). Raises `ValueError` if
    any `rv_pred <= 0` (log of a non-positive forecast is undefined -- a
    model producing one is a bug worth surfacing loudly) or any `rv_true <
    0` (RV proxies are non-negative by construction; a negative "realized"
    value would mean a bug upstream, not a valid degenerate input).
    """
    y, f = _as_arrays(rv_true, rv_pred)
    if np.any(f <= 0):
        raise ValueError(f"qlike needs rv_pred > 0 everywhere, got min={f.min()}")
    if np.any(y < 0):
        raise ValueError(f"qlike needs rv_true >= 0 everywhere, got min={y.min()}")
    return float(np.mean(np.log(f) + y / f))


def rmse(rv_true, rv_pred) -> float:
    """Root-mean-squared error between realized and predicted RV, same
    variance scale as everything else in this module."""
    y, f = _as_arrays(rv_true, rv_pred)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mincer_zarnowitz(rv_true, rv_pred) -> dict:
    """Mincer-Zarnowitz regression: OLS of `rv_true` on `[1, rv_pred]`
    (`numpy.linalg.lstsq`, no `statsmodels` -- same hand-rolled-OLS
    convention `models.har_rv_coefficients` uses). A well-calibrated,
    unbiased forecaster has intercept≈0, slope≈1, R²≈1; a forecast that is a
    constant multiple or additive shift away from the truth shows up as a
    slope != 1 or intercept != 0 respectively.

    Returns `{"intercept": ..., "slope": ..., "r_squared": ...}`. Raises
    `ValueError` with fewer than 2 observations (can't fit 2 parameters).
    """
    y, f = _as_arrays(rv_true, rv_pred)
    n = len(y)
    if n < 2:
        raise ValueError(f"mincer_zarnowitz needs at least 2 observations, got {n}")

    design = np.column_stack([np.ones(n), f])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])

    fitted = design @ coefficients
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {"intercept": intercept, "slope": slope, "r_squared": r_squared}


# --- Diebold-Mariano / HLN ---------------------------------------------------
#
# Ported from `timeseries_showdown/src/timeseries_showdown/stats.py::dm_test`
# essentially verbatim (same HLN-corrected formula, same degenerate-variance
# guard) -- see this module's own docstring for vol-scout's `h` story, which
# differs from that sibling project's h=1-only framing.

# HORIZON_DM_H: module docstring's binding `h` choice per headline horizon.
# `headline_pair_stats` reads from here rather than hardcoding either number
# inline, so the design decision lives in exactly one place.
HORIZON_DM_H: dict[int, int] = {1: 1, 22: 22}


@dataclass
class DMResult:
    """Diebold-Mariano/HLN test result. See module docstring for what `h`
    means here for vol-scout's two headline horizons.

    `dm_stat`/`dm_stat_hln`/`p_value` are `None` in the degenerate case
    where the loss differential series has zero variance (e.g. `losses_a`
    and `losses_b` are identical, or differ by a constant offset for every
    window) -- there is then no variability for a test statistic to use.
    `mean_loss_diff` is always populated.
    """

    n: int
    dm_stat: float | None
    dm_stat_hln: float | None
    p_value: float | None
    mean_loss_diff: float


def dm_test(losses_a: list[float], losses_b: list[float], h: int) -> DMResult:
    """Diebold-Mariano test, HLN small-sample corrected. See module
    docstring for what `h` means here and vol-scout's own two headline
    choices (`HORIZON_DM_H`).

    Loss differential `d_t = losses_a[t] - losses_b[t]`. Variance of the
    mean is estimated from `d`'s autocovariances up to lag `h-1` (standard
    DM); the corrected statistic is `DM* = DM * sqrt((n + 1 - 2h +
    h(h-1)/n) / n)`, and the p-value comes from the Student-t distribution
    with `n-1` degrees of freedom (NOT the normal -- that is what
    "small-sample correction" means: the DM statistic is asymptotically
    normal, but HLN showed it is oversized at small n and the t-distribution
    is the fix).
    """
    a = np.asarray(losses_a, dtype=float)
    b = np.asarray(losses_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"losses_a and losses_b must have the same length, got {a.shape} vs {b.shape}"
        )
    n = len(a)
    if n < 2:
        raise ValueError(f"dm_test needs at least 2 windows, got {n}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")

    d = a - b
    dbar = float(np.mean(d))

    # gamma_0: lag-0 autocovariance (population variance, ddof=0) of d.
    gamma_0 = float(np.mean((d - dbar) ** 2))
    autocov_sum = 0.0
    for k in range(1, h):
        if k >= n:
            break
        autocov_sum += float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))

    v_hat = (gamma_0 + 2 * autocov_sum) / n

    if v_hat <= 0.0:
        return DMResult(n=n, dm_stat=None, dm_stat_hln=None, p_value=None, mean_loss_diff=dbar)

    dm_stat = dbar / math.sqrt(v_hat)

    correction = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if correction <= 0.0:
        raise ValueError(
            f"HLN correction factor is non-positive ({correction:.4f}) for n={n}, h={h}; "
            "h is too large relative to n for the correction to be meaningful"
        )
    dm_stat_hln = dm_stat * math.sqrt(correction)

    df = n - 1
    p_value = float(2 * scipy_stats.t.sf(abs(dm_stat_hln), df))

    return DMResult(
        n=n, dm_stat=dm_stat, dm_stat_hln=dm_stat_hln, p_value=p_value, mean_loss_diff=dbar
    )


def dm_verdict(dm_result: DMResult, alpha: float = 0.05) -> str:
    """Editorial tie rule (spec's Evaluation section, "differences within
    noise are ties"), turned into one small testable function instead of a
    README-writing convention someone has to remember correctly every time.

    Returns `"tie"` whenever the DM/HLN test cannot distinguish the two
    sides at the `alpha` level -- including the fully-degenerate case where
    `dm_result.p_value is None` (zero-variance loss differential, `dm_test`'s
    own docstring). Otherwise returns `"a"` or `"b"`, whichever side has the
    strictly lower mean loss: `dm_result.mean_loss_diff` is `mean(losses_a -
    losses_b)`, so negative means `a`'s losses were lower on average (lower
    loss = the better forecast). This function only knows about the two
    generic sides `dm_test` was given -- callers with real model names
    (`headline_pair_stats`) map `"a"`/`"b"` back to `model_a`/`model_b`.
    """
    if dm_result.p_value is None or dm_result.p_value >= alpha:
        return "tie"
    return "a" if dm_result.mean_loss_diff < 0 else "b"


def per_window_spread(losses: list[float]) -> dict:
    """`{"mean", "std", "min", "max"}` of a per-window loss list -- the
    per-window spread reported alongside every metrics.json bucket
    (mirrors `timeseries_showdown.baselines`'s `MASE_std/min/max` fields).
    `std` uses `ddof=1` (sample standard deviation), matching `dm_test`'s
    sibling `paired_summary`-style convention elsewhere in this project.
    """
    values = np.asarray(losses, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def headline_pair_stats(
    per_model_losses: dict[str, dict[int, list[float]]],
    pairs: list[tuple[str, str]],
) -> dict:
    """DM/HLN + tie-verdict + per-window spread for each `(model_a,
    model_b)` in `pairs`, for both of vol-scout's headline horizons
    (`HORIZON_DM_H`'s keys) -- the `results/metrics.json` `"headline"`
    block (plan §7).

    `per_model_losses` is `{model_name: {horizon: [loss_window_0, ...]}}`
    -- per-origin losses (any metric the caller chooses, typically QLIKE;
    this function is metric-agnostic, it just needs matched per-origin
    numbers). Reads the CORRECT `h` for each horizon from `HORIZON_DM_H`
    (module docstring) rather than ever hardcoding `h=1`, which is what
    makes the h=22 pair's overlap-aware correction actually apply instead
    of silently defaulting to the h=1 pair's answer.

    Returns `{"<model_a>_vs_<model_b>": {"h1": {...}, "h22": {...}}}`, each
    horizon bucket `{"dm": asdict(DMResult), "verdict": <model name or
    "tie">, "spread_a": per_window_spread(losses_a), "spread_b": ...}`.
    Raises `ValueError` (propagated from `dm_test`) if a pair's two loss
    lists for a horizon differ in length -- a mismatch there means the two
    models were not actually evaluated on the same set of origins, which
    would silently corrupt the comparison.
    """
    headline: dict[str, dict] = {}
    for model_a, model_b in pairs:
        key = f"{model_a}_vs_{model_b}"
        headline[key] = {}
        for horizon, h in HORIZON_DM_H.items():
            losses_a = per_model_losses[model_a][horizon]
            losses_b = per_model_losses[model_b][horizon]
            dm_result = dm_test(losses_a, losses_b, h=h)
            verdict_side = dm_verdict(dm_result)
            verdict = {"a": model_a, "b": model_b, "tie": "tie"}[verdict_side]
            headline[key][f"h{horizon}"] = {
                "dm": asdict(dm_result),
                "verdict": verdict,
                "spread_a": per_window_spread(losses_a),
                "spread_b": per_window_spread(losses_b),
            }
    return headline
