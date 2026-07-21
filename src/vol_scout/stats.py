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
"""

from __future__ import annotations

import numpy as np


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
