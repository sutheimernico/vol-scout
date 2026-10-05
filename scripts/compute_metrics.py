"""Recompute `results/metrics.json` from the local `results/
forecasts.csv` (plan §7's persistence contract): QLIKE (primary)/RMSE/
Mincer-Zarnowitz per `(model, ticker, horizon)`, plus the headline
model-vs-HAR-RV DM/HLN verdicts on SPY (the series every headline README
number is about, plan §1).

Thin wiring script, no metric math of its own: `stats.py` defines every
metric, `backtest.py` defines the window/origin contracts -- this script
only reads the stored forecasts back and calls them, so `results/
metrics.json` is regenerable from `results/forecasts.csv` alone, with no
model refitting (same split the sibling `timeseries_showdown` project's
`scripts/compute_pairwise_stats.py` uses, for the identical reason).

Only successful rows (`error == ""`) feed any metric -- a model's failure
at some origins does not corrupt its numbers on the origins where it
succeeded. `headline_pair_stats` (called on SPY's per-origin QLIKE losses)
still requires every model in a pair to have the SAME SET of successful
origins there -- if it doesn't, `dm_test`'s own length-mismatch guard
raises loudly rather than silently pairing mismatched windows. Per the
plan's own M5.4 guidance ("a mismatch means one model silently skipped
origins -- investigate before trusting any number downstream"), that raise
is the intended behavior, not a bug to work around here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vol_scout.data import TICKERS
from vol_scout.stats import (
    HORIZON_DM_H,
    headline_pair_stats,
    mincer_zarnowitz,
    per_window_spread,
    qlike,
    rmse,
)

FORECASTS_PATH = Path(__file__).resolve().parent.parent / "results" / "forecasts.csv"
METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.json"

# The pairs the README leads with -- every contender vs. HAR-RV (the
# standard to beat), on SPY only.
HEADLINE_MODEL = "har_rv"
HEADLINE_TICKER = "SPY"


def _bucket_metrics(rows: pd.DataFrame) -> dict:
    """One `(model, ticker, horizon)` bucket: QLIKE (primary), RMSE,
    per-window QLIKE spread, Mincer-Zarnowitz, mean fit+predict time --
    plan §7's exact field list. `{"n": 0}` if every row in this bucket
    failed (or there are none at all -- e.g. a ticker Chronos-2 hasn't been
    run on yet).
    """
    ok = rows[rows["error"] == ""]
    if ok.empty:
        return {"n": 0}

    per_origin_qlike = [
        qlike([y], [f]) for y, f in zip(ok["rv_true"], ok["rv_pred"], strict=True)
    ]
    spread = per_window_spread(per_origin_qlike)
    return {
        "n": len(ok),
        "QLIKE": qlike(ok["rv_true"], ok["rv_pred"]),
        "QLIKE_std": spread["std"],
        "QLIKE_min": spread["min"],
        "QLIKE_max": spread["max"],
        "RMSE": rmse(ok["rv_true"], ok["rv_pred"]),
        "MZ": mincer_zarnowitz(ok["rv_true"], ok["rv_pred"]),
        "mean_fit_predict_seconds": float(ok["fit_predict_seconds"].mean()),
    }


def main() -> None:
    forecasts = pd.read_csv(FORECASTS_PATH, dtype={"error": str})
    forecasts["error"] = forecasts["error"].fillna("")

    models = sorted(forecasts["model"].unique())
    metrics: dict = {}
    per_model_qlike_losses: dict[str, dict[int, list[float]]] = {}

    for model in models:
        metrics[model] = {}
        for ticker in TICKERS:
            rows = forecasts[(forecasts["model"] == model) & (forecasts["ticker"] == ticker)]
            per_horizon = {}
            for horizon in (1, 22):
                bucket_rows = rows[rows["horizon"] == horizon].sort_values("origin")
                per_horizon[f"h{horizon}"] = _bucket_metrics(bucket_rows)
                if ticker == HEADLINE_TICKER:
                    ok = bucket_rows[bucket_rows["error"] == ""]
                    per_model_qlike_losses.setdefault(model, {})[horizon] = [
                        qlike([y], [f])
                        for y, f in zip(ok["rv_true"], ok["rv_pred"], strict=True)
                    ]
            metrics[model][ticker] = per_horizon

    pairs = [(model, HEADLINE_MODEL) for model in models if model != HEADLINE_MODEL]
    metrics["headline"] = headline_pair_stats(per_model_qlike_losses, pairs)

    total_failures = int((forecasts["error"] != "").sum())
    failed_rows = forecasts[forecasts["error"] != ""]
    failures_by_model = {str(k): int(v) for k, v in failed_rows.groupby("model").size().items()}
    metrics["meta"] = {
        "n_forecast_rows": len(forecasts),
        "n_failures": total_failures,
        "failures_by_model": failures_by_model,
        "horizon_dm_h": HORIZON_DM_H,
        "headline_model": HEADLINE_MODEL,
        "headline_ticker": HEADLINE_TICKER,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {METRICS_PATH} ({total_failures} total failures across {len(forecasts)} rows)")


if __name__ == "__main__":
    main()
