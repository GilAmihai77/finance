"""Turning a PD into an Approve / Review / Reject decision using business costs.

Expected cost of approving = p * C_FN, of rejecting = (1 - p) * C_FP, so with calibrated PDs the
theoretical cut-off is p* = C_FP / (C_FP + C_FN). The threshold actually used is the empirical
cost minimum on out-of-fold predictions (should be close to p*), optionally tightened by a
risk-appetite cap. A manual-review band of `review_share` of applicants is placed around it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

APPROVE, REVIEW, REJECT = "APPROVE", "REVIEW", "REJECT"


def cost_table(y, p, cost_fn: float, cost_fp: float, grid=None) -> pd.DataFrame:
    """Binary rule `reject if p >= t` evaluated on a threshold grid (one sort + cumulative sums)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    grid = np.round(np.arange(0.005, 1.0, 0.005), 3) if grid is None else np.asarray(grid, dtype=float)
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    cum_bad = np.concatenate([[0.0], np.cumsum(y_sorted)])          # defaulters among the k lowest PDs
    n, n_bad = len(y), y.sum()
    n_appr = np.searchsorted(p_sorted, grid, side="left")           # approved = p < t
    fn = cum_bad[n_appr]                                            # approved defaulters
    tp = n_bad - fn
    fp = (n - n_appr) - tp
    return pd.DataFrame({
        "threshold": grid,
        "reject_rate": (n - n_appr) / n,
        "recall": tp / max(n_bad, 1),                               # share of defaulters caught
        "precision": tp / np.maximum(n - n_appr, 1),
        "cost_per_app": (cost_fn * fn + cost_fp * fp) / n,
        "default_rate_approved": fn / np.maximum(n_appr, 1),        # risk of the approved book
    }).set_index("threshold")


def choose_thresholds(y, p, cfg: dict) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    c_fn, c_fp = float(cfg["cost_fn"]), float(cfg["cost_fp"])
    table = cost_table(y, p, c_fn, c_fp)

    t_bayes = c_fp / (c_fp + c_fn)
    t_cost = float(table["cost_per_app"].idxmin())
    t, binding = t_cost, "cost"

    cap = cfg.get("max_approved_default_rate")
    if cap is not None:
        feasible = table[table["default_rate_approved"] <= cap]
        if feasible.empty:
            raise ValueError(f"No threshold keeps the approved-book default rate <= {cap}")
        # Checked at the cut-off t, not at t_low: even if underwriters approve every REVIEW case below t,
        # the approved book still meets the cap (the auto-approved book, p < t_low, is safer still).
        t_cap = float(feasible.index.max())                  # the loosest threshold that meets the cap
        if t_cap < t:
            t, binding = t_cap, "risk_appetite_cap"

    # Review band: `review_share` of applicants centred (in rank) on the cut-off
    share = float(cfg.get("review_share", 0.0))
    q = float((p < t).mean())
    t_low = float(np.quantile(p, max(q - share / 2, 0.0)))
    t_high = float(np.quantile(p, min(q + share / 2, 1.0)))
    t_low, t_high = min(t_low, t), max(t_high, t)

    row = table.loc[t]
    th = {"t_low": t_low, "t_high": t_high}
    return {
        "threshold": t, "binding_rule": binding, "t_bayes": t_bayes, "t_cost": t_cost,
        "t_low": t_low, "t_high": t_high, "cost_fn": c_fn, "cost_fp": c_fp,
        # Binary rule at the cut-off t (= centre of the review band), i.e. if REVIEW were split at t
        "oof_at_threshold": {"reject_rate": float(row["reject_rate"]), "recall": float(row["recall"]),
                             "default_rate_approved": float(row["default_rate_approved"])},
        # What the deployed three-band rule actually produces
        "oof_bands": decision_metrics(y, p, th),
    }


def decide(p, th: dict) -> np.ndarray:
    p = np.asarray(p)
    return np.where(p >= th["t_high"], REJECT, np.where(p < th["t_low"], APPROVE, REVIEW))


def decision_metrics(y, p, th: dict) -> dict:
    """How the bands perform on labelled data (used on the holdout set)."""
    y, d = np.asarray(y), decide(p, th)
    out = {}
    for band in (APPROVE, REVIEW, REJECT):
        m = d == band
        out[f"{band.lower()}_share"] = float(m.mean())
        out[f"{band.lower()}_default_rate"] = float(y[m].mean()) if m.any() else None
    return out
