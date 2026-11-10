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
    y, p = np.asarray(y), np.asarray(p)
    grid = np.round(np.arange(0.005, 0.605, 0.005), 3) if grid is None else grid
    rows = []
    for t in grid:
        rej = p >= t
        tp = int((rej & (y == 1)).sum()); fp = int((rej & (y == 0)).sum()); fn = int((~rej & (y == 1)).sum())
        n_appr = int((~rej).sum())
        rows.append({
            "threshold": float(t),
            "reject_rate": float(rej.mean()),
            "recall": tp / max(tp + fn, 1),                    # share of defaulters caught
            "precision": tp / max(tp + fp, 1),
            "cost_per_app": (cost_fn * fn + cost_fp * fp) / len(y),
            "default_rate_approved": fn / max(n_appr, 1),     # risk of the approved book
        })
    return pd.DataFrame(rows).set_index("threshold")


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
    decisions = decide(p, {"t_low": t_low, "t_high": t_high})
    return {
        "threshold": t, "binding_rule": binding, "t_bayes": t_bayes, "t_cost": t_cost,
        "t_low": t_low, "t_high": t_high, "cost_fn": c_fn, "cost_fp": c_fp,
        "oof_reject_rate": float(row["reject_rate"]),
        "oof_recall": float(row["recall"]),
        "oof_default_rate_approved": float(row["default_rate_approved"]),
        "oof_band_shares": {d: float((decisions == d).mean()) for d in (APPROVE, REVIEW, REJECT)},
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
