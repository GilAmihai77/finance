"""SHAP explanations via LightGBM's built-in TreeSHAP (pred_contrib=True), so no `shap` dependency."""
from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row contributions in log-odds; row sum + bias = model margin."""
    contrib = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def global_importance(booster: lgb.Booster, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature (log-odds). Less biased than split gain towards high-cardinality features."""
    return shap_values(booster, X).abs().mean().sort_values(ascending=False)


def _json_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def reason_codes(booster: lgb.Booster, X: pd.DataFrame, k: int = 3) -> list:
    """Top-k features pushing each applicant towards default (for adverse-action notices)."""
    sv = booster.predict(X, pred_contrib=True)[:, :-1]
    top = np.argsort(-sv, axis=1)[:, :k]                      # one sort for all rows
    cols = X.columns
    out = []
    for i, idx in enumerate(top):
        out.append([{"feature": cols[j], "value": _json_value(X.iat[i, j]), "impact": round(float(sv[i, j]), 4)}
                    for j in idx if sv[i, j] > 0])           # only features pushing towards default
    return out
