"""Credit-risk model metrics: AUC, Gini, KS, PR-AUC, Brier, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    """Max distance between the score CDFs of defaulters and non-defaulters (= max(TPR - FPR))."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def classification_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),          # the usual banking name for the same ranking power
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(p.mean()),        # vs base_rate: overall calibration
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per score decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    """Direction-free single-feature AUC (categoricals target-rate encoded, NaN -> median)."""
    if not pd.api.types.is_numeric_dtype(x):
        x = x.astype(str).map(y.groupby(x.astype(str)).mean())
    x = x.fillna(x.median())
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def sagemaker_model_quality(m: dict) -> dict:
    """Format metrics as SageMaker Model Registry 'model quality' JSON (shown in Studio)."""
    return {"binary_classification_metrics": {
        k: {"value": v, "standard_deviation": "NaN"} for k, v in m.items() if k not in ("n",)
    }}
