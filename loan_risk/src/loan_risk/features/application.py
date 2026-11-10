"""Cleaning and derived features on the main application table (one row per loan application)."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_EMPLOYED_PLACEHOLDER = 365243   # ~1000 years; used for pensioners / unemployed

RAW_USED = ["DAYS_EMPLOYED", "DAYS_BIRTH", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
            "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

ENGINEERED = ["DAYS_EMPLOYED_ANOM", "AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO",
              "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "EXT_SOURCE_MEAN"]


def prep_application(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as the EDA notebook's prep(). Tolerates missing columns, which matters at inference time."""
    df = df.copy()
    for c in RAW_USED:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_EMPLOYED_PLACEHOLDER)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25

    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = df["AMT_CREDIT"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / credit          # ~ 1 / number of payments
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    return df.drop(columns=["DAYS_BIRTH"])
