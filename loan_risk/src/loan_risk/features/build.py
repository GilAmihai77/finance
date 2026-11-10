"""Single source of truth for features: used by the processing job, training, and the endpoint."""
from __future__ import annotations

import gc
from typing import Callable

import pandas as pd

from .application import prep_application
from .history import (ID, bureau_features, credit_card_features, installments_features,
                      pos_cash_features, previous_application_features)

# Only the columns each aggregation needs (keeps memory low: bureau_balance alone has 27M rows)
HISTORY_COLUMNS = {
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE",
               "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT"],
    "bureau_balance": ["SK_ID_BUREAU", "STATUS"],
    "previous_application": ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
                             "DAYS_DECISION", "CNT_PAYMENT"],
    "installments_payments": ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
                              "AMT_PAYMENT"],
    "POS_CASH_balance": ["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "MONTHS_BALANCE"],
    "credit_card_balance": ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD_DEF",
                            "AMT_DRAWINGS_CURRENT"],
}


def build_history(read: Callable[[str, list], pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """Aggregate all history tables into one frame indexed by SK_ID_CURR.

    `read(table_name, columns)` returns the raw table. Tables are loaded one at a time and released.
    In production this frame is what gets written to a feature store (online lookup by SK_ID_CURR).
    """
    def load(name):
        df = read(name, HISTORY_COLUMNS[name])
        if verbose:
            print(f"  loaded {name}: {len(df):,} rows")
        return df

    parts = [bureau_features(load("bureau"), load("bureau_balance"))]
    gc.collect()
    for name, fn in [("previous_application", previous_application_features),
                     ("installments_payments", installments_features),
                     ("POS_CASH_balance", pos_cash_features),
                     ("credit_card_balance", credit_card_features)]:
        parts.append(fn(load(name)))
        gc.collect()
    hist = pd.concat(parts, axis=1)
    hist.index.name = ID
    return hist


def build_features(app: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Application features + history aggregates (left join: clients without history get NaN)."""
    out = prep_application(app)
    return out.merge(history, left_on=ID, right_index=True, how="left")
