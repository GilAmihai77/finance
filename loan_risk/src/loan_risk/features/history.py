"""Per-client (SK_ID_CURR) aggregates of the six history tables. Each function returns a frame indexed by SK_ID_CURR.

Dates are relative to the application (DAYS_* / MONTHS_BALANCE, 0 = application date). Records dated after
the application are dropped (`_before_application`), so the aggregates are point-in-time correct and do not
leak post-application information. In the Kaggle snapshot no rows are dropped; the guard is there for live data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID = "SK_ID_CURR"


def _before_application(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return df[~(df[date_col] > 0)]          # keeps NaN dates; drops anything after the application


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Credits at other institutions (Credit Bureau) plus the worst monthly delinquency status."""
    bureau = _before_application(bureau, "DAYS_CREDIT")
    bureau_balance = _before_application(bureau_balance, "MONTHS_BALANCE")
    active = bureau["CREDIT_ACTIVE"].eq("Active")
    b = pd.DataFrame({
        ID: bureau[ID],
        "SK_ID_BUREAU": bureau["SK_ID_BUREAU"],
        "active": active,
        "overdue": bureau["CREDIT_DAY_OVERDUE"].gt(0),
        "debt_active": bureau["AMT_CREDIT_SUM_DEBT"].where(active),
        "DAYS_CREDIT": bureau["DAYS_CREDIT"],
    })
    agg = b.groupby(ID).agg(
        BUR_N_CREDITS=("SK_ID_BUREAU", "size"),
        BUR_N_ACTIVE=("active", "sum"),
        BUR_ACTIVE_SHARE=("active", "mean"),
        BUR_ANY_OVERDUE=("overdue", "max"),
        BUR_DEBT_ACTIVE_SUM=("debt_active", "sum"),
        BUR_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),   # most recent bureau credit
        BUR_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),   # oldest
    )

    # bureau_balance STATUS: 0 = on time, 1..5 = 1-30 .. 120+ days past due, C = closed, X = unknown
    dpd = pd.to_numeric(bureau_balance["STATUS"], errors="coerce")
    worst = dpd.groupby(bureau_balance["SK_ID_BUREAU"]).max()
    agg["BUR_WORST_DPD_STATUS"] = b["SK_ID_BUREAU"].map(worst).groupby(b[ID]).max()
    return agg.astype("float64")


def previous_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Earlier applications at Home Credit: refusals, how much of the request was granted, recency."""
    prev = _before_application(prev, "DAYS_DECISION")
    status = prev["NAME_CONTRACT_STATUS"]
    p = pd.DataFrame({
        ID: prev[ID],
        "refused": status.eq("Refused"),
        "approved": status.eq("Approved"),
        "credit_to_app": prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan),
        "DAYS_DECISION": prev["DAYS_DECISION"],
        "CNT_PAYMENT": prev["CNT_PAYMENT"],
    })
    return p.groupby(ID).agg(
        PREV_N_APPS=("refused", "size"),
        PREV_REFUSED_SHARE=("refused", "mean"),
        PREV_N_REFUSED=("refused", "sum"),
        PREV_APPROVED_SHARE=("approved", "mean"),
        PREV_CREDIT_TO_APP_MEAN=("credit_to_app", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    ).astype("float64")


def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Actual repayment behaviour on previous Home Credit loans.

    An instalment with no payment record (DAYS_ENTRY_PAYMENT / AMT_PAYMENT missing, ~2.9k rows) was due
    before the application but never paid: it is counted as missed and as underpaid. Lateness is measured
    on the instalments that were paid.
    """
    inst = _before_application(inst, "DAYS_INSTALMENT")
    missed = inst["DAYS_ENTRY_PAYMENT"].isna() | inst["AMT_PAYMENT"].isna()
    days_late = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    i = pd.DataFrame({
        ID: inst[ID],
        "late": (days_late > 0).astype(float).where(~missed),  # NaN for missed -> excluded from the late share
        "days_late": days_late.clip(lower=0),
        "missed": missed,
        "underpaid": missed | (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"] - 1),
    })
    return i.groupby(ID).agg(
        INST_N=("missed", "size"),
        INST_LATE_SHARE=("late", "mean"),
        INST_DAYS_LATE_MEAN=("days_late", "mean"),
        INST_DAYS_LATE_MAX=("days_late", "max"),
        INST_MISSED_SHARE=("missed", "mean"),
        INST_UNDERPAID_SHARE=("underpaid", "mean"),
    ).astype("float64")


def pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    """Monthly snapshots of previous POS / cash loans."""
    pos = _before_application(pos, "MONTHS_BALANCE")
    p = pd.DataFrame({ID: pos[ID], "dpd_def": pos["SK_DPD_DEF"] > 0, "SK_DPD": pos["SK_DPD"],
                      "MONTHS_BALANCE": pos["MONTHS_BALANCE"]})
    return p.groupby(ID).agg(
        POS_N_MONTHS=("MONTHS_BALANCE", "size"),
        POS_DPD_DEF_SHARE=("dpd_def", "mean"),
        POS_DPD_MAX=("SK_DPD", "max"),
    ).astype("float64")


def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    """Monthly snapshots of previous credit cards: utilisation, delinquency, drawings."""
    cc = _before_application(cc, "MONTHS_BALANCE")
    c = pd.DataFrame({
        ID: cc[ID],
        "util": cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan),
        "dpd_def": cc["SK_DPD_DEF"] > 0,
        "drawings": cc["AMT_DRAWINGS_CURRENT"],
    })
    return c.groupby(ID).agg(
        CC_N_MONTHS=("util", "size"),
        CC_UTIL_MEAN=("util", "mean"),
        CC_UTIL_MAX=("util", "max"),
        CC_DPD_DEF_SHARE=("dpd_def", "mean"),
        CC_DRAWINGS_MEAN=("drawings", "mean"),
    ).astype("float64")
