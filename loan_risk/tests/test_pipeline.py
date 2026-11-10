"""Fast unit tests on synthetic data (no Kaggle files needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loan_risk import load_config  # noqa: E402
from loan_risk.decision import APPROVE, REJECT, REVIEW, choose_thresholds, decide  # noqa: E402
from loan_risk.features import FeatureSpec, build_features, prep_application  # noqa: E402
from loan_risk.features.history import bureau_features, installments_features  # noqa: E402
from loan_risk.scoring import LoanRiskModel  # noqa: E402
from loan_risk.train import run_training  # noqa: E402


def make_app(n=2000, seed=0):
    r = np.random.default_rng(seed)
    ext = r.uniform(0, 1, n)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "DAYS_BIRTH": -r.integers(20 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": np.where(r.uniform(size=n) < 0.18, 365243, -r.integers(0, 40 * 365, n)),
        "AMT_INCOME_TOTAL": r.uniform(30e3, 400e3, n),
        "AMT_CREDIT": r.uniform(50e3, 2e6, n),
        "AMT_ANNUITY": r.uniform(2e3, 80e3, n),
        "AMT_GOODS_PRICE": r.uniform(40e3, 2e6, n),
        "EXT_SOURCE_1": np.where(r.uniform(size=n) < 0.5, np.nan, r.uniform(size=n)),
        "EXT_SOURCE_2": ext,
        "EXT_SOURCE_3": r.uniform(size=n),
        "NAME_CONTRACT_TYPE": r.choice(["Cash loans", "Revolving loans"], n),
        "OCCUPATION_TYPE": r.choice(["Laborers", "Managers", None], n),
        "CODE_GENDER": r.choice(["M", "F"], n),
    })
    df["TARGET"] = (r.uniform(size=n) < 0.25 * (1 - ext)).astype(int)   # PD driven by EXT_SOURCE_2
    return df


def test_prep_handles_placeholder_and_missing_columns():
    out = prep_application(pd.DataFrame({"DAYS_EMPLOYED": [365243, -3650], "DAYS_BIRTH": [-20000, -10000]}))
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [1, 0]
    assert np.isnan(out["DAYS_EMPLOYED"].iloc[0])
    assert out["YEARS_EMPLOYED"].iloc[1] == pytest.approx(10, rel=1e-3)
    assert "DAYS_BIRTH" not in out and "CREDIT_INCOME_RATIO" in out   # missing inputs -> NaN, no crash


def test_history_one_row_per_client():
    bureau = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "SK_ID_BUREAU": [10, 11, 12],
                           "CREDIT_ACTIVE": ["Active", "Closed", "Active"], "CREDIT_DAY_OVERDUE": [0, 5, 0],
                           "AMT_CREDIT_SUM_DEBT": [100.0, 50.0, np.nan], "DAYS_CREDIT": [-10, -500, -30]})
    bb = pd.DataFrame({"SK_ID_BUREAU": [10, 10, 11, 12], "STATUS": ["0", "2", "C", "X"]})
    b = bureau_features(bureau, bb)
    assert b.index.is_unique and len(b) == 2
    assert b.loc[1, "BUR_WORST_DPD_STATUS"] == 2 and np.isnan(b.loc[2, "BUR_WORST_DPD_STATUS"])
    assert b.loc[1, "BUR_DEBT_ACTIVE_SUM"] == 100 and b.loc[1, "BUR_ANY_OVERDUE"] == 1

    inst = pd.DataFrame({"SK_ID_CURR": [1, 1], "DAYS_INSTALMENT": [-100.0, -70.0], "DAYS_ENTRY_PAYMENT": [-95.0, -75.0],
                         "AMT_INSTALMENT": [100.0, 100.0], "AMT_PAYMENT": [100.0, 50.0]})
    i = installments_features(inst)
    assert i.loc[1, "INST_LATE_SHARE"] == 0.5 and i.loc[1, "INST_UNDERPAID_SHARE"] == 0.5


def test_spec_unseen_category_and_missing_column():
    train = pd.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
    spec = FeatureSpec.fit(train, drop=[])
    X = spec.transform(pd.DataFrame({"c": ["z", "x"]}))
    assert list(X.columns) == ["a", "c"] and X["a"].isna().all()
    assert pd.isna(X["c"].iloc[0]) and X["c"].iloc[1] == "x"
    assert list(X["c"].cat.categories) == ["x", "y"]


def test_thresholds_ordering_and_bands():
    r = np.random.default_rng(1)
    p = r.uniform(0, 0.5, 20000)
    y = (r.uniform(size=p.size) < p).astype(int)           # perfectly calibrated
    th = choose_thresholds(y, p, {"cost_fn": 0.6, "cost_fp": 0.12, "review_share": 0.1})
    assert th["t_low"] <= th["threshold"] <= th["t_high"]
    assert abs(th["t_cost"] - th["t_bayes"]) < 0.03         # calibrated -> empirical ~ theoretical
    assert th["oof_band_shares"]["REVIEW"] == pytest.approx(0.1, abs=0.02)
    d = decide([0.0, th["threshold"], 0.99], th)
    assert d[0] == APPROVE and d[1] in (REVIEW, REJECT) and d[2] == REJECT


def test_train_score_roundtrip(tmp_path):
    cfg = load_config()
    cfg["model"].update(n_folds=3, num_boost_round=200, early_stopping_rounds=20)
    app = make_app()
    feats = build_features(app, pd.DataFrame(index=pd.Index([], name="SK_ID_CURR")))
    meta = run_training(feats, cfg, tmp_path / "model", tmp_path / "reports")
    assert meta["cv"]["auc"] > 0.6
    spec = json.loads((tmp_path / "model" / "feature_spec.json").read_text())
    assert "CODE_GENDER" not in spec["features"] and "TARGET" not in spec["features"]

    m = LoanRiskModel.load(tmp_path / "model")
    raw = app.drop(columns="TARGET").head(5)
    s_raw = m.score_raw(raw)
    s_feat = m.score_features(feats.head(5))
    np.testing.assert_allclose(s_raw["pd"], s_feat["pd"], rtol=1e-12)   # raw path == batch path
    assert set(s_raw["decision"]) <= {APPROVE, REVIEW, REJECT}
    assert all(len(rc) <= 3 for rc in s_raw["reason_codes"])

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import entry_serve
    body = json.dumps({"instances": json.loads(raw.to_json(orient="records", double_precision=15))})
    out, ctype = entry_serve.output_fn(entry_serve.predict_fn(entry_serve.input_fn(body), m))
    preds = json.loads(out)["predictions"]
    np.testing.assert_allclose([x["pd"] for x in preds], s_raw["pd"], rtol=1e-9)   # JSON round trip
