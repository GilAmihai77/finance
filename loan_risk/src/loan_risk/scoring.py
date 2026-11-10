"""Loading a trained model directory and scoring applicants. Batch, local, and endpoint scoring all use this."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .decision import decide
from .explain import reason_codes
from .features import FeatureSpec, prep_application

MODEL_FILE, SPEC_FILE, THRESH_FILE, META_FILE = "model.txt", "feature_spec.json", "thresholds.json", "model_meta.json"


class LoanRiskModel:
    def __init__(self, booster: lgb.Booster, spec: FeatureSpec, thresholds: dict, meta: dict | None = None):
        self.booster, self.spec, self.thresholds, self.meta = booster, spec, thresholds, meta or {}

    @classmethod
    def load(cls, model_dir: str | Path) -> "LoanRiskModel":
        d = Path(model_dir)
        meta = json.loads((d / META_FILE).read_text()) if (d / META_FILE).exists() else {}
        return cls(lgb.Booster(model_file=str(d / MODEL_FILE)), FeatureSpec.load(d / SPEC_FILE),
                   json.loads((d / THRESH_FILE).read_text()), meta)

    def predict_pd(self, features: pd.DataFrame):
        return self.booster.predict(self.spec.transform(features))

    def score_features(self, features: pd.DataFrame, explain: bool = True, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
        """`features` = output of build_features (already prepped)."""
        X = self.spec.transform(features)
        p = self.booster.predict(X)                       # = predict_pd, but X is reused for reason codes
        out = pd.DataFrame({"pd": p, "decision": decide(p, self.thresholds)}, index=features.index)
        if id_col in features.columns:
            out.insert(0, id_col, features[id_col].values)
        if explain:
            out["reason_codes"] = reason_codes(self.booster, X)
        out["model_version"] = self.meta.get("model_version", "unknown")
        return out

    def score_raw(self, raw: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        """`raw` = raw application fields + precomputed history aggregates (as sent to the endpoint)."""
        return self.score_features(prep_application(raw), explain=explain)
