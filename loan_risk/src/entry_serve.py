"""SageMaker inference handlers for the SKLearn serving container.

Request  (application/json): {"instances": [{<raw application fields> + <history aggregates>}, ...]}
         (a single object or a bare list also works)
Response (application/json): {"predictions": [{"SK_ID_CURR", "pd", "decision", "reason_codes", "model_version"}]}
"""
from __future__ import annotations

import json

import pandas as pd

from loan_risk.scoring import LoanRiskModel


def model_fn(model_dir):
    return LoanRiskModel.load(model_dir)


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type}")
    payload = json.loads(body)
    if isinstance(payload, dict):
        payload = payload.get("instances", [payload])
    return pd.DataFrame(payload)


def predict_fn(df, model):
    return model.score_raw(df)


def output_fn(pred, accept="application/json"):
    return json.dumps({"predictions": json.loads(pred.to_json(orient="records", double_precision=15))}), \
        "application/json"
