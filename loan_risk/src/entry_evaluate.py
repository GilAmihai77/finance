"""Evaluation step: score the untouched holdout and write evaluation.json.
The pipeline's quality gate reads binary_classification_metrics.auc.value from this file.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from loan_risk import load_config
from loan_risk.data import read_dir
from loan_risk.decision import decision_metrics
from loan_risk.evaluate import calibration_table, classification_metrics, sagemaker_model_quality
from loan_risk.scoring import LoanRiskModel

P = "/opt/ml/processing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{P}/model")
    ap.add_argument("--holdout", default=f"{P}/holdout")
    ap.add_argument("--output", default=f"{P}/evaluation")
    a = ap.parse_args()
    cfg = load_config()

    model_dir = Path(a.model)
    tar = model_dir / "model.tar.gz"
    if tar.exists():                       # SageMaker hands over the packed training artifact
        with tarfile.open(tar) as t:
            if hasattr(tarfile, "data_filter"):   # Python >= 3.9.17 / 3.12: block path traversal
                t.extractall(model_dir, filter="data")
            else:
                root = model_dir.resolve()
                if any(not (model_dir / m.name).resolve().is_relative_to(root) for m in t.getmembers()):
                    raise ValueError("unsafe path in model.tar.gz")
                t.extractall(model_dir)
    model = LoanRiskModel.load(model_dir)

    ho = read_dir(a.holdout)
    y = ho[cfg["target"]].astype(int)
    p = model.predict_pd(ho)
    m = classification_metrics(y, p)

    report = sagemaker_model_quality(m)
    report["decision_bands"] = decision_metrics(y, p, model.thresholds)
    report["thresholds"] = model.thresholds
    report["calibration_deciles"] = calibration_table(y, p).round(5).reset_index().to_dict(orient="records")
    report["model_version"] = model.meta.get("model_version")

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}))
    print(json.dumps(report["decision_bands"]))


if __name__ == "__main__":
    main()
