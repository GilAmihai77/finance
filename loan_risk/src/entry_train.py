"""Training step. As a SageMaker Training job it reads the 'train' channel and writes to SM_MODEL_DIR,
which SageMaker packs into model.tar.gz. Locally pass --train / --model-dir / --report-dir.
"""
from __future__ import annotations

import argparse
import os

from loan_risk import load_config
from loan_risk.data import read_dir
from loan_risk.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--report-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    # SageMaker hyperparameters arrive as CLI args; these override config.yaml
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--exclude", default=None, help="comma-separated features to exclude ('' = none)")
    a = ap.parse_args()

    cfg = load_config()
    if a.n_folds:
        cfg["model"]["n_folds"] = a.n_folds
    if a.learning_rate:
        cfg["model"]["lgbm_params"]["learning_rate"] = a.learning_rate
    if a.exclude is not None:
        cfg["features"]["exclude"] = [c for c in a.exclude.split(",") if c]

    run_training(read_dir(a.train), cfg, a.model_dir, a.report_dir)


if __name__ == "__main__":
    main()
