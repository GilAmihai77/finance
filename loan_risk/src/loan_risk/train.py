"""Stratified K-fold CV (out-of-fold PDs for evaluation and thresholds) and a final refit on all data."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .decision import choose_thresholds, cost_table
from .evaluate import calibration_table, classification_metrics
from .explain import global_importance
from .features import FeatureSpec
from .scoring import META_FILE, MODEL_FILE, SPEC_FILE, THRESH_FILE


def cross_validate(X: pd.DataFrame, y: pd.Series, params: dict, cfg: dict) -> tuple:
    folds = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(X))
    fold_auc, best_iters = [], []
    for k, (tr, va) in enumerate(folds.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], y.iloc[va], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=cfg["num_boost_round"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        fold_auc.append(classification_metrics(y.iloc[va], oof[va])["auc"])
        best_iters.append(b.best_iteration)
        print(f"fold {k}: auc={fold_auc[-1]:.4f} trees={b.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    return oof, {"fold_auc": fold_auc, "fold_auc_std": float(np.std(fold_auc)), "best_iters": best_iters}


def run_training(train_df: pd.DataFrame, config: dict, model_dir: str | Path,
                 report_dir: str | Path | None = None) -> dict:
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    id_col, target, mcfg = config["id_col"], config["target"], config["model"]
    params = {**mcfg["lgbm_params"], "seed": mcfg["seed"]}

    spec = FeatureSpec.fit(train_df, drop=[id_col, target] + list(config["features"]["exclude"]))
    X, y = spec.transform(train_df), train_df[target].astype(int)
    print(f"training rows={len(X):,} features={len(spec.features)} "
          f"(categorical={len(spec.categorical)}) default_rate={y.mean():.4f}", flush=True)

    oof, cv_info = cross_validate(X, y, params, mcfg)
    cv_metrics = classification_metrics(y, oof)
    thresholds = choose_thresholds(y, oof, config["decision"])

    n_rounds = int(np.mean(cv_info["best_iters"]) * mcfg["refit_rounds_multiplier"])
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_rounds)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    meta = {"model_version": version, "n_train": int(len(X)), "n_rounds": n_rounds,
            "cv": {**cv_metrics, **cv_info}, "excluded_features": config["features"]["exclude"],
            "lightgbm_version": lgb.__version__}
    booster.save_model(str(model_dir / MODEL_FILE))
    spec.save(model_dir / SPEC_FILE)
    (model_dir / THRESH_FILE).write_text(json.dumps(thresholds, indent=1))
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=1))

    if report_dir:
        r = Path(report_dir); r.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({id_col: train_df[id_col].values, "y": y.values, "oof_pd": oof}).to_parquet(r / "oof.parquet")
        cost_table(y, oof, thresholds["cost_fn"], thresholds["cost_fp"]).to_csv(r / "threshold_table.csv")
        calibration_table(y, oof).to_csv(r / "calibration_oof.csv")
        gain = pd.Series(booster.feature_importance("gain"), index=spec.features)
        sample = X.sample(min(len(X), 20_000), random_state=mcfg["seed"])
        pd.DataFrame({"mean_abs_shap": global_importance(booster, sample), "gain_share": gain / gain.sum()})             .sort_values("mean_abs_shap", ascending=False).to_csv(r / "feature_importance.csv")

    # SageMaker metric_definitions parse these lines into CloudWatch / Experiments
    print(f"cv_auc={cv_metrics['auc']:.5f}")
    print(f"cv_gini={cv_metrics['gini']:.5f}")
    print(f"cv_ks={cv_metrics['ks']:.5f}")
    print(f"threshold={thresholds['threshold']:.4f} (bayes {thresholds['t_bayes']:.4f}) "
          f"band=[{thresholds['t_low']:.4f}, {thresholds['t_high']:.4f})", flush=True)
    return meta
