# Loan default risk (PD) model: local package + SageMaker MLOps

Predicts the probability of default (PD) for a loan application (Home Credit dataset). It turns the PD
into an **Approve / Review / Reject** decision with a cost-based threshold, and explains each decision
with SHAP reason codes. Deployed as a SageMaker Pipeline with a quality gate, Model Registry approval,
and a monitored endpoint.

## Results (local run, CODE_GENDER excluded)
| | AUC | Gini | KS | Brier | mean PD vs default rate |
|---|---|---|---|---|---|
| 5-fold CV (246k) | 0.778 | 0.557 | 0.419 | | |
| Untouched holdout (61.5k) | **0.782** | **0.563** | **0.427** | 0.066 | 0.0798 vs 0.0807 (calibrated) |

Decision layer (C_FN = LGD 0.60, C_FP = margin 0.12): theoretical cut-off 0.167, empirical OOF optimum 0.155,
review band [0.124, 0.203). On the holdout:
| band | share | default rate |
|---|---|---|
| APPROVE | 81.5% | 4.6% |
| REVIEW | 9.9% | 16.0% |
| REJECT | 8.7% | 31.5% |

## Layout
```
src/loan_risk/            package (shipped to every SageMaker job as source_dir)
  config.yaml             all settings: features to exclude, LGBM params, costs, AUC gate
  data.py                 parquet/CSV readers, CSV -> parquet
  features/application.py application cleaning + ratios (365243 placeholder fix, age, credit/income ...)
  features/history.py     per-client aggregates of the 6 history tables
  features/build.py       single source of truth for features (processing + serving)
  features/spec.py        frozen feature contract (columns + categorical levels) -> no train/serve skew
  train.py                stratified 5-fold CV -> OOF PDs -> thresholds -> refit on all data
  evaluate.py             AUC, Gini, KS, PR-AUC, Brier, calibration
  decision.py             cost-based threshold, risk-appetite cap, manual-review band
  explain.py              TreeSHAP reason codes (LightGBM pred_contrib, no shap dependency)
  scoring.py              LoanRiskModel: load artifacts, score raw or prepared rows
src/entry_*.py            SageMaker entry points (same scripts run locally)
pipelines/build_pipeline.py   SageMaker Pipeline definition
pipelines/manage_model.py     registry list/approve, deploy, delete
pipelines/invoke_endpoint.py  send real applications, compare local vs endpoint PDs
tests/                    unit tests on synthetic data
```

## Run locally (conda env `loanrisk`)
```bash
python -m loan_risk.data --src ../data --dst ../data/parquet               # with PYTHONPATH=src
python src/entry_preprocess.py --input ../data/parquet --output artifacts/features
python src/entry_train.py --train artifacts/features/train --model-dir artifacts/model --report-dir artifacts/reports
python src/entry_evaluate.py --model artifacts/model --holdout artifacts/features/holdout --output artifacts/evaluation
python pipelines/invoke_endpoint.py --local-model artifacts/model
pytest -q
```

## AWS architecture
```
S3 raw/ (parquet) ─► SageMaker Pipeline "loan-risk-pipeline"
   BuildFeatures (Processing) ─► TrainModel (Training, 5-fold CV) ─► EvaluateModel (holdout)
        ─► AucQualityGate: AUC >= 0.77 ? ─► RegisterModel (Model Registry, PendingManualApproval)
                                        └► AucBelowGate (Fail)
Human approval (model risk / four-eyes) ─► real-time endpoint + Data Capture ─► Model Monitor
```
