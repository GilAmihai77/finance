# Loan default risk (PD) model: local package + SageMaker MLOps

Predicts the probability of default (PD) for a loan application (Home Credit dataset). It turns the PD
into an **Approve / Review / Reject** decision with a cost-based threshold, and explains each decision
with SHAP reason codes. Deployed as a SageMaker Pipeline with a quality gate, Model Registry approval,
and a monitored endpoint.

## Results (local run, CODE_GENDER excluded)
| | AUC | Gini | KS | Brier | mean PD vs default rate |
|---|---|---|---|---|---|
| 5-fold CV (246k) | 0.780 | 0.559 | 0.420 | | |
| Untouched holdout (61.5k) | **0.782** | **0.564** | **0.425** | 0.066 | 0.0797 vs 0.0807 (calibrated) |

Fairness: what each exclusion costs (holdout AUC):
| features excluded | CV AUC | holdout AUC |
|---|---|---|
| none | 0.781 | 0.784 |
| CODE_GENDER (production) | 0.780 | 0.782 |
| CODE_GENDER + AGE_YEARS | 0.778 | 0.780 |

Decision layer (C_FN = LGD 0.60, C_FP = margin 0.12): theoretical cut-off 0.167, empirical OOF optimum 0.165,
review band [0.131, 0.220). On the holdout (OOF shares: 82.6% / 10.0% / 7.4%):
| band | share | default rate |
|---|---|---|
| APPROVE | 82.7% | 4.7% |
| REVIEW | 9.9% | 17.1% |
| REJECT | 7.5% | 33.6% |

How to read the thresholds: the cost-optimal cut-off `t` is the centre of the review band. The model only
auto-rejects at `p >= t_high` and auto-approves at `p < t_low`. `thresholds.json` reports both the binary rule at
`t` (`oof_at_threshold`) and the shares and default rates of the three bands that are actually deployed
(`oof_bands`). The optional risk-appetite cap is checked at `t`. That means the cap still holds if underwriters
approve every REVIEW case below `t`.

## Design notes and known limitations
- **Thresholds are set on out-of-fold PDs but applied to the refit model.** The refit model sees all the data and
  uses 1.1x the mean number of trees, so its PD distribution can shift slightly. The holdout band shares
  (approve / review / reject) match the OOF shares to within 0.2pp, which shows the shift is small.
- **Early stopping uses the same fold as the out-of-fold predictions.** This makes OOF AUC very slightly
  optimistic. The quality gate uses the untouched holdout, which is unbiased (holdout AUC is in fact higher than CV).
- **Unseen categorical levels map to NaN** (missing), not to an "other" level. LightGBM sends them down the
  learned missing-value branch.
- **Point-in-time features:** history records dated after the application are dropped before aggregation.
  None are dropped in the Kaggle snapshot. An instalment with no payment is counted as missed and underpaid.
- **Fairness:** `CODE_GENDER` is excluded. Age (`AGE_YEARS`) is kept. Under fair-lending rules, age can be
  used in an empirically derived scoring system provided older applicants are not scored down for their age.
  The AUC cost of each exclusion is in the Results table.
- **Deliberately not built for the demo:** KMS keys for S3 and volumes, VPC-only jobs with network isolation,
  PII masking in data capture (it records 100% of requests). In production these are required. They are
  parameters on the same SDK calls (`output_kms_key`, `volume_kms_key`, `network_config`, `subnets` /
  `security_group_ids`, and capture with a KMS key and lower sampling).
- **S3 layout:** raw data is stored as immutable snapshots (`raw/v1`, `raw/v2`, ...). Step caching keys on step
  arguments, not on S3 contents, so a new snapshot needs a new prefix. Features are written per execution
  (`features/<execution-id>/...`) for lineage.

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
python src/entry_train.py ... --exclude CODE_GENDER,AGE_YEARS          # fairness comparison (--exclude "" = none)
pytest -q
```

## AWS architecture
```
S3 raw/v1/ (parquet) ─► SageMaker Pipeline "loan-risk-pipeline"
   BuildFeatures (Processing) ─► TrainModel (Training, 5-fold CV) ─► EvaluateModel (holdout)
        ─► AucQualityGate: AUC >= 0.77 ? ─► RegisterModel (Model Registry, PendingManualApproval)
                                        └► AucBelowGate (Fail)
Human approval (model risk / four-eyes) ─► real-time endpoint + Data Capture ─► Model Monitor
```
