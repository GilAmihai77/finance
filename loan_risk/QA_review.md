# QA review: loan_risk package and SageMaker pipeline

Scope: every source file under `loan_risk/`, checked against `plan.txt`. No code was changed.
The saved artifacts were used to confirm finding #1. The tests were not run.

Overall the code is lean and in good shape. There are a few real correctness issues and a handful of cleanups.

## High priority (bugs / wrong numbers)

| # | Issue | Where | Why it matters / suggestion |
|---|---|---|---|
| 1 | **The reported threshold stats don't match the decisions the model actually makes.** `oof_reject_rate`, `oof_recall` and `oof_default_rate_approved` are calculated at the single cut-off `t`. But `decide()` rejects only at `p ≥ t_high` and approves only at `p < t_low`. The current artifact shows it: `oof_reject_rate = 13.6%` but `oof_band_shares.REJECT = 8.6%`. The risk-appetite cap is also checked at `t`, not at the real approve boundary `t_low`. | [decision.py:60-69](src/loan_risk/decision.py#L60-L69) | A validator who reads `thresholds.json` or the registry sees a reject rate the model never produces. Report the stats for the actual bands, and either document `t` as only the centre of the review band, or check the cap at `t_low`. |
| 2 | **Pipeline caching can silently reuse old features.** SageMaker decides whether to reuse a step from its arguments, not from the S3 contents. If new raw data lands at the same `InputDataUri` within 30 days, `BuildFeatures` (and then `TrainModel`) is skipped and the old features are reused. | [build_pipeline.py:45](pipelines/build_pipeline.py#L45), [:62](pipelines/build_pipeline.py#L62), [:76](pipelines/build_pipeline.py#L76) | This breaks the plan's promise that thresholds are recomputed on every retrain. Version the raw input path by date or snapshot, or turn caching off for `BuildFeatures`. |
| 3 | **Feature outputs go to a fixed S3 path** (`{base}/features/{train,…}`) that every run overwrites. | [build_pipeline.py:58-60](pipelines/build_pipeline.py#L58-L60) | Two runs at once will clash, and you lose the record of which data trained which model (this matters for model-risk audits). Add `ExecutionVariables.PIPELINE_EXECUTION_ID` to the path, as is already done for evaluation with `TrainingJobName`. |
| 4 | **Missed payments count as "on time".** If `DAYS_ENTRY_PAYMENT` or `AMT_PAYMENT` is NaN (the instalment was never paid), then `days_late > 0` and `AMT_PAYMENT < …` are both False. | [history.py:66-72](src/loan_risk/features/history.py#L66-L72) | The worst repayment behaviour is recorded as good behaviour. Few rows are affected, but it's a modelling error. Add an `INST_MISSED_SHARE` feature, or treat NaN as late or underpaid. |

## Medium priority

| # | Issue | Where | Why / suggestion |
|---|---|---|---|
| 5 | The same fold is used for both early stopping and the OOF predictions. | [train.py:28-30](src/loan_risk/train.py#L28-L30) | OOF AUC is slightly optimistic, and the thresholds are tuned on those optimistic PDs. The holdout gate is still unbiased, so this is mostly something to be ready to explain. A cleaner option is an inner split for early stopping. |
| 6 | Thresholds come from the OOF predictions of the 5 fold models, but they're applied to the refit model, which sees more data and uses 1.1× the trees. | [train.py:50-53](src/loan_risk/train.py#L50-L53) | The probability distribution can shift a little. The holdout band results (8.7% reject) show it's small. Worth one sentence in the README. |
| 7 | `config.yaml: evaluation.auc_gate` is never read. The pipeline hardcodes `0.77` separately. | [config.yaml:39](src/loan_risk/config.yaml#L39), [build_pipeline.py:43](pipelines/build_pipeline.py#L43) | Two places to keep in sync. Either read the config value in `build_pipeline` or delete the key. |
| 8 | `reason_codes` runs a Python loop over every row, calling `.iloc` and sorting each one. | [explain.py:35-39](src/loan_risk/explain.py#L35-L39) | Fine for 5 rows sent to the endpoint, but slow for batch scoring of 48k rows. Use `np.argsort` on the contribution matrix once. |
| 9 | Security settings in the plan aren't implemented: no KMS keys (`output_kms_key`, `volume_kms_key`), no VPC or network isolation, and data capture records 100% of requests, including personal data. | [build_pipeline.py](pipelines/build_pipeline.py), [manage_model.py:67-70](pipelines/manage_model.py#L67-L70) | The plan sells this as the bank security story. At least be ready to say it's deliberately not built. |
| 10 | Fairness: `CODE_GENDER` is excluded, but `AGE_YEARS` (age) is still a feature. The "AUC cost of excluding them" comparison from plan §3 isn't built. | [application.py:26](src/loan_risk/features/application.py#L26) | An interviewer may ask about it. A quick run with `--exclude CODE_GENDER,AGE_YEARS` would give the number. |

## Low priority (robustness)

| # | Issue | Where |
|---|---|---|
| 11 | `packages(...)[0]` raises a bare `IndexError` when no package is pending or approved. It should give a clear message. | [manage_model.py:51,59](pipelines/manage_model.py#L51) |
| 12 | `input_fn` requires the content type to be exactly `"application/json"`. It rejects `application/json; charset=utf-8`. | [entry_serve.py:21](src/entry_serve.py#L21) |
| 13 | `tarfile.extractall` is called without `filter="data"`. That's a path-traversal risk, and it raises a deprecation warning on Python 3.12+. | [entry_evaluate.py:31-32](src/entry_evaluate.py#L31-L32) |
| 14 | The threshold grid stops at 0.6, so cost settings where `C_FP/(C_FP+C_FN)` is above 0.6 silently pick the edge of the grid. | [decision.py:18](src/loan_risk/decision.py#L18) |
| 15 | `--region` defaults to `boto3.Session().region_name`, which can be `None`, and the error appears later and is confusing. | [build_pipeline.py:129](pipelines/build_pipeline.py#L129), [manage_model.py:30](pipelines/manage_model.py#L30) |
| 16 | The "history is dated before the application" rule is only in a docstring. No assertion or filter enforces it. | [history.py:3](src/loan_risk/features/history.py#L3) |
| 17 | `SK_ID_CURR` is hardcoded in `history.ID`, `scoring.score_features` and `HISTORY_COLUMNS`, although the config has `id_col`. | several files |

## Redundant / over-engineered (cleanup)

| # | Item | Where |
|---|---|---|
| 18 | Dead code: `univariate_auc`, `global_importance` and `ENGINEERED` are never used. Remove them, or use `global_importance` in the reports instead of gain. | [evaluate.py:37](src/loan_risk/evaluate.py#L37), [explain.py:17](src/loan_risk/explain.py#L17), [application.py:12](src/loan_risk/features/application.py#L12) |
| 19 | The `score` and `history` pipeline outputs are written but no pipeline step reads them. `history` is only used locally by `invoke_endpoint.py`. Keep them only if batch scoring or a feature store is part of the story. | [build_pipeline.py:60](pipelines/build_pipeline.py#L60) |
| 20 | `--n-folds`, `--learning-rate` and `--exclude` are never passed by the pipeline. They're fine for local experiments, but they're unused in SageMaker. | [entry_train.py:20-22](src/entry_train.py#L20-L22) |
| 21 | `Join(on="/", values=[base, "features", n])` joins only constant strings. A plain f-string does the same. | [build_pipeline.py:59](pipelines/build_pipeline.py#L59) |
| 22 | `predict_pd` repeats the first two lines of `score_features`. | [scoring.py:28-34](src/loan_risk/scoring.py#L28-L34) |
| 23 | `HISTORY_COLUMNS` in `build.py` lists the same columns each `history.py` function uses, so there are two places to keep in sync. It could live next to each function. | [build.py:14-25](src/loan_risk/features/build.py#L14-L25) |
| 24 | Small leftovers: `.astype(int)` on `BUR_ANY_OVERDUE` is undone by the `astype("float64")` that follows. The second `sys.path.insert` in the test is redundant. `gc.collect()` after each table is mostly cosmetic. `got_less` is a misleading name for a ratio. | [history.py:33](src/loan_risk/features/history.py#L33), [test_pipeline.py:105](tests/test_pipeline.py#L105), [build.py:41-47](src/loan_risk/features/build.py#L41-L47) |
| 25 | `cost_table` loops over 120 thresholds. It's fast enough, but a single sort plus cumulative sums would be clearer and O(n log n). | [decision.py:20-31](src/loan_risk/decision.py#L20-L31) |

## Gaps against plan.txt (not bugs)

- **Different names:** the plan's `inference.py` is now `entry_serve.py`, `deploy.py` is now `manage_model.py`, and the outputs are `train/holdout/score` instead of `train/val/test`. That's fine, but the plan is out of date.
- **Not built:**
  - The scorecard challenger (it's below the cut line).
  - The fairness AUC comparison (#10).
  - Plan verification step 6, the teardown check.
- **Unseen categories:** the plan says unseen levels map to `"other"`, but the code maps them to NaN, which LightGBM treats as missing. That's defensible and matches the code's docstring, but the plan and README should say the same thing.
- **Tests:** plan verification step 1 lists a "no duplicate `SK_ID_CURR` after `build_features`" test. It isn't explicitly tested; only the history index uniqueness is.

## Suggested order

1. Fix #1 and #4 first: they change numbers you would present.
2. Then #2 and #3: pipeline correctness.
3. Then #7 and the dead-code cleanup (#18–#24).
4. Have answers ready for #5, #6, #9 and #10, since an interviewer is likely to raise them.
