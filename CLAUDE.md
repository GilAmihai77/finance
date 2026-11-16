# CLAUDE.md

Home Credit default-risk (PD) model → Approve/Review/Reject decision, deployed as a SageMaker pipeline. Overview and results: `README.md`.

## Layout
- `loan_risk/src/loan_risk/`: core package (features, train, decision, scoring, explain). Settings in `config.yaml`.
- `loan_risk/src/entry_*.py`: SageMaker job entry points. Default paths are `/opt/ml/...`, so pass explicit args when running locally.
- `loan_risk/pipelines/`: build/deploy the SageMaker pipeline, manage models, invoke the endpoint.

## Commands
- Tests (synthetic data, no Kaggle files needed): `cd loan_risk && pytest`
- Install: `pip install -e "loan_risk[dev]"` (add `aws` for SageMaker)

## Rules
- Never commit Kaggle data, CSV/parquet files, `.env`, or notebook outputs. Redistributing the data is not allowed. Notebooks are stripped by `tools/strip_notebook.py` (git filter).
- `data/` is raw input. Read only.
- `CODE_GENDER` is excluded from the production model on purpose (fairness). Don't add it back.
- Keep code compatible with Python 3.9 (`requires-python >=3.9`).
