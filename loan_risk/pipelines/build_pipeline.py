"""Define (upsert) and optionally start the SageMaker Pipeline:

  Preprocess -> Train -> Evaluate -> [AUC >= gate ?] -> Register (PendingManualApproval) | Fail

Usage:
  python pipelines/build_pipeline.py --role-arn <SageMakerExecutionRoleArn> [--bucket B] [--start]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

SRC = str(Path(__file__).resolve().parents[1] / "src")   # shipped to every job: package + entry scripts
sys.path.insert(0, SRC)
from loan_risk import load_config  # noqa: E402

FRAMEWORK = "1.2-1"
P = "/opt/ml/processing"


def get_pipeline(role: str, bucket: str, region: str, prefix: str, name: str, model_group: str) -> Pipeline:
    sess = PipelineSession(boto_session=boto3.Session(region_name=region), default_bucket=bucket)
    base = f"s3://{bucket}/{prefix}"

    # Raw data is stored as immutable snapshots: new data goes to a new prefix (raw/v2, ...), passed as
    # InputDataUri. Caching keys on step arguments, not on S3 contents, so overwriting raw/v1 in place
    # would let BuildFeatures be served from cache with stale features.
    input_uri = ParameterString("InputDataUri", default_value=f"{base}/raw/v1")
    proc_type = ParameterString("ProcessingInstanceType", default_value="ml.m5.2xlarge")
    train_type = ParameterString("TrainingInstanceType", default_value="ml.m5.2xlarge")
    auc_gate = ParameterFloat("AucThreshold", default_value=float(load_config()["evaluation"]["auc_gate"]))
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    cache = CacheConfig(enable_caching=True, expire_after="P30D")   # re-runs (e.g. new gate) skip unchanged steps

    def processor(job):
        return FrameworkProcessor(estimator_cls=SKLearn, framework_version=FRAMEWORK, py_version="py3",
                                  role=role, instance_type=proc_type, instance_count=1, volume_size_in_gb=50,
                                  base_job_name=f"{prefix}-{job}", sagemaker_session=sess)

    # 1. Feature engineering
    step_process = ProcessingStep(
        name="BuildFeatures",
        step_args=processor("features").run(
            code="entry_preprocess.py", source_dir=SRC,
            inputs=[ProcessingInput(source=input_uri, destination=f"{P}/input/raw")],
            outputs=[ProcessingOutput(output_name=n, source=f"{P}/output/{n}",
                                      # one folder per execution: no clashes, and lineage of which
                                      # features trained which model (on a cache hit the earlier
                                      # execution's folder is reused, consistently)
                                      destination=Join(on="/", values=[base, "features",
                                                                       ExecutionVariables.PIPELINE_EXECUTION_ID, n]))
                     for n in ("train", "holdout", "score", "history")],
        ),
        cache_config=cache,
    )
    out = step_process.properties.ProcessingOutputConfig.Outputs

    # 2. Training (5-fold CV + refit + thresholds)
    estimator = SKLearn(
        entry_point="entry_train.py", source_dir=SRC, framework_version=FRAMEWORK, py_version="py3",
        role=role, instance_type=train_type, instance_count=1, base_job_name=f"{prefix}-train",
        output_path=f"{base}/models", sagemaker_session=sess,
        metric_definitions=[{"Name": m, "Regex": f"{m}=([0-9\\.]+)"} for m in ("cv_auc", "cv_gini", "cv_ks")],
    )
    step_train = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit({"train": TrainingInput(s3_data=out["train"].S3Output.S3Uri)}),
        cache_config=cache,
    )
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts

    # 3. Evaluation on the untouched holdout
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="EvaluateModel",
        step_args=processor("eval").run(
            code="entry_evaluate.py", source_dir=SRC,
            inputs=[ProcessingInput(source=model_data, destination=f"{P}/model"),
                    ProcessingInput(source=out["holdout"].S3Output.S3Uri, destination=f"{P}/holdout")],
            outputs=[ProcessingOutput(output_name="evaluation", source=f"{P}/evaluation",
                                      destination=Join(on="/", values=[base, "evaluation",
                                                                        step_train.properties.TrainingJobName]))],
        ),
        property_files=[eval_report],
    )
    eval_uri = step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri

    # 4. Register in the Model Registry (governance: needs a human approval before deployment)
    model = SKLearnModel(model_data=model_data, entry_point="entry_serve.py", source_dir=SRC,
                         framework_version=FRAMEWORK, py_version="py3", role=role, sagemaker_session=sess)
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"], response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"], transform_instances=["ml.m5.xlarge"],
            model_package_group_name=model_group, approval_status=approval,
            description="LightGBM PD model, Home Credit default risk",
            model_metrics=ModelMetrics(model_statistics=MetricsSource(
                s3_uri=Join(on="/", values=[eval_uri, "evaluation.json"]), content_type="application/json")),
        ),
    )
    step_fail = FailStep(name="AucBelowGate",
                         error_message=Join(on=" ", values=["Holdout AUC below the gate of", auc_gate]))

    step_gate = ConditionStep(
        name="AucQualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                         json_path="binary_classification_metrics.auc.value"),
            right=auc_gate)],
        if_steps=[step_register], else_steps=[step_fail],
    )

    return Pipeline(name=name, parameters=[input_uri, proc_type, train_type, auc_gate, approval],
                    steps=[step_process, step_train, step_eval, step_gate], sagemaker_session=sess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--region", default=boto3.Session().region_name)
    ap.add_argument("--bucket", default=None, help="default: sagemaker-<region>-<account>")
    ap.add_argument("--prefix", default="loan-risk")
    ap.add_argument("--name", default="loan-risk-pipeline")
    ap.add_argument("--model-group", default="loan-risk")
    ap.add_argument("--start", action="store_true", help="start an execution after upsert")
    ap.add_argument("--auc-threshold", type=float, default=None, help="override the gate for this execution")
    ap.add_argument("--print-definition", action="store_true")
    a = ap.parse_args()
    if not a.region:
        ap.error("no AWS region: pass --region or run 'aws configure set region il-central-1'")

    bucket = a.bucket or sagemaker.Session(boto3.Session(region_name=a.region)).default_bucket()
    pipe = get_pipeline(a.role_arn, bucket, a.region, a.prefix, a.name, a.model_group)
    if a.print_definition:
        print(pipe.definition())
    pipe.upsert(role_arn=a.role_arn, description="Loan default PD model: features -> train -> eval -> gate -> registry")
    print(f"Pipeline '{a.name}' upserted (bucket s3://{bucket}/{a.prefix})")
    if a.start:
        params = {"AucThreshold": a.auc_threshold} if a.auc_threshold is not None else {}
        ex = pipe.start(parameters=params)
        print("Started execution:", ex.arn)


if __name__ == "__main__":
    main()
