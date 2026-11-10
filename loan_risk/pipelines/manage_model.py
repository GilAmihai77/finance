"""Model Registry and endpoint lifecycle.

  python pipelines/manage_model.py list
  python pipelines/manage_model.py approve [--arn ARN]            # latest pending package by default
  python pipelines/manage_model.py deploy --role-arn R [--serverless]
  python pipelines/manage_model.py delete                         # teardown: endpoint + config + model
"""
from __future__ import annotations

import argparse
import json

import boto3
import sagemaker
from sagemaker import ModelPackage
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.serverless import ServerlessInferenceConfig


def packages(sm, group, status=None):
    kw = {"ModelPackageGroupName": group, "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 50}
    if status:
        kw["ModelApprovalStatus"] = status
    return sm.list_model_packages(**kw)["ModelPackageSummaryList"]


def latest(sm, group, status) -> str:
    found = packages(sm, group, status)
    if not found:
        raise SystemExit(f"No model package with status {status} in group '{group}'. Run 'list' to see what exists.")
    return found[0]["ModelPackageArn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "approve", "deploy", "delete"])
    ap.add_argument("--region", default=boto3.Session().region_name)
    ap.add_argument("--model-group", default="loan-risk")
    ap.add_argument("--endpoint-name", default="loan-risk-endpoint")
    ap.add_argument("--arn", default=None)
    ap.add_argument("--role-arn", default=None)
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--serverless", action="store_true", help="serverless endpoint (no data capture)")
    ap.add_argument("--prefix", default="loan-risk")
    a = ap.parse_args()
    if not a.region:
        ap.error("no AWS region: pass --region or run 'aws configure set region il-central-1'")

    boto = boto3.Session(region_name=a.region)
    sm = boto.client("sagemaker")

    if a.action == "list":
        for p in packages(sm, a.model_group):
            d = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
            metrics = d.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {}).get("S3Uri", "-")
            print(f"v{p['ModelPackageVersion']:<3} {p['ModelApprovalStatus']:<22} {p['CreationTime']:%Y-%m-%d %H:%M}"
                  f"  {p['ModelPackageArn']}\n     metrics: {metrics}")

    elif a.action == "approve":
        arn = a.arn or latest(sm, a.model_group, "PendingManualApproval")
        sm.update_model_package(ModelPackageArn=arn, ModelApprovalStatus="Approved",
                                ApprovalDescription="Reviewed holdout AUC/KS, calibration and decision bands")
        print("Approved:", arn)

    elif a.action == "deploy":
        if not a.role_arn:
            ap.error("--role-arn is required for deploy")
        arn = a.arn or latest(sm, a.model_group, "Approved")
        sess = sagemaker.Session(boto)
        model = ModelPackage(role=a.role_arn, model_package_arn=arn, sagemaker_session=sess)
        print(f"Deploying {arn} -> {a.endpoint_name} (takes ~5-8 min)")
        if a.serverless:
            model.deploy(endpoint_name=a.endpoint_name,
                         serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=2))
        else:
            capture = f"s3://{sess.default_bucket()}/{a.prefix}/data-capture"
            model.deploy(endpoint_name=a.endpoint_name, initial_instance_count=1, instance_type=a.instance_type,
                         data_capture_config=DataCaptureConfig(enable_capture=True, sampling_percentage=100,
                                                               destination_s3_uri=capture))
            print("Data capture ->", capture)
        print(json.dumps(sm.describe_endpoint(EndpointName=a.endpoint_name)["EndpointStatus"]))

    elif a.action == "delete":
        ep = sm.describe_endpoint(EndpointName=a.endpoint_name)
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        sm.delete_endpoint(EndpointName=a.endpoint_name)
        sm.delete_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        for v in cfg["ProductionVariants"]:
            sm.delete_model(ModelName=v["ModelName"])
        print("Deleted endpoint, endpoint config and model:", a.endpoint_name)


if __name__ == "__main__":
    main()
