"""Send real applications (application_test + their history aggregates) to the endpoint and/or a local model.

  python pipelines/invoke_endpoint.py --local-model artifacts/model                    # local only
  python pipelines/invoke_endpoint.py --endpoint-name loan-risk-endpoint --local-model artifacts/model  # compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_payload(data_dir: Path, history_path: Path, n: int) -> list:
    raw = pd.read_parquet(data_dir / "application_test.parquet").head(n) if (data_dir / "application_test.parquet").exists() \
        else pd.read_csv(data_dir / "application_test.csv", nrows=n)
    hist = pd.read_parquet(history_path)
    req = raw.merge(hist, on="SK_ID_CURR", how="left")      # in production: online feature-store lookup
    return json.loads(req.to_json(orient="records", double_precision=15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-name", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "parquet"))
    ap.add_argument("--history", default=str(ROOT / "artifacts" / "features" / "history" / "history.parquet"))
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    records = build_payload(Path(a.data_dir), Path(a.history), a.n)
    body = json.dumps({"instances": records})
    print(f"request: {len(records)} applications, {len(records[0])} fields each, {len(body) / 1024:.1f} KB")

    preds = {}
    if a.local_model:
        import entry_serve
        model = entry_serve.model_fn(a.local_model)
        out, _ = entry_serve.output_fn(entry_serve.predict_fn(entry_serve.input_fn(body), model))
        preds["local"] = json.loads(out)["predictions"]
    if a.endpoint_name:
        import boto3
        rt = boto3.Session(region_name=a.region).client("sagemaker-runtime")
        t0 = time.time()
        resp = rt.invoke_endpoint(EndpointName=a.endpoint_name, ContentType="application/json",
                                  Accept="application/json", Body=body)
        print(f"endpoint latency: {(time.time() - t0) * 1000:.0f} ms")
        preds["endpoint"] = json.loads(resp["Body"].read())["predictions"]

    for src, p in preds.items():
        print(f"\n== {src} ==")
        for r in p:
            reasons = ", ".join(f"{c['feature']}={c['value']}" for c in r["reason_codes"])
            print(f"{r['SK_ID_CURR']}  PD={r['pd']:.4f}  {r['decision']:<8} because: {reasons}")
    if len(preds) == 2:
        diff = np.max(np.abs(np.array([r["pd"] for r in preds["local"]]) - [r["pd"] for r in preds["endpoint"]]))
        print(f"\nmax |PD local - PD endpoint| = {diff:.2e}  ->  {'MATCH' if diff < 1e-6 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
