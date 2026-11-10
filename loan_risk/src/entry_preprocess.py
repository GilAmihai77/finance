"""Processing step: raw tables -> features. Outputs train / holdout / scoring sets and the history snapshot.

Runs unchanged locally and as a SageMaker Processing job (defaults are the container paths).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from loan_risk import load_config
from loan_risk.data import read_table
from loan_risk.features import build_features, build_history

P = "/opt/ml/processing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=f"{P}/input/raw")
    ap.add_argument("--output", default=f"{P}/output")
    a = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    out = Path(a.output)

    print("building history aggregates ...", flush=True)
    history = build_history(lambda name, cols: read_table(a.input, name, cols))
    print(f"history: {history.shape} ({time.time() - t0:.0f}s)", flush=True)

    train_raw = read_table(a.input, "application_train")
    test_raw = read_table(a.input, "application_test")
    feats = build_features(train_raw, history)
    test_feats = build_features(test_raw, history)

    tr, ho = train_test_split(feats, test_size=cfg["split"]["holdout_share"], stratify=feats[cfg["target"]],
                              random_state=cfg["split"]["seed"])
    for name, df in [("train", tr), ("holdout", ho), ("score", test_feats)]:
        (out / name).mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_parquet(out / name / f"{name}.parquet", index=False)
    (out / "history").mkdir(parents=True, exist_ok=True)
    history.reset_index().to_parquet(out / "history" / "history.parquet", index=False)

    summary = {"train": tr.shape, "holdout": ho.shape, "score": test_feats.shape, "history": history.shape,
               "train_default_rate": float(tr[cfg["target"]].mean()),
               "holdout_default_rate": float(ho[cfg["target"]].mean()), "seconds": round(time.time() - t0)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
