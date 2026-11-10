"""Reading raw tables (parquet preferred, CSV fallback) and one-off CSV -> parquet conversion."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLES = [
    "application_train", "application_test", "bureau", "bureau_balance", "previous_application",
    "POS_CASH_balance", "credit_card_balance", "installments_payments",
]


def read_table(base: str | Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read `name` from `base`, preferring `<name>.parquet` over `<name>.csv`. Loads only `columns`."""
    base = Path(base)
    pq = base / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq, columns=columns)
    csv = base / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv, usecols=columns)
    raise FileNotFoundError(f"Neither {pq} nor {csv} exists")


def read_dir(path: str | Path) -> pd.DataFrame:
    """Read a parquet file, or all parquet files in a directory (SageMaker channels are directories)."""
    path = Path(path)
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def csv_to_parquet(src: str | Path, dst: str | Path) -> None:
    """Convert the Kaggle CSVs to parquet (about 4x smaller and much faster to read, in S3 too)."""
    import pyarrow.csv as pv
    import pyarrow.parquet as pq

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        out = dst / f"{name}.parquet"
        if out.exists():
            print(f"skip {out.name} (exists)")
            continue
        table = pv.read_csv(src / f"{name}.csv")
        pq.write_table(table, out, compression="snappy")
        print(f"{name}: {table.num_rows:,} rows -> {out.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert raw Home Credit CSVs to parquet")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    csv_to_parquet(a.src, a.dst)
