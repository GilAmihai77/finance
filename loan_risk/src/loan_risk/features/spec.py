"""Frozen feature contract: the column list and categorical levels seen at training time.

Applying the same spec at training and at inference prevents training/serving skew. Missing columns
become NaN, and unseen category levels become NaN, which LightGBM treats as missing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    features: list = field(default_factory=list)
    categorical: dict = field(default_factory=dict)   # column -> sorted list of training levels

    @classmethod
    def fit(cls, df: pd.DataFrame, drop: list) -> "FeatureSpec":
        feats = [c for c in df.columns if c not in set(drop)]
        cats = {}
        for c in feats:
            if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
                cats[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        return cls(features=feats, categorical=cats)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.features)
        for c in self.features:
            if c in self.categorical:
                s = X[c]
                s = s.where(s.isna(), s.astype(str))
                X[c] = pd.Categorical(s, categories=self.categorical[c])
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)
        return X

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
