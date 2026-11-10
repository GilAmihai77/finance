"""Loan default risk (PD) model: features, training, decisioning, explanations, serving."""
from __future__ import annotations

from pathlib import Path

import yaml

__version__ = "0.1.0"

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
