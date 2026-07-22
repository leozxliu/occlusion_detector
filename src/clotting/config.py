"""Tiny config loader shared across stages."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    # src/clotting/config.py -> project root is two parents up from src/clotting
    return Path(__file__).resolve().parents[2]
