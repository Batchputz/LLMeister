"""Load config.yaml (single source of truth for system config)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}
