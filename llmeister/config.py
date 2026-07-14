"""Load config.yaml (single source of truth for system config)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from llmeister import PROJECT_ROOT
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}
