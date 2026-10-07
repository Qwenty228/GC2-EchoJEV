"""Load a YAML run configuration (`configs/*.yaml`). The `system` key names the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg
