"""Load typed module configuration from project-level YAML."""
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T")
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_config(cls: type[T], **overrides: object) -> T:
    name = cls.__name__.removesuffix("Config").lower()
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return cls(**(values | overrides))
