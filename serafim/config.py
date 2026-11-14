"""Load config/regions.yaml: the watched regions and every pipeline setting."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"


def load_config(path: Path | None = None) -> dict:
    """Return the config as a dict with the keys `settings` and `regions`."""
    with open(path or CONFIG_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)
