"""The repo skeleton every role builds on: config, modules and CLI."""

import importlib
from pathlib import Path

import pytest
import yaml

from serafim.cli import main

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load((ROOT / "config" / "regions.yaml").read_text(encoding="utf-8"))


def test_config_has_the_pole():
    regions = {r["id"]: r for r in load_config()["regions"]}
    assert regions["pole"]["ra"] == pytest.approx(270.0)
    assert regions["pole"]["dec"] == pytest.approx(66.56)


def test_region_ids_are_unique():
    ids = [r["id"] for r in load_config()["regions"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "module",
    [
        "serafim.contract",
        "serafim.bronze",
        "serafim.silver",
        "serafim.gold",
        "serafim.validation.catalogs",
        "serafim.validation.status",
        "serafim.ai.predict",
    ],
)
def test_module_imports(module):
    importlib.import_module(module)


def test_cli_help_works():
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
