"""The repo skeleton every role builds on: config, modules and CLI."""

import importlib
from datetime import date
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


def test_the_three_watched_regions():
    assert [r["id"] for r in load_config()["regions"]] == ["pole", "asteroid", "variable_star"]


@pytest.mark.parametrize("region", load_config()["regions"], ids=lambda r: r["id"])
def test_region_is_complete(region):
    assert {"id", "name", "ra", "dec", "why"} <= set(region)
    assert 0 <= region["ra"] < 360
    assert -90 <= region["dec"] <= 90


@pytest.mark.parametrize("region", load_config()["regions"], ids=lambda r: r["id"])
def test_key_window_is_two_ordered_utc_dates(region):
    if "key_window_utc" not in region:
        pytest.skip("no key window")
    start, end = (date.fromisoformat(str(day)) for day in region["key_window_utc"])
    assert start <= end


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
