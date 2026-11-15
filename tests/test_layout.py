"""The repo skeleton every role builds on: config, modules and CLI."""

import importlib

import pytest

from serafim.cli import main
from serafim.config import CONFIG_PATH, ROOT, load_config


def test_the_pipeline_reads_config_targets_yaml():
    assert CONFIG_PATH == ROOT / "config" / "targets.yaml"
    assert not (ROOT / "config" / "regions.yaml").exists(), "regions.yaml was retired"


def test_config_has_the_five_challenge_targets():
    config = load_config()
    ids = [target["id"] for target in config["targets"]]
    assert ids == ["barnard", "eps_indi_b", "wise_0855", "rw_aur", "r_dra"]
    for target in config["targets"]:
        assert 0 <= target["ra_now"] < 360
        assert -90 <= target["dec_now"] <= 90
    assert {"cutout_size_deg", "max_frames_per_target", "collections"} <= set(config["settings"])


@pytest.mark.parametrize(
    "module",
    [
        "serafim.contract",
        "serafim.config",
        "serafim.bronze",
        "serafim.bronze.download",
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
