"""The whole pipeline (task D9): which steps run for which targets, and publish.yml. No network:
every step is replaced by a function that only writes down that it was called."""

from pathlib import Path

import pytest
import yaml

from serafim import contract
from serafim.cli import main
from serafim.config import ROOT, load_config
from serafim.pipeline import publish_targets, run


def test_only_checked_targets_are_published():
    config = load_config()
    assert publish_targets(config) == ["barnard", "wise_0855"]
    assert set(config["settings"]["publish"]) <= {t["id"] for t in config["targets"]}


def test_publish_targets_keep_the_order_of_the_config_and_default_to_all():
    config = load_config()
    config["settings"]["publish"] = ["r_dra", "barnard"]
    assert publish_targets(config) == ["barnard", "r_dra"]
    del config["settings"]["publish"]
    assert publish_targets(config) == [t["id"] for t in config["targets"]]
    config["settings"]["publish"] = ["pluto"]
    with pytest.raises(ValueError, match="pluto"):
        publish_targets(config)


@pytest.fixture
def calls(monkeypatch):
    log = []

    def record(step):
        def function(target=None, **kwargs):
            log.append((step, target))
            return {"targets.parquet": 1} if step == "gold" else 0
        return function

    monkeypatch.setattr("serafim.bronze.download.download", record("bronze"))
    monkeypatch.setattr("serafim.silver.frames.build_frames", record("frames"))
    monkeypatch.setattr("serafim.silver.sums.build_sums", record("sums"))
    monkeypatch.setattr("serafim.silver.differences.build_differences", record("differences"))
    monkeypatch.setattr("serafim.gold.build.build_gold", record("gold"))
    return log


def test_from_bronze_runs_every_step_and_gold_has_every_published_target(calls):
    run()
    published = publish_targets(load_config())
    steps_per_target = [(step, name) for name in published
                        for step in ("bronze", "frames", "sums", "differences")]
    assert calls == [*steps_per_target, ("gold", published)]


def test_from_silver_does_not_download(calls):
    run(start="silver")
    steps = [step for step, _ in calls]
    assert "bronze" not in steps and steps[-1] == "gold"
    assert steps.count("frames") == len(publish_targets(load_config()))


def test_from_gold_only_rebuilds_gold(calls):
    run(start="gold")
    assert calls == [("gold", publish_targets(load_config()))]


def test_a_target_that_is_not_published_is_refused(calls):
    with pytest.raises(ValueError, match="settings.publish"):
        run("r_dra")
    assert calls == []


def test_the_command_returns_1_when_gold_fails_its_check(monkeypatch):
    def failing(*args, **kwargs):
        raise contract.GoldValidationError(["catalogs.parquet: catalog must be one of ..."])

    monkeypatch.setattr("serafim.pipeline.run", failing)
    assert main(["run", "--from", "gold"]) == 1


def test_the_command_returns_0_when_the_pipeline_works(monkeypatch):
    monkeypatch.setattr("serafim.pipeline.run", lambda *args, **kwargs: {"targets.parquet": 1})
    assert main(["run"]) == 0


# --- publish.yml --------------------------------------------------------------------------------
WORKFLOW = yaml.safe_load(Path(ROOT / ".github" / "workflows" / "publish.yml").read_text())


def steps(job):
    return WORKFLOW["jobs"][job]["steps"]


def test_publish_builds_real_gold_and_never_the_simulated_one():
    commands = " ".join(str(step.get("run", "")) for step in steps("build"))
    assert "python -m serafim run" in commands
    assert "mock" not in commands


def test_publish_does_not_cancel_a_run_in_progress():
    # Runs of one branch wait for each other; other branches never cancel them.
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False
    assert "github.ref" in WORKFLOW["concurrency"]["group"]
    # Only deploys share the Pages group, and they wait too.
    deploy = WORKFLOW["jobs"]["deploy"]["concurrency"]
    assert deploy == {"group": "pages", "cancel-in-progress": False}


def test_bronze_is_saved_even_when_the_run_fails_or_is_cancelled():
    save = next(s for s in steps("build") if s.get("name") == "Save bronze and the catalog cache")
    assert save["if"].startswith("always()")
    assert "data/bronze" in save["with"]["path"] and "data/catalog_cache" in save["with"]["path"]


def test_deploy_needs_a_good_build_and_can_be_switched_off_by_hand():
    deploy = WORKFLOW["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    # PyYAML reads the key `on` as True.
    trigger = WORKFLOW.get("on", WORKFLOW.get(True))
    assert trigger["workflow_dispatch"]["inputs"]["deploy"]["default"] is True
    assert "inputs.deploy" in deploy["if"]
