"""The whole pipeline in one command (task D9): bronze, silver and gold.

    python -m serafim run [--target X] [--from bronze|silver|gold]

The targets are `settings.publish` in config/targets.yaml: the ones whose silver has been
checked end to end. A target is added there when it is ready; the site never shows one that
was not checked. `--target X` runs bronze and silver for X only, but gold always has every
published target, so a partial run never drops a target from the site.

- from bronze (default): download what is missing (nothing is downloaded twice), then silver
  and gold.
- from silver: silver and gold from the bronze on disk.
- from gold: gold from the silver on disk.

Gold is checked with validate_gold() before it replaces the previous one (gold/build.py), so a
failed run leaves the previous gold and publish.yml does not deploy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from serafim.config import ROOT, load_config

log = logging.getLogger("serafim")

STEPS = ("bronze", "silver", "gold")


def publish_targets(config: dict) -> list[str]:
    """`settings.publish`, in the order of config/targets.yaml; every target if it is not set."""
    known = [t["id"] for t in config["targets"]]
    wanted = config["settings"].get("publish")
    if wanted is None:
        return known
    unknown = [t for t in wanted if t not in known]
    if unknown:
        raise ValueError(f"settings.publish has {unknown[0]}, which is not a target.")
    return [t for t in known if t in wanted]


def run(target_id: str | None = None, *, start: str = "bronze", root: Path | None = None) -> dict:
    """Run the steps from `start`; return the rows of every gold table."""
    from serafim.bronze.download import download
    from serafim.gold.build import build_gold
    from serafim.silver.differences import build_differences
    from serafim.silver.frames import build_frames
    from serafim.silver.sums import build_sums

    if start not in STEPS:
        raise ValueError(f"--from must be one of {STEPS}")
    root = root or ROOT
    published = publish_targets(load_config(root / "config" / "targets.yaml"))
    if target_id is not None and target_id not in published:
        raise ValueError(f"{target_id} is not in settings.publish ({', '.join(published)}).")
    targets = [target_id] if target_id else published
    first = STEPS.index(start)

    for name in targets:
        if first <= 0:
            log.info("target=%s: bronze", name)
            download(name, root=root)
        if first <= 1:
            log.info("target=%s: silver", name)
            build_frames(name, root=root)
            build_sums(name, root=root)
            build_differences(name, root=root)
    log.info("gold for %s", ", ".join(published))
    return build_gold(published, root=root)
