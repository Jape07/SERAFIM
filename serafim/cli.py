"""Command line: `python -m serafim <command>`.

- `python -m serafim mock [--all-targets]`: simulated gold in data/gold/.
- `python -m serafim bronze [--target X]`: IRSA cutouts in data/bronze/ (task D3).
- `python -m serafim silver [--target X] [--force]`: frames, sums, differences (D4 to D6).
- `python -m serafim alignment [--target X]`: sums against Gaia, in docs/alignment.md (D5).
- `python -m serafim png [--target X]`: PNG of sums, differences and color in data/gold/ (D7).
- `python -m serafim gold [--target X]`: real gold (tables, meta.json, PNGs), validated (D8).
- `python -m serafim run [--target X] [--from silver]`: the whole pipeline for the targets in
  settings.publish (task D9); what publish.yml runs.

Targets are the ids in config/targets.yaml (barnard, eps_indi_b, wise_0855, rw_aur, r_dra).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from serafim import contract
from serafim.config import load_config

log = logging.getLogger("serafim")

TARGET_HELP = "Only this target (an id from config/targets.yaml)."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="serafim", description="SERAFIM pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    mock = commands.add_parser("mock", help="Write simulated gold to data/gold/.")
    mock.add_argument("--out", default="data/gold", help="Gold folder (default: data/gold).")
    mock.add_argument(
        "--all-targets",
        action="store_true",
        help="Write every target in config/targets.yaml (the published site).",
    )

    bronze_cmd = commands.add_parser(
        "bronze", help="Download IRSA cutouts to data/bronze/ (task D3)."
    )
    bronze_cmd.add_argument("--target", help=TARGET_HELP)

    silver_cmd = commands.add_parser(
        "silver", help="Frames (D4), sums (D5) and differences (D6) in data/silver/."
    )
    silver_cmd.add_argument("--target", help=TARGET_HELP)
    silver_cmd.add_argument(
        "--force", action="store_true", help="Reproject frames that are already in silver."
    )

    alignment_cmd = commands.add_parser(
        "alignment", help="Measure the sums against Gaia and rewrite docs/alignment.md (D5)."
    )
    alignment_cmd.add_argument("--target", help=TARGET_HELP)

    png_cmd = commands.add_parser(
        "png", help="PNG of every sum, difference and color image in data/gold/png/ (D7)."
    )
    png_cmd.add_argument("--target", help=TARGET_HELP)
    png_cmd.add_argument("--out", default="data/gold", help="Gold folder (default: data/gold).")

    gold_cmd = commands.add_parser(
        "gold", help="Real gold: tables, meta.json and PNGs, checked with validate_gold (D8)."
    )
    gold_cmd.add_argument("--target", help="Only this target (default: every target in silver).")
    gold_cmd.add_argument("--out", default="data/gold", help="Gold folder (default: data/gold).")
    gold_cmd.add_argument("--no-catalogs", action="store_true",
                          help="Write catalogs.parquet empty (no Gaia/SIMBAD/SkyBoT queries).")
    gold_cmd.add_argument("--no-spectra", action="store_true",
                          help="Write spectra.parquet empty (no photometry on bronze).")

    run = commands.add_parser(
        "run", help="Bronze, silver and gold for the targets in settings.publish (task D9)."
    )
    run.add_argument("--target", help="Only this published target for bronze and silver "
                                      "(gold always has every published target).")
    run.add_argument(
        "--from",
        dest="start",
        choices=["bronze", "silver", "gold"],
        default="bronze",
        help="Layer to rebuild from (bronze is never re-downloaded if already there).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "mock":
        return mock(Path(args.out), all_targets=args.all_targets)
    if args.command == "bronze":
        return bronze(args.target)
    if args.command == "silver":
        return silver(args.target, force=args.force)
    if args.command == "alignment":
        return alignment(args.target)
    if args.command == "gold":
        return gold(args.target, Path(args.out), catalogs=not args.no_catalogs,
                    spectra=not args.no_spectra)
    if args.command == "png":
        return png(args.target, Path(args.out))
    return full_run(args.target, args.start)


def full_run(target_id: str | None, start: str) -> int:
    """The whole pipeline; 1 if a step fails (gold is then left as it was)."""
    from serafim.bronze.download import ArchiveError
    from serafim.pipeline import run

    try:
        counts = run(target_id, start=start)
    except (ArchiveError, FileNotFoundError, ValueError) as error:  # incl. GoldValidationError
        log.error("%s", error)
        return 1
    log.info("Pipeline done: %s", counts)
    return 0


def bronze(target_id: str | None) -> int:
    """Download bronze cutouts. Silver and gold come in later tasks."""
    from serafim.bronze.download import ArchiveError, download

    try:
        download(target_id)
    except ArchiveError as error:
        log.error("%s", error)
        return 1
    return 0


def silver(target_id: str | None, *, force: bool = False) -> int:
    """Silver 1 (frames), silver 2 (sums) and silver 3 (differences)."""
    from serafim.silver.differences import build_differences
    from serafim.silver.frames import build_frames
    from serafim.silver.sums import build_sums

    try:
        written = build_frames(target_id, force=force)
        sums = build_sums(target_id)
        differences = build_differences(target_id)
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 1
    log.info("Silver: %d frames reprojected now, %d sums, %d differences.", written, sums,
             differences)
    return 0


def gold(target_id: str | None, gold_dir: Path, *, catalogs: bool = True,
         spectra: bool = True) -> int:
    """Real gold from silver and AI's tables; the previous gold stays if the check fails."""
    from serafim.gold import build

    try:
        build.build_gold(target_id, gold_dir=gold_dir,
                         catalogs=build.ai_catalogs if catalogs else None,
                         spectra=build.ai_spectra if spectra else None)
    except (FileNotFoundError, ValueError) as error:  # GoldValidationError is a ValueError
        log.error("%s", error)
        return 1
    log.info("validate_gold: %s follows the contract", gold_dir)
    return 0


def png(target_id: str | None, gold_dir: Path) -> int:
    """Gold PNGs from silver: sums, differences and color images."""
    from serafim.gold.png import build_pngs

    try:
        written = build_pngs(target_id, gold_dir=gold_dir)
    except FileNotFoundError as error:
        log.error("%s", error)
        return 1
    log.info("Gold: %d PNG in %s.", written, gold_dir)
    return 0


def alignment(target_id: str | None) -> int:
    """Measure the sums against Gaia; docs/alignment.md says if they meet the goal."""
    from serafim.silver.alignment import GOAL_PX, run
    from serafim.validation.catalogs import CatalogUnavailable

    try:
        worst = run(target_id)
    except (FileNotFoundError, StopIteration) as error:
        log.error("No sums to measure (%s): run `python -m serafim silver` first.", error)
        return 1
    except CatalogUnavailable as error:
        log.error("No Gaia to measure against (ESA and VizieR did not answer): %s", error)
        return 1
    if worst >= GOAL_PX:
        log.warning("Largest offset %.3f px is over the goal of %.1f px: tell AI.", worst, GOAL_PX)
    return 0


def mock(gold_dir: Path, *, all_targets: bool = False) -> int:
    """Simulated gold, checked against the contract like real gold will be."""
    from serafim.gold.mock import write_all_targets, write_mock_gold

    if all_targets:
        write_all_targets(gold_dir, load_config())
    else:
        write_mock_gold(gold_dir, load_config())
    try:
        contract.validate_gold(gold_dir)
    except contract.GoldValidationError as error:
        log.error("%s", error)
        return 1
    log.info("validate_gold: %s follows the contract", gold_dir)
    return 0
