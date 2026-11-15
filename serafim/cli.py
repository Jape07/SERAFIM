"""Command line: `python -m serafim <command>`.

- `python -m serafim mock [--all-targets]`: simulated gold in data/gold/.
- `python -m serafim bronze [--target X]`: IRSA cutouts in data/bronze/ (task D3).
- `python -m serafim run [--target X] [--from silver]`: full pipeline (task D9).

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

    run = commands.add_parser("run", help="Run the full pipeline (task D9).")
    run.add_argument("--target", help=TARGET_HELP)
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
    log.error("Command '%s' is not implemented yet (task D9).", args.command)
    return 1


def bronze(target_id: str | None) -> int:
    """Download bronze cutouts. Silver and gold come in later tasks."""
    from serafim.bronze.download import ArchiveError, download

    try:
        download(target_id)
    except ArchiveError as error:
        log.error("%s", error)
        return 1
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
