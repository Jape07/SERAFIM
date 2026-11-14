"""Command line: `python -m serafim <command>`.

- `python -m serafim mock`: simulated gold in data/gold/ (task D1, data engineering).
- `python -m serafim run [--region X] [--from silver]`: full pipeline (task D8).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from serafim import contract
from serafim.config import load_config

log = logging.getLogger("serafim")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="serafim", description="SERAFIM pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    mock = commands.add_parser("mock", help="Write simulated gold to data/gold/ (task D1).")
    mock.add_argument("--out", default="data/gold", help="Gold folder (default: data/gold).")

    run = commands.add_parser("run", help="Run the full pipeline (task D8).")
    run.add_argument("--region", help="Only this region (an id from config/regions.yaml).")
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
        return mock(Path(args.out))
    log.error("Command '%s' is not implemented yet (task D8).", args.command)
    return 1


def mock(gold_dir: Path) -> int:
    """Simulated gold, checked against the contract like real gold will be."""
    from serafim.gold.mock import write_mock_gold  # matplotlib only loads when needed

    write_mock_gold(gold_dir, load_config())
    try:
        contract.validate_gold(gold_dir)
    except contract.GoldValidationError as error:
        log.error("%s", error)
        return 1
    log.info("validate_gold: %s follows the contract", gold_dir)
    return 0
