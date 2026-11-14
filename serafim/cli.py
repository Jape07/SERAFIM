"""Command line: `python -m serafim <command>`.

- `python -m serafim mock`: simulated gold in data/gold/ (task D1, data engineering).
- `python -m serafim run [--region X] [--from silver]`: full pipeline (task D8).
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger("serafim")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="serafim", description="SERAFIM pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("mock", help="Write simulated gold to data/gold/ (task D1).")

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
    log.error("Command '%s' is not implemented yet (tasks D1 and D8).", args.command)
    return 1
