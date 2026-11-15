"""Write a gold folder following serafim/contract.py.

Used by the simulated gold now (task D2) and by the real pipeline later (task D8),
so both always write exactly the same files with exactly the same types.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from serafim import contract


def build_meta(frames: pd.DataFrame, collections: list[str], commit: str | None = None) -> dict:
    """meta.json content, taken from the frames table so it always agrees with it.

    Args:
        frames: the frames.parquet table (needs `date_utc`); it cannot be empty.
        collections: the SIA collections that were searched (`settings.collections`).
        commit: git SHA; by default the current one.
    """
    if frames.empty:
        raise ValueError("Cannot build meta.json: frames is empty")
    last_frame = frames.date_utc.max().to_pydatetime()
    return {
        "run_date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit or current_commit(),
        "collections": list(collections),
        "last_frame_date_utc": last_frame.isoformat(timespec="seconds"),
        "n_frames": len(frames),
    }


def current_commit() -> str:
    """Git SHA of this run: GITHUB_SHA in Actions, `git rev-parse` locally, else "unknown"."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def to_contract_table(data: pd.DataFrame | pa.Table, schema: pa.Schema) -> pa.Table:
    """Columns in contract order and types cast to the contract schema."""
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data[schema.names], schema=schema, preserve_index=False)
    return data.select(schema.names).cast(schema)


def write_gold(gold_dir: Path, tables: dict[str, pd.DataFrame | pa.Table], meta: dict) -> None:
    """Write the 7 Parquet tables and meta.json into gold_dir.

    `tables` is keyed by file name, as in contract.GOLD_TABLES ("frames.parquet", ...).
    A table with no rows is fine (write it empty with its schema so the site does not fail).
    PNG files are written by whoever makes them; this only writes the tables.
    """
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in contract.GOLD_TABLES.items():
        table = to_contract_table(tables[filename], schema)
        pq.write_table(table, gold_dir / filename)
    with open(gold_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
        file.write("\n")
