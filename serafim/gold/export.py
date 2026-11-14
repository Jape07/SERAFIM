"""Write a gold folder following serafim/contract.py.

Used by the simulated gold now (task D1) and by the real pipeline later (task D7),
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


def build_meta(n_frames: int, commit: str | None = None) -> dict:
    """meta.json content: when the run happened, from which commit and how many frames."""
    return {
        "run_date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit or current_commit(),
        "n_frames": int(n_frames),
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
    """Write the 4 Parquet tables, objects.csv and meta.json into gold_dir.

    `tables` is keyed by file name, as in contract.GOLD_TABLES ("frames.parquet", ...).
    PNG files are written by whoever makes them; this only writes the tables.
    """
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in contract.GOLD_TABLES.items():
        table = to_contract_table(tables[filename], schema)
        pq.write_table(table, gold_dir / filename)
        if filename == "objects.parquet":
            table.to_pandas().to_csv(gold_dir / "objects.csv", index=False)
    with open(gold_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
        file.write("\n")
