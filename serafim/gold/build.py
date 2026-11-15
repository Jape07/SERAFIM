"""Real gold: the tables, meta.json and PNGs the site publishes (task D8).

    python -m serafim gold [--target X] [--no-catalogs] [--no-spectra] [--out data/gold]

Gold has only the targets with real data: the ones that have sums in silver (agreed with web:
a target that is not in targets.parquet shows as "Coming soon"; the simulated gold is never
published). From silver and from AI & validation:

    targets.parquet      config/targets.yaml + the grid of the target (silver/grid.py)
    frames.parquet       data/silver/frames.parquet          (D4)
    sums.parquet         data/silver/sums.parquet            (D5)
    differences.parquet  data/silver/differences.parquet     (D6)
    catalogs.parquet     validation.layers.catalog_layers(targets, frames)     (AI, A2)
    spectra.parquet      validation.spectra.spectra(targets, frames=frames)    (AI, A4)
    trajectory.parquet   validation.trajectory.trajectory(targets)            (AI, A5)
An AI table with no rows (or skipped with None) is written empty, with its schema.
    meta.json            gold.export.build_meta()
    png/                 gold.png.build_pngs()                (D7)

Everything is written to a new folder next to gold and checked with validate_gold(). Only if it
passes does the new folder replace gold, so a failed run never leaves broken or half gold: the
previous gold stays (publish.yml relies on that).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from serafim import contract
from serafim.config import ROOT, load_config
from serafim.gold.export import build_meta, write_gold
from serafim.gold.png import build_pngs
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import differences_table_path, frames_table_path, sums_table_path

log = logging.getLogger("serafim.gold")

RowsFunction = Callable[[list[dict], pd.DataFrame], pd.DataFrame]


# --- What AI & validation gives (each can be replaced in tests) -------------------------------
def ai_catalogs(targets: list[dict], frames: pd.DataFrame) -> pd.DataFrame:
    from serafim.validation.layers import catalog_layers

    return catalog_layers(targets, frames)


def ai_spectra(targets: list[dict], frames: pd.DataFrame) -> pd.DataFrame:
    from serafim.validation.spectra import spectra

    return spectra(targets, frames=frames)


def ai_trajectory(targets: list[dict], frames: pd.DataFrame) -> pd.DataFrame:
    from serafim.validation.trajectory import trajectory  # reads the sums with load_sum()

    return trajectory(targets)


def empty(filename: str) -> pd.DataFrame:
    return contract.GOLD_TABLES[filename].empty_table().to_pandas()


# --- Pure: the targets table -------------------------------------------------------------------
def targets_table(targets: list[dict], size_deg: float) -> pd.DataFrame:
    """targets.parquet rows; a target added by coordinates has no proper motion (0)."""
    rows = []
    for t in targets:
        size = float(t.get("cutout_size_deg") or size_deg)
        grid = target_wcs(float(t["ra_now"]), float(t["dec_now"]), size)
        rows.append({
            "target_id": t["id"],
            "name": t.get("name") or t["id"],
            "kind": t.get("kind") or "custom",
            "ra_now": float(t["ra_now"]),
            "dec_now": float(t["dec_now"]),
            "pmra_mas_yr": float(t.get("pmra_mas_yr") or 0.0),
            "pmdec_mas_yr": float(t.get("pmdec_mas_yr") or 0.0),
            "size_deg": size,
            "wcs_json": wcs_to_json(grid),
        })
    return pd.DataFrame(rows)


# --- Disk ---------------------------------------------------------------------------------------
def build_gold(
    target_id: str | list[str] | None = None,
    *,
    root: Path | None = None,
    gold_dir: Path | None = None,
    catalogs: RowsFunction | None = ai_catalogs,
    spectra: RowsFunction | None = ai_spectra,
    trajectory: RowsFunction | None = ai_trajectory,
) -> dict[str, int]:
    """Build and check gold; return the rows of every table. None skips an AI table (empty).

    `target_id` is one target, a list of them, or None for every target with silver sums.

    Raises:
        FileNotFoundError: silver is missing.
        contract.GoldValidationError: the new gold does not follow the contract; the previous
            gold is left as it was and the new one stays next to it for inspection.
    """
    root = root or ROOT
    gold_dir = Path(gold_dir or root / "data" / "gold")
    config = load_config(root / "config" / "targets.yaml")
    for path in (frames_table_path(root), sums_table_path(root), differences_table_path(root)):
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} is missing in silver: run the silver step first")
    sums = pq.read_table(sums_table_path(root)).to_pandas()
    if target_id is None:
        real = sorted(set(sums.target_id))
    else:
        real = [target_id] if isinstance(target_id, str) else list(target_id)
    known = {t["id"]: t for t in config["targets"]}
    unknown = [t for t in real if t not in known or t not in set(sums.target_id)]
    if unknown:
        raise ValueError(f"No silver sums for {unknown[0]}: run the silver step for it first.")
    targets = [known[t] for t in real]

    def only(path: Path) -> pd.DataFrame:
        table = pq.read_table(path).to_pandas()
        return table[table.target_id.isin(real)].reset_index(drop=True)

    frames = only(frames_table_path(root))
    tables = {
        "targets.parquet": targets_table(targets, float(config["settings"]["cutout_size_deg"])),
        "frames.parquet": frames,
        "sums.parquet": only(sums_table_path(root)),
        "differences.parquet": only(differences_table_path(root)),
    }
    for filename, function in (("catalogs.parquet", catalogs), ("spectra.parquet", spectra),
                               ("trajectory.parquet", trajectory)):
        rows = function(targets, frames) if function is not None else empty(filename)
        tables[filename] = rows if len(rows) else empty(filename)
        log.info("%s: %d rows", filename, len(tables[filename]))

    new = gold_dir.with_name(gold_dir.name + ".new")
    if new.exists():
        shutil.rmtree(new)
    for name in real:
        build_pngs(name, root=root, gold_dir=new)
    write_gold(new, tables, build_meta(frames, config["settings"]["collections"]))
    contract.validate_gold(new)  # raises before touching the previous gold

    if gold_dir.exists():
        shutil.rmtree(gold_dir)
    new.replace(gold_dir)
    counts = {name: len(table) for name, table in tables.items()}
    log.info("gold: %s for %s in %s", counts, ", ".join(real), gold_dir)
    return counts
