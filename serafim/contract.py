"""Gold contract: the schema of every file the website reads.

Agreed by the whole team at H0. Any change to this file needs approval from all
three roles (data engineering, AI & validation, web). If the website needs a new
column, it is requested with a PR here; it is never invented in the web code.

Gold files (one row per ...):
    regions.parquet     region
    frames.parquet      image at one date
    detections.parquet  change found in a difference image
    objects.parquet     grouped object
    meta.json           pipeline run
    objects.csv         same rows as objects.parquet, for download

Pixel convention (x_px, y_px):
    0-based position in the frame PNG, origin at the top-left corner (x to the
    right, y down), so the web places markers without any conversion. Every PNG
    of a region has the size of the region grid and is drawn north up.
    wcs_json describes that same grid in the FITS convention (pixel 1,1 at the
    bottom-left): fits_x = x_px + 1 and fits_y = NAXIS2 - y_px.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Allowed values for objects.status -------------------------------------
# Wording matters: never call a candidate a "new object".
KNOWN = "Known"
CANDIDATE = "Unidentified candidate"
DISCARDED = "Discarded"
STATUSES = (KNOWN, CANDIDATE, DISCARDED)

# --- Schemas ----------------------------------------------------------------
_UTC = pa.timestamp("us", tz="UTC")


def _required(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=False)


def _optional(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=True)


REGIONS = pa.schema(
    [
        _required("region_id", pa.string()),
        _required("name", pa.string()),
        _required("ra", pa.float64()),  # deg, ICRS, region center
        _required("dec", pa.float64()),  # deg, ICRS, region center
        _required("size_deg", pa.float64()),
        _required("wcs_json", pa.string()),  # WCS of the common pixel grid, as JSON
    ]
)

FRAMES = pa.schema(
    [
        _required("frame_id", pa.string()),
        _required("region_id", pa.string()),
        _required("obs_id", pa.string()),  # IRSA observation id
        _required("date_utc", _UTC),
        _required("wavelength_um", pa.float64()),  # between 0.75 and 5.0
        _required("bandwidth_um", pa.float64()),
        _required("png", pa.string()),  # path relative to gold/
        _optional("png_diff", pa.string()),  # null when the frame has no reference
    ]
)

DETECTIONS = pa.schema(
    [
        _required("detection_id", pa.string()),
        _required("frame_id", pa.string()),
        _required("object_id", pa.string()),
        _required("ra", pa.float64()),
        _required("dec", pa.float64()),
        _required("x_px", pa.float64()),  # on the region grid; the web draws markers here
        _required("y_px", pa.float64()),
        _required("snr", pa.float64()),
        _required("sign", pa.int8()),  # +1 got brighter, -1 got fainter
        _optional("p_real", pa.float64()),  # CNN probability in [0, 1]; null until task D10
    ]
)

OBJECTS = pa.schema(
    [
        _required("object_id", pa.string()),
        _required("region_id", pa.string()),
        _required("name", pa.string()),
        _required("type", pa.string()),
        _required("status", pa.string()),  # one of STATUSES
        _required("reason", pa.string()),  # why it got that status
        _optional("catalog", pa.string()),  # SkyBoT, SIMBAD or Gaia when Known
        _optional("url", pa.string()),  # link to the catalog entry
        _required("n_detections", pa.int32()),
        _required("first_date_utc", _UTC),
        _required("last_date_utc", _UTC),
    ]
)

GOLD_TABLES: dict[str, pa.Schema] = {
    "regions.parquet": REGIONS,
    "frames.parquet": FRAMES,
    "detections.parquet": DETECTIONS,
    "objects.parquet": OBJECTS,
}

# meta.json keys: run_date_utc is ISO 8601 in UTC, commit is the git SHA.
META_KEYS = ("run_date_utc", "commit", "n_frames")

# SPHEREx sees 0.75 to 5.0 micrometers.
WAVELENGTH_RANGE_UM = (0.75, 5.0)


# --- Validation (task D1) ---------------------------------------------------
class GoldValidationError(ValueError):
    """Gold does not follow the contract. `errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        listed = "\n- ".join(errors)
        super().__init__(f"Gold does not follow the contract ({len(errors)} problems):\n- {listed}")


def validate_gold(gold_dir: Path) -> None:
    """Check that a gold folder follows this contract; raise if it does not.

    Checks files, columns, types, required values, allowed values, links between
    tables, PNG files and meta.json. It collects every problem before raising, so
    one run shows everything that has to be fixed.

    publish.yml stops if this fails, so the site keeps the previous version.

    Raises:
        GoldValidationError: with the list of problems in `.errors`.
    """
    gold_dir = Path(gold_dir)
    errors: list[str] = []

    tables = {}
    for filename, schema in GOLD_TABLES.items():
        table = _read_table(gold_dir / filename, schema, errors)
        if table is not None:
            tables[filename] = table
    meta = _read_meta(gold_dir / "meta.json", errors)

    _check_values(tables, errors)
    _check_links(tables, errors)
    if "frames.parquet" in tables:
        _check_pngs(gold_dir, tables["frames.parquet"], errors)
        if meta is not None and meta.get("n_frames") != len(tables["frames.parquet"]):
            errors.append(
                f"meta.json: n_frames is {meta.get('n_frames')!r} but frames.parquet has "
                f"{len(tables['frames.parquet'])} rows"
            )

    if errors:
        raise GoldValidationError(errors)


def _read_table(path: Path, schema: pa.Schema, errors: list[str]) -> pd.DataFrame | None:
    """Read one table and check its columns, types and required values."""
    name = path.name
    if not path.is_file():
        errors.append(f"{name}: file not found")
        return None
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        errors.append(f"{name}: cannot be read as Parquet ({error})")
        return None

    missing = [c for c in schema.names if c not in table.column_names]
    extra = [c for c in table.column_names if c not in schema.names]
    if missing:
        errors.append(f"{name}: missing columns {missing}")
    if extra:
        errors.append(f"{name}: columns not in the contract {extra}")
    if missing:
        return None  # The other checks need every column.
    if not extra and table.column_names != schema.names:
        errors.append(f"{name}: columns must follow the contract order {schema.names}")

    for field in schema:
        column = table.column(field.name)
        if not column.type.equals(field.type):
            errors.append(f"{name}: {field.name} is {column.type}, the contract says {field.type}")
        if not field.nullable and column.null_count:
            errors.append(f"{name}: {field.name} has {column.null_count} empty values (required)")
    return table.to_pandas()


def _read_meta(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append("meta.json: file not found")
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"meta.json: not valid JSON ({error})")
        return None
    if not isinstance(meta, dict):
        errors.append("meta.json: must be a JSON object")
        return None

    missing = [key for key in META_KEYS if key not in meta]
    if missing:
        errors.append(f"meta.json: missing keys {missing}")
    if "run_date_utc" in meta:
        try:
            when = datetime.fromisoformat(str(meta["run_date_utc"]))
            if when.tzinfo is None:
                errors.append("meta.json: run_date_utc must include the UTC offset")
        except ValueError:
            errors.append(f"meta.json: run_date_utc {meta['run_date_utc']!r} is not ISO 8601")
    if "commit" in meta and not (isinstance(meta["commit"], str) and meta["commit"]):
        errors.append("meta.json: commit must be a non-empty string")
    return meta


def _report(errors: list[str], message: str, bad: pd.Series, examples: pd.Series) -> None:
    """Add one error line if any row is bad, with up to 3 example values."""
    count = int(bad.sum())
    if count:
        shown = ", ".join(str(v) for v in examples[bad].head(3))
        errors.append(f"{message}: {count} rows (e.g. {shown})")


def _check_values(tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    """Unique ids and allowed values in each table."""
    for name, table in tables.items():
        id_column = GOLD_TABLES[name].names[0]
        ids = table[id_column]
        _report(errors, f"{name}: repeated {id_column}", ids.duplicated(keep=False), ids)

    def out_of_range(series: pd.Series, low: float, high: float, include_high: bool) -> pd.Series:
        inclusive = "both" if include_high else "left"
        return series.notna() & ~series.between(low, high, inclusive=inclusive)

    for name in ("regions.parquet", "detections.parquet"):
        if name in tables:
            t, ids = tables[name], tables[name].iloc[:, 0]
            bad_ra = out_of_range(t.ra, 0, 360, False)
            bad_dec = out_of_range(t.dec, -90, 90, True)
            _report(errors, f"{name}: ra outside [0, 360)", bad_ra, ids)
            _report(errors, f"{name}: dec outside [-90, 90]", bad_dec, ids)

    if "regions.parquet" in tables:
        t = tables["regions.parquet"]
        _report(errors, "regions.parquet: size_deg must be > 0", ~(t.size_deg > 0), t.region_id)

    if "frames.parquet" in tables:
        t = tables["frames.parquet"]
        low, high = WAVELENGTH_RANGE_UM
        bad = out_of_range(t.wavelength_um, low, high, True)
        _report(errors, f"frames.parquet: wavelength_um outside [{low}, {high}]", bad, t.frame_id)
        bad = ~(t.bandwidth_um > 0)
        _report(errors, "frames.parquet: bandwidth_um must be > 0", bad, t.frame_id)

    if "detections.parquet" in tables:
        t = tables["detections.parquet"]
        _report(errors, "detections.parquet: sign must be +1 or -1", ~t.sign.isin([1, -1]),
                t.detection_id)
        bad = out_of_range(t.p_real, 0, 1, True)
        _report(errors, "detections.parquet: p_real outside [0, 1]", bad, t.detection_id)

    if "objects.parquet" in tables:
        t = tables["objects.parquet"]
        bad = t.status.notna() & ~t.status.isin(STATUSES)
        _report(errors, f"objects.parquet: status must be one of {list(STATUSES)}", bad, t.status)
        _report(errors, "objects.parquet: n_detections must be >= 1", ~(t.n_detections >= 1),
                t.object_id)
        bad = t.first_date_utc > t.last_date_utc
        _report(errors, "objects.parquet: first_date_utc after last_date_utc", bad, t.object_id)


def _check_links(tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    """Every id that points to another table exists there; n_detections adds up."""
    links = [
        ("frames.parquet", "region_id", "regions.parquet"),
        ("objects.parquet", "region_id", "regions.parquet"),
        ("detections.parquet", "frame_id", "frames.parquet"),
        ("detections.parquet", "object_id", "objects.parquet"),
    ]
    for child, column, parent in links:
        if child in tables and parent in tables:
            values = tables[child][column]
            bad = values.notna() & ~values.isin(tables[parent][column])
            _report(errors, f"{child}: {column} not found in {parent}", bad, values)

    if "objects.parquet" in tables and "detections.parquet" in tables:
        objects = tables["objects.parquet"]
        counts = tables["detections.parquet"].object_id.value_counts()
        expected = objects.object_id.map(counts).fillna(0)
        bad = objects.n_detections != expected
        _report(errors, "objects.parquet: n_detections does not match detections.parquet", bad,
                objects.object_id)


def _check_pngs(gold_dir: Path, frames: pd.DataFrame, errors: list[str]) -> None:
    """PNG paths are relative to gold/ (the site does not live at the domain root) and exist."""
    def is_relative(path: str) -> bool:
        posix = PurePosixPath(path)
        return "\\" not in path and not posix.is_absolute() and ".." not in posix.parts

    for column in ("png", "png_diff"):
        paths = frames[column]
        relative = pd.Series([isinstance(p, str) and is_relative(p) for p in paths], paths.index)
        exists = pd.Series(
            [bool(ok) and (gold_dir / p).is_file() for p, ok in zip(paths, relative, strict=True)],
            paths.index,
        )
        present = paths.notna()
        _report(errors, f"frames.parquet: {column} must be a relative path with /",
                present & ~relative, paths)
        _report(errors, f"frames.parquet: {column} file not found in gold",
                present & relative & ~exists, paths)
