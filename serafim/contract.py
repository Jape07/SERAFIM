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
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

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


def validate_gold(gold_dir: Path) -> None:
    """Check that a gold folder follows this contract; raise if it does not.

    Task D1 (data engineering). publish.yml stops if this fails, so the site
    keeps the previous version.
    """
    raise NotImplementedError("Task D1")
