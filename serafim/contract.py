"""Gold contract: the schema of every file the website reads.

Agreed by the whole team (issue #18, task D2). Any change to this file needs approval from
all three roles (data engineering, AI & validation, web). If the website needs a new column,
it is requested with a PR here; it is never invented in the web code.

Gold files (one row per ...):
    targets.parquet      target
    frames.parquet       single exposure
    sums.parquet         detector x season: the image the site shows
    differences.parquet  detector x pair of seasons
    catalogs.parquet     catalog source in one season        (filled by AI & validation)
    spectra.parquet      photometry point                    (filled by AI & validation)
    trajectory.parquet   target x season                     (filled by AI & validation)
    meta.json            pipeline run

Two columns use 0 for "all":
    detector = 0   the white image: every detector of that season added together
    season = 0     every season of that detector added together (in catalogs.parquet: the
                   catalog layer at the mean date of every frame)
A single exposure always has one detector (1 to 6) and one season (1 or more). Seasons
count from 1 in time order. If a season has no frames for a detector, that row does not
exist (no sum, no difference, no PNG): the site skips it. Never invent a cell.

The id of a frame is always frame_id(target_id, obs_id), `<target_id>_<obs_id>`: silver,
AI's spectra and the gold frames build it the same way, so spectra.frame_id finds its frame.

PNG names (relative to gold/; the site builds the URL from these pieces, so they never
change without telling web). See sum_png(), difference_png() and color_png():
    png/<target>/sum_d<detector>_s<season>.png
    png/<target>/diff_d<detector>_s<a>-s<b>.png     season b minus season a
    png/<target>/color_s<season>.png                exists when sums has detectors 1, 3 and 6

`flags` is text: the flag names separated by commas, never SOURCE (almost every frame has
it), for example "BLOOM,OVERFLOW"; empty text when there are none. In pandas read it as
df["flags"]: df.flags is a pandas attribute, not the column. In frames, `saturated` says
OVERFLOW or NONLINEAR within 2 px of the target. In sums, it says the center of the target
(within 2 px) is painted as saturated: in silver, `saturated` (every frame saturated) or
n_saturated > 0 and n_saturated >= n_valid. The site shows "saturated center".

Pixel convention:
    0-based position in the PNG, origin at the top-left corner (x to the right, y down).
    Every PNG of a target has the size of the target grid and is drawn north up.
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

# --- Allowed values ----------------------------------------------------------
KINDS = ("motion", "brightness", "spectrum", "custom")  # custom: targets added by coordinates
CATALOG_NAMES = ("gaia", "catwise", "simbad", "skybot")
# Photometric band of catalogs.mag, set by the catalog (it is set even when mag is empty).
MAG_BANDS = {"gaia": "G", "catwise": "W2", "simbad": "V", "skybot": "V"}

# Detector number -> wavelength range in micrometers (docs/real_fits.md, measured on IRSA).
DETECTOR_BANDS_UM = {
    1: (0.746, 1.117),
    2: (1.103, 1.650),
    3: (1.630, 2.435),
    4: (2.403, 3.847),
    5: (3.811, 4.430),
    6: (4.412, 5.010),
}
WAVELENGTH_RANGE_UM = (DETECTOR_BANDS_UM[1][0], DETECTOR_BANDS_UM[6][1])
# The bands above are SIA's em_min/em_max. At the edge of a detector IRSA's spectral WCS gives a
# wavelength up to 0.014 um outside them (measured at the target in 18 of the 874 bronze frames),
# so a frame may be this far outside its band.
BAND_MARGIN_UM = 0.02
COLOR_DETECTORS = (1, 3, 6)  # blue, green and red of the color image

# --- Schemas -----------------------------------------------------------------
_UTC = pa.timestamp("us", tz="UTC")


def _required(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=False)


def _optional(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=True)


TARGETS = pa.schema(
    [
        _required("target_id", pa.string()),
        _required("name", pa.string()),
        _required("kind", pa.string()),  # one of KINDS
        _required("ra_now", pa.float64()),  # deg, ICRS, position today: the grid center
        _required("dec_now", pa.float64()),
        _required("pmra_mas_yr", pa.float64()),  # 0 for targets added by coordinates
        _required("pmdec_mas_yr", pa.float64()),
        _required("size_deg", pa.float64()),
        _required("wcs_json", pa.string()),  # WCS of the common pixel grid, as JSON
    ]
)

FRAMES = pa.schema(
    [
        _required("frame_id", pa.string()),
        _required("target_id", pa.string()),
        _required("obs_id", pa.string()),  # IRSA observation id
        _required("collection", pa.string()),  # SIA collection, e.g. spherex_dr1
        _required("date_utc", _UTC),  # mean date of the exposure
        _required("season", pa.int32()),  # 1 or more
        _required("detector", pa.int8()),  # 1 to 6
        _required("wavelength_um", pa.float64()),  # at the target, inside its detector's band
        _required("bandwidth_um", pa.float64()),
        _required("flags", pa.string()),  # names separated by commas, never SOURCE
        _required("saturated", pa.bool_()),
    ]
)

SUMS = pa.schema(
    [
        _required("sum_id", pa.string()),
        _required("target_id", pa.string()),
        _required("detector", pa.int8()),  # 0 (white image) to 6
        _required("season", pa.int32()),  # 0 (all seasons) or more
        _required("mean_date_utc", _UTC),
        _required("n_frames", pa.int32()),  # frames of frames.parquet that went into the sum
        _required("background_noise", pa.float64()),  # MJy/sr per pixel, away from sources
        _required("flags", pa.string()),
        _required("saturated", pa.bool_()),  # the target center is painted as saturated
        _required("png", pa.string()),
    ]
)

DIFFERENCES = pa.schema(
    [
        _required("difference_id", pa.string()),
        _required("target_id", pa.string()),
        _required("detector", pa.int8()),  # 1 to 6
        _required("season_a", pa.int32()),  # the earlier season
        _required("season_b", pa.int32()),  # the later season; the image is b minus a
        _required("png", pa.string()),
    ]
)

CATALOGS = pa.schema(
    [
        _required("target_id", pa.string()),
        _required("catalog", pa.string()),  # one of CATALOG_NAMES
        _required("name", pa.string()),
        _required("season", pa.int32()),  # 0 (all seasons) or more
        # The date the marker is drawn at; ra and dec are already moved to it. Gaia, CatWISE
        # and SIMBAD: the mean date of the season's frames (season 0: of every frame).
        # SkyBoT: one visit, so a season can have several rows per asteroid.
        _required("date_utc", _UTC),
        _required("ra", pa.float64()),
        _required("dec", pa.float64()),
        _optional("mag", pa.float64()),
        _required("mag_band", pa.string()),  # a value of MAG_BANDS
        _optional("url", pa.string()),  # empty for Gaia and CatWISE
    ]
)

SPECTRA = pa.schema(
    [
        _required("target_id", pa.string()),
        _required("frame_id", pa.string()),
        _required("season", pa.int32()),
        _required("date_utc", _UTC),
        _required("wavelength_um", pa.float64()),
        _required("flux_mjy", pa.float64()),
        _required("error_mjy", pa.float64()),
        _optional("flag", pa.string()),  # null when the point is fine
    ]
)

TRAJECTORY = pa.schema(
    [
        _required("target_id", pa.string()),
        _required("season", pa.int32()),
        _optional("ra_measured", pa.float64()),  # null when the target could not be measured
        _optional("dec_measured", pa.float64()),
        _optional("error_arcsec", pa.float64()),
        _required("ra_catalog", pa.float64()),  # where the catalogs say it is in that season
        _required("dec_catalog", pa.float64()),
    ]
)

GOLD_TABLES: dict[str, pa.Schema] = {
    "targets.parquet": TARGETS,
    "frames.parquet": FRAMES,
    "sums.parquet": SUMS,
    "differences.parquet": DIFFERENCES,
    "catalogs.parquet": CATALOGS,
    "spectra.parquet": SPECTRA,
    "trajectory.parquet": TRAJECTORY,
}

# Columns that identify a row: no two rows share them.
KEYS: dict[str, tuple[str, ...]] = {
    "targets.parquet": ("target_id",),
    "frames.parquet": ("frame_id",),
    "sums.parquet": ("sum_id",),
    "differences.parquet": ("difference_id",),
    "catalogs.parquet": ("target_id", "catalog", "name", "season", "date_utc"),
    "spectra.parquet": ("target_id", "frame_id"),
    "trajectory.parquet": ("target_id", "season"),
}

# meta.json: run_date_utc and last_frame_date_utc are ISO 8601 with the UTC offset, commit
# is the git SHA, collections the SIA collections that were searched, n_frames the rows of
# frames.parquet.
META_KEYS = ("run_date_utc", "commit", "collections", "last_frame_date_utc", "n_frames")


# --- Ids (every layer builds them the same way) ------------------------------
def frame_id(target_id: str, obs_id: str) -> str:
    """The frame_id of one exposure: `<target_id>_<obs_id>`.

    IRSA's obs_id has only letters, digits and underscores, and the 1975 obs_id of the 5
    targets are all different, so this id is unique and safe in a file name.
    """
    return f"{target_id}_{obs_id}"


# --- PNG names (the site builds URLs from these pieces) ----------------------
def sum_png(target_id: str, detector: int, season: int) -> str:
    return f"png/{target_id}/sum_d{detector}_s{season}.png"


def difference_png(target_id: str, detector: int, season_a: int, season_b: int) -> str:
    return f"png/{target_id}/diff_d{detector}_s{season_a}-s{season_b}.png"


def color_png(target_id: str, season: int) -> str:
    return f"png/{target_id}/color_s{season}.png"


# --- Validation (tasks D1 and D2) --------------------------------------------
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
    _check_pngs(gold_dir, tables, errors)
    if meta is not None and "frames.parquet" in tables:
        _check_meta(meta, tables["frames.parquet"], errors)

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
    for key in ("run_date_utc", "last_frame_date_utc"):
        if key in meta and _parse_utc(meta[key]) is None:
            errors.append(f"meta.json: {key} {meta[key]!r} must be ISO 8601 with the UTC offset")
    if "commit" in meta and not (isinstance(meta["commit"], str) and meta["commit"]):
        errors.append("meta.json: commit must be a non-empty string")
    if "collections" in meta:
        names = meta["collections"]
        if not (isinstance(names, list) and names and all(isinstance(c, str) and c for c in names)):
            errors.append("meta.json: collections must be a list of collection names")
    if "n_frames" in meta and not (
        isinstance(meta["n_frames"], int) and not isinstance(meta["n_frames"], bool)
    ):
        errors.append("meta.json: n_frames must be an integer")
    return meta


def _parse_utc(value) -> datetime | None:
    """A datetime if `value` is ISO 8601 text with a UTC offset, else None."""
    try:
        when = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return when if when.tzinfo is not None else None


def _report(errors: list[str], message: str, bad: pd.Series, examples: pd.Series) -> None:
    """Add one error line if any row is bad, with up to 3 example values."""
    count = int(bad.sum())
    if count:
        shown = ", ".join(str(v) for v in examples[bad].head(3))
        errors.append(f"{message}: {count} rows (e.g. {shown})")


def _labels(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """One readable text per row, from the key columns, for the examples in error lines."""
    rows = ["/".join(str(v) for v in row) for row in table[list(columns)].itertuples(index=False)]
    return pd.Series(rows, index=table.index, dtype="object")


FLAG_LIST = r"([A-Z][A-Z0-9_]*(,[A-Z][A-Z0-9_]*)*)?"


def _check_values(tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    """Unique keys and allowed values in each table."""
    labels = {name: _labels(t, KEYS[name]) for name, t in tables.items()}
    for name, table in tables.items():
        keys = list(KEYS[name])
        bad = table.duplicated(subset=keys, keep=False)
        _report(errors, f"{name}: repeated {' + '.join(keys)}", bad, labels[name])

    def check(name: str, message: str, bad: pd.Series) -> None:
        _report(errors, f"{name}: {message}", bad, labels[name])

    def outside(series: pd.Series, low: float, high: float, include_high: bool) -> pd.Series:
        inclusive = "both" if include_high else "left"
        return series.notna() & ~series.between(low, high, inclusive=inclusive)

    def sky(name: str, ra: str, dec: str) -> None:
        t = tables[name]
        check(name, f"{ra} outside [0, 360)", outside(t[ra], 0, 360, False))
        check(name, f"{dec} outside [-90, 90]", outside(t[dec], -90, 90, True))

    def positive(series: pd.Series) -> pd.Series:
        return series.notna() & ~(series > 0)

    def flags(name: str) -> None:
        text = tables[name]["flags"].fillna("")  # df.flags is a pandas attribute, not the column
        check(name, "flags must be names separated by commas (like BLOOM,OVERFLOW) or empty",
              ~text.str.fullmatch(FLAG_LIST))
        check(name, "flags must not include SOURCE",
              text.str.split(",").map(lambda names: "SOURCE" in names))

    if "targets.parquet" in tables:
        t = tables["targets.parquet"]
        check("targets.parquet", f"kind must be one of {list(KINDS)}", ~t.kind.isin(KINDS))
        check("targets.parquet", "size_deg must be > 0", positive(t.size_deg))
        sky("targets.parquet", "ra_now", "dec_now")

    if "frames.parquet" in tables:
        t = tables["frames.parquet"]
        name = "frames.parquet"
        check(name, "detector must be 1 to 6", ~t.detector.between(1, 6))
        check(name, "season must be 1 or more", t.season < 1)
        # Each wavelength has to fall inside the band of its own detector.
        low = t.detector.map({d: band[0] for d, band in DETECTOR_BANDS_UM.items()})
        high = t.detector.map({d: band[1] for d, band in DETECTOR_BANDS_UM.items()})
        margin = BAND_MARGIN_UM
        inside = (t.wavelength_um >= low - margin) & (t.wavelength_um <= high + margin)
        check(name, f"wavelength_um outside the band of its detector (+/- {BAND_MARGIN_UM} um)",
              low.notna() & ~inside)
        check(name, "bandwidth_um must be > 0", positive(t.bandwidth_um))
        flags(name)
        expected = [frame_id(r.target_id, r.obs_id) for r in t.itertuples()]
        check(name, "frame_id must be <target_id>_<obs_id>",
              t.frame_id != pd.Series(expected, index=t.index, dtype="object"))

    if "sums.parquet" in tables:
        t = tables["sums.parquet"]
        name = "sums.parquet"
        check(name, "detector must be 0 to 6", ~t.detector.between(0, 6))
        check(name, "season must be 0 or more", t.season < 0)
        check(name, "n_frames must be >= 1", ~(t.n_frames >= 1))
        check(name, "background_noise must be > 0", positive(t.background_noise))
        flags(name)
        expected = [sum_png(r.target_id, r.detector, r.season) for r in t.itertuples()]
        check(name, "png must be png/<target>/sum_d<detector>_s<season>.png",
              t.png != pd.Series(expected, index=t.index, dtype="object"))

    if "differences.parquet" in tables:
        t = tables["differences.parquet"]
        name = "differences.parquet"
        check(name, "detector must be 1 to 6", ~t.detector.between(1, 6))
        check(name, "season_a must be 1 or more", t.season_a < 1)
        check(name, "season_a must be less than season_b", ~(t.season_a < t.season_b))
        expected = [
            difference_png(r.target_id, r.detector, r.season_a, r.season_b) for r in t.itertuples()
        ]
        check(name, "png must be png/<target>/diff_d<detector>_s<a>-s<b>.png",
              t.png != pd.Series(expected, index=t.index, dtype="object"))

    if "catalogs.parquet" in tables:
        t = tables["catalogs.parquet"]
        check("catalogs.parquet", f"catalog must be one of {list(CATALOG_NAMES)}",
              ~t.catalog.isin(CATALOG_NAMES))
        check("catalogs.parquet", "season must be 0 or more", t.season < 0)
        expected = t.catalog.map(MAG_BANDS)  # empty for an unknown catalog: reported above
        check("catalogs.parquet", "mag_band must match its catalog (gaia G, catwise W2, "
              "simbad and skybot V)", expected.notna() & (t.mag_band != expected))
        sky("catalogs.parquet", "ra", "dec")

    if "spectra.parquet" in tables:
        t = tables["spectra.parquet"]
        # A point is one frame: the same margin as frames.parquet at the edges of the range.
        low, high = WAVELENGTH_RANGE_UM[0] - BAND_MARGIN_UM, WAVELENGTH_RANGE_UM[1] + BAND_MARGIN_UM
        check("spectra.parquet", f"wavelength_um outside [{low:.3f}, {high:.3f}]",
              outside(t.wavelength_um, low, high, True))
        check("spectra.parquet", "season must be 1 or more", t.season < 1)
        check("spectra.parquet", "error_mjy must be >= 0", t.error_mjy < 0)

    if "trajectory.parquet" in tables:
        t = tables["trajectory.parquet"]
        check("trajectory.parquet", "season must be 1 or more", t.season < 1)
        check("trajectory.parquet", "error_arcsec must be >= 0", t.error_arcsec < 0)
        for ra, dec in (("ra_measured", "dec_measured"), ("ra_catalog", "dec_catalog")):
            sky("trajectory.parquet", ra, dec)


def _check_links(tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    """Every id points to a row that exists; each sum counts the frames it really has."""

    def missing(child: str, column: str, known: pd.Series, name: str | None = None) -> None:
        values = tables[child][column]
        bad = values.notna() & ~values.isin(known)
        _report(errors, f"{child}: {column} not found in {name or 'the parent table'}", bad, values)

    if "targets.parquet" in tables:
        for child in tables:
            if child != "targets.parquet":
                missing(child, "target_id", tables["targets.parquet"].target_id, "targets.parquet")
    if "spectra.parquet" in tables and "frames.parquet" in tables:
        missing("spectra.parquet", "frame_id", tables["frames.parquet"].frame_id, "frames.parquet")

    sums = tables.get("sums.parquet")
    if sums is not None and "frames.parquet" in tables:
        frames = tables["frames.parquet"]
        counts = []
        for row in sums.itertuples():
            cell = frames[frames.target_id == row.target_id]
            if row.detector:
                cell = cell[cell.detector == row.detector]
            if row.season:
                cell = cell[cell.season == row.season]
            counts.append(len(cell))
        expected = pd.Series(counts, index=sums.index, dtype="int64")
        _report(errors, "sums.parquet: n_frames does not match the frames of its detector and "
                "season in frames.parquet", sums.n_frames != expected, sums.sum_id)

    if sums is not None and "differences.parquet" in tables:
        differences = tables["differences.parquet"]
        cells = set(zip(sums.target_id, sums.detector, sums.season, strict=True))
        for column in ("season_a", "season_b"):
            exists = [
                (r.target_id, r.detector, getattr(r, column)) in cells
                for r in differences.itertuples()
            ]
            _report(errors, f"differences.parquet: {column} has no sum of that detector",
                    ~pd.Series(exists, index=differences.index, dtype=bool),
                    differences.difference_id)


def _check_pngs(gold_dir: Path, tables: dict[str, pd.DataFrame], errors: list[str]) -> None:
    """PNG paths are relative to gold/ (the site does not live at the domain root) and exist."""

    def is_relative(path: str) -> bool:
        posix = PurePosixPath(path)
        return "\\" not in path and not posix.is_absolute() and ".." not in posix.parts

    for name in ("sums.parquet", "differences.parquet"):
        if name not in tables:
            continue
        paths = tables[name].png
        relative = pd.Series([isinstance(p, str) and is_relative(p) for p in paths], paths.index)
        exists = pd.Series(
            [bool(ok) and (gold_dir / p).is_file() for p, ok in zip(paths, relative, strict=True)],
            paths.index,
        )
        present = paths.notna()
        _report(errors, f"{name}: png must be a relative path with /", present & ~relative, paths)
        _report(errors, f"{name}: png file not found in gold", present & relative & ~exists, paths)

    # The color image exists exactly when the season has the three detectors it needs.
    if "sums.parquet" in tables:
        sums = tables["sums.parquet"]
        have: dict[tuple[str, int], set[int]] = {}
        for row in sums.itertuples():
            have.setdefault((row.target_id, row.season), set()).add(row.detector)
        for (target_id, season), detectors in sorted(have.items()):
            path = color_png(target_id, season)
            needed = season >= 1 and set(COLOR_DETECTORS) <= detectors
            if needed and not (gold_dir / path).is_file():
                errors.append(f"{path}: color image not found, but {target_id} has detectors "
                              f"{list(COLOR_DETECTORS)} in season {season}")


def _check_meta(meta: dict, frames: pd.DataFrame, errors: list[str]) -> None:
    """meta.json agrees with frames.parquet."""
    if "n_frames" in meta and meta["n_frames"] != len(frames):
        errors.append(
            f"meta.json: n_frames is {meta['n_frames']!r} but frames.parquet has "
            f"{len(frames)} rows"
        )
    last = _parse_utc(meta.get("last_frame_date_utc"))
    if last is not None and len(frames):
        newest = frames.date_utc.max().to_pydatetime()
        if abs((last - newest).total_seconds()) >= 1:
            errors.append(
                f"meta.json: last_frame_date_utc is {meta['last_frame_date_utc']} but the last "
                f"frame in frames.parquet is {newest.isoformat()}"
            )
    if isinstance(meta.get("collections"), list):
        unknown = sorted(set(frames.collection) - set(meta["collections"]))
        if unknown:
            errors.append(f"meta.json: collections does not list {unknown}, which frames use")
