"""Catalog queries with an on-disk cache (task A1, AI & validation).

- skybot(center, radius, date): known asteroids and comets at that exact date.
- simbad(center, radius): stars, galaxies and other catalogued objects.
- gaia(center, radius): Gaia DR3 sources, with their variability flag.
- fetch_for_frames(frames, regions): everything classify() needs for a run.

The three catalogs return the same columns (COLUMNS), so classify() can match
detections against all of them in the same way.

Every query is cached in data/catalog_cache/ (empty answers too), has a timeout
and 3 attempts. If a service does not answer and there is no cache, nothing can
become a candidate: fetch_for_frames() returns None and classify() marks those
objects Discarded with reason "not validated".
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import astropy.units as u
import numpy as np
import pandas as pd
import pyvo
import requests
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/catalog_cache")
TIMEOUT_S = 60
RETRIES = 3
BACKOFF_S = 1.0  # wait 1 s after the first failure, 2 s after the second

COLUMNS = ["catalog", "name", "ra", "dec", "type", "is_variable", "mag", "url"]

# SIMBAD TAP services, tried in this order. In November 2026 the main CDS site
# in Strasbourg went offline after a security incident; CDS lists these mirrors.
SIMBAD_TAP_SERVERS = (
    "https://simbad.cds.unistra.fr/simbad/sim-tap",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap",
    "https://simbad.idia.ac.za/simbad/sim-tap",
)
GAIA_TAP_SERVER = "https://gea.esac.esa.int/tap-server/tap"
GAIA_MAX_MAG = 19.0  # G magnitude limit: bright enough to leave residuals or be flagged

# SIMBAD object types that mean "this source changes brightness". In the current
# SIMBAD hierarchy many of them do not hang from V* (RR Lyrae sit under Ev*,
# eclipsing binaries under **), so they are listed one by one. Taken from the
# otypedef table. HV* is a high-velocity star, not a variable, and is left out.
SIMBAD_VARIABLE_TYPES = frozenset(
    {
        "V*", "V*?", "var", "Er*", "Er?", "Ir*", "Or*", "Pu*", "Pu?", "Ro*", "Ro?",
        "RR*", "RR?", "Ce*", "Ce?", "cC*", "WV*", "WV?", "RV*", "RV?",
        "LP*", "LP?", "Mi*", "Mi?", "dS*", "gD*", "SX*", "bC*", "bC?",
        "a2*", "a2?", "RC*", "RC?", "EB*", "EB?", "El*", "El?", "BY*", "BY?",
        "RS*", "RS?", "CV*", "CV?", "No*", "No?", "Sy*", "Sy?",
        "XB*", "XB?", "LXB", "HXB", "TT*", "TT?", "SN*", "SN?",
    }
)  # fmt: skip


class CatalogUnavailable(RuntimeError):
    """A catalog did not answer after every attempt and there was no cache."""


# --- Public API ---------------------------------------------------------------


def skybot(center: SkyCoord, radius: Angle | u.Quantity, date: Time) -> pd.DataFrame:
    """Solar system objects (asteroids, comets, planets) inside the cone at that date.

    Positions are geocentric; seen from SPHEREx's orbit (~700 km up) the parallax
    of a main-belt asteroid is far below the 12 arcsec match radius.
    """
    radius = Angle(radius)
    params = {**_cone(center, radius), "date_utc": date.utc.isot}
    return _cached("skybot", params, lambda: _from_skybot(_query_skybot(center, radius, date)))


def simbad(center: SkyCoord, radius: Angle | u.Quantity) -> pd.DataFrame:
    """SIMBAD objects inside the cone; is_variable comes from SIMBAD_VARIABLE_TYPES."""
    radius = Angle(radius)
    params = _cone(center, radius)
    return _cached("simbad", params, lambda: _from_simbad(*_query_simbad(center, radius)))


def gaia(
    center: SkyCoord, radius: Angle | u.Quantity, max_mag: float = GAIA_MAX_MAG
) -> pd.DataFrame:
    """Gaia DR3 sources brighter than max_mag (G band) inside the cone."""
    radius = Angle(radius)
    params = {**_cone(center, radius), "max_mag": max_mag}
    return _cached("gaia", params, lambda: _from_gaia(_query_gaia(center, radius, max_mag)))


def fetch_for_frames(frames: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame | None:
    """Every catalog source classify() needs for these frames.

    SIMBAD and Gaia are queried once per region; SkyBoT once per frame, because
    each frame has its own date and asteroids move.

    Args:
        frames: frame_id, region_id and date_utc (contract.FRAMES).
        regions: region_id, ra, dec and size_deg (contract.REGIONS).

    Returns:
        COLUMNS plus region_id and frame_id (frame_id is only set for SkyBoT
        rows), or None if a catalog did not answer and there was no cache.
    """
    parts = []
    try:
        for region in regions.itertuples(index=False):
            center = SkyCoord(region.ra, region.dec, unit="deg")
            radius = _region_radius(region.size_deg)
            for table in (simbad(center, radius), gaia(center, radius)):
                parts.append(table.assign(region_id=region.region_id, frame_id=None))
            region_frames = frames[frames["region_id"] == region.region_id]
            for frame in region_frames.itertuples(index=False):
                table = skybot(center, radius, _to_time(frame.date_utc))
                parts.append(table.assign(region_id=region.region_id, frame_id=frame.frame_id))
    except CatalogUnavailable as error:
        log.error("Catalogs unavailable, every object will be 'not validated': %s", error)
        return None

    columns = [*COLUMNS, "region_id", "frame_id"]
    parts = [part for part in parts if len(part)]
    if not parts:
        return pd.DataFrame({name: pd.Series(dtype=_DTYPES.get(name, "str")) for name in columns})
    return pd.concat(parts, ignore_index=True)[columns]


# --- Network calls (replaced by fakes in the tests) ------------------------------


class _TimeoutSession(requests.Session):
    """requests session that never waits forever."""

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", TIMEOUT_S)
        return super().request(*args, **kwargs)


def _run_tap(server: str, adql: str) -> Table:
    result = pyvo.dal.TAPService(server, session=_TimeoutSession()).run_sync(adql)
    if result.query_status == "OVERFLOW":
        raise RuntimeError(f"{server} truncated the result; narrow the query")
    return result.to_table()


def _query_skybot(center: SkyCoord, radius: Angle, date: Time) -> Table:
    from astroquery.exceptions import NoResultsWarning
    from astroquery.imcce import Skybot, conf

    with conf.set_temp("timeout", TIMEOUT_S), warnings.catch_warnings():
        warnings.simplefilter("ignore", NoResultsWarning)  # an empty cone is a valid answer
        return Skybot.cone_search(center.icrs, radius, date, location="500", cache=False)


def _query_simbad(center: SkyCoord, radius: Angle) -> tuple[Table, str]:
    adql = f"""
        SELECT basic.main_id, basic.ra, basic.dec, basic.otype, allfluxes.V
        FROM basic LEFT JOIN allfluxes ON basic.oid = allfluxes.oidref
        WHERE CONTAINS(POINT('ICRS', basic.ra, basic.dec), {_adql_circle(center, radius)}) = 1
    """
    errors = []
    for server in SIMBAD_TAP_SERVERS:
        try:
            return _run_tap(server, adql), server
        except Exception as error:  # any failure: try the next mirror
            log.info("SIMBAD server %s failed: %s", server, error)
            errors.append(f"{server}: {error}")
    raise RuntimeError("every SIMBAD server failed: " + " | ".join(errors))


def _query_gaia(center: SkyCoord, radius: Angle, max_mag: float) -> Table:
    adql = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_variable_flag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {_adql_circle(center, radius)})
          AND phot_g_mean_mag < {float(max_mag)}
    """
    return _run_tap(GAIA_TAP_SERVER, adql)


# --- Pure conversions to COLUMNS --------------------------------------------------


def _from_skybot(table: Table) -> pd.DataFrame:
    if len(table) == 0:
        return _empty()
    return _frame(
        catalog="SkyBoT",
        name=[str(name) for name in table["Name"]],
        ra=_floats(table["RA"], u.deg),
        dec=_floats(table["DEC"], u.deg),
        type=[str(kind) for kind in table["Type"]],
        is_variable=True,  # it moves, so it always shows up in a difference image
        mag=_floats(table["V"], u.mag),
        url=[str(link) for link in table["externallink"]],
    )


def _from_simbad(table: Table, server: str) -> pd.DataFrame:
    if len(table) == 0:
        return _empty()
    site = server.removesuffix("/simbad/sim-tap")
    names = [str(name).strip() for name in table["main_id"]]
    types = [str(kind).strip() for kind in table["otype"]]
    return _frame(
        catalog="SIMBAD",
        name=names,
        ra=_floats(table["ra"]),
        dec=_floats(table["dec"]),
        type=types,
        is_variable=[kind in SIMBAD_VARIABLE_TYPES for kind in types],
        mag=_floats(table["V"]),
        url=[f"{site}/simbad/sim-id?Ident={quote(name)}" for name in names],
    )


def _from_gaia(table: Table) -> pd.DataFrame:
    if len(table) == 0:
        return _empty()
    flags = [str(flag).strip() for flag in table["phot_variable_flag"]]
    return _frame(
        catalog="Gaia",
        name=[f"Gaia DR3 {source_id}" for source_id in table["source_id"]],
        ra=_floats(table["ra"]),
        dec=_floats(table["dec"]),
        type=["VARIABLE" if flag == "VARIABLE" else "source" for flag in flags],
        is_variable=[flag == "VARIABLE" for flag in flags],
        mag=_floats(table["phot_g_mean_mag"]),
        url=None,  # the Gaia archive has no page per source
    )


# --- Cache and retries ----------------------------------------------------------------


def _cached(catalog: str, params: dict, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = _cache_path(catalog, params)
    if path.exists():
        log.debug("%s from cache %s", catalog, params)
        return pd.read_parquet(path)

    table = _with_retries(catalog, params, fetch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: a crash never leaves half a file in the cache
    log.info("%s: %d rows for %s", catalog, len(table), params)
    return table


def _with_retries(catalog: str, params: dict, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fetch()
        except Exception as error:  # any failure of a remote service counts
            last_error = error
            log.warning("%s attempt %d/%d failed %s: %s", catalog, attempt, RETRIES, params, error)
            if attempt < RETRIES:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
    raise CatalogUnavailable(f"{catalog} did not answer {params}") from last_error


def _cache_path(catalog: str, params: dict) -> Path:
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    return CACHE_DIR / catalog / f"{key}.parquet"


# --- Small helpers ----------------------------------------------------------------

_DTYPES = {"ra": "float64", "dec": "float64", "mag": "float64", "is_variable": "bool"}


def _frame(**columns) -> pd.DataFrame:
    n = len(columns["ra"])
    data = {}
    for name in COLUMNS:
        value = columns[name]
        if value is None or isinstance(value, str | bool):
            value = [value] * n
        data[name] = pd.Series(value, dtype=_DTYPES.get(name, "str"))
    return pd.DataFrame(data)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=_DTYPES.get(name, "str")) for name in COLUMNS})


def _floats(column, unit: u.Unit | None = None) -> np.ndarray:
    """Plain float array from an astropy column; masked values become NaN."""
    has_unit = unit is not None and hasattr(column, "unit")
    if hasattr(column, "filled"):
        column = column.filled(np.nan * unit if has_unit else np.nan)
    values = column.to_value(unit) if has_unit and hasattr(column, "to_value") else column
    return np.asarray(values, dtype=float)


def _cone(center: SkyCoord, radius: Angle) -> dict:
    icrs = center.icrs
    return {
        "ra": round(float(icrs.ra.deg), 5),
        "dec": round(float(icrs.dec.deg), 5),
        "radius_deg": round(float(radius.deg), 5),
    }


def _adql_circle(center: SkyCoord, radius: Angle) -> str:
    icrs = center.icrs
    return f"CIRCLE('ICRS', {icrs.ra.deg:.6f}, {icrs.dec.deg:.6f}, {radius.deg:.6f})"


def _region_radius(size_deg: float) -> Angle:
    """Cone that covers the whole square cutout (half its diagonal) plus a small margin."""
    return Angle(size_deg * np.sqrt(2) / 2 + 0.01, u.deg)


def _to_time(value) -> Time:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return Time(stamp.to_pydatetime(), scale="utc")
