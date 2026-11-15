"""A fixed copy of Gaia DR3 around each target, committed in config/gaia_snapshot/.

Gaia DR3 never changes, and every query the pipeline makes is a cone around a target's
ra_now/dec_now (0.187 deg for the catalog layers, 110 arcsec for the neighbours of a fit) with
G < 18 or G < 19. One cone of SNAPSHOT_RADIUS_DEG with G < SNAPSHOT_MAX_MAG per target covers
all of them, so gold does not have to wait for ESA's archive (down for days in November 2026).

    python -m serafim gaia-snapshot [--target X]

- save_snapshots(): writes config/gaia_snapshot/<target>.parquet from a VizieR mirror of
  Gaia DR3 (I/355/gaiadr3), or from ESA's archive if no mirror answers. Commit the files.
- from_snapshot(center, radius, max_mag): the sources of that query, cut from the copy that
  covers it, in the columns of validation.catalogs.gaia(); None if no copy covers it.

Positions are ICRS at epoch 2016.0 with their proper motion and parallax, as in ESA's archive,
so layers.move() takes them to any date.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import Angle, SkyCoord

from serafim.config import ROOT, load_config

log = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("config") / "gaia_snapshot"
SNAPSHOT_RADIUS_DEG = 0.2  # > 0.187 deg, the cone of the catalog layers for a 0.25 deg cutout
SNAPSHOT_MAX_MAG = 19.0  # the faintest G any caller asks for
METADATA_KEY = b"serafim_gaia_snapshot"

# Mirrors of VizieR that serve Gaia DR3, tried in order. The main CDS site was offline in
# November 2026; the Harvard mirror answers most of the time.
VIZIER_SERVERS = ("vizier.cfa.harvard.edu", "vizier.cds.unistra.fr", "vizier.iucaa.in")
VIZIER_COLUMNS = ["Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "Gmag", "VarFlag"]


def snapshot_path(target_id: str, root: Path | None = None) -> Path:
    return (root or ROOT) / SNAPSHOT_DIR / f"{target_id}.parquet"


# --- Reading ---------------------------------------------------------------------------------
def from_snapshot(center: SkyCoord, radius: Angle | u.Quantity, max_mag: float, *,
                  root: Path | None = None) -> pd.DataFrame | None:
    """The Gaia sources of this cone brighter than max_mag, from the copy that covers it.

    None if no saved copy holds the whole cone down to max_mag (the caller then asks the
    archive as before).
    """
    radius = Angle(radius)
    folder = (root or ROOT) / SNAPSHOT_DIR
    for path in sorted(folder.glob("*.parquet")):
        meta = read_metadata(path)
        snap_center = SkyCoord(meta["ra"] * u.deg, meta["dec"] * u.deg)
        inside = snap_center.separation(center.icrs).deg + radius.deg <= meta["radius_deg"] + 1e-6
        if not inside or max_mag > meta["max_mag"]:
            continue
        table = pd.read_parquet(path)
        stars = SkyCoord(table["ra"].to_numpy() * u.deg, table["dec"].to_numpy() * u.deg)
        keep = (center.icrs.separation(stars).deg <= radius.deg) & (table["mag"] < max_mag)
        log.debug("gaia from %s: %d of %d sources", path.name, int(keep.sum()), len(table))
        return table[keep].reset_index(drop=True)
    return None


def read_metadata(path: Path) -> dict:
    """Center, radius, magnitude limit, source and date of a saved copy."""
    metadata = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError(f"{path} is not a Gaia snapshot (no {METADATA_KEY.decode()} metadata)")
    return json.loads(metadata[METADATA_KEY])


# --- Writing ---------------------------------------------------------------------------------
def save_snapshots(target_id: str | None = None, *, root: Path | None = None,
                   fetch=None) -> list[Path]:
    """Write config/gaia_snapshot/<target>.parquet for every target (or one).

    `fetch(center, radius, max_mag) -> (DataFrame, source)` replaces the network in tests.
    A target whose query fails is skipped with an error; the others are still written.
    """
    from serafim.bronze.download import _targets

    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    fetch = fetch or fetch_gaia
    radius = Angle(SNAPSHOT_RADIUS_DEG, u.deg)
    written = []
    for target in _targets(config, target_id):
        center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
        try:
            table, source = fetch(center, radius, SNAPSHOT_MAX_MAG)
        except Exception as error:  # any failure of a remote service
            log.error("target=%s: no Gaia snapshot saved: %s", target["id"], error)
            continue
        if table.empty:
            log.error("target=%s: no Gaia snapshot saved: %s answered no sources",
                      target["id"], source)
            continue
        meta = {
            "target_id": str(target["id"]),
            "ra": round(float(center.ra.deg), 6),
            "dec": round(float(center.dec.deg), 6),
            "radius_deg": SNAPSHOT_RADIUS_DEG,
            "max_mag": SNAPSHOT_MAX_MAG,
            "catalog": "Gaia DR3",
            "source": source,
            "saved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = snapshot_path(str(target["id"]), root)
        write(table, meta, path)
        log.info("target=%s: %d Gaia sources saved in %s (from %s)", target["id"], len(table),
                 path, source)
        written.append(path)
    return written


def write(table: pd.DataFrame, meta: dict, path: Path) -> None:
    """The table sorted by magnitude, with the metadata in the Parquet schema."""
    from serafim.validation import catalogs

    table = table[catalogs.COLUMNS].sort_values("mag", kind="stable").reset_index(drop=True)
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    arrow = arrow.replace_schema_metadata({**(arrow.schema.metadata or {}),
                                           METADATA_KEY: json.dumps(meta).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, path)


def fetch_gaia(center: SkyCoord, radius: Angle, max_mag: float) -> tuple[pd.DataFrame, str]:
    """Gaia DR3 from a VizieR mirror; from ESA's archive if every mirror fails."""
    from serafim.validation import catalogs

    try:
        return from_vizier(center, radius, max_mag)
    except catalogs.CatalogUnavailable as error:
        log.warning("%s; asking ESA's Gaia archive", error)
    table = catalogs._from_gaia(catalogs._query_gaia(center, radius, max_mag))
    return table, catalogs.GAIA_TAP_SERVER


def from_vizier(center: SkyCoord, radius, max_mag: float) -> tuple[pd.DataFrame, str]:
    """Gaia DR3 sources from the first VizieR mirror that answers, in the columns of
    validation.catalogs.gaia(); also return the mirror. Positions are ICRS at epoch 2016.0."""
    from astroquery.vizier import Vizier

    from serafim.validation import catalogs

    last = None
    for server in VIZIER_SERVERS:
        vizier = Vizier(columns=VIZIER_COLUMNS, row_limit=-1,
                        column_filters={"Gmag": f"<{max_mag}"}, vizier_server=server, timeout=60)
        try:
            found = vizier.query_region(center, radius=Angle(radius), catalog="I/355/gaiadr3")
        except Exception as error:  # any failure: try the next mirror
            last = f"{server}: {error}"
            continue
        if not len(found) or not len(found[0]):  # an empty answer once came back: next mirror
            last = f"{server} answered no rows"
            continue
        rows = vizier_rows(found[0].to_pandas())
        log.info("Gaia DR3 from %s: %d sources", server, len(rows))
        return rows, server
    raise catalogs.CatalogUnavailable(f"Gaia DR3 at VizieR did not answer: {last}")


def vizier_rows(table: pd.DataFrame) -> pd.DataFrame:
    """VizieR's I/355/gaiadr3 columns as validation.catalogs._from_gaia() writes ESA's."""
    from serafim.validation import catalogs

    flags = table["VarFlag"].astype(str).str.strip() if "VarFlag" in table else ""
    variable = np.asarray(flags == "VARIABLE", dtype=bool)
    pmra = table["pmRA"].astype(float).to_numpy()
    pmdec = table["pmDE"].astype(float).to_numpy()
    has_motion = np.isfinite(pmra) & np.isfinite(pmdec)
    return catalogs._frame(
        catalog="Gaia",
        name=[f"Gaia DR3 {source}" for source in table["Source"]],
        ra=table["RA_ICRS"].astype(float).to_numpy(),
        dec=table["DE_ICRS"].astype(float).to_numpy(),
        type=np.where(variable, "VARIABLE", "source"),
        is_variable=variable,
        mag=table["Gmag"].astype(float).to_numpy(),
        url=None,  # the Gaia archive has no page per source
        pmra_mas_yr=pmra,
        pmdec_mas_yr=pmdec,
        parallax_mas=table["Plx"].astype(float).to_numpy(),
        epoch_jyear=np.where(has_motion, catalogs.GAIA_EPOCH_JYEAR, np.nan),
    )
