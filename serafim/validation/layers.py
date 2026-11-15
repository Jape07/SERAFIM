"""Catalog layers for the viewer (task A2, AI & validation).

For each challenge target, the Gaia DR3, CatWISE2020, SIMBAD and SkyBoT sources in
its field, each one moved to the mean date of every season, so the markers move
with the video. These are the rows of gold `catalogs.parquet`; data engineering
writes them in task D8 with the frames of that run:

    from serafim.validation.layers import catalog_layers
    rows = catalog_layers(config["targets"], frames)  # frames: target_id, season, date_utc

- Season 0 means "all seasons": the mean date of every frame of the target, the
  same convention as the sums.
- gaia, catwise and simbad: one query per target. A source with a proper motion is
  moved from its catalog epoch with SkyCoord.apply_space_motion (Gaia 2016.0,
  CatWISE 2015.4, SIMBAD 2000.0); one without stays where the catalog puts it.
- catwise: entries within 30 arcsec of the target's path between 2010 and 2019
  (the years of the WISE images CatWISE combines) are left out. Along that path
  CatWISE splits fast movers, gets blended motions wrong (Epsilon Indi B comes out
  at 1.1 instead of 4.7 arcsec/yr) and leaves artifacts behind a saturated star
  (about 40 false "movers" along Barnard's path). The target itself is in the
  simbad layer, with its exact motion.
- skybot: one query per visit (frames less than 2 h apart) at the mean time of the
  visit, because asteroids cross the field within a season. Only asteroids with
  V < 20; skipped for targets more than 40 degrees from the ecliptic, where there
  are almost none.

Every query goes through the catalogs.py cache, so a second run makes no network
call. A catalog that does not answer is logged and left out; the rest go on.

    python -m serafim.validation.layers [--target barnard] [--no-skybot]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import Angle, Distance, GeocentricTrueEcliptic, SkyCoord
from astropy.time import Time
from erfa import ErfaWarning

from serafim.validation import catalogs, targets

log = logging.getLogger(__name__)

LAYER_COLUMNS = [
    "target_id", "catalog", "name", "season", "date_utc", "ra", "dec", "mag", "mag_band", "url",
]  # fmt: skip
ALL_SEASONS = 0
DEFAULT_OUT = Path("data/validation/catalog_layers.parquet")

GAIA_MAX_MAG = 18.0  # G: up to ~1600 sources in the densest field (Barnard)
CATWISE_TARGET_RADIUS_ARCSEC = 30.0
# CatWISE2020 combines WISE and NEOWISE images from 2010 to 2018: a target that moves
# fast leaves a trail of bad entries along its path over those years (Barnard: 83 arcsec).
CATWISE_YEARS = (2010.0, 2019.0)
VISIT_GAP_HOURS = 2.0
SKYBOT_MAX_ECLIPTIC_LAT_DEG = 40.0
# V: fainter asteroids are below what SPHEREx shows (most SkyBoT answers are V ~ 22)
SKYBOT_MAX_MAG = 20.0
SKYBOT_MARGIN_DEG = 0.05  # a main-belt asteroid moves ~1 arcmin/h: room for the visit length
NO_DISTANCE_MAS = 0.1  # unknown or negative parallax: 10 kpc, as targets.propagate()

MAG_BANDS = {"gaia": "G", "catwise": "W2", "simbad": "V", "skybot": "V"}
_MJD_ZERO = pd.Timestamp("1858-11-17", tz="UTC")


# --- Public API ----------------------------------------------------------------------------


def catalog_layers(
    target_list: list[dict],
    frames: pd.DataFrame | None = None,
    *,
    settings: dict | None = None,
    skybot: bool = True,
    workers: int = 4,
) -> pd.DataFrame:
    """Rows of catalogs.parquet (LAYER_COLUMNS) for these targets.

    Args:
        target_list: targets from config/targets.yaml (id, ra_now, dec_now and, for
            the CatWISE cut, ra, dec, pmra_mas_yr, pmdec_mas_yr, parallax_mas).
        frames: target_id, season and date_utc (or t_mid_mjd), one row per frame. If
            None, every SPHEREx frame of each target in IRSA, with split_seasons().
        settings: the settings block of config/targets.yaml (read from it if None).
        skybot: False leaves the asteroids out (faster).
        workers: SkyBoT queries at once.
    """
    settings = settings or _read_settings()
    if frames is None:
        frames = frames_from_archive(target_list, settings)
    radius = catalogs._region_radius(float(settings["cutout_size_deg"]))
    dates = season_dates(frames)
    visit_table = visits(frames)

    parts = []
    for target in target_list:
        target_id = target["id"]
        when = dates[dates["target_id"] == target_id]
        if when.empty:
            log.warning("%s: no frames, no catalog layers", target_id)
            continue
        center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
        for name in ("gaia", "catwise", "simbad"):
            try:
                sources = _sources(name, target, center, radius)
            except catalogs.CatalogUnavailable as error:
                log.error("%s: %s layer left out: %s", target_id, name, error)
                continue
            parts.append(at_seasons(target_id, name, sources, when))
        if skybot:
            mine = visit_table[visit_table["target_id"] == target_id]
            parts.append(_skybot_rows(target_id, center, radius, mine, workers))

    parts = [part for part in parts if len(part)]
    if not parts:
        return _empty()
    rows = pd.concat(parts, ignore_index=True)
    log.info("catalog layers: %d rows for %d targets", len(rows), rows["target_id"].nunique())
    return rows[LAYER_COLUMNS]


def move(sources: pd.DataFrame, when: Time) -> tuple[np.ndarray, np.ndarray]:
    """RA and Dec (deg) of each source at `when`, from catalogs.COLUMNS.

    Sources without a proper motion (or without an epoch) are returned unchanged.
    The parallax only sets the distance for apply_space_motion; the parallax shift
    seen from SPHEREx (under 1 arcsec) is left out, as in targets.propagate().
    """
    ra = sources["ra"].to_numpy(dtype=float).copy()
    dec = sources["dec"].to_numpy(dtype=float).copy()
    pmra = sources["pmra_mas_yr"].to_numpy(dtype=float)
    pmdec = sources["pmdec_mas_yr"].to_numpy(dtype=float)
    parallax = sources["parallax_mas"].to_numpy(dtype=float)
    epoch = sources["epoch_jyear"].to_numpy(dtype=float)
    moving = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(epoch)
    for start_epoch in np.unique(epoch[moving]):
        chosen = moving & (epoch == start_epoch)
        known = np.nan_to_num(parallax[chosen], nan=-1.0)
        distance_mas = np.where(known > 0, known, NO_DISTANCE_MAS)
        start = SkyCoord(
            ra=ra[chosen] * u.deg,
            dec=dec[chosen] * u.deg,
            pm_ra_cosdec=pmra[chosen] * u.mas / u.yr,
            pm_dec=pmdec[chosen] * u.mas / u.yr,
            distance=Distance(parallax=distance_mas * u.mas),
            obstime=Time(start_epoch, format="jyear"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            moved = start.apply_space_motion(new_obstime=when)
        ra[chosen], dec[chosen] = moved.ra.deg, moved.dec.deg
    return ra, dec


def at_seasons(
    target_id: str, catalog: str, sources: pd.DataFrame, when: pd.DataFrame
) -> pd.DataFrame:
    """One row per source and season, each at that season's mean date."""
    if sources.empty:
        return _empty()
    parts = []
    for season, mjd in zip(when["season"], when["mjd"], strict=True):
        ra, dec = move(sources, Time(float(mjd), format="mjd", scale="utc"))
        parts.append(
            pd.DataFrame(
                {
                    "target_id": target_id,
                    "catalog": catalog,
                    "name": sources["name"].to_numpy(),
                    "season": int(season),
                    "date_utc": _utc(float(mjd)),
                    "ra": ra,
                    "dec": dec,
                    "mag": sources["mag"].to_numpy(dtype=float),
                    "mag_band": MAG_BANDS[catalog],
                    "url": sources["url"].to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def season_dates(frames: pd.DataFrame) -> pd.DataFrame:
    """target_id, season and mjd: the mean date of each season, plus season 0 (all)."""
    times = frames[["target_id", "season"]].assign(mjd=_mjd(frames))
    per_season = times.groupby(["target_id", "season"], as_index=False)["mjd"].mean()
    every = times.groupby("target_id", as_index=False)["mjd"].mean().assign(season=ALL_SEASONS)
    dates = pd.concat([every, per_season], ignore_index=True)
    dates["season"] = dates["season"].astype("int64")
    return dates.sort_values(["target_id", "season"], ignore_index=True)[
        ["target_id", "season", "mjd"]
    ]


def visits(frames: pd.DataFrame, gap_hours: float = VISIT_GAP_HOURS) -> pd.DataFrame:
    """target_id, season and mjd of each visit: frames less than gap_hours apart."""
    times = frames[["target_id", "season"]].assign(mjd=_mjd(frames))
    rows = []
    for target_id, group in times.groupby("target_id", sort=False):
        group = group.sort_values("mjd")
        mjd = group["mjd"].to_numpy(dtype=float)
        season = group["season"].to_numpy()
        starts = np.r_[0, np.flatnonzero(np.diff(mjd) > gap_hours / 24) + 1]
        ends = np.r_[starts[1:], len(mjd)]
        for first, last in zip(starts, ends, strict=True):
            rows.append(
                {
                    "target_id": target_id,
                    "season": int(season[first]),
                    "mjd": float(mjd[first:last].mean()),
                }
            )
    return pd.DataFrame(rows, columns=["target_id", "season", "mjd"])


def frames_from_archive(target_list: list[dict], settings: dict) -> pd.DataFrame:
    """Every SPHEREx frame of each target in IRSA (target_id, season, t_mid_mjd)."""
    seasons = settings.get("seasons", {})
    parts = []
    for target in target_list:
        center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
        found = targets.spherex_frames(center, list(settings["collections"]))
        parts.append(
            pd.DataFrame(
                {
                    "target_id": target["id"],
                    "season": targets.split_seasons(found["t_mid_mjd"], **seasons),
                    "t_mid_mjd": found["t_mid_mjd"].to_numpy(dtype=float),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["target_id", "season", "t_mid_mjd"])
    return pd.concat(parts, ignore_index=True)


# --- One catalog for one target --------------------------------------------------------------


def _sources(name: str, target: dict, center: SkyCoord, radius: Angle) -> pd.DataFrame:
    if name == "gaia":
        return catalogs.gaia(center, radius, max_mag=GAIA_MAX_MAG)
    if name == "simbad":
        return catalogs.simbad(center, radius)
    if name == "catwise":
        return _away_from_target(catalogs.catwise(center, radius), target)
    raise ValueError(f"unknown catalog {name}")


def _away_from_target(sources: pd.DataFrame, target: dict) -> pd.DataFrame:
    """CatWISE entries farther than CATWISE_TARGET_RADIUS_ARCSEC from the target's path
    while CatWISE took its images (CATWISE_YEARS)."""
    if sources.empty:
        return sources
    years = np.arange(CATWISE_YEARS[0], CATWISE_YEARS[1] + 0.01, 0.25)
    path = _target_at(target, Time(years, format="jyear"))
    positions = SkyCoord(sources["ra"].to_numpy() * u.deg, sources["dec"].to_numpy() * u.deg)
    closest = positions[:, None].separation(path[None, :]).arcsec.min(axis=1)
    far = closest > CATWISE_TARGET_RADIUS_ARCSEC
    log.info("%s: %d CatWISE entries on the target's path left out", target["id"],
             int((~far).sum()))  # fmt: skip
    return sources[far].reset_index(drop=True)


def _target_at(target: dict, when: Time) -> SkyCoord:
    """The target at `when`; a target without catalog motion (added by coordinates) stays put."""
    full = {
        "ra": target.get("ra", target["ra_now"]),
        "dec": target.get("dec", target["dec_now"]),
        "pmra_mas_yr": target.get("pmra_mas_yr") or 0.0,
        "pmdec_mas_yr": target.get("pmdec_mas_yr") or 0.0,
        "parallax_mas": target.get("parallax_mas") or 0.0,
    }
    moved = targets.propagate(full, when)
    return SkyCoord(ra=moved.ra, dec=moved.dec)


def _skybot_rows(
    target_id: str, center: SkyCoord, radius: Angle, visit_rows: pd.DataFrame, workers: int
) -> pd.DataFrame:
    latitude = center.transform_to(GeocentricTrueEcliptic()).lat.deg
    if abs(latitude) > SKYBOT_MAX_ECLIPTIC_LAT_DEG:
        log.info("%s: SkyBoT skipped, %.0f deg from the ecliptic", target_id, latitude)
        return _empty()
    cone = Angle(radius.deg + SKYBOT_MARGIN_DEG, u.deg)
    dates = [Time(float(mjd), format="mjd", scale="utc") for mjd in visit_rows["mjd"]]

    def ask(date: Time) -> pd.DataFrame | None:
        try:
            return catalogs.skybot(center, cone, date)
        except catalogs.CatalogUnavailable as error:
            log.error("%s: SkyBoT at %s left out: %s", target_id, date.isot, error)
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        answers = list(pool.map(ask, dates))

    parts = []
    for season, mjd, found in zip(visit_rows["season"], visit_rows["mjd"], answers, strict=True):
        if found is None:
            continue
        found = found[~(found["mag"] >= SKYBOT_MAX_MAG)]  # no V (some comets): kept
        if found.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "target_id": target_id,
                    "catalog": "skybot",
                    "name": found["name"].to_numpy(),
                    "season": int(season),
                    "date_utc": _utc(float(mjd)),
                    "ra": found["ra"].to_numpy(dtype=float),
                    "dec": found["dec"].to_numpy(dtype=float),
                    "mag": found["mag"].to_numpy(dtype=float),
                    "mag_band": MAG_BANDS["skybot"],
                    "url": found["url"].to_numpy(),
                }
            )
        )
    log.info("%s: SkyBoT %d visits, %d asteroid positions", target_id, len(dates),
             sum(len(part) for part in parts))  # fmt: skip
    return pd.concat(parts, ignore_index=True) if parts else _empty()


# --- Small helpers -------------------------------------------------------------------------


def _mjd(frames: pd.DataFrame) -> np.ndarray:
    if "t_mid_mjd" in frames:
        return frames["t_mid_mjd"].to_numpy(dtype=float)
    stamps = pd.to_datetime(frames["date_utc"], utc=True)
    return ((stamps - _MJD_ZERO) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def _utc(mjd: float) -> pd.Timestamp:
    return (_MJD_ZERO + pd.to_timedelta(mjd, unit="D")).round("s")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": pd.Series(dtype="str"),
            "catalog": pd.Series(dtype="str"),
            "name": pd.Series(dtype="str"),
            "season": pd.Series(dtype="int64"),
            "date_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "ra": pd.Series(dtype="float64"),
            "dec": pd.Series(dtype="float64"),
            "mag": pd.Series(dtype="float64"),
            "mag_band": pd.Series(dtype="str"),
            "url": pd.Series(dtype="str"),
        }
    )


def _read_settings() -> dict:
    return targets.read_settings(targets.TARGETS_PATH.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m serafim.validation.layers",
        description="Gaia, CatWISE, SIMBAD and SkyBoT around each target, at every season.",
    )
    parser.add_argument("--target", help="Only this target id (default: all 5).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-skybot", action="store_true", help="Leave the asteroids out.")
    args = parser.parse_args(argv)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    text = targets.TARGETS_PATH.read_text(encoding="utf-8")
    target_list = yaml.safe_load(text)["targets"]
    if args.target:
        target_list = [target for target in target_list if target["id"] == args.target]
        if not target_list:
            parser.error(f"no target '{args.target}' in {targets.TARGETS_PATH}")
    rows = catalog_layers(target_list, settings=targets.read_settings(text),
                          skybot=not args.no_skybot)  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out, index=False)
    summary = rows.groupby(["target_id", "catalog"], sort=False).agg(
        sources=("name", "nunique"), rows=("name", "size"), seasons=("season", "nunique")
    )
    print(summary.to_string())
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
