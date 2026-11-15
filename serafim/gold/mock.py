"""Simulated gold for the targets (task D2).

Lets the website work before real data exists. It follows serafim/contract.py exactly. For
each target it simulates the three seasons of SPHEREx, six detectors and three frames per
cell, so the tables and PNGs have the same shape as real gold:

- one cell is missing on purpose (detector 6 in season 1, like WISE 0855), so the site
  can test the hole. That season has no color image either, because it needs detector 6;
- the center is saturated in detectors 1 and 2: `saturated = true`, flags "BLOOM,OVERFLOW"
  and the empty pixels painted with the brightest color of the scale (decision of AI,
  task D7), so the star does not look like a hole;
- a `motion` target moves between seasons (exaggerated: 5 px per season, so the video and
  the difference images show it) and a `brightness` target changes its flux;
- `catalogs`, `spectra` and `trajectory` have example rows, with the nulls the contract allows
  (no `mag` or `url` for a source, no `flag` on a good point, no measurement in season 2).
  Their names and values are placeholders: AI & validation fills the real ones (A2, A4, A5).

Everything comes from a fixed seed, so two runs write the same tables and PNGs (only
meta.json changes: it records the run date).

    python -m serafim mock                 # first target, in data/gold/
    python -m serafim mock --all-targets   # every target (the published site)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from matplotlib import image as mpimg

from serafim import contract
from serafim.gold.export import build_meta, write_gold
from serafim.validation.targets import PIXEL_SCALE_ARCSEC

log = logging.getLogger(__name__)

SEED = 20261114
N_STARS = 40
FRAMES_PER_CELL = 3

# Season number -> (first day UTC, SIA collection that serves it).
SEASONS = {
    1: ("2025-08-15", "spherex_dr1"),
    2: ("2026-03-11", "spherex_dr1"),
    3: ("2026-08-16", "spherex_qr3"),
}
HOLE = (6, 1)  # (detector, season) without frames
UNMEASURED_SEASON = 2  # trajectory has no measurement (null) for this season
SATURATED_DETECTORS = (1, 2)
SATURATED_FLAGS = "BLOOM,OVERFLOW"
SATURATED_RADIUS_PX = 2.2

DETECTOR_GAIN = {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.5, 5: 0.35, 6: 0.25}  # fainter at long wavelengths
BACKGROUND = 10.0
FRAME_NOISE = 1.0  # one frame; a sum of n frames has FRAME_NOISE / sqrt(n)
TARGET_FLUX = 150.0
MOTION_PX_PER_SEASON = 5.0
BRIGHTNESS_BY_SEASON = {1: 1.0, 2: 1.8, 3: 0.7}  # for `brightness` targets

GAIA_URL = "https://gea.esac.esa.int/archive/"
SIMBAD_URL = "https://simbad.cds.unistra.fr/simbad/"
SKYBOT_URL = "https://ssp.imcce.fr/webservices/skybot/"


@dataclass(frozen=True)
class Png:
    """An image to save as PNG: values plus how to color them (RGB images need neither)."""

    data: np.ndarray
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class MockGold:
    tables: dict[str, pd.DataFrame]  # keyed like contract.GOLD_TABLES
    pngs: dict[str, Png]  # keyed by path relative to gold/


# --- Pure functions: no disk, no network ------------------------------------
def target_wcs(ra: float, dec: float, size_deg: float) -> WCS:
    """TAN grid centered on the target, one SPHEREx pixel per pixel, north up."""
    n = int(round(size_deg * 3600 / PIXEL_SCALE_ARCSEC))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600, PIXEL_SCALE_ARCSEC / 3600]
    wcs.pixel_shape = (n, n)
    return wcs


def wcs_to_json(wcs: WCS) -> str:
    header = wcs.to_header()
    values = {key: header[key] for key in header}
    values["NAXIS1"], values["NAXIS2"] = wcs.pixel_shape
    return json.dumps(values, sort_keys=True)


def png_to_sky(wcs: WCS, x_px, y_px) -> tuple[np.ndarray, np.ndarray]:
    """PNG pixel (origin top-left) -> (ra, dec) in degrees."""
    n_rows = wcs.pixel_shape[1]
    ra, dec = wcs.wcs_pix2world(x_px, n_rows - 1 - np.asarray(y_px), 0)
    return np.mod(ra, 360.0), dec


def target_xy(kind: str, n: int, season: int) -> tuple[float, float]:
    """Target position in the PNG: the grid center, moving north (up) for `motion` targets.

    Season 0 (all seasons) is the middle of the path, where the mean date of every frame is.
    """
    x = y = (n - 1) / 2
    if kind == "motion":
        y -= MOTION_PX_PER_SEASON * ((season - 1) if season else (len(SEASONS) - 1) / 2)
    return x, y


def brightness(kind: str, season: int) -> float:
    return BRIGHTNESS_BY_SEASON[season] if kind == "brightness" else 1.0


def _resolving_power(wavelength_um: float) -> float:
    """Approximate SPHEREx spectral resolution per detector band."""
    if wavelength_um < 2.42:
        return 41.0
    if wavelength_um < 3.83:
        return 35.0
    if wavelength_um < 4.42:
        return 112.0
    return 128.0


def build_mock(target: dict, size_deg: float) -> MockGold:
    """All simulated gold for one target of config/targets.yaml, in memory."""
    rng = np.random.default_rng(SEED)
    target_id, kind = target["id"], target["kind"]
    wcs = target_wcs(float(target["ra_now"]), float(target["dec_now"]), size_deg)
    n = wcs.pixel_shape[0]

    targets = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "name": target["name"],
                "kind": kind,
                "ra_now": float(target["ra_now"]),
                "dec_now": float(target["dec_now"]),
                "pmra_mas_yr": float(target["pmra_mas_yr"]),
                "pmdec_mas_yr": float(target["pmdec_mas_yr"]),
                "size_deg": float(size_deg),
                "wcs_json": wcs_to_json(wcs),
            }
        ]
    )
    stars_xy = rng.uniform(5, n - 5, (N_STARS, 2))
    stars_flux = rng.lognormal(2.5, 0.8, N_STARS)

    frames = _frames(target_id, rng)
    sums = _sums(target_id, frames)
    differences = _differences(target_id, sums)
    pngs = _render(target, sums, differences, n, stars_xy, stars_flux, rng)
    return MockGold(
        tables={
            "targets.parquet": targets,
            "frames.parquet": frames,
            "sums.parquet": sums,
            "differences.parquet": differences,
            "catalogs.parquet": _catalogs(target, frames, wcs, n, stars_xy),
            "spectra.parquet": _spectra(target, frames, rng),
            "trajectory.parquet": _trajectory(target, wcs, n, rng),
        },
        pngs=pngs,
    )


def _frames(target_id: str, rng: np.random.Generator) -> pd.DataFrame:
    """Three frames in every (detector, season) cell except the hole."""
    rows = []
    for season, (start, collection) in SEASONS.items():
        for detector in range(1, 7):
            if (detector, season) == HOLE:
                continue
            low, high = contract.DETECTOR_BANDS_UM[detector]
            saturated = detector in SATURATED_DETECTORS
            for k in range(FRAMES_PER_CELL):
                # Frames 9 days apart; each detector sees the target one orbit (98 min) later.
                date = pd.Timestamp(start, tz="UTC") + pd.Timedelta(
                    days=9 * k, minutes=98 * detector
                )
                wavelength = round(float(low + (high - low) * rng.uniform(0.15, 0.85)), 4)
                rows.append(
                    {
                        "frame_id": f"{target_id}_s{season}_d{detector}_f{k}",
                        "target_id": target_id,
                        "obs_id": f"MOCK_{date:%Y%m%d}_{len(rows):03d}",
                        "collection": collection,
                        "date_utc": date,
                        "season": season,
                        "detector": detector,
                        "wavelength_um": wavelength,
                        "bandwidth_um": round(wavelength / _resolving_power(wavelength), 4),
                        "flags": SATURATED_FLAGS if saturated else "",
                        "saturated": saturated,
                    }
                )
    return pd.DataFrame(rows).sort_values("date_utc", ignore_index=True)


def _union_flags(flags: pd.Series) -> str:
    return ",".join(sorted({name for text in flags for name in text.split(",") if name}))


def _sums(target_id: str, frames: pd.DataFrame) -> pd.DataFrame:
    """One sum per (detector, season), plus the white image (detector 0) and all seasons (0)."""
    rows = []
    for detector in range(0, 7):
        for season in range(0, len(SEASONS) + 1):
            if (detector, season) == (0, 0):
                continue
            cell = frames
            if detector:
                cell = cell[cell.detector == detector]
            if season:
                cell = cell[cell.season == season]
            if cell.empty:
                continue  # The hole: no frames, so no sum.
            rows.append(
                {
                    "sum_id": f"{target_id}_d{detector}_s{season}",
                    "target_id": target_id,
                    "detector": detector,
                    "season": season,
                    "mean_date_utc": cell.date_utc.mean().floor("s"),
                    "n_frames": len(cell),
                    "background_noise": round(FRAME_NOISE / np.sqrt(len(cell)), 4),
                    "flags": _union_flags(cell["flags"]),  # df.flags is a pandas attribute
                    "saturated": bool(cell.saturated.any()),
                    "png": contract.sum_png(target_id, detector, season),
                }
            )
    return pd.DataFrame(rows)


def _differences(target_id: str, sums: pd.DataFrame) -> pd.DataFrame:
    """Season b minus a: consecutive seasons and last minus first, where both sums exist."""
    last = len(SEASONS)
    pairs = [(s, s + 1) for s in range(1, last)] + [(1, last)]
    cells = set(zip(sums.detector, sums.season, strict=True))
    rows = []
    for detector in range(1, 7):
        for a, b in pairs:
            if (detector, a) in cells and (detector, b) in cells:
                rows.append(
                    {
                        "difference_id": f"{target_id}_d{detector}_s{a}-s{b}",
                        "target_id": target_id,
                        "detector": detector,
                        "season_a": a,
                        "season_b": b,
                        "png": contract.difference_png(target_id, detector, a, b),
                    }
                )
    return pd.DataFrame(rows)


def _render(
    target: dict,
    sums: pd.DataFrame,
    differences: pd.DataFrame,
    n: int,
    stars_xy: np.ndarray,
    stars_flux: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Png]:
    """Sum PNGs (asinh stretch, magma), differences (diverging) and color images.

    Limits are fixed per detector, so the background looks the same in every season.
    """
    target_id, kind = target["id"], target["kind"]
    yy, xx = np.mgrid[0:n, 0:n]

    def blob(x: float, y: float, flux: float, sigma: float = 1.4) -> np.ndarray:
        return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

    stars = sum(blob(x, y, f) for (x, y), f in zip(stars_xy, stars_flux, strict=True))

    # The image of every (detector 1-6, season >= 1) cell, and where its pixels are empty.
    images: dict[tuple[int, int], np.ndarray] = {}
    empty: dict[tuple[int, int], np.ndarray] = {}
    single = sums[(sums.detector > 0) & (sums.season > 0)]
    for row in single.itertuples():
        x, y = target_xy(kind, n, row.season)
        flux = TARGET_FLUX * brightness(kind, row.season) * (3 if row.saturated else 1)
        signal = BACKGROUND + DETECTOR_GAIN[row.detector] * (stars + blob(x, y, flux))
        noise = rng.normal(0, FRAME_NOISE / np.sqrt(row.n_frames), (n, n))
        images[row.detector, row.season] = signal + noise
        empty[row.detector, row.season] = (
            ((xx - x) ** 2 + (yy - y) ** 2 <= SATURATED_RADIUS_PX**2)
            if row.saturated
            else np.zeros((n, n), dtype=bool)
        )

    # All seasons of one detector, and all detectors of one season, are averages of those.
    singles = list(images)
    for row in sums[(sums.detector == 0) | (sums.season == 0)].itertuples():
        keys = [k for k in singles if row.detector in (0, k[0]) and row.season in (0, k[1])]
        images[row.detector, row.season] = np.mean([images[k] for k in keys], axis=0)
        empty[row.detector, row.season] = np.any([empty[k] for k in keys], axis=0)

    stretched = {k: np.arcsinh((image - BACKGROUND) / FRAME_NOISE) for k, image in images.items()}
    limits = {}
    for detector in range(0, 7):
        values = [stretched[k][~empty[k]] for k in stretched if k[0] == detector]
        if values:
            low, high = np.percentile(np.concatenate(values), [1, 99.9])
            limits[detector] = (float(low), float(high))

    pngs: dict[str, Png] = {}
    for row in sums.itertuples():
        key = (row.detector, row.season)
        low, high = limits[row.detector]
        # Saturated pixels get the brightest color, not a hole.
        pngs[row.png] = Png(np.where(empty[key], high, stretched[key]), "magma", low, high)

    # Differences: season b minus a. A pixel empty in either season counts as zero.
    diffs = {}
    for row in differences.itertuples():
        a, b = (row.detector, row.season_a), (row.detector, row.season_b)
        diff = images[b] - images[a]
        diff[empty[a] | empty[b]] = 0.0
        diffs[row.png] = (row.detector, diff)
    for detector in sorted({d for d, _ in diffs.values()}):
        group = {path: diff for path, (d, diff) in diffs.items() if d == detector}
        noise = FRAME_NOISE * np.sqrt(2 / FRAMES_PER_CELL)
        limit = float(max(5 * noise, np.percentile(np.abs(np.stack(list(group.values()))), 99.9)))
        pngs |= {path: Png(diff, "RdBu_r", -limit, limit) for path, diff in group.items()}

    # Color: detector 6 red, 3 green and 1 blue, each with its own limits.
    for season in SEASONS:
        if all((d, season) in stretched for d in contract.COLOR_DETECTORS):
            channels = []
            for detector in reversed(contract.COLOR_DETECTORS):
                key = (detector, season)
                low, high = limits[detector]
                scaled = np.where(empty[key], high, stretched[key])
                channels.append(np.clip((scaled - low) / (high - low), 0.0, 1.0))
            pngs[contract.color_png(target_id, season)] = Png(np.dstack(channels))
    return pngs


def _catalogs(
    target: dict, frames: pd.DataFrame, wcs: WCS, n: int, stars_xy: np.ndarray
) -> pd.DataFrame:
    """Example rows shaped like AI's `catalog_layers()` (task A2).

    Gaia, CatWISE and SIMBAD have one row per source for season 0 (all seasons, at the mean
    date of every frame) and for each season (at that season's mean date). SkyBoT has one row
    per asteroid and visit, so an asteroid has several rows in a season and none in season 0.
    """
    rows = []
    first, last = frames.date_utc.min(), frames.date_utc.max()
    mean_date = {0: frames.date_utc.mean().floor("s")} | {
        season: frames[frames.season == season].date_utc.mean().floor("s") for season in SEASONS
    }

    def add(catalog, name, season, date, x, y, mag, url):
        ra, dec = png_to_sky(wcs, x, y)
        rows.append(
            {
                "target_id": target["id"],
                "catalog": catalog,
                "name": name,
                "season": season,
                "date_utc": date,
                "ra": round(float(ra), 6),
                "dec": round(float(dec), 6),
                "mag": mag,
                "mag_band": contract.MAG_BANDS[catalog],
                "url": url,
            }
        )

    for season, date in mean_date.items():
        x, y = target_xy(target["kind"], n, season)
        add("gaia", f"MOCK Gaia {target['name']}", season, date, x, y, 9.5, GAIA_URL)
        add("simbad", target["name"], season, date, x, y, 9.5, SIMBAD_URL)
        for i in (0, 1):
            add("gaia", f"MOCK Gaia neighbor {i + 1}", season, date, *stars_xy[i], 14.0 + i,
                GAIA_URL)
        # CatWISE has no mag for this source and, like Gaia, no url.
        add("catwise", "MOCK CatWISE neighbor", season, date, *stars_xy[2], None, None)

    # An asteroid crossing the field, seen at the first and the last frame of each season.
    for season in SEASONS:
        dates = frames[frames.season == season].date_utc
        for visit in (dates.min().floor("s"), dates.max().floor("s")):
            progress = (visit - first) / (last - first)
            add("skybot", "MOCK asteroid", season, visit, 20 + 100 * progress,
                120 - 80 * progress, 17.8, SKYBOT_URL)
    return pd.DataFrame(rows)


def _spectra(target: dict, frames: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One photometry point per frame, except saturated ones (the site says "saturated center")."""
    rows = []
    for frame in frames[~frames.saturated].itertuples():
        flux = 500.0 * frame.wavelength_um**-2 * brightness(target["kind"], frame.season)
        flux *= 1 + rng.normal(0, 0.03)
        rows.append(
            {
                "target_id": target["id"],
                "frame_id": frame.frame_id,
                "season": frame.season,
                "date_utc": frame.date_utc,
                "wavelength_um": frame.wavelength_um,
                "flux_mjy": round(float(flux), 3),
                "error_mjy": round(float(0.05 * flux), 3),
                "flag": "LOW_SNR" if frame.detector >= 5 else None,  # placeholder name
            }
        )
    return pd.DataFrame(rows)


def _trajectory(target: dict, wcs: WCS, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """The catalog position of the target each season, and a measurement within ~1 arcsec.

    One season (UNMEASURED_SEASON) has no measurement, so the site can test a gap.
    """
    rows = []
    for season in SEASONS:
        ra, dec = png_to_sky(wcs, *target_xy(target["kind"], n, season))
        error = round(float(rng.uniform(0.8, 1.6)), 2)
        d_ra = rng.normal(0, 0.5) / 3600 / np.cos(np.radians(dec))
        d_dec = rng.normal(0, 0.5) / 3600
        measured = season != UNMEASURED_SEASON
        rows.append(
            {
                "target_id": target["id"],
                "season": season,
                "ra_measured": round(float(ra + d_ra), 6) if measured else None,
                "dec_measured": round(float(dec + d_dec), 6) if measured else None,
                "error_arcsec": error if measured else None,
                "ra_catalog": round(float(ra), 6),
                "dec_catalog": round(float(dec), 6),
            }
        )
    return pd.DataFrame(rows)


# --- Writing to disk ---------------------------------------------------------
def write_mock_gold(gold_dir: Path, config: dict) -> MockGold:
    """Build simulated gold for the first target in the config and write it to gold_dir."""
    size_deg = config["settings"]["cutout_size_deg"]
    gold = build_mock(config["targets"][0], size_deg)
    _write([gold], gold_dir, config)
    return gold


def write_all_targets(gold_dir: Path, config: dict) -> None:
    """Simulated gold for every target in the config, for the published site.

    ``write_mock_gold`` stays one target: that is what the contract tests check.
    """
    size_deg = config["settings"]["cutout_size_deg"]
    _write([build_mock(t, size_deg) for t in config["targets"]], gold_dir, config)


def _write(pieces: list[MockGold], gold_dir: Path, config: dict) -> None:
    gold_dir = Path(gold_dir)
    tables = {
        name: pd.concat([piece.tables[name] for piece in pieces], ignore_index=True)
        for name in contract.GOLD_TABLES
    }
    for piece in pieces:
        for path, png in piece.pngs.items():
            target = gold_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(target, png.data, cmap=png.cmap, vmin=png.vmin, vmax=png.vmax)
    meta = build_meta(tables["frames.parquet"], config["settings"]["collections"])
    write_gold(gold_dir, tables, meta)
    log.info(
        "simulated gold in %s: %d targets, %d frames, %d sums, %d differences, %d PNG",
        gold_dir,
        len(tables["targets.parquet"]),
        len(tables["frames.parquet"]),
        len(tables["sums.parquet"]),
        len(tables["differences.parquet"]),
        sum(len(piece.pngs) for piece in pieces),
    )
