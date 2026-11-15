"""How well the sums of a target line up with the sky (task D5, asked by AI & validation).

    python -m serafim alignment [--target barnard]     # rewrites docs/alignment.md

A difference between seasons only shows real motion if the sums of both seasons sit on the same
grid: an error of 0.2 px leaves a false dipole on every fixed star. So, for each sum of one
detector and one season, the bright stars found in the sum are compared with where Gaia puts
them on the mean date of the sum (Gaia G 10-15 from the copy in config/gaia_snapshot/, or the
query that catalog_layers() uses, moved with serafim.validation.layers.move()). The goal is a
median offset under GOAL_PX in x and in y. Nothing is corrected automatically: a cell over the
goal is reported and talked over with AI.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from serafim.config import ROOT, load_config
from serafim.silver.frames import target_position
from serafim.silver.store import load_sum, sums_table_path

log = logging.getLogger("serafim.silver")

GOAL_PX = 0.1
GAIA_MAG = (10.0, 15.0)  # bright, but not saturated
MATCH_PX = 1.5
TARGET_EXCLUSION_PX = 8.0
DETECTION_SIGMA = 8.0
DOC_PATH = Path("docs") / "alignment.md"


def detect_stars(image, mask) -> tuple[np.ndarray, np.ndarray]:
    """Windowed centroids (x, y) of the stars in an image, in pixels.

    The detection threshold uses the noise measured in the background of the image, not the
    propagated variance: in a crowded field (Barnard in D1-D3) the faint stars make the real
    background several times noisier than the variance says, and a threshold from the variance
    turns the whole image into a few huge blobs.
    """
    mask = np.ascontiguousarray(mask | ~np.isfinite(image))
    data = np.ascontiguousarray(np.where(mask, 0.0, image), dtype=np.float32)
    background = sep.Background(data, mask=mask, bw=32, bh=32)
    subtracted = np.ascontiguousarray(data - background.back(), dtype=np.float32)
    error = np.ascontiguousarray(background.rms(), dtype=np.float32)
    found = sep.extract(subtracted, DETECTION_SIGMA, err=error, mask=mask, minarea=4)
    if len(found) == 0:
        return np.array([]), np.array([])
    x, y, _ = sep.winpos(subtracted, found["x"], found["y"], np.full(len(found), 1.0), mask=mask)
    return np.asarray(x), np.asarray(y)


def match_offsets(x, y, ref_x, ref_y, target_xy=None) -> dict:
    """Median (x, y) - (ref_x, ref_y) of the nearest pairs within MATCH_PX, and how many."""
    dx, dy = [], []
    if len(x):
        for px, py in zip(np.atleast_1d(ref_x), np.atleast_1d(ref_y), strict=True):
            if target_xy is not None and np.hypot(px - target_xy[0], py - target_xy[1]) < (
                TARGET_EXCLUSION_PX
            ):
                continue
            distance = np.hypot(x - px, y - py)
            k = int(np.argmin(distance))
            if distance[k] < MATCH_PX:
                dx.append(x[k] - px)
                dy.append(y[k] - py)
    if not dx:
        return {"n_stars": 0, "dx_px": np.nan, "dy_px": np.nan}
    return {"n_stars": len(dx), "dx_px": float(np.median(dx)), "dy_px": float(np.median(dy))}


def measure_offsets(image, mask, wcs, ra, dec, target_xy=None) -> dict:
    """Median (found - catalog) position of the matched stars, in pixels, and how many matched.

    `ra`, `dec` (degrees) are the catalog positions already moved to the date of the image.
    Stars within TARGET_EXCLUSION_PX of `target_xy` are left out (the target moves).
    """
    x, y = detect_stars(image, mask)
    cat_x, cat_y = wcs.world_to_pixel(SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg))
    return match_offsets(x, y, cat_x, cat_y, target_xy)


def measure_between_seasons(target_id: str, *, root: Path | None = None) -> pd.DataFrame:
    """Offset of season b against season a for every difference, on the stars of the sums.

    What a difference needs is that both seasons sit on the same grid; this measures exactly
    that, without any catalog. Used when Gaia does not answer. Columns: detector, season
    ("a-b"), n_stars, dx_px, dy_px (season b minus season a).
    """
    from serafim.silver.differences import difference_pairs

    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    target = next(t for t in config["targets"] if t["id"] == target_id)
    sums = pq.read_table(sums_table_path(root)).to_pandas()
    mine = sums[sums.target_id == target_id]
    rows = []
    for detector, a, b in difference_pairs(zip(mine.detector, mine.season, strict=True)):
        first = load_sum(target_id, detector, a, root)
        second = load_sum(target_id, detector, b, root)
        target_xy = first["wcs"].world_to_pixel(target_position(target, first["mean_mjd"]))
        ref_x, ref_y = detect_stars(first["image"], first["mask"])
        x, y = detect_stars(second["image"], second["mask"])
        found = match_offsets(x, y, ref_x, ref_y, target_xy)
        rows.append({"detector": detector, "season": f"{a}-{b}", **found})
    return pd.DataFrame(rows, columns=["detector", "season", "n_stars", "dx_px", "dy_px"])


def gaia_sources(target: dict, size_deg: float, *, root: Path | None = None) -> pd.DataFrame:
    """Gaia DR3 around the target: the copy in config/gaia_snapshot/ if it covers the cutout,
    else ESA's archive (or its cache), else a VizieR mirror."""
    from serafim.bronze.gaia_snapshot import from_snapshot, from_vizier
    from serafim.validation import catalogs, layers

    center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
    radius = catalogs._region_radius(size_deg)
    saved = from_snapshot(center, radius, layers.GAIA_MAX_MAG, root=root)
    if saved is not None:
        return saved
    try:
        return catalogs.gaia(center, radius, max_mag=layers.GAIA_MAX_MAG)
    except catalogs.CatalogUnavailable as error:
        log.warning("target=%s: %s; trying Gaia DR3 at VizieR", target["id"], error)
        return from_vizier(center, radius, GAIA_MAG[1])[0]


def measure_alignment(target_id: str, *, root: Path | None = None, sources=None) -> pd.DataFrame:
    """One row per sum of one detector and one season: n_stars, dx_px, dy_px.

    `sources` is a catalogs.COLUMNS table (for tests); by default Gaia from gaia_sources().
    """
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    target = next(t for t in config["targets"] if t["id"] == target_id)
    if sources is None:
        size = float(target.get("cutout_size_deg") or config["settings"]["cutout_size_deg"])
        sources = gaia_sources(target, size, root=root)
    sources = sources[sources["mag"].between(*GAIA_MAG)].reset_index(drop=True)

    from serafim.validation.layers import move

    sums = pq.read_table(sums_table_path(root)).to_pandas()
    sums = sums[(sums.target_id == target_id) & (sums.detector > 0) & (sums.season > 0)]
    rows = []
    for row in sums.sort_values(["detector", "season"]).itertuples():
        arrays = load_sum(target_id, int(row.detector), int(row.season), root)
        when = Time(arrays["mean_mjd"], format="mjd", scale="utc")
        ra, dec = move(sources, when)
        target_xy = arrays["wcs"].world_to_pixel(target_position(target, arrays["mean_mjd"]))
        found = measure_offsets(arrays["image"], arrays["mask"], arrays["wcs"], ra, dec,
                                target_xy)
        rows.append({"detector": int(row.detector), "season": int(row.season), **found})
    return pd.DataFrame(rows, columns=["detector", "season", "n_stars", "dx_px", "dy_px"])


BETWEEN_SEASONS_NOTE = (
    "Gaia did not answer (ESA and the VizieR mirrors), so this is measured without a catalog: "
    "for every difference, the stars of the season b sum against the same stars in the season "
    "a sum (Season = a-b, offsets of b minus a). That is what a difference needs; the "
    "comparison with Gaia is still to do."
)


def section(target_id: str, table: pd.DataFrame, note: str | None = None) -> str:
    """The markdown section of one target in docs/alignment.md."""
    offsets = np.abs(table[["dx_px", "dy_px"]].to_numpy(dtype=float))
    worst = float(np.nanmax(offsets)) if len(table) else np.nan
    verdict = "meets" if worst < GOAL_PX else "DOES NOT meet"
    lines = [
        f"## {target_id}",
        "",
        *([note, ""] if note else []),
        f"Largest median offset: **{worst:.3f} px**, which {verdict} the goal of {GOAL_PX} px.",
        "",
        "| Detector | Season | Stars | dx (px) | dy (px) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in table.itertuples():
        lines.append(f"| {row.detector} | {row.season} | {row.n_stars} | {row.dx_px:+.3f} | "
                     f"{row.dy_px:+.3f} |")
    return "\n".join(lines) + "\n"


HEADER = f"""# Alignment of the silver sums

Written by `python -m serafim alignment`; do not edit by hand.

For each sum of one detector and one season, the stars found in the sum (sep,
{DETECTION_SIGMA:g} times the measured background rms, windowed centroids) are matched within
{MATCH_PX} px to Gaia stars with G between {GAIA_MAG[0]:g} and {GAIA_MAG[1]:g}, moved to the
mean date of the sum with
`serafim.validation.layers.move()`. dx and dy are the median of (found - Gaia) in pixels of the
target grid (6.15 arcsec), x to the west and y to the north. Stars within
{TARGET_EXCLUSION_PX:g} px of the target are left out. The goal (agreed with AI & validation) is
under {GOAL_PX} px: above that, differences between seasons fake dipoles on fixed stars.
"""


def write_section(target_id: str, table: pd.DataFrame, path: Path,
                  note: str | None = None) -> None:
    """Replace (or add) the section of this target in docs/alignment.md; keep the others."""
    sections = {}
    if path.is_file():
        for block in re.split(r"(?m)^(?=## )", path.read_text(encoding="utf-8"))[1:]:
            sections[block.split("\n", 1)[0][3:].strip()] = block.rstrip() + "\n"
    sections[target_id] = section(target_id, table, note)
    body = "\n".join(sections[name] for name in sorted(sections))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "\n" + body, encoding="utf-8")


def run(target_id: str | None = None, *, root: Path | None = None) -> float:
    """Measure the targets that have sums, rewrite their sections; return the largest offset."""
    root = root or ROOT
    sums = pq.read_table(sums_table_path(root)).to_pandas()
    wanted = [target_id] if target_id else sorted(set(sums.target_id))
    worst = 0.0
    from serafim.validation.catalogs import CatalogUnavailable

    for name in wanted:
        note = None
        try:
            table = measure_alignment(name, root=root)
        except CatalogUnavailable as error:
            log.warning("target=%s: %s; measuring between seasons instead", name, error)
            table, note = measure_between_seasons(name, root=root), BETWEEN_SEASONS_NOTE
        write_section(name, table, root / DOC_PATH, note)
        target_worst = float(np.nanmax(np.abs(table[["dx_px", "dy_px"]].to_numpy(dtype=float))))
        worst = max(worst, target_worst)
        log.info("target=%s: largest median offset %.3f px over %d sums", name, target_worst,
                 len(table))
    return worst
