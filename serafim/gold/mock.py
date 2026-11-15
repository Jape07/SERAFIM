"""Simulated gold for one region (task D1).

Lets the website work from H2, before real data exists. It follows
serafim/contract.py exactly: 10 frames with PNGs, 20 detections and 5 objects
covering the 3 statuses. Everything comes from a fixed seed, so two runs write
the same tables and PNGs (only meta.json changes: it records the run date).

The PNGs show fixed stars plus a source wherever there is a detection, so the
web can check that markers land on something visible (task W5).

    python -m serafim mock            # writes data/gold/
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

log = logging.getLogger(__name__)

SEED = 20261114
PIXEL_SCALE_ARCSEC = 6.15  # SPHEREx pixel, from the IRSA data products docs
N_STARS = 40

# 10 frames: (date UTC, wavelength in um, has a difference image).
# Gaps mix one orbit (~98 min) with weeks and months, like the real pole field.
MOCK_FRAMES = [
    ("2025-05-03T10:12:00", 0.95, False),
    ("2025-05-03T11:50:00", 0.95, True),
    ("2025-05-03T13:28:00", 1.35, True),
    ("2025-05-10T09:00:00", 1.35, True),
    ("2025-06-14T22:30:00", 2.05, False),
    ("2025-06-15T00:08:00", 2.05, True),
    ("2025-08-20T05:45:00", 3.10, True),
    ("2025-08-20T07:23:00", 3.10, True),
    ("2025-11-02T16:00:00", 4.10, False),
    ("2025-11-02T17:38:00", 4.70, True),
]

# 5 objects, 20 detections in total. Detections only happen on frames with a
# difference image. Positions are PNG pixels (see the convention in contract.py):
# the object moves in a straight line from `start` to `end` over its frames.
MOCK_OBJECTS = [
    {
        "name": "MOCK asteroid",
        "type": "asteroid",
        "status": contract.KNOWN,
        "reason": "Within 2 px of a SkyBoT object on that date",
        "catalog": "SkyBoT",
        "url": "https://ssp.imcce.fr/webservices/skybot/",
        "frames": [1, 2, 3, 5, 6, 7],
        "start": (60, 230),
        "end": (235, 70),
        "signs": [1],
        "p_real": 0.93,
        "base_flux": 0.0,
    },
    {
        "name": "MOCK variable star",
        "type": "variable star",
        "status": contract.KNOWN,
        "reason": "Within 2 px of a SIMBAD variable star (V*)",
        "catalog": "SIMBAD",
        "url": "https://simbad.cds.unistra.fr/simbad/",
        "frames": [2, 3, 6, 7, 9],
        "start": (150, 140),
        "end": (150, 140),
        "signs": [1, -1],
        "p_real": None,  # Real gold has no p_real until task D10: the web must handle it.
        "base_flux": 40.0,
    },
    {
        "name": "MOCK candidate",
        "type": "unknown",
        "status": contract.CANDIDATE,
        "reason": "No match in SkyBoT, SIMBAD or Gaia",
        "catalog": None,
        "url": None,
        "frames": [1, 2, 6, 7],
        "start": (205, 205),
        "end": (205, 205),
        "signs": [1],
        "p_real": 0.71,
        "base_flux": 0.0,
    },
    {
        "name": "MOCK artifact (edge)",
        "type": "artifact",
        "status": contract.DISCARDED,
        "reason": "Touches flagged pixels or the edge",
        "catalog": None,
        "url": None,
        "frames": [3, 5, 7, 9],
        "start": (6, 120),
        "end": (6, 120),
        "signs": [1, -1],
        "p_real": 0.08,
        "base_flux": 0.0,
    },
    {
        "name": "MOCK artifact (single)",
        "type": "artifact",
        "status": contract.DISCARDED,
        "reason": "Seen in a single exposure",
        "catalog": None,
        "url": None,
        "frames": [9],
        "start": (95, 255),
        "end": (95, 255),
        "signs": [1],
        "p_real": 0.35,
        "base_flux": 0.0,
    },
]


@dataclass(frozen=True)
class Png:
    """An image to save as PNG: values plus how to color them."""

    data: np.ndarray
    cmap: str
    vmin: float
    vmax: float


@dataclass(frozen=True)
class MockGold:
    tables: dict[str, pd.DataFrame]  # keyed like contract.GOLD_TABLES
    pngs: dict[str, Png]  # keyed by path relative to gold/


# --- Pure functions: no disk, no network ------------------------------------
def region_wcs(ra: float, dec: float, size_deg: float) -> WCS:
    """TAN grid centered on the region, one SPHEREx pixel per pixel, north up."""
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


def png_to_sky(wcs: WCS, x_px: np.ndarray, y_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PNG pixel (origin top-left) -> (ra, dec) in degrees."""
    n_rows = wcs.pixel_shape[1]
    ra, dec = wcs.wcs_pix2world(x_px, n_rows - 1 - np.asarray(y_px), 0)
    return np.mod(ra, 360.0), dec


def _resolving_power(wavelength_um: float) -> float:
    """Approximate SPHEREx spectral resolution per detector band."""
    if wavelength_um < 2.42:
        return 41.0
    if wavelength_um < 3.83:
        return 35.0
    if wavelength_um < 4.42:
        return 112.0
    return 128.0


def build_mock(region: dict, size_deg: float) -> MockGold:
    """All simulated gold for one region (a target, see `target_as_region`), in memory."""
    rng = np.random.default_rng(SEED)
    region_id = region["id"]
    wcs = region_wcs(region["ra"], region["dec"], size_deg)

    regions = pd.DataFrame(
        [
            {
                "region_id": region_id,
                "name": region["name"],
                "ra": float(region["ra"]),
                "dec": float(region["dec"]),
                "size_deg": float(size_deg),
                "wcs_json": wcs_to_json(wcs),
            }
        ]
    )
    frames = _frames(region_id)
    detections, objects = _detections_and_objects(region_id, frames, wcs, rng)
    pngs = _render(frames, detections, wcs.pixel_shape[0], rng)
    return MockGold(
        tables={
            "regions.parquet": regions,
            "frames.parquet": frames,
            "detections.parquet": detections,
            "objects.parquet": objects,
        },
        pngs=pngs,
    )


def _frames(region_id: str) -> pd.DataFrame:
    rows = []
    for i, (date, wavelength, has_diff) in enumerate(MOCK_FRAMES):
        frame_id = f"{region_id}_f{i:03d}"
        rows.append(
            {
                "frame_id": frame_id,
                "region_id": region_id,
                "obs_id": f"MOCK_{date[:10].replace('-', '')}_{i:03d}",
                "date_utc": pd.Timestamp(date, tz="UTC"),
                "wavelength_um": wavelength,
                "bandwidth_um": round(wavelength / _resolving_power(wavelength), 4),
                "png": f"png/{frame_id}.png",
                "png_diff": f"png/{frame_id}_diff.png" if has_diff else None,
            }
        )
    return pd.DataFrame(rows)


def _detections_and_objects(
    region_id: str, frames: pd.DataFrame, wcs: WCS, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for o, spec in enumerate(MOCK_OBJECTS):
        k = len(spec["frames"])
        xs = np.linspace(spec["start"][0], spec["end"][0], k) + rng.normal(0, 0.3, k)
        ys = np.linspace(spec["start"][1], spec["end"][1], k) + rng.normal(0, 0.3, k)
        for j, frame_index in enumerate(spec["frames"]):
            frame = frames.iloc[frame_index]
            assert pd.notna(frame.png_diff), "detections only happen on difference images"
            rows.append(
                {
                    "frame_id": frame.frame_id,
                    "object_id": f"{region_id}_o{o:03d}",
                    "x_px": round(float(xs[j]), 2),
                    "y_px": round(float(ys[j]), 2),
                    "snr": round(float(rng.uniform(5, 30)), 1),
                    "sign": spec["signs"][j % len(spec["signs"])],
                    "p_real": spec["p_real"],
                    "base_flux": spec["base_flux"],
                }
            )

    detections = pd.DataFrame(rows).sort_values(["frame_id", "object_id"], ignore_index=True)
    detections.insert(0, "detection_id", [f"{region_id}_d{i:03d}" for i in detections.index])
    ra, dec = png_to_sky(wcs, detections.x_px.to_numpy(), detections.y_px.to_numpy())
    detections.insert(3, "ra", np.round(ra, 6))
    detections.insert(4, "dec", np.round(dec, 6))
    detections["p_real"] = detections.p_real.astype("float64")

    dates = detections.frame_id.map(frames.set_index("frame_id").date_utc)
    per_object = detections.assign(date=dates).groupby("object_id").date
    objects = pd.DataFrame(
        [
            {
                "object_id": f"{region_id}_o{o:03d}",
                "region_id": region_id,
                "name": spec["name"],
                "type": spec["type"],
                "status": spec["status"],
                "reason": spec["reason"],
                "catalog": spec["catalog"],
                "url": spec["url"],
            }
            for o, spec in enumerate(MOCK_OBJECTS)
        ]
    )
    objects["n_detections"] = objects.object_id.map(per_object.count())
    objects["first_date_utc"] = objects.object_id.map(per_object.min())
    objects["last_date_utc"] = objects.object_id.map(per_object.max())
    return detections, objects


def _render(
    frames: pd.DataFrame, detections: pd.DataFrame, n: int, rng: np.random.Generator
) -> dict[str, Png]:
    """Frame PNGs (asinh stretch, magma) and difference PNGs (diverging scale)."""
    yy, xx = np.mgrid[0:n, 0:n]

    def blob(x: float, y: float, flux: float, sigma: float = 1.4) -> np.ndarray:
        return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

    background, noise, change = 10.0, 1.0, 25.0
    static = np.full((n, n), background)
    for (x, y), flux in zip(
        rng.uniform(5, n - 5, (N_STARS, 2)), rng.lognormal(2.5, 0.8, N_STARS), strict=True
    ):
        static += blob(x, y, flux)
    for obj in detections[detections.base_flux > 0].drop_duplicates("object_id").itertuples():
        static += blob(obj.x_px, obj.y_px, obj.base_flux)  # e.g. the variable star itself

    frame_images, diff_images = {}, {}
    for frame in frames.itertuples():
        image = static + rng.normal(0, noise, (n, n))
        diff = rng.normal(0, noise * np.sqrt(2), (n, n))
        for det in detections[detections.frame_id == frame.frame_id].itertuples():
            image += blob(det.x_px, det.y_px, det.sign * change)
            diff += blob(det.x_px, det.y_px, det.sign * change)
        frame_images[frame.png] = np.arcsinh((image - background) / noise)
        if pd.notna(frame.png_diff):
            diff_images[frame.png_diff] = diff

    # Fixed limits for the whole region, so the background looks the same in every PNG.
    vmin, vmax = (float(v) for v in np.percentile(np.stack(list(frame_images.values())), [1, 99.9]))
    diff_stack = np.abs(np.stack(list(diff_images.values())))
    limit = float(max(5 * noise * np.sqrt(2), np.percentile(diff_stack, 99.9)))
    pngs = {path: Png(data, "magma", vmin, vmax) for path, data in frame_images.items()}
    pngs |= {path: Png(data, "RdBu_r", -limit, limit) for path, data in diff_images.items()}
    return pngs


# --- Writing to disk ---------------------------------------------------------
def target_as_region(target: dict) -> dict:
    """A target from config/targets.yaml in the shape the current contract calls a region.

    Until the new gold contract (task D2), each target is written as one region,
    centered on its position today (`ra_now`, `dec_now`).
    """
    return {
        "id": target["id"],
        "name": target["name"],
        "ra": float(target["ra_now"]),
        "dec": float(target["dec_now"]),
    }


def write_mock_gold(gold_dir: Path, config: dict) -> MockGold:
    """Build simulated gold for the first target in the config and write it to gold_dir."""
    region = target_as_region(config["targets"][0])
    size_deg = config["settings"]["cutout_size_deg"]
    gold = build_mock(region, size_deg)

    gold_dir = Path(gold_dir)
    for path, png in gold.pngs.items():
        target = gold_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(target, png.data, cmap=png.cmap, vmin=png.vmin, vmax=png.vmax)
    frames = gold.tables["frames.parquet"]
    write_gold(gold_dir, gold.tables, build_meta(n_frames=len(frames)))
    log.info(
        "region=%s simulated gold in %s: %d frames, %d detections, %d objects, %d PNG",
        region["id"],
        gold_dir,
        len(frames),
        len(gold.tables["detections.parquet"]),
        len(gold.tables["objects.parquet"]),
        len(gold.pngs),
    )
    return gold


def write_all_targets(gold_dir: Path, config: dict) -> None:
    """Simulated gold for every target in the config, for the published site.

    ``write_mock_gold`` stays one target: that is what the contract tests check.
    """
    size_deg = config["settings"]["cutout_size_deg"]
    pieces = [build_mock(target_as_region(t), size_deg) for t in config["targets"]]
    tables = {
        name: pd.concat([piece.tables[name] for piece in pieces], ignore_index=True)
        for name in contract.GOLD_TABLES
    }
    gold_dir = Path(gold_dir)
    for piece in pieces:
        for path, png in piece.pngs.items():
            target = gold_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(target, png.data, cmap=png.cmap, vmin=png.vmin, vmax=png.vmax)
    write_gold(gold_dir, tables, build_meta(n_frames=len(tables["frames.parquet"])))
    log.info(
        "simulated gold for %d targets in %s (%d frames)",
        len(pieces),
        gold_dir,
        len(tables["frames.parquet"]),
    )
