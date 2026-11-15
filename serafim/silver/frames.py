"""Silver 1: every bronze frame cleaned, masked and reprojected to its target's grid (task D4).

    python -m serafim silver [--target barnard] [--force]

For each frame in `data/bronze/manifest.parquet`:

1. image = IMAGE - ZODI (MJy/sr; IRSA's IMAGE still has the zodiacal light).
2. The mask comes from the FLAGS bits, read by name from the `MSKN*` keys of the header (never
   from a fixed table and never from the NaN of IMAGE: most flagged pixels are finite).
   Masked: MASKED_FLAGS. SOURCE is never masked; BLOOM and HALO are not masked either (they are
   light of the star) but they are reported in `flags`.
3. Bad pixels are set to NaN and IMAGE, VARIANCE and the saturation bits are reprojected to the
   grid of the target (serafim/silver/grid.py) with `reproject_exact`, which shares the flux by
   overlapping area (it is not a bilinear interpolation). An output pixel touched by a bad input
   pixel therefore becomes NaN and is masked: no bad value leaks into a neighbor.
4. Per frame, `data/silver/frames.parquet` gets the columns of contract.FRAMES: detector,
   wavelength and bandwidth at the target (from the spectral WCS, key W), mean date (DATE-AVG,
   UTC), season (from the manifest), the flag names within `settings.validation.match_radius_px`
   of the target, and `saturated` (OVERFLOW or NONLINEAR there).

`.npz` of a frame: data/silver/<target>/frames/<frame_id>.npz, frame_id = contract.frame_id()

    image      float32  MJy/sr        IMAGE - ZODI on the grid; NaN where nothing valid arrived
    variance   float32  (MJy/sr)^2    VARIANCE on the grid. Approximate: reprojecting correlates
                                      neighboring pixels, so it is not the variance of `image`
                                      pixel by pixel. AI scales errors with the measured sky noise.
    mask       bool                   True = do not use: a masked flag, NaN, or footprint < 0.5
    saturated  bool                   True = OVERFLOW or NONLINEAR reached this pixel
    footprint  float32  0 to 1        fraction of the pixel covered by the cutout
    wcs_json   str                    the grid, the same text as targets.parquet `wcs_json`

Arrays are array[y, x], indices from 0, row 0 = south, and
WCS(json.loads(wcs_json)).pixel_to_world(x, y) is the position of pixel (x, y).
Read them with serafim.silver.load_frame(), or with numpy.load(path) anywhere.

A frame whose `.npz` exists is not reprojected again (use --force); its row is always rebuilt,
because its season can change when new frames arrive.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_exact

from serafim import contract
from serafim.bronze.download import read_manifest
from serafim.config import ROOT, load_config
from serafim.gold.export import to_contract_table
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import frame_path, frames_table_path
from serafim.validation.targets import SATURATION_FLAGS, propagate

log = logging.getLogger("serafim.silver")

MASKED_FLAGS = ("OVERFLOW", "NONLINEAR", "NONFUNC", "MISSING_DATA", "HOT", "COLD")
NOT_REPORTED = ("SOURCE",)  # almost every star pixel has it
MIN_FOOTPRINT = 0.5


# --- Pure functions: an open FITS in, arrays and a row out ----------------------------------
def flag_bits(header: fits.Header) -> dict[str, int]:
    """Flag name -> bit number, from the `MSKN<bit>` keys of the FLAGS header."""
    return {str(header[key]).strip(): int(key[4:]) for key in header if key.startswith("MSKN")}


def has_flags(flags: np.ndarray, names, bits: dict[str, int]) -> np.ndarray:
    """True where any of `names` is set (names missing from the header are ignored)."""
    found = np.zeros(flags.shape, dtype=bool)
    for name in names:
        if name in bits:
            found |= ((flags >> bits[name]) & 1).astype(bool)
    return found


def flags_near(flags: np.ndarray, x: float, y: float, radius: float, bits: dict[str, int]) -> list:
    """Sorted flag names set within `radius` pixels of (x, y), SOURCE left out."""
    rows, cols = np.mgrid[0 : flags.shape[0], 0 : flags.shape[1]]
    near = (cols - x) ** 2 + (rows - y) ** 2 <= radius**2
    if not near.any():
        return []
    value = int(np.bitwise_or.reduce(flags[near].ravel()))
    return sorted(
        name for name, bit in bits.items() if (value >> bit) & 1 and name not in NOT_REPORTED
    )


def spectral_wcs(hdul: fits.HDUList) -> WCS:
    """The WAVE-TAB WCS (key W). The SIP keys belong to the sky WCS, so they are left out."""
    header = hdul["IMAGE"].header.copy()
    for key in [k for k in header if "_" in k and k.split("_")[0] in ("A", "B", "AP", "BP")]:
        del header[key]
    spectral = WCS(header, fobj=hdul, key="W")
    spectral.sip = None
    return spectral


def target_position(target: dict, mjd: float) -> SkyCoord:
    """Where the target is at `mjd`: moved with its proper motion, or fixed (added by coords)."""
    if all(key in target for key in ("ra", "dec", "pmra_mas_yr", "pmdec_mas_yr")):
        moved = propagate(target, Time(mjd, format="mjd", scale="utc"))
        return SkyCoord(moved.ra, moved.dec)
    return SkyCoord(float(target["ra_now"]), float(target["dec_now"]), unit="deg")


def frame_arrays(hdul: fits.HDUList, grid: WCS) -> dict[str, np.ndarray]:
    """The arrays of the `.npz` (without wcs_json) for one open bronze FITS."""
    sky = WCS(hdul["IMAGE"].header)
    flags = hdul["FLAGS"].data.astype(np.int64)
    bits = flag_bits(hdul["FLAGS"].header)
    image = hdul["IMAGE"].data.astype(np.float64) - hdul["ZODI"].data.astype(np.float64)
    variance = hdul["VARIANCE"].data.astype(np.float64)
    bad = has_flags(flags, MASKED_FLAGS, bits) | ~np.isfinite(image) | ~np.isfinite(variance)
    image[bad] = np.nan
    variance[bad] = np.nan
    saturated = has_flags(flags, SATURATION_FLAGS, bits).astype(np.float64)

    shape = (grid.pixel_shape[1], grid.pixel_shape[0])  # (rows, columns)
    image_out, footprint = reproject_exact((image, sky), grid, shape_out=shape)
    variance_out, _ = reproject_exact((variance, sky), grid, shape_out=shape)
    saturated_out, _ = reproject_exact((saturated, sky), grid, shape_out=shape)
    footprint = np.nan_to_num(footprint, nan=0.0)
    mask = ~np.isfinite(image_out) | ~np.isfinite(variance_out) | (footprint < MIN_FOOTPRINT)
    return {
        "image": image_out.astype(np.float32),
        "variance": variance_out.astype(np.float32),
        "mask": mask,
        "saturated": np.nan_to_num(saturated_out, nan=0.0) > 0,
        "footprint": footprint.astype(np.float32),
    }


def frame_row(hdul: fits.HDUList, target: dict, frame: dict, radius_px: float) -> dict:
    """The frames.parquet row of one open bronze FITS. `frame` is its manifest row."""
    header = hdul["IMAGE"].header
    detector = int(header["DETECTOR"])
    if detector != int(frame["detector"]):
        raise ValueError(f"{frame['obs_id']}: DETECTOR {detector} but manifest says "
                         f"{frame['detector']}")
    mjd = float(header["MJD-AVG"])
    x, y = WCS(header).world_to_pixel(target_position(target, mjd))
    wavelength, bandwidth = spectral_wcs(hdul).pixel_to_world_values(x, y)
    flags = hdul["FLAGS"].data.astype(np.int64)
    names = flags_near(flags, float(x), float(y), radius_px, flag_bits(hdul["FLAGS"].header))
    return {
        "frame_id": contract.frame_id(target["id"], frame["obs_id"]),
        "target_id": target["id"],
        "obs_id": frame["obs_id"],
        "collection": frame["collection"],
        "date_utc": pd.Timestamp(header["DATE-AVG"], tz="UTC"),
        "season": int(frame["season"]),
        "detector": detector,
        "wavelength_um": float(wavelength),
        "bandwidth_um": float(bandwidth),
        "flags": ",".join(names),
        "saturated": any(name in SATURATION_FLAGS for name in names),
    }


# --- Disk -------------------------------------------------------------------------------------
def build_frames(target_id: str | None = None, *, root: Path | None = None,
                 force: bool = False) -> int:
    """Silver 1 for the targets in bronze. Return how many `.npz` were written."""
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    settings = config["settings"]
    radius = float((settings.get("validation") or {}).get("match_radius_px", 2))
    manifest = pd.DataFrame(read_manifest(root))
    if manifest.empty:
        raise FileNotFoundError("data/bronze/manifest.parquet is empty: run the bronze step first")
    targets = {t["id"]: t for t in config["targets"]}
    wanted = [target_id] if target_id else [t for t in targets if t in set(manifest.target_id)]
    unknown = [t for t in wanted if t not in targets]
    if unknown:
        raise ValueError(f"Unknown target {unknown[0]}. Config has: {', '.join(targets)}.")

    written = 0
    rows: list[dict] = []
    for name in wanted:
        target = targets[name]
        size_deg = float(target.get("cutout_size_deg") or settings["cutout_size_deg"])
        grid = target_wcs(float(target["ra_now"]), float(target["dec_now"]), size_deg)
        wcs_text = np.array(wcs_to_json(grid))
        mine = manifest[manifest.target_id == name].sort_values(["t_mid_mjd", "obs_id"])
        new = 0
        for frame in mine.to_dict("records"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                with fits.open(root / frame["path"]) as hdul:
                    row = frame_row(hdul, target, frame, radius)
                    path = frame_path(name, row["frame_id"], root)
                    if force or not path.is_file():
                        _save_npz(path, {**frame_arrays(hdul, grid), "wcs_json": wcs_text})
                        new += 1
            rows.append(row)
        written += new
        log.info("target=%s: %d frames in silver, %d reprojected now", name, len(mine), new)

    _write_frames_table(pd.DataFrame(rows), wanted, root)
    return written


def _save_npz(path: Path, arrays: dict) -> None:
    """Write the `.npz` through a temporary file, so a cut run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as file:
        np.savez_compressed(file, **arrays)
    partial.replace(path)


def _write_frames_table(rows: pd.DataFrame, targets: list[str], root: Path) -> None:
    """Replace the rows of these targets in data/silver/frames.parquet; keep the others."""
    path = frames_table_path(root)
    if path.is_file():
        old = pq.read_table(path).to_pandas()
        rows = pd.concat([old[~old.target_id.isin(targets)], rows], ignore_index=True)
    rows = rows.sort_values(["target_id", "date_utc", "frame_id"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(rows, contract.FRAMES), path)
