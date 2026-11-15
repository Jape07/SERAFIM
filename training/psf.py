"""The PSF of each detector, measured on the real silver sums (task A9, stage 2, version 2).

    from training.psf import build_psfs, Psf
    psfs = build_psfs()          # {detector: Psf}, also saved to data/dataset/psf.npz

The injected sources of the CNN's dataset use these PSFs instead of a Gaussian: they include the
real core, the resampling of silver and the wings. For every detector, isolated Gaia stars that do
not move (G 10 to 14, no other Gaia star brighter than G 16 within ISOLATION_PX, no masked or
saturated pixel nearby) are cut from every sum of the 5 targets, background-subtracted, normalized
to unit flux, and their pixels are binned by their offset from the star's catalog position (the
sums are aligned with Gaia to 0.03 px) on a grid OVERSAMPLE times finer. The median of each bin is
the PSF; beyond RADIUS_PX it is taken as 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from serafim.validation import targets

log = logging.getLogger(__name__)

PSF_PATH = targets.ROOT / "data" / "dataset" / "psf.npz"
RADIUS_PX = 7
OVERSAMPLE = 4
MAG_RANGE = (10.0, 14.0)
ISOLATION_PX = 5.0
NEIGHBOUR_MAX_MAG = 16.0
FLUX_RADIUS_PX = 5.0
KEEP_PX = 3.0  # the measured PSF is kept as it is out to here
TAPER_PX = 5.0  # and goes smoothly to 0 here
MIN_STARS = 20


@dataclass
class Psf:
    """An oversampled PSF: grid[i, j] is the value of the pixel whose center is at offset
    ((j - c) / OVERSAMPLE, (i - c) / OVERSAMPLE) px from a source of flux 1, c = (size - 1) / 2."""

    grid: np.ndarray
    stars: int

    def at(self, xs: np.ndarray, ys: np.ndarray, x0: float, y0: float) -> np.ndarray:
        """The PSF of a source of flux 1 at (x0, y0), at pixel centers (xs, ys)."""
        center = (self.grid.shape[0] - 1) / 2
        rows = (np.asarray(ys) - y0) * OVERSAMPLE + center
        cols = (np.asarray(xs) - x0) * OVERSAMPLE + center
        return ndimage.map_coordinates(self.grid, [rows, cols], order=1, mode="constant", cval=0)

    def image(self, shape: tuple[int, int], x0: float, y0: float) -> np.ndarray:
        rows, cols = np.indices(shape)
        return self.at(cols, rows, x0, y0)


def stars_of(sum_arrays: dict, layers: pd.DataFrame, wcs) -> list[tuple[float, float]]:
    """Pixel positions of the isolated, unsaturated Gaia stars of one sum."""
    gaia = layers[(layers["catalog"] == "gaia") & layers["mag"].notna()]
    if gaia.empty:
        return []
    xs, ys = wcs.world_to_pixel_values(gaia["ra"].to_numpy(), gaia["dec"].to_numpy())
    mags = gaia["mag"].to_numpy()
    image = np.asarray(sum_arrays["image"], float)
    bad = np.asarray(sum_arrays["mask"], bool) | ~np.isfinite(image)
    bad |= np.asarray(sum_arrays.get("n_saturated", 0)) > 0
    blocked = ndimage.binary_dilation(bad, iterations=RADIUS_PX + 1)
    ny, nx = image.shape
    out = []
    for x, y, mag in zip(xs, ys, mags, strict=True):
        if not MAG_RANGE[0] <= mag <= MAG_RANGE[1]:
            continue
        if not (
            RADIUS_PX + 2 <= x < nx - RADIUS_PX - 2 and RADIUS_PX + 2 <= y < ny - RADIUS_PX - 2
        ):
            continue
        if blocked[int(round(y)), int(round(x))]:
            continue
        near = (np.hypot(xs - x, ys - y) < ISOLATION_PX) & (mags < NEIGHBOUR_MAX_MAG)
        if near.sum() > 1:  # itself
            continue
        out.append((float(x), float(y)))
    return out


def measure(cuts: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    """The oversampled PSF from (image cut, star x, star y in the cut) triplets."""
    size = 2 * RADIUS_PX * OVERSAMPLE + 1
    center = (size - 1) / 2
    bins: list[list[float]] = [[] for _ in range(size * size)]
    for cut, x0, y0 in cuts:
        rows, cols = np.indices(cut.shape)
        distance = np.hypot(cols - x0, rows - y0)
        ring = (distance > RADIUS_PX - 1) & (distance <= RADIUS_PX + 1)
        values = cut - np.median(cut[ring])
        flux = values[distance <= FLUX_RADIUS_PX].sum()
        if not flux > 0:
            continue
        values = values / flux
        i = np.round((rows - y0) * OVERSAMPLE + center).astype(int)
        j = np.round((cols - x0) * OVERSAMPLE + center).astype(int)
        inside = (i >= 0) & (i < size) & (j >= 0) & (j < size)
        for ii, jj, v in zip(i[inside], j[inside], values[inside], strict=True):
            bins[ii * size + jj].append(v)
    grid = np.array([np.median(b) if b else np.nan for b in bins]).reshape(size, size)
    filled = np.where(np.isfinite(grid), grid, 0.0)
    weight = ndimage.uniform_filter(np.isfinite(grid).astype(float), 3)
    smooth = ndimage.uniform_filter(filled, 3) / np.maximum(weight, 1e-9)
    grid = np.where(np.isfinite(grid), grid, smooth)
    rows, cols = np.indices(grid.shape)
    radius = np.hypot(rows - center, cols - center) / OVERSAMPLE
    # beyond KEEP_PX neighbours and background errors dominate the median: taper to 0 at TAPER_PX
    taper = np.clip((TAPER_PX - radius) / (TAPER_PX - KEEP_PX), 0, 1)
    grid = np.clip(grid, 0, None) * np.sin(taper * np.pi / 2) ** 2
    # grid holds the pixel value of a unit-flux source at each offset: the pixels of one source
    # sample it every OVERSAMPLE bins, so they add up to grid.sum() / OVERSAMPLE**2 = 1
    return grid / (grid.sum() / OVERSAMPLE**2)


def build_psfs(root: Path | None = None, gold_dir: Path | None = None) -> dict[int, Psf]:
    """{detector: Psf} from every season sum (season >= 1) of every target in silver and gold."""
    from serafim.silver.store import load_sum
    from serafim.validation import dipoles, verification

    tables, _ = verification.read_gold(gold_dir or dipoles.GOLD_DIR)
    cuts: dict[int, list] = {}
    for target_id in sorted(tables["targets"]["target_id"]):
        layers = tables["catalogs"][tables["catalogs"]["target_id"] == target_id]
        sums = tables["sums"][tables["sums"]["target_id"] == target_id]
        sums = sums[(sums["detector"] >= 1) & (sums["season"] >= 1)]
        for cell in sums.itertuples():
            arrays = load_sum(target_id, int(cell.detector), int(cell.season), root)
            here = layers[layers["season"] == cell.season]
            image = np.asarray(arrays["image"], float)
            for x, y in stars_of(arrays, here, arrays["wcs"]):
                ix, iy = int(round(x)), int(round(y))
                r = RADIUS_PX + 1
                cut = image[iy - r : iy + r + 1, ix - r : ix + r + 1]
                cuts.setdefault(int(cell.detector), []).append((cut, x - (ix - r), y - (iy - r)))
    psfs = {}
    for detector, found in sorted(cuts.items()):
        if len(found) < MIN_STARS:
            log.warning("detector %d: only %d stars, no PSF", detector, len(found))
            continue
        psfs[detector] = Psf(measure(found), len(found))
        log.info("detector %d: PSF from %d star cuts", detector, len(found))
    return psfs


def save(psfs: dict[int, Psf], path: Path = PSF_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"d{d}": p.grid for d, p in psfs.items()},
                        stars=np.array([[d, p.stars] for d, p in psfs.items()]))  # fmt: skip


def load(path: Path = PSF_PATH) -> dict[int, Psf]:
    data = np.load(path)
    stars = {int(d): int(n) for d, n in data["stars"]}
    return {d: Psf(data[f"d{d}"], stars[d]) for d in stars}
