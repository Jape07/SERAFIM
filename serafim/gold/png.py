"""Gold PNGs for people (task D7): every sum, difference and color image of a target.

    python -m serafim png [--target barnard]          # writes data/gold/png/<target>/

Names come from serafim/contract.py and never change without telling web:

    png/<target>/sum_d<detector>_s<season>.png      every sum (detector 0 = white image)
    png/<target>/diff_d<detector>_s<a>-s<b>.png     every difference (season b minus a)
    png/<target>/color_s<season>.png                seasons with sums of detectors 1, 3 and 6

Every PNG has the size of the target grid and is drawn north up with the origin at the top-left
(silver arrays have row 0 = south, so they are flipped here). The PNGs are only for people:
measurements and AI read the silver `.npz`, never these files. Nothing in silver changes.

How values become colors:
- Sums: each image minus its own background level (sep), divided by the typical background
  noise of its detector, through asinh. The limits are fixed per target and detector, so the
  seasons of one detector share the same background in the video. `magma`.
- Saturated centers: pixels with the paint rule of silver/sums.py (every frame saturated, or
  n_saturated > 0 and n_saturated >= n_valid) get the brightest color, so a saturated star does
  not look like a hole. Other pixels without data (outside the cutout) are NEUTRAL gray.
- Differences: asinh of difference / its background rms, symmetric limits per target and
  detector, `RdBu_r` (red = brighter in season b, blue = brighter in season a). A pixel masked or
  saturated in either season is drawn as zero (the white middle of the scale).
- Color: detector 6 red, 3 green and 1 blue, each scaled like its sum; saturated pixels are white
  and a pixel without data in any of the three is gray.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sep
from matplotlib import colormaps
from matplotlib import image as mpimg

from serafim import contract
from serafim.config import ROOT
from serafim.silver.store import (
    differences_table_path,
    load_difference,
    load_sum,
    sums_table_path,
)
from serafim.silver.sums import painted_saturated

log = logging.getLogger("serafim.gold")

SUM_CMAP = "magma"
DIFFERENCE_CMAP = "RdBu_r"
NEUTRAL = np.array([128, 128, 128, 255], dtype=np.uint8)
SUM_FLOOR = -1.0  # asinh units (about 1 sigma below the background): darkest color
TOP_PERCENTILE = 99.9
MIN_DIFFERENCE_LIMIT = 5.0  # asinh units: a noise-only difference stays pale


@dataclass(frozen=True)
class Stretch:
    """How one detector of one target maps values to [0, 1]."""

    noise: float
    low: float
    high: float


# --- Pure functions ---------------------------------------------------------------------------
def background_level(image: np.ndarray, mask: np.ndarray) -> float:
    data = np.ascontiguousarray(np.where(mask, 0.0, image), dtype=np.float32)
    return float(sep.Background(data, mask=np.ascontiguousarray(mask), bw=32, bh=32).globalback)


def background_rms(image: np.ndarray, mask: np.ndarray) -> float:
    data = np.ascontiguousarray(np.where(mask, 0.0, image), dtype=np.float32)
    return float(sep.Background(data, mask=np.ascontiguousarray(mask), bw=32, bh=32).globalrms)


def asinh_sigma(image: np.ndarray, mask: np.ndarray, noise: float) -> np.ndarray:
    """Image minus its background, in units of `noise`, through asinh."""
    return np.arcsinh((image - background_level(image, mask)) / noise)


def sum_stretch(sums: list[dict], noise: float) -> Stretch:
    """Fixed limits for the sums of one detector (every season), from their valid pixels."""
    values = []
    for s in sums:
        usable = ~s["mask"] & ~painted_saturated(s) & np.isfinite(s["image"])
        values.append(asinh_sigma(s["image"], s["mask"], noise)[usable])
    high = float(np.percentile(np.concatenate(values), TOP_PERCENTILE))
    return Stretch(noise=noise, low=SUM_FLOOR, high=max(high, SUM_FLOOR + 1.0))


def difference_stretch(differences: list[dict]) -> Stretch:
    """Symmetric limits for the differences of one detector, in asinh units of their rms."""
    noise = float(np.median([background_rms(d["difference"], _hidden(d)) for d in differences]))
    noise = noise if noise > 0 else 1.0
    values = [np.abs(np.arcsinh(d["difference"] / noise))[~_hidden(d)] for d in differences]
    high = float(np.percentile(np.concatenate(values), TOP_PERCENTILE))
    high = max(high, MIN_DIFFERENCE_LIMIT)
    return Stretch(noise=noise, low=-high, high=high)


def scaled_sum(s: dict, stretch: Stretch) -> np.ndarray:
    """A sum in [0, 1] (NaN where there is no data), saturated pixels at 1."""
    values = (asinh_sigma(s["image"], s["mask"], stretch.noise) - stretch.low) / (
        stretch.high - stretch.low
    )
    values = np.clip(values, 0.0, 1.0)
    values[~np.isfinite(s["image"])] = np.nan
    values[painted_saturated(s)] = 1.0
    return values


def sum_rgba(s: dict, stretch: Stretch) -> np.ndarray:
    """RGBA uint8, north up."""
    values = scaled_sum(s, stretch)
    rgba = colormaps[SUM_CMAP](np.nan_to_num(values), bytes=True)
    rgba[np.isnan(values)] = NEUTRAL
    return rgba[::-1]


def difference_rgba(d: dict, stretch: Stretch) -> np.ndarray:
    """RGBA uint8, north up; masked or saturated pixels at the middle of the scale."""
    values = np.arcsinh(d["difference"] / stretch.noise) / stretch.high
    values = 0.5 + 0.5 * np.clip(values, -1.0, 1.0)
    values[_hidden(d) | ~np.isfinite(values)] = 0.5
    return colormaps[DIFFERENCE_CMAP](values, bytes=True)[::-1]


def color_rgba(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    """RGBA uint8 from three sums already scaled to [0, 1] (NaN where there is no data)."""
    rgb = np.dstack([red, green, blue])
    missing = np.isnan(rgb).any(axis=2)
    rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = np.round(np.nan_to_num(rgb) * 255)
    rgba[..., 3] = 255
    rgba[missing] = NEUTRAL
    return rgba[::-1]


def _hidden(d: dict) -> np.ndarray:
    return d["mask"] | d["saturated"] | ~np.isfinite(d["difference"])


# --- Disk -------------------------------------------------------------------------------------
def build_pngs(target_id: str | None = None, *, root: Path | None = None,
               gold_dir: Path | None = None) -> int:
    """PNGs of every sum, difference and color image in silver. Return how many were written."""
    root = root or ROOT
    gold_dir = gold_dir or root / "data" / "gold"
    if not sums_table_path(root).is_file():
        raise FileNotFoundError("data/silver/sums.parquet is missing: run the silver step first")
    sums = pq.read_table(sums_table_path(root)).to_pandas()
    path = differences_table_path(root)
    differences = pq.read_table(path).to_pandas() if path.is_file() else sums.iloc[0:0]
    wanted = [target_id] if target_id else sorted(set(sums.target_id))

    written = 0
    for name in wanted:
        mine = sums[sums.target_id == name]
        made: set[str] = set()
        scaled: dict[tuple[int, int], np.ndarray] = {}
        for detector, group in mine.groupby("detector"):
            seasons = group[group.season > 0] if (group.season > 0).any() else group
            noise = float(seasons.background_noise.median())
            loaded = {int(r.season): load_sum(name, int(detector), int(r.season), root)
                      for r in group.itertuples()}
            stretch = sum_stretch(list(loaded.values()), noise)
            for season, s in loaded.items():
                relative = contract.sum_png(name, int(detector), season)
                _save(gold_dir / relative, sum_rgba(s, stretch))
                made.add(relative)
                scaled[int(detector), season] = scaled_sum(s, stretch)

        mine_differences = differences[differences.target_id == name]
        for detector, group in mine_differences.groupby("detector"):
            loaded = {(int(r.season_a), int(r.season_b)):
                      load_difference(name, int(detector), int(r.season_a), int(r.season_b), root)
                      for r in group.itertuples()}
            stretch = difference_stretch(list(loaded.values()))
            for (a, b), d in loaded.items():
                relative = contract.difference_png(name, int(detector), a, b)
                _save(gold_dir / relative, difference_rgba(d, stretch))
                made.add(relative)

        for season in sorted({s for _, s in scaled if s >= 1}):
            if all((d, season) in scaled for d in contract.COLOR_DETECTORS):
                blue, green, red = (scaled[d, season] for d in contract.COLOR_DETECTORS)
                relative = contract.color_png(name, season)
                _save(gold_dir / relative, color_rgba(blue, green, red))
                made.add(relative)

        # A PNG of a cell that no longer exists would show the site something that is not there.
        for stale in (gold_dir / "png" / name).glob("*.png"):
            if f"png/{name}/{stale.name}" not in made:
                stale.unlink()
        written += len(made)
        log.info("target=%s: %d PNG in %s", name, len(made), gold_dir / "png" / name)
    return written


def _save(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, rgba)
