"""Silver 2: the frames of each detector and season added together (task D5).

    python -m serafim silver [--target barnard]      # frames (D4), then these sums

From `data/silver/frames.parquet` and the frame `.npz` of D4, one sum per cell:

    (detector, season)   every (detector, season) that has frames
    (0, season)          the white image: every detector of that season
    (detector, 0)        every season of that detector
A cell without frames has no sum (nothing is made up) and there is no (0, 0).

The sum is the inverse-variance weighted mean of the frames, pixel by pixel, using only the
pixels that are not masked. It stays in MJy/sr: nothing is stretched or clipped.

`.npz` of a sum: data/silver/<target>/sums/d<detector>_s<season>.npz

    image        float32  MJy/sr        sum(w * x) / sum(w), w = 1 / variance; NaN where n_valid = 0
    variance     float32  (MJy/sr)^2    1 / sum(w); as approximate as the frame variances
    n_valid      int16                  frames that gave a valid value to the pixel
    mask         bool                   True where n_valid = 0
    saturated    bool                   True where every frame was OVERFLOW or NONLINEAR
    n_saturated  int16                  frames where the pixel was OVERFLOW or NONLINEAR. A pixel
                                        with n_saturated > 0 has a value but it is biased low: the
                                        mean left out the brightest frames
    mean_mjd     float64  (0-d)         mean date of the frames of the sum
    wcs_json     str                    the grid of the target, as in the frames

Arrays are array[y, x], indices from 0, row 0 = south, and
WCS(json.loads(wcs_json)).pixel_to_world(x, y) is the position of pixel (x, y).
Read them with serafim.silver.load_sum().

PNG rule (agreed with AI & validation, used by D7): a pixel is painted with the brightest colour
where `saturated`, or where n_saturated > 0 and n_saturated >= n_valid.

`data/silver/sums.parquet` has one row per sum with the columns of contract.SUMS:
- background_noise: the global rms of the sum's background (sep.Background, 32 px boxes, masked
  pixels left out), in MJy/sr per pixel. It falls as 1/sqrt(N) only where the background is
  random noise. In a crowded field it falls less: the faint stars below the detection limit are
  the same in every frame, and each frame sees another wavelength through SPHEREx's variable
  filter, so stars change between frames. Measured on Barnard: the random part (the difference of
  two half sums) follows 1/sqrt(N) in D6 and almost in D5, while the total rms of a season sum is
  1.5 (D6) to 2.9 (D1-D3) times above 1/sqrt(N). The propagated variance does not include this:
  use the measured rms for thresholds.
- saturated: the target, at its position on the mean date, has a pixel painted bright (rule above)
  within settings.validation.match_radius_px.
Everything is recomputed on every run: it is cheap and it follows the frames when seasons move.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sep

from serafim import contract
from serafim.config import ROOT, load_config
from serafim.gold.export import to_contract_table
from serafim.silver.frames import target_position
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import (
    frames_table_path,
    load_frame,
    save_npz,
    silver_dir,
    sum_path,
    sums_table_path,
)

log = logging.getLogger("serafim.silver")

_MJD_ZERO = pd.Timestamp("1858-11-17", tz="UTC")


# --- Pure functions ---------------------------------------------------------------------------
def sum_cells(frames: pd.DataFrame) -> list[tuple[int, int, pd.DataFrame]]:
    """(detector, season, frames of that sum), sorted, for one target's frames."""
    cells = [(int(d), int(s), g) for (d, s), g in frames.groupby(["detector", "season"])]
    cells += [(0, int(s), g) for s, g in frames.groupby("season")]
    cells += [(int(d), 0, g) for d, g in frames.groupby("detector")]
    return sorted(cells, key=lambda cell: (cell[0], cell[1]))


def combine(frames: list[dict]) -> dict[str, np.ndarray]:
    """Inverse-variance weighted mean of silver frames (dicts as load_frame() returns them)."""
    images = np.stack([f["image"] for f in frames]).astype(np.float64)
    variances = np.stack([f["variance"] for f in frames]).astype(np.float64)
    masks = np.stack([f["mask"] for f in frames])
    valid = ~masks & np.isfinite(images) & np.isfinite(variances) & (variances > 0)
    weights = np.where(valid, 1.0 / np.where(valid, variances, 1.0), 0.0)
    total = weights.sum(axis=0)
    n_valid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(n_valid > 0, (weights * np.where(valid, images, 0.0)).sum(axis=0) / total,
                         np.nan)
        variance = np.where(n_valid > 0, 1.0 / total, np.nan)
    n_saturated = np.stack([f["saturated"] for f in frames]).sum(axis=0)
    return {
        "image": image.astype(np.float32),
        "variance": variance.astype(np.float32),
        "n_valid": n_valid.astype(np.int16),
        "mask": n_valid == 0,
        "saturated": n_saturated == len(frames),
        "n_saturated": n_saturated.astype(np.int16),
    }


def painted_saturated(arrays: dict) -> np.ndarray:
    """Pixels the PNG paints with the brightest colour (rule agreed with AI & validation)."""
    n_saturated, n_valid = arrays["n_saturated"], arrays["n_valid"]
    return arrays["saturated"] | ((n_saturated > 0) & (n_saturated >= n_valid))


def background_noise(image: np.ndarray, mask: np.ndarray) -> float:
    """Global rms of the background of a sum, in its units, masked pixels left out."""
    data = np.ascontiguousarray(np.where(mask, 0.0, image), dtype=np.float32)
    return float(sep.Background(data, mask=np.ascontiguousarray(mask), bw=32, bh=32).globalrms)


def center_saturated(arrays: dict, wcs, x: float, y: float, radius_px: float) -> bool:
    """True if a pixel within radius_px of (x, y) is painted as saturated."""
    rows, cols = np.mgrid[0 : arrays["mask"].shape[0], 0 : arrays["mask"].shape[1]]
    near = (cols - x) ** 2 + (rows - y) ** 2 <= radius_px**2
    return bool((painted_saturated(arrays) & near).any())


def mjd(dates: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(dates, utc=True) - _MJD_ZERO) / pd.Timedelta(days=1)).to_numpy(float)


def union_flags(flags: pd.Series) -> str:
    return ",".join(sorted({name for text in flags for name in str(text).split(",") if name}))


# --- Disk -------------------------------------------------------------------------------------
def build_sums(target_id: str | None = None, *, root: Path | None = None) -> int:
    """Silver 2 for the targets in silver 1. Return how many sums were written."""
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    settings = config["settings"]
    radius = float((settings.get("validation") or {}).get("match_radius_px", 2))
    path = frames_table_path(root)
    if not path.is_file():
        raise FileNotFoundError("data/silver/frames.parquet is missing: run the frames step first")
    frames = pq.read_table(path).to_pandas()
    targets = {t["id"]: t for t in config["targets"]}
    wanted = [target_id] if target_id else sorted(set(frames.target_id) & set(targets))
    unknown = [t for t in wanted if t not in targets]
    if unknown:
        raise ValueError(f"Unknown target {unknown[0]}. Config has: {', '.join(targets)}.")

    rows: list[dict] = []
    for name in wanted:
        target = targets[name]
        mine = frames[frames.target_id == name]
        if mine.empty:
            log.warning("target=%s: no frames in silver, no sums", name)
            continue
        size_deg = float(target.get("cutout_size_deg") or settings["cutout_size_deg"])
        grid = target_wcs(float(target["ra_now"]), float(target["dec_now"]), size_deg)
        wcs_text = np.array(wcs_to_json(grid))
        loaded = {fid: load_frame(name, fid, root) for fid in mine.frame_id}
        made = set()
        for detector, season, cell in sum_cells(mine):
            arrays = combine([loaded[fid] for fid in cell.frame_id])
            mean_mjd = float(mjd(cell.date_utc).mean())
            x, y = grid.world_to_pixel(target_position(target, mean_mjd))
            save_npz(sum_path(name, detector, season, root),
                     {**arrays, "mean_mjd": np.float64(mean_mjd), "wcs_json": wcs_text})
            made.add(sum_path(name, detector, season, root).name)
            rows.append({
                "sum_id": f"{name}_d{detector}_s{season}",
                "target_id": name,
                "detector": detector,
                "season": season,
                "mean_date_utc": cell.date_utc.mean().floor("us"),
                "n_frames": len(cell),
                "background_noise": background_noise(arrays["image"], arrays["mask"]),
                "flags": union_flags(cell["flags"]),  # df.flags is a pandas attribute
                "saturated": center_saturated(arrays, grid, float(x), float(y), radius),
                "png": contract.sum_png(name, detector, season),
            })
        # A sum of a cell that no longer exists would mislead whoever reads silver.
        for stale in (silver_dir(root) / name / "sums").glob("d*_s*.npz"):
            if stale.name not in made:
                stale.unlink()
        log.info("target=%s: %d sums from %d frames", name, len(made), len(mine))

    _write_sums_table(pd.DataFrame(rows), wanted, root)
    return len(rows)


def _write_sums_table(rows: pd.DataFrame, targets: list[str], root: Path) -> None:
    """Replace the rows of these targets in data/silver/sums.parquet; keep the others."""
    path = sums_table_path(root)
    if path.is_file():
        old = pq.read_table(path).to_pandas()
        rows = pd.concat([old[~old.target_id.isin(targets)], rows], ignore_index=True)
    if rows.empty:
        return
    rows = rows.sort_values(["target_id", "detector", "season"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(rows, contract.SUMS), path)
