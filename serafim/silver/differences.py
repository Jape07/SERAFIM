"""Silver 3: one season minus another, per detector (task D6).

    python -m serafim silver [--target barnard]      # frames (D4), sums (D5), then these

Which differences exist is agreed with AI & validation and lives in difference_pairs(), so the
simulated gold and the real pipeline make exactly the same ones (93 for the 5 targets).

`.npz` of a difference: data/silver/<target>/differences/d<detector>_s<a>-s<b>.npz

    difference  float32  MJy/sr        sum of season b minus sum of season a, not normalised;
                                       NaN where `mask`
    error       float32  MJy/sr        sqrt(variance a + variance b); NaN where `mask`. It comes
                                       from the propagated variance, which underestimates the real
                                       noise in a crowded field (up to ~7x in Barnard D1-D3): use
                                       the measured rms for thresholds
    mask        bool                   mask of a OR mask of b
    saturated   bool                   painted as saturated in a OR in b (rule of sums.py): the
                                       value there is biased low, so skip it when looking for
                                       motion
    season_a    int (0-d)              the earlier season
    season_b    int (0-d)              the later season
    wcs_json    str                    the grid of the target

Arrays are array[y, x], indices from 0, row 0 = south, and
WCS(json.loads(wcs_json)).pixel_to_world(x, y) is the position of pixel (x, y).
Read them with serafim.silver.load_difference().

A source that moved leaves a dipole: negative where it was (season a), positive where it went
(season b). `data/silver/differences.parquet` has one row per difference with the columns of
contract.DIFFERENCES. Everything is recomputed on every run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from serafim import contract
from serafim.config import ROOT
from serafim.gold.export import to_contract_table
from serafim.silver.store import (
    difference_path,
    differences_table_path,
    load_sum,
    save_npz,
    silver_dir,
    sum_path,
    sums_table_path,
)

log = logging.getLogger("serafim.silver")


def difference_pairs(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(detector, season_a, season_b) of every difference to make, in that order.

    `cells` are the (detector, season) cells that exist; the white image (detector 0) and
    "all seasons" (season 0) are not cells. Per detector, with the seasons it has:

    - each season minus the one before it (n - 1 differences), and
    - if it has 3 or more seasons, also the last minus the first.

    The image is season_b minus season_a, and season_a < season_b. A detector with 2 seasons
    has one difference (last minus first is the same pair), and with 1 season none. A season
    the detector does not have is skipped: nothing is made up for a missing cell.
    """
    by_detector: dict[int, set[int]] = {}
    for detector, season in cells:
        if detector >= 1 and season >= 1:
            by_detector.setdefault(int(detector), set()).add(int(season))
    pairs: list[tuple[int, int, int]] = []
    for detector in sorted(by_detector):
        seasons = sorted(by_detector[detector])
        found = list(zip(seasons, seasons[1:], strict=False))
        if len(seasons) >= 3:
            found.append((seasons[0], seasons[-1]))
        pairs += [(detector, a, b) for a, b in found]
    return pairs


def subtract(a: dict, b: dict) -> dict[str, np.ndarray]:
    """Sum b minus sum a (dicts as serafim.silver.load_sum() returns them)."""
    from serafim.silver.sums import painted_saturated  # sums needs reproject; mock does not

    mask = a["mask"] | b["mask"]
    with np.errstate(invalid="ignore"):
        difference = b["image"].astype(np.float64) - a["image"].astype(np.float64)
        error = np.sqrt(a["variance"].astype(np.float64) + b["variance"].astype(np.float64))
    return {
        "difference": np.where(mask, np.nan, difference).astype(np.float32),
        "error": np.where(mask, np.nan, error).astype(np.float32),
        "mask": mask,
        "saturated": painted_saturated(a) | painted_saturated(b),
    }


def build_differences(target_id: str | None = None, *, root: Path | None = None) -> int:
    """Silver 3 for the targets in silver 2. Return how many differences were written."""
    root = root or ROOT
    path = sums_table_path(root)
    if not path.is_file():
        raise FileNotFoundError("data/silver/sums.parquet is missing: run the sums step first")
    sums = pq.read_table(path).to_pandas()
    wanted = [target_id] if target_id else sorted(set(sums.target_id))

    rows: list[dict] = []
    for name in wanted:
        mine = sums[sums.target_id == name]
        made = set()
        for detector, season_a, season_b in difference_pairs(zip(mine.detector, mine.season,
                                                                 strict=True)):
            a = load_sum(name, detector, season_a, root)
            b = load_sum(name, detector, season_b, root)
            with np.load(sum_path(name, detector, season_a, root)) as npz:
                wcs_text = npz["wcs_json"]
            out = difference_path(name, detector, season_a, season_b, root)
            save_npz(out, {**subtract(a, b), "season_a": np.int16(season_a),
                           "season_b": np.int16(season_b), "wcs_json": wcs_text})
            made.add(out.name)
            rows.append({
                "difference_id": f"{name}_d{detector}_s{season_a}-s{season_b}",
                "target_id": name,
                "detector": detector,
                "season_a": season_a,
                "season_b": season_b,
                "png": contract.difference_png(name, detector, season_a, season_b),
            })
        # A difference of a pair that no longer exists would mislead whoever reads silver.
        for stale in (silver_dir(root) / name / "differences").glob("d*_s*-s*.npz"):
            if stale.name not in made:
                stale.unlink()
        log.info("target=%s: %d differences", name, len(made))

    _write_differences_table(pd.DataFrame(rows), wanted, root)
    return len(rows)


def _write_differences_table(rows: pd.DataFrame, targets: list[str], root: Path) -> None:
    """Replace the rows of these targets in data/silver/differences.parquet; keep the others."""
    path = differences_table_path(root)
    if path.is_file():
        old = pq.read_table(path).to_pandas()
        rows = pd.concat([old[~old.target_id.isin(targets)], rows], ignore_index=True)
    if rows.empty:
        return
    rows = rows.sort_values(["target_id", "detector", "season_a", "season_b"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(rows, contract.DIFFERENCES), path)
