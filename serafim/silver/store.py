"""Where silver lives on disk and how to read it back (for D5, D6, D8 and AI & validation).

Every function finds `data/silver/` in the repo by itself; `root` is only for tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from serafim.config import ROOT
from serafim.silver.grid import wcs_from_json


def silver_dir(root: Path | None = None) -> Path:
    return (root or ROOT) / "data" / "silver"


def frame_path(target_id: str, frame_id: str, root: Path | None = None) -> Path:
    return silver_dir(root) / target_id / "frames" / f"{frame_id}.npz"


def frames_table_path(root: Path | None = None) -> Path:
    return silver_dir(root) / "frames.parquet"


def load_frame(target_id: str, frame_id: str, root: Path | None = None) -> dict:
    """One silver frame: image, variance, mask, saturated, footprint and wcs.

    Arrays are array[y, x], indices from 0, row 0 = south, and wcs.pixel_to_world(x, y) is the
    position of pixel (x, y). `wcs` is the grid of the target (targets.parquet `wcs_json`).
    Units and meaning of each array: see serafim/silver/frames.py.
    """
    return _load(frame_path(target_id, frame_id, root))


def _load(path: Path) -> dict:
    with np.load(path) as npz:  # no pickle: plain numpy arrays only
        arrays = {key: npz[key] for key in npz.files}
    arrays["wcs"] = wcs_from_json(str(arrays.pop("wcs_json")))
    return arrays
