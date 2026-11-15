"""The input of the dipole CNN (task A9, stage 2): 3 x 32 x 32 stamps from two season sums.

Training (training/dataset.py) and the pipeline (serafim/ai/predict.py) build the stamps with
these same functions, so the network always sees the same thing:

    channel 0  asinh(sum A / noise A)        the earlier season
    channel 1  asinh(sum B / noise B)        the later season
    channel 2  asinh((B - A) / noise B-A)    their difference: a mover leaves a dipole
    channel 3  1 where there is no data      masked, saturated in some frame, or beyond the edge

Each image has its background (median) subtracted and is divided by its measured noise (1.4826
MAD: the propagated variance underestimates it up to 7 times in crowded fields). asinh keeps faint
things linear and squeezes bright stars, so one network sees both. Masked pixels, and pixels where
some frame saturated (`n_saturated > 0`, biased low), are 0 in the first three channels and 1 in the
fourth: without it a saturated core and empty sky look the same (version 1 of the CNN had only three
channels and could not tell a saturated star that moves from one that does not).

Arrays are silver's: array[y, x], row 0 = south.
"""

from __future__ import annotations

import numpy as np

STAMP = 32
HALF = STAMP // 2
CHANNELS = ("sum_a", "sum_b", "difference", "no_data")


def unusable(arrays: dict) -> np.ndarray:
    """Pixels the network does not see: masked, not finite, or saturated in some frame."""
    image = np.asarray(arrays["image"], float)
    bad = np.asarray(arrays["mask"], bool) | ~np.isfinite(image)
    if "n_saturated" in arrays:
        bad |= np.asarray(arrays["n_saturated"]) > 0
    return bad


def scaled(image: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """(image - median) / (1.4826 MAD), with bad pixels at 0."""
    values = image[~bad]
    if not len(values):
        return np.zeros_like(image, dtype=np.float32)
    level = float(np.median(values))
    noise = float(1.4826 * np.median(np.abs(values - level))) or 1.0
    return np.where(bad, 0.0, (image - level) / noise).astype(np.float32)


def cube(image_a: np.ndarray, image_b: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """The 4 channels on the whole grid (4, ny, nx), float32. `bad`: unusable in A or in B."""
    image_a = np.asarray(image_a, float)
    image_b = np.asarray(image_b, float)
    out = np.stack([scaled(image_a, bad), scaled(image_b, bad), scaled(image_b - image_a, bad)])
    return np.concatenate([np.arcsinh(out), bad[None].astype(float)]).astype(np.float32)


def cube_from_sums(sum_a: dict, sum_b: dict) -> tuple[np.ndarray, np.ndarray]:
    """(the 4 channels, the unusable pixels) from two silver sums of one detector."""
    bad = unusable(sum_a) | unusable(sum_b)
    return cube(sum_a["image"], sum_b["image"], bad), bad


def cut(channels: np.ndarray, x: float, y: float) -> np.ndarray:
    """The STAMP x STAMP stamp centered on pixel (x, y). Beyond the edge: 0, and 1 in the
    no-data channel."""
    _, ny, nx = channels.shape
    cx, cy = int(round(x)), int(round(y))
    out = np.zeros((channels.shape[0], STAMP, STAMP), np.float32)
    if channels.shape[0] == len(CHANNELS):
        out[CHANNELS.index("no_data")] = 1.0
    x0, y0 = cx - HALF, cy - HALF
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + STAMP, nx), min(y0 + STAMP, ny)
    if xs < xe and ys < ye:
        out[:, ys - y0 : ye - y0, xs - x0 : xe - x0] = channels[:, ys:ye, xs:xe]
    return out
