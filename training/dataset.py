"""Stamps of the motion-dipole CNN (task A9, stage 2).

    python -m training.dataset     # data/dataset/dipoles.npz, labels.parquet and real.npz

For every silver difference (one detector, seasons a and b) its two season sums are read and
3 x 32 x 32 stamps are cut with serafim/ai/stamps.py (the functions the pipeline uses):

- dipole (label 1): a point source injected into sum A at p and into sum B at p + shift, with the
  PSF of that detector (a Gaussian of the EPSF width widened by silver's resampling, as in task
  A5, plus 10 % of the light in wide wings). Shift 0.3 to 3 px in any direction; peak S/N in the
  difference log-uniform from 3 to 300; its brightness may change by up to 15 %.
- saturated_dipole (label 1): the same with a source bright enough to saturate: its core is
  masked (0 in the stamp) and only the wings show, like Barnard's Star.
- Negatives (label 0), hard ones included:
  - background: random places of the real sums;
  - star: centered on a catalog star that does not move (its residual: each season sees it at
    another wavelength, so its PSF and brightness change a little);
  - brightness: a source injected at the same place, 1.5 to 4 times brighter or fainter in B
    (RW Aurigae, R Draconis);
  - pixel_pair: a positive and a negative single pixel side by side (the two candidates the rule
    left, task A9 stage 1);
  - saturated_still: a saturated source that does not move but changes brightness (up to 4
    times) or PSF width (10 %, another wavelength);
  - saturated_core: the center of a real saturated target that does not move.
- Every random place keeps away from the edge, from masked or saturated pixels, and from the real
  movers (any catalog source that moves between a and b, and the moving targets).

Real stamps, only for the test (real.npz): every moving target at its catalog midpoint in every
difference (label 1 if it moved 0.3 px or more), and the saturated targets that do not move.

The split is by target (training/train.py), never at random: neighbor stamps overlap.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from scipy import ndimage

from serafim.ai import stamps
from serafim.validation import dipoles, targets, trajectory

log = logging.getLogger(__name__)

OUT_DIR = targets.ROOT / "data" / "dataset"
GOLD_DIR = targets.ROOT / "data" / "gold"
SEED = 0
PER_DIFFERENCE = {"dipole": 32, "saturated_dipole": 8, "background": 15, "star": 15,
                  "brightness": 5, "pixel_pair": 5, "saturated_still": 8}  # fmt: skip
WING_FRACTION = 0.1
WING_SIGMA_PX = 2.5
SATURATED_SNR = (300.0, 5000.0)  # peak S/N of a source bright enough to saturate
CORE_LEVEL = (0.005, 0.2)  # of the peak: the saturated core is where the profile is above it
PSF_CHANGE = 0.1  # a star seen at another wavelength: up to 10 % wider or narrower
SATURATED_CORES = 5  # per difference of a saturated target that does not move
SHIFT_PX = (0.3, 3.0)
SNR = (3.0, 300.0)
BRIGHTNESS_JITTER = 0.15
BRIGHTNESS_CHANGE = (1.5, 4.0)
PIXEL_SNR = (8.0, 40.0)
KEEP_AWAY_PX = 3  # from masked or saturated pixels
MOVER_PX = 10.0  # from a real mover
STAR_MAX_MAG = {"gaia": 16.0, "catwise": 14.0}
CENTER_JITTER_PX = 1.0
MOVING_TARGETS = ("barnard", "eps_indi_b", "wise_0855")
STILL_SATURATED = ("rw_aur", "r_dra")


def psf_sigma(detector: int) -> float:
    return float(np.hypot(trajectory.PSF_SIGMA_PX.get(detector, 0.65),
                          trajectory.RESAMPLING_SIGMA_PX))  # fmt: skip


def point(shape: tuple[int, int], x: float, y: float, sigma: float) -> np.ndarray:
    """A point source of total flux 1 at pixel (x, y): a Gaussian core and WING_FRACTION of the
    light in a wide Gaussian, like the wings of the archive EPSF (what a saturated star shows)."""
    rows, cols = np.indices(shape)
    r2 = (cols - x) ** 2 + (rows - y) ** 2
    core = np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    wing = np.exp(-r2 / (2 * WING_SIGMA_PX**2)) / (2 * np.pi * WING_SIGMA_PX**2)
    return (1 - WING_FRACTION) * core + WING_FRACTION * wing


def saturate(bad: np.ndarray, *profiles: np.ndarray, level: float) -> np.ndarray:
    """`bad` plus the core of each bright source: the pixels above `level` (silver masks the
    pixels where some frame saturated, so the stamp has 0 there)."""
    out = bad.copy()
    for profile in profiles:
        out |= profile > level
    return out


def noise(image: np.ndarray, bad: np.ndarray) -> float:
    values = image[~bad]
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def free_places(bad: np.ndarray, avoid: list[tuple[float, float]], rng, count: int) -> np.ndarray:
    """`count` random (x, y) away from the edge, bad pixels and the movers in `avoid`."""
    ny, nx = bad.shape
    blocked = ndimage.binary_dilation(bad, iterations=KEEP_AWAY_PX)
    out = []
    for _ in range(count * 50):
        x = rng.uniform(stamps.HALF, nx - stamps.HALF - 1)
        y = rng.uniform(stamps.HALF, ny - stamps.HALF - 1)
        if blocked[int(round(y)), int(round(x))]:
            continue
        if any(np.hypot(x - ax, y - ay) < MOVER_PX for ax, ay in avoid):
            continue
        out.append((x, y))
        if len(out) == count:
            break
    return np.array(out).reshape(-1, 2)


def log_uniform(rng, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# --- One difference -----------------------------------------------------------------------------


def difference_stamps(
    sum_a: dict,
    sum_b: dict,
    detector: int,
    sources: pd.DataFrame,
    avoid: list,
    rng,
    saturated_center: tuple | None = None,
) -> tuple:
    """(stamps (N, 3, 32, 32), labels table) for one difference."""  # fmt: skip
    image_a = np.asarray(sum_a["image"], float)
    image_b = np.asarray(sum_b["image"], float)
    bad = stamps.unusable(sum_a) | stamps.unusable(sum_b)
    base = stamps.cube(image_a, image_b, bad)
    sigma = psf_sigma(detector)
    noise_d = noise(image_b - image_a, bad)
    shape = image_a.shape
    out, rows = [], []

    def add(stamp, kind, label, x, y, snr=np.nan, shift=np.nan):
        out.append(stamp)
        rows.append({"kind": kind, "label": label, "x": x, "y": y, "snr": snr, "shift_px": shift})

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["dipole"]):
        shift, angle = rng.uniform(*SHIFT_PX), rng.uniform(0, 2 * np.pi)
        dx, dy = shift * np.cos(angle), shift * np.sin(angle)
        snr = log_uniform(rng, *SNR)
        flux = snr * noise_d * 2 * np.pi * sigma**2
        a = image_a + flux * point(shape, x, y, sigma)
        b = image_b + flux * rng.uniform(1 - BRIGHTNESS_JITTER, 1 + BRIGHTNESS_JITTER) * point(
            shape, x + dx, y + dy, sigma
        )
        cx = x + dx / 2 + rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX)
        cy = y + dy / 2 + rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX)
        add(stamps.cut(stamps.cube(a, b, bad), cx, cy), "dipole", 1, cx, cy, snr, shift)

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["saturated_dipole"]):
        shift, angle = rng.uniform(*SHIFT_PX), rng.uniform(0, 2 * np.pi)
        dx, dy = shift * np.cos(angle), shift * np.sin(angle)
        snr = log_uniform(rng, *SATURATED_SNR)
        flux = snr * noise_d * 2 * np.pi * sigma**2
        star_a = flux * point(shape, x, y, sigma)
        star_b = (
            flux
            * rng.uniform(1 - BRIGHTNESS_JITTER, 1 + BRIGHTNESS_JITTER)
            * point(shape, x + dx, y + dy, sigma)
        )
        level = star_a.max() * log_uniform(rng, *CORE_LEVEL)
        cut_bad = saturate(bad, star_a, star_b, level=level)
        cx = x + dx / 2 + rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX)
        cy = y + dy / 2 + rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX)
        cube = stamps.cube(image_a + star_a, image_b + star_b, cut_bad)
        add(stamps.cut(cube, cx, cy), "saturated_dipole", 1, cx, cy, snr, shift)

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["saturated_still"]):
        snr = log_uniform(rng, *SATURATED_SNR)
        flux = snr * noise_d * 2 * np.pi * sigma**2
        change = rng.uniform(1.0, BRIGHTNESS_CHANGE[1]) ** rng.choice([-1, 1])
        width = sigma * rng.uniform(1 - PSF_CHANGE, 1 + PSF_CHANGE)
        star_a = flux * point(shape, x, y, sigma)
        star_b = flux * change * point(shape, x, y, width)
        level = star_a.max() * log_uniform(rng, *CORE_LEVEL)
        cube = stamps.cube(image_a + star_a, image_b + star_b,
                           saturate(bad, star_a, star_b, level=level))  # fmt: skip
        jx, jy = rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX, 2)
        add(stamps.cut(cube, x + jx, y + jy), "saturated_still", 0, x + jx, y + jy, snr)

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["background"]):
        add(stamps.cut(base, x, y), "background", 0, x, y)

    bright_enough = sources["catalog"].map(STAR_MAX_MAG).fillna(-99) >= sources["mag"]
    fixed = sources[(sources["moved_px"] < dipoles.MOVING_PX) & bright_enough]
    blocked = ndimage.binary_dilation(bad, iterations=KEEP_AWAY_PX)
    ny, nx = shape
    inside = fixed[fixed["xa"].between(stamps.HALF, nx - stamps.HALF - 1)
                   & fixed["ya"].between(stamps.HALF, ny - stamps.HALF - 1)]  # fmt: skip
    inside = inside[[not blocked[int(round(y)), int(round(x))]
                     and all(np.hypot(x - ax, y - ay) >= MOVER_PX for ax, ay in avoid)
                     for x, y in zip(inside["xa"], inside["ya"], strict=True)]]  # fmt: skip
    if len(inside):
        chosen = inside.sample(
            min(PER_DIFFERENCE["star"], len(inside)), random_state=rng.integers(1e9)
        )
        for x, y in zip(chosen["xa"], chosen["ya"], strict=True):
            jx, jy = rng.uniform(-CENTER_JITTER_PX, CENTER_JITTER_PX, 2)
            add(stamps.cut(base, x + jx, y + jy), "star", 0, x + jx, y + jy)

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["brightness"]):
        snr = log_uniform(rng, *SNR)
        flux = snr * noise_d * 2 * np.pi * sigma**2
        change = rng.uniform(*BRIGHTNESS_CHANGE) ** rng.choice([-1, 1])
        star = point(shape, x, y, sigma)
        add(stamps.cut(stamps.cube(image_a + flux * star, image_b + flux * change * star, bad),
                       x, y), "brightness", 0, x, y, snr)  # fmt: skip

    for x, y in free_places(bad, avoid, rng, PER_DIFFERENCE["pixel_pair"]):
        ix, iy = int(round(x)), int(round(y))
        step = [(1, 0), (0, 1), (1, 1)][rng.integers(3)]
        value = rng.uniform(*PIXEL_SNR) * noise_d
        a, b = image_a.copy(), image_b.copy()
        b[iy, ix] += value
        a[iy + step[1], ix + step[0]] += value
        add(stamps.cut(stamps.cube(a, b, bad), ix, iy), "pixel_pair", 0, ix, iy)

    if saturated_center is not None:
        for _ in range(SATURATED_CORES):
            jx, jy = rng.uniform(-2, 2, 2)
            x, y = saturated_center[0] + jx, saturated_center[1] + jy
            add(stamps.cut(base, x, y), "saturated_core", 0, x, y)

    return np.array(out, np.float32).reshape(-1, 3, stamps.STAMP, stamps.STAMP), pd.DataFrame(rows)


# --- Every difference ---------------------------------------------------------------------------


def target_place(target: dict, mjd: float, wcs) -> tuple[float, float]:
    from serafim.validation.spectra import _full

    where = trajectory.catalog_position(_full(target), Time(mjd, format="mjd", scale="utc"))
    x, y = wcs.world_to_pixel(where)
    return float(x), float(y)


def build(root: Path | None = None, gold_dir: Path = GOLD_DIR, seed: int = SEED) -> tuple:
    """(stamps, labels, real stamps, real labels) for every difference in silver."""
    from serafim.silver.store import load_sum
    from serafim.validation import verification

    rng = np.random.default_rng(seed)
    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in document["targets"]}
    tables, _ = verification.read_gold(gold_dir)
    parts, labels, real, real_rows = [], [], [], []
    for target_id in sorted(tables["targets"]["target_id"]):
        target = by_id[target_id]
        layers = tables["catalogs"][tables["catalogs"]["target_id"] == target_id]
        for detector, a, b in dipoles.difference_cells(target_id, root):
            sum_a = load_sum(target_id, detector, a, root)
            sum_b = load_sum(target_id, detector, b, root)
            wcs = sum_a["wcs"]
            sources = dipoles.catalog_moves(layers, wcs, a, b)
            movers = sources[sources["moved_px"] >= dipoles.MOVING_PX]
            avoid = list(zip(movers["xa"], movers["ya"], strict=True)) + list(
                zip(movers["xb"], movers["yb"], strict=True)
            )
            xa, ya = target_place(target, float(sum_a["mean_mjd"]), wcs)
            xb, yb = target_place(target, float(sum_b["mean_mjd"]), wcs)
            if target_id in MOVING_TARGETS:
                avoid += [(xa, ya), (xb, yb)]
            still = (xa, ya) if target_id in STILL_SATURATED else None
            found, rows = difference_stamps(sum_a, sum_b, detector, sources, avoid, rng, still)
            parts.append(found)
            labels.append(rows.assign(target_id=target_id, detector=detector, season_a=a,
                                      season_b=b))  # fmt: skip
            channels, _ = stamps.cube_from_sums(sum_a, sum_b)
            shift = float(np.hypot(xb - xa, yb - ya))
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            real.append(stamps.cut(channels, cx, cy))
            real_rows.append({"target_id": target_id, "detector": detector, "season_a": a,
                              "season_b": b, "x": cx, "y": cy, "shift_px": shift,
                              "label": int(target_id in MOVING_TARGETS
                                           and shift >= dipoles.MOVING_PX)})  # fmt: skip
        log.info("%s: %d stamps", target_id, sum(len(r) for r in labels if
                                                (r["target_id"] == target_id).all()))  # fmt: skip
    return (np.concatenate(parts), pd.concat(labels, ignore_index=True), np.array(real),
            pd.DataFrame(real_rows))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.dataset")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    found, labels, real, real_labels = build(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "dipoles.npz", stamps=found.astype(np.float16))
    labels.to_parquet(args.out / "labels.parquet", index=False)
    np.savez_compressed(args.out / "real.npz", stamps=real.astype(np.float32))
    real_labels.to_parquet(args.out / "real.parquet", index=False)
    counts = labels.groupby(["target_id", "kind"]).size().unstack(fill_value=0)
    log.info("stamps per target and kind:\n%s", counts.to_string())
    log.info("real test stamps: %d (%d movers)", len(real_labels), int(real_labels["label"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
