"""Can SPHEREx see the targets? A check on real bronze frames, before task A6.

`python -m serafim.validation.visibility [--target wise_0855 barnard]` reads the bronze
cutouts of each target (data/bronze/, from `python -m serafim bronze`) and writes
docs/visibility.md plus one small image per target in docs/img/.

For every frame, a stamp of STAMP_SIZE pixels (6.15 arcsec each) is cut with
`reproject_interp`, either around the target's position at that frame's date (co-moving:
the target stays at the center even if it moves) or on a grid fixed on the sky. Stamps
are averaged with inverse-variance weights, leaving out flagged pixels.

- Brightness: forced photometry with a Gaussian matched filter of the archive PSF width
  (EPSF extension). The signal-to-noise ratio is empirical: the same filter is run on the
  empty sky around the target, so correlated noise from the resampling does not inflate it.
- Position: a 2D Gaussian fit that ignores saturated pixels and their spill-over (BLOOM),
  for targets with a saturated center like Barnard's Star.

The forced photometry is the base of task A4 (spectra and light curves).
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import convolve
from scipy.optimize import least_squares

from serafim.validation import targets

log = logging.getLogger(__name__)

PIXEL_ARCSEC = targets.PIXEL_SCALE_ARCSEC
PIXEL_SR = (PIXEL_ARCSEC / 206264.806) ** 2
MJY_PER_PIXEL_UNIT = 1e9 * PIXEL_SR  # mJy for 1 MJy/sr over one 6.15 arcsec pixel
STAMP_SIZE = 41
INTERPOLATION_SIGMA = 0.41  # px: bilinear resampling widens a point source by ~1/sqrt(6) px
BAD_FLAGS = ("OVERFLOW", "NONLINEAR", "NONFUNC", "MISSING_DATA", "HOT", "COLD")
UNRELIABLE_FLAGS = ("OVERFLOW", "NONLINEAR", "BLOOM")  # for position fits: the saturated core
SEEN, DOUBTFUL = 5.0, 3.0  # signal-to-noise thresholds of the verdict
REPORT_PATH = targets.ROOT / "docs" / "visibility.md"
IMAGE_DIR = targets.ROOT / "docs" / "img"


@dataclass
class Frame:
    """One bronze cutout, ready to measure."""

    obs_id: str
    detector: int
    season: int
    mjd: float
    image: np.ndarray  # IMAGE - ZODI, MJy/sr
    variance: np.ndarray
    bad: np.ndarray  # BAD_FLAGS or not finite
    unreliable: np.ndarray  # UNRELIABLE_FLAGS
    wcs: WCS
    psf_sigma: float  # px, from the EPSF extension


# --- Reading ------------------------------------------------------------------------------


def flag_bits(header: fits.Header) -> dict[str, int]:
    """FLAGS bit numbers by name, from the MSKN* keys (docs/real_fits.md)."""
    return {str(header[key]).strip(): int(key[4:]) for key in header if key.startswith("MSKN")}


def any_flag(flags: np.ndarray, bits: dict[str, int], names: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(flags.shape, dtype=bool)
    for name in names:
        if name in bits:
            mask |= (flags >> bits[name]) & 1 == 1
    return mask


def epsf_sigma(hdu: fits.BinTableHDU) -> float:
    """Median Gaussian sigma (native pixels) of the archive PSF, from its second moments."""
    over = float(hdu.header.get("OVSMPX", 1))
    sigmas = []
    for row in hdu.data:
        psf = np.clip(np.asarray(row["EPSF"], dtype=float), 0, None)
        side = int(round(np.sqrt(psf.size)))
        psf = psf.reshape(side, side)
        y, x = np.mgrid[:side, :side]
        total = psf.sum()
        cx, cy = (psf * x).sum() / total, (psf * y).sum() / total
        second = (psf * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / total
        sigmas.append(np.sqrt(second / 2) / over)
    return float(np.median(sigmas))


def read_frame(path: Path, season: int) -> Frame:
    with fits.open(path) as hdul, warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        header = hdul["IMAGE"].header
        image = hdul["IMAGE"].data.astype(float) - hdul["ZODI"].data.astype(float)
        variance = hdul["VARIANCE"].data.astype(float)
        flags = hdul["FLAGS"].data.astype(np.int64)
        bits = flag_bits(hdul["FLAGS"].header)
        bad = any_flag(flags, bits, BAD_FLAGS) | ~np.isfinite(image) | ~(variance > 0)
        return Frame(
            obs_id=str(header["OBSID"]),
            detector=int(header["DETECTOR"]),
            season=season,
            mjd=float(header["MJD-AVG"]),
            image=image,
            variance=variance,
            bad=bad,
            unreliable=any_flag(flags, bits, UNRELIABLE_FLAGS),
            wcs=WCS(header, naxis=2),
            psf_sigma=epsf_sigma(hdul["EPSF"]),
        )


# --- Stamps and stacks (pure) ----------------------------------------------------------


def grid_wcs(center: SkyCoord, size: int = STAMP_SIZE) -> WCS:
    """North-up TAN grid of `size` pixels of 6.15 arcsec centered on `center`."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.icrs.ra.deg, center.icrs.dec.deg]
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.cdelt = [-PIXEL_ARCSEC / 3600, PIXEL_ARCSEC / 3600]
    return wcs


def stamp(frame: Frame, center: SkyCoord, size: int = STAMP_SIZE) -> dict | None:
    """The frame resampled onto a grid centered on `center`, background removed.

    Returns None when the center falls outside the frame. Flagged pixels stay in the
    stamp as `bad` (NaN): a saturated center is measured on its ring.
    """
    from reproject import reproject_interp

    x, y = frame.wcs.world_to_pixel(center)
    ny, nx = frame.image.shape
    if not (0 <= x < nx - 0.5 and 0 <= y < ny - 0.5):
        return None
    out = grid_wcs(center, size)
    shape = (size, size)
    clean = np.where(frame.bad, np.nan, frame.image)
    image, footprint = reproject_interp((clean, frame.wcs), out, shape_out=shape)
    variance, _ = reproject_interp((np.where(frame.bad, np.nan, frame.variance), frame.wcs),
                                   out, shape_out=shape)  # fmt: skip
    bad, _ = reproject_interp((frame.bad.astype(float), frame.wcs), out, shape_out=shape, order=0)
    unreliable, _ = reproject_interp((frame.unreliable.astype(float), frame.wcs), out,
                                     shape_out=shape, order=0)  # fmt: skip
    bad = (np.nan_to_num(bad, nan=1.0) > 0) | (footprint < 1) | ~np.isfinite(image)
    bad |= ~(np.nan_to_num(variance) > 0)
    image = np.where(bad, np.nan, image)
    image = image - _background(image)
    return {
        "image": image,
        "variance": np.where(bad, np.nan, variance),
        "bad": bad,
        "unreliable": np.nan_to_num(unreliable, nan=0.0) > 0,
        "mjd": frame.mjd,
        "psf_sigma": frame.psf_sigma,
    }


def _background(image: np.ndarray, inner: float = 6.0) -> float:
    """Sigma-clipped median of the stamp away from its center."""
    size = image.shape[0]
    y, x = np.mgrid[:size, :size]
    radius = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2)
    values = image[(radius > inner) & np.isfinite(image)]
    if values.size < 20:
        return 0.0
    return float(sigma_clipped_stats(values, sigma=3.0)[1])


def stack(stamps: list[dict]) -> dict:
    """Inverse-variance mean of stamps; flagged pixels are left out."""
    images = np.array([s["image"] for s in stamps])
    weights = np.array([np.where(s["bad"], 0.0, 1.0 / s["variance"]) for s in stamps])
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)
    total = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(total > 0, (np.nan_to_num(images) * weights).sum(axis=0) / total, np.nan)
        variance = np.where(total > 0, 1.0 / total, np.nan)
    unreliable = np.array([s["unreliable"] for s in stamps]).any(axis=0)
    return {
        "image": image,
        "variance": variance,
        "n_valid": (weights > 0).sum(axis=0),
        "unreliable": unreliable,
        "n_frames": len(stamps),
        "mjd": float(np.mean([s["mjd"] for s in stamps])),
        "psf_sigma": float(np.median([s["psf_sigma"] for s in stamps])),
    }


def gaussian(size: int, sigma: float, x0: float | None = None, y0: float | None = None):
    """Gaussian of unit sum on a size x size grid (center by default)."""
    center = (size - 1) / 2
    x0 = center if x0 is None else x0
    y0 = center if y0 is None else y0
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return psf / psf.sum()


def forced_flux(stacked: dict, sigma: float | None = None) -> dict:
    """Matched-filter flux at the stamp center, in mJy, with an empirical noise.

    The same filter runs over every pixel; its scatter away from the center (robust,
    so other stars barely count) is the noise. snr = flux / that scatter.
    """
    sigma = sigma or np.hypot(stacked["psf_sigma"], INTERPOLATION_SIGMA)
    image = np.nan_to_num(stacked["image"], nan=0.0)
    valid = np.isfinite(stacked["image"]).astype(float)
    half = int(np.ceil(3 * sigma))
    kernel = gaussian(2 * half + 1, sigma)
    numerator = convolve(image, kernel[::-1, ::-1], mode="constant")
    norm = convolve(valid, (kernel**2)[::-1, ::-1], mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_map = np.where(norm > 0.5 * (kernel**2).sum(), numerator / norm, np.nan)
    size = image.shape[0]
    c = size // 2
    y, x = np.mgrid[:size, :size]
    radius = np.hypot(x - c, y - c)
    around = flux_map[(radius > 4 * sigma + 2) & np.isfinite(flux_map)]
    noise = 1.4826 * np.median(np.abs(around - np.median(around))) if around.size > 30 else np.nan
    flux = flux_map[c, c]
    return {
        "flux_mjy": float(flux * MJY_PER_PIXEL_UNIT),
        "error_mjy": float(noise * MJY_PER_PIXEL_UNIT),
        "snr": float(flux / noise) if noise and np.isfinite(noise) and noise > 0 else float("nan"),
        "sigma_px": float(sigma),
    }


def fit_position(stacked: dict, guess: tuple[float, float], radius: float = 6.0) -> dict | None:
    """Center of a 2D Gaussian + constant fitted to the reliable pixels near `guess`.

    Saturated pixels and their spill-over (`unreliable`) are left out, so a star with an
    empty center is fitted on its ring.
    """
    image, variance = stacked["image"], stacked["variance"]
    size = image.shape[0]
    y, x = np.mgrid[:size, :size]
    near = np.hypot(x - guess[0], y - guess[1]) < radius
    use = near & np.isfinite(image) & np.isfinite(variance) & ~stacked["unreliable"]
    if use.sum() < 12:
        return None
    xs, ys, values, errors = x[use], y[use], image[use], np.sqrt(variance[use])
    start_sigma = max(stacked["psf_sigma"], 0.7) * 1.5
    start = [np.nanmax(values) * 2 * np.pi * start_sigma**2, guess[0], guess[1], start_sigma, 0.0]

    def residuals(p):
        amplitude, x0, y0, sigma, background = p
        model = amplitude / (2 * np.pi * sigma**2)
        model = model * np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma**2)) + background
        return (values - model) / errors

    lower = [0, guess[0] - 3, guess[1] - 3, 0.3, -np.inf]
    upper = [np.inf, guess[0] + 3, guess[1] + 3, 10.0, np.inf]
    result = least_squares(residuals, start, bounds=(lower, upper))
    if not result.success:
        return None
    return {"x": float(result.x[1]), "y": float(result.x[2]), "sigma": float(result.x[3]),
            "pixels_used": int(use.sum())}  # fmt: skip


def _psf_at(xs, ys, x0, y0, sigma):
    return np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def blend_fit(
    stacked: dict,
    target_xy: tuple[float, float],
    neighbour_xy: list[tuple[float, float]],
    sigma: float | None = None,
    radius: float = 5.0,
    free_target: bool = False,
) -> dict | None:
    """The target and its neighbours (known positions) fitted together as point sources.

    Fluxes come from weighted linear least squares; with `free_target` the target's position
    is fitted too. The flux error is the formal one scaled to the empirical noise of the stack
    (forced_flux()), so the resampling's correlated noise does not shrink it.
    """
    sigma = sigma or float(np.hypot(stacked["psf_sigma"], INTERPOLATION_SIGMA))
    image, variance = stacked["image"], stacked["variance"]
    size = image.shape[0]
    y, x = np.mgrid[:size, :size]
    near = np.hypot(x - target_xy[0], y - target_xy[1]) < radius
    use = near & np.isfinite(image) & np.isfinite(variance) & ~stacked["unreliable"]
    if use.sum() < 12:
        return None
    xs, ys, values = x[use].astype(float), y[use].astype(float), image[use]
    weights = 1.0 / variance[use]

    def design(tx, ty):
        columns = [_psf_at(xs, ys, tx, ty, sigma)]
        columns += [_psf_at(xs, ys, nx, ny, sigma) for nx, ny in neighbour_xy]
        columns.append(np.ones_like(xs))
        return np.column_stack(columns)

    tx, ty = target_xy
    if free_target:
        start = np.linalg.lstsq(design(tx, ty) * np.sqrt(weights)[:, None],
                                values * np.sqrt(weights), rcond=None)[0]  # fmt: skip

        def residuals(p):
            model = design(p[0], p[1]) @ p[2:]
            return (values - model) * np.sqrt(weights)

        lower = [tx - 2, ty - 2] + [-np.inf] * len(start)
        upper = [tx + 2, ty + 2] + [np.inf] * len(start)
        fitted = least_squares(residuals, [tx, ty, *start], bounds=(lower, upper))
        tx, ty = float(fitted.x[0]), float(fitted.x[1])

    matrix = design(tx, ty) * np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(matrix, values * np.sqrt(weights), rcond=None)[0]
    covariance = np.linalg.pinv(matrix.T @ matrix)
    single = 1.0 / np.sqrt(np.sum(matrix[:, 0] ** 2))  # formal error if the target were alone
    empirical = forced_flux(stacked, sigma)["error_mjy"] / MJY_PER_PIXEL_UNIT
    scale = max(empirical / single, 1.0) if np.isfinite(empirical) else 1.0
    error = float(np.sqrt(covariance[0, 0]) * scale * MJY_PER_PIXEL_UNIT)
    flux = float(coefficients[0] * MJY_PER_PIXEL_UNIT)
    return {
        "flux_mjy": flux,
        "error_mjy": error,
        "snr": flux / error if error > 0 else float("nan"),
        "neighbour_mjy": [float(c * MJY_PER_PIXEL_UNIT) for c in coefficients[1:-1]],
        "x": tx,
        "y": ty,
    }


def neighbours(target: dict, season_mjds: list[float], radius_arcsec: float = 20.0,
               max_mag: float = 19.0, cone_deg: float | None = None):  # fmt: skip
    """Gaia stars that come within `radius_arcsec` of the target during these seasons.

    The target's own Gaia entry (within 2 arcsec of it at the Gaia epoch) is left out.
    `cone_deg` picks the Gaia query (default: radius + 90 arcsec); passing the catalog-layer
    cone of task A2 reuses its cache. If Gaia does not answer and nothing is cached, the
    problem is logged and no neighbour is returned, so a measurement still runs.
    """
    from serafim.validation import catalogs, layers

    center = SkyCoord(target["ra_now"] * u.deg, target["dec_now"] * u.deg)
    cone = cone_deg if cone_deg is not None else (radius_arcsec + 90) / 3600
    try:
        found = catalogs.gaia(center, cone * u.deg, max_mag=max_mag)
    except catalogs.CatalogUnavailable as error:
        log.error("%s: no Gaia neighbours (Gaia unavailable): %s", target.get("id"), error)
        return catalogs._empty()
    if found.empty:
        return found
    keep = np.zeros(len(found), dtype=bool)
    itself = np.zeros(len(found), dtype=bool)
    gaia_epoch = target_at(target, Time(catalogs.GAIA_EPOCH_JYEAR, format="jyear").mjd)
    gaia_sky = SkyCoord(found["ra"].to_numpy() * u.deg, found["dec"].to_numpy() * u.deg)
    itself |= gaia_epoch.separation(gaia_sky).arcsec < 2.0
    for mjd in season_mjds:
        ra, dec = layers.move(found, Time(mjd, format="mjd"))
        here = target_at(target, mjd)
        keep |= here.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec < radius_arcsec
    return found[keep & ~itself].reset_index(drop=True)


# --- Measuring one target --------------------------------------------------------------


def target_frames(target_id: str, root: Path | None = None) -> list[Frame]:
    from serafim.bronze.download import read_manifest

    root = root or targets.ROOT
    frames = []
    for row in read_manifest(root):
        if row["target_id"] != target_id:
            continue
        path = root / row["path"]
        if path.exists():
            frames.append(read_frame(path, int(row["season"])))
    log.info("%s: %d bronze frames", target_id, len(frames))
    return frames


def target_at(target: dict, mjd: float) -> SkyCoord:
    moved = targets.propagate(target, Time(mjd, format="mjd", scale="utc"))
    return SkyCoord(ra=moved.ra, dec=moved.dec)


def comoving_stacks(target: dict, frames: list[Frame]) -> dict:
    """Per detector (all seasons) and per (detector, season): stacks that follow the target."""
    stamps: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        made = stamp(frame, target_at(target, frame.mjd))
        if made is not None:
            stamps.setdefault((frame.detector, frame.season), []).append(made)
    result = {}
    for detector in range(1, 7):
        cells = {season: s for (d, season), s in stamps.items() if d == detector}
        if not cells:
            continue
        everything = [one for season in sorted(cells) for one in cells[season]]
        result[detector] = {
            "all": stack(everything),
            "seasons": {season: stack(cells[season]) for season in sorted(cells)},
        }
    return result


def fixed_stacks(target: dict, frames: list[Frame]) -> dict:
    """Per (detector, season) on a grid fixed on the sky at ra_now/dec_now."""
    center = SkyCoord(target["ra_now"] * u.deg, target["dec_now"] * u.deg)
    cells: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        made = stamp(frame, center)
        if made is not None:
            cells.setdefault((frame.detector, frame.season), []).append(made)
    return {key: stack(value) for key, value in cells.items()}


def motion_check(target: dict, frames: list[Frame]) -> list[dict]:
    """Measured position per (detector, season) against the catalog motion."""
    center = SkyCoord(target["ra_now"] * u.deg, target["dec_now"] * u.deg)
    grid = grid_wcs(center)
    rows = []
    for (detector, season), stacked in sorted(fixed_stacks(target, frames).items()):
        expected = grid.world_to_pixel(target_at(target, stacked["mjd"]))
        guess = (float(expected[0]), float(expected[1]))
        fitted = fit_position(stacked, guess)
        rows.append({"detector": detector, "season": season, "n_frames": stacked["n_frames"],
                     "expected": guess, "fitted": fitted})  # fmt: skip
    return rows


def deblended(target: dict, frames: list[Frame], near) -> dict:
    """Per (detector, season), on the fixed grid: the target fitted together with `near`."""
    from serafim.validation import layers

    grid = grid_wcs(SkyCoord(target["ra_now"] * u.deg, target["dec_now"] * u.deg))
    # A saturated star's empty core and spill-over cover most of the inner 5 px: fit its ring
    radius = 8.0 if saturates(target) else 5.0
    out = {}
    for (detector, season), stacked in sorted(fixed_stacks(target, frames).items()):
        tx, ty = (float(v) for v in grid.world_to_pixel(target_at(target, stacked["mjd"])))
        ra, dec = layers.move(near, Time(stacked["mjd"], format="mjd"))
        nx, ny = grid.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        others = list(zip(np.atleast_1d(nx).astype(float), np.atleast_1d(ny).astype(float),
                          strict=True))  # fmt: skip
        fit = blend_fit(stacked, (tx, ty), others, radius=radius)
        free = blend_fit(stacked, (tx, ty), others, radius=radius, free_target=True)
        if not (fit and fit["snr"] >= SEEN):
            free = None  # too faint in this cell to fit a position
        out[(detector, season)] = {"fit": fit, "free": free, "expected": (tx, ty),
                                   "n_frames": stacked["n_frames"]}  # fmt: skip
    return out


def deblended_rows(cells: dict) -> list[dict]:
    """Per detector: the seasons' deblended fluxes combined with inverse-variance weights."""
    rows = []
    for detector in sorted({d for d, _ in cells}):
        mine = {s: c for (d, s), c in cells.items() if d == detector and c["fit"]}
        fits = [c["fit"] for c in mine.values() if c["fit"]["error_mjy"] > 0]
        if not fits:
            continue
        weights = np.array([1 / f["error_mjy"] ** 2 for f in fits])
        flux = float(np.sum([f["flux_mjy"] for f in fits] * weights) / weights.sum())
        error = float(1 / np.sqrt(weights.sum()))
        neighbour = [f["neighbour_mjy"][0] for f in fits if f["neighbour_mjy"]]
        rows.append({
            "detector": detector,
            "n_frames": int(sum(c["n_frames"] for c in mine.values())),
            "flux_mjy": flux,
            "error_mjy": error,
            "snr": flux / error,
            "verdict": verdict(flux / error),
            "season_snr": {s: round(c["fit"]["snr"], 1) for s, c in sorted(mine.items())},
            "neighbour_mjy": float(np.mean(neighbour)) if neighbour else None,
        })  # fmt: skip
    return rows


def deblended_motion(cells: dict) -> list[dict]:
    return [
        {"detector": d, "season": s, "n_frames": c["n_frames"], "expected": c["expected"],
         "fitted": {"x": c["free"]["x"], "y": c["free"]["y"]} if c["free"] else None}
        for (d, s), c in sorted(cells.items())
    ]  # fmt: skip


def saturates(target: dict) -> bool:
    """The target pixel was saturated in some checked frame (config/targets.yaml, task A1)."""
    return any(row.get("saturated", 0) > 0 for row in target.get("saturation", []))


def verdict(snr: float) -> str:
    if not np.isfinite(snr):
        return "no data"
    if snr >= SEEN:
        return "seen"
    if snr >= DOUBTFUL:
        return "doubtful"
    return "not seen"


# --- Report -----------------------------------------------------------------------------


def brightness_rows(stacks: dict) -> list[dict]:
    rows = []
    for detector, cells in sorted(stacks.items()):
        whole = forced_flux(cells["all"])
        per_season = {season: forced_flux(s) for season, s in cells["seasons"].items()}
        season_snr = {s: round(v["snr"], 1) for s, v in per_season.items()}
        rows.append({"detector": detector, "n_frames": cells["all"]["n_frames"], **whole,
                     "verdict": verdict(whole["snr"]), "season_snr": season_snr})  # fmt: skip
    return rows


def shift_summary(rows: list[dict]) -> list[dict]:
    """First-to-last-season shift per detector, measured against expected (pixels)."""
    out = []
    for detector in sorted({row["detector"] for row in rows}):
        mine = [r for r in rows if r["detector"] == detector and r["fitted"]]
        if len(mine) < 2:
            continue
        first, last = mine[0], mine[-1]
        measured = np.hypot(last["fitted"]["x"] - first["fitted"]["x"],
                            last["fitted"]["y"] - first["fitted"]["y"])  # fmt: skip
        expected = np.hypot(last["expected"][0] - first["expected"][0],
                            last["expected"][1] - first["expected"][1])  # fmt: skip
        offsets = [
            np.hypot(r["fitted"]["x"] - r["expected"][0], r["fitted"]["y"] - r["expected"][1])
            for r in mine
        ]
        out.append({"detector": detector, "seasons": f"{first['season']}->{last['season']}",
                    "measured_px": float(measured), "expected_px": float(expected),
                    "max_offset_px": float(max(offsets))})  # fmt: skip
    return out


def write_image(target: dict, stacks: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = sorted(stacks)
    fig, axes = plt.subplots(1, len(detectors), figsize=(1.9 * len(detectors), 2.2), dpi=80)
    axes = np.atleast_1d(axes)
    for ax, detector in zip(axes, detectors, strict=True):
        image = stacks[detector]["all"]["image"]
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [5, 99.5]) if finite.size else (0, 1)
        ax.imshow(np.arcsinh((image - low) / max(high - low, 1e-6) * 10), origin="lower",
                  cmap="magma")  # fmt: skip
        center = (image.shape[1] // 2, image.shape[0] // 2)
        ax.add_patch(plt.Circle(center, 3.5, fill=False, color="cyan", lw=1))  # leaves it visible
        ax.set_title(f"D{detector}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{target['name']}: every season, following the target (circle)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def report(results: dict) -> str:
    lines = [
        "# Can SPHEREx see the targets? (visibility check, before task A6)",
        "",
        "Generated by `python -m serafim.validation.visibility` from the bronze cutouts of",
        "`python -m serafim bronze` (real IRSA frames, at most 12 per season and detector).",
        "",
        "- **Brightness:** forced photometry at the target's position on each frame's date,",
        "  stacked with inverse-variance weights (a Gaussian matched filter of the EPSF width).",
        "  The noise is empirical: the same filter on the empty sky around the target.",
        f"  Verdict: seen if S/N >= {SEEN:g}, doubtful if >= {DOUBTFUL:g}, not seen below.",
        "- **Motion:** a 2D Gaussian fitted to the pixels that are not saturated (nor BLOOM),",
        "  on a grid fixed on the sky, per detector and season.",
        "",
    ]
    for target_id, result in results.items():
        target = result["target"]
        lines += [f"## {target['name']} (`{target_id}`)", ""]
        mags = target.get("mag", {})
        lines += [f"Catalog: W1 = {mags.get('W1')}, W2 = {mags.get('W2')} (Vega). "
                  f"{result['n_frames']} bronze frames.", ""]  # fmt: skip
        near = result.get("neighbours")
        if near is not None and len(near):
            lines += [
                "**Neighbour stars fitted together with the target** (Gaia DR3, within 20″ of",
                "the target in some season; their light is not the target's):",
                "",
            ]
            for _, star in near.iterrows():
                lines.append(f"- {star['name']}, G = {star['mag']:.2f}")
            lines += [""]
        if saturates(target):
            lines += [
                "**Brightness is not measured here:** the center is saturated in every detector,",
                "so a point-source flux would be meaningless. It is obviously seen. Its brightness",
                "comes from task A4 (the unsaturated wings, with the EPSF).",
                "",
            ]
            result = {**result, "brightness": []}
        naive = result.get("naive_snr")
        header = "| Detector | Frames | Flux (mJy) | S/N, all seasons | Verdict | S/N per season |"
        rule = "| --- | --- | --- | --- | --- | --- |"
        if naive:
            header += " S/N without the neighbour model |"
            rule += " --- |"
        if result["brightness"]:
            lines += [header, rule]
        for row in result["brightness"]:
            per = ", ".join(f"s{s}: {v}" for s, v in row["season_snr"].items())
            line = (
                f"| {row['detector']} | {row['n_frames']} | {row['flux_mjy']:.3f} "
                f"± {row['error_mjy']:.3f} | {row['snr']:.1f} | {row['verdict']} | {per} |"
            )
            if naive:
                line += f" {naive.get(row['detector'], float('nan')):.1f} |"
            lines.append(line)
        if result.get("controls"):
            lines += [
                "",
                "Control stars (CatWISE, W2 close to the target's, same frames):",
                "",
                "| Star | W2 | Detector | S/N, all seasons |",
                "| --- | --- | --- | --- |",
            ]
            for row in result["controls"]:
                lines.append(
                    f"| {row['name']} | {row['w2']:.2f} | {row['detector']} | {row['snr']:.1f} |"
                )
        if result.get("shifts"):
            lines += [
                "",
                "| Detector | Seasons | Measured shift (px) | Expected (px) | Largest offset "
                "from the catalog position (px) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for row in result["shifts"]:
                lines.append(
                    f"| {row['detector']} | {row['seasons']} | {row['measured_px']:.2f} | "
                    f"{row['expected_px']:.2f} | {row['max_offset_px']:.2f} |"
                )
        lines += ["", f"![{target['name']}](img/visibility_{target_id}.png)", ""]
        if near is not None and len(near) and not saturates(target):
            lines += [
                "In the detectors where the target is not seen, the dot inside the circle is the",
                "neighbour star listed above, smeared because the stack follows the target.",
                "",
            ]
    return "\n".join(lines) + "\n"


def control_stars(target: dict, frames: list[Frame], w2: float, limit: int = 4) -> list[dict]:
    """CatWISE stars with W2 within 0.5 mag of the target's, measured like the target."""
    from serafim.validation import catalogs, layers

    center = SkyCoord(target["ra_now"] * u.deg, target["dec_now"] * u.deg)
    found = catalogs.catwise(center, catalogs._region_radius(0.2))
    mid = float(np.mean([frame.mjd for frame in frames]))
    ra, dec = layers.move(found, Time(mid, format="mjd"))  # CatWISE movers moved to today
    sky = SkyCoord(ra * u.deg, dec * u.deg)
    near_mag = ((found["mag"] - w2).abs() < 0.5).to_numpy()
    # Away from the target where it is now: CatWISE also lists the target itself at 2015
    away = target_at(target, mid).separation(sky).arcsec > 60
    chosen = found[near_mag & away].head(limit)
    rows = []
    for _, star in chosen.iterrows():
        cells: dict[int, list[dict]] = {}
        for frame in frames:
            if frame.detector not in (5, 6):
                continue
            ra, dec = layers.move(star.to_frame().T, Time(frame.mjd, format="mjd"))
            made = stamp(frame, SkyCoord(ra[0] * u.deg, dec[0] * u.deg))
            if made is not None:
                cells.setdefault(frame.detector, []).append(made)
        for detector, stamps in sorted(cells.items()):
            rows.append({"name": star["name"], "w2": float(star["mag"]), "detector": detector,
                         "snr": forced_flux(stack(stamps))["snr"]})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m serafim.validation.visibility")
    parser.add_argument("--target", nargs="+", default=["wise_0855", "barnard"])
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import yaml

    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in document["targets"]}
    results = {}
    for target_id in args.target:
        target = by_id[target_id]
        frames = target_frames(target_id)
        if not frames:
            log.error("%s: no bronze frames; run python -m serafim bronze --target %s",
                      target_id, target_id)  # fmt: skip
            continue
        stacks = comoving_stacks(target, frames)
        naive = brightness_rows(stacks)
        result = {"target": target, "n_frames": len(frames), "brightness": naive}
        season_mjds = [float(np.mean([f.mjd for f in frames if f.season == s]))
                       for s in sorted({f.season for f in frames})]  # fmt: skip
        near = neighbours(target, season_mjds)
        if len(near):
            cells = deblended(target, frames, near)
            result["brightness"] = deblended_rows(cells)
            result["naive_snr"] = {row["detector"]: row["snr"] for row in naive}
            result["neighbours"] = near
            if target["kind"] == "motion":
                result["shifts"] = shift_summary(deblended_motion(cells))
        elif target["kind"] == "motion":
            result["shifts"] = shift_summary(motion_check(target, frames))
        w2 = (target.get("mag") or {}).get("W2")
        if w2 and w2 > 8:  # unsaturated in WISE: compare with stars of the same W2
            result["controls"] = control_stars(target, frames, w2)
        write_image(target, stacks, IMAGE_DIR / f"visibility_{target_id}.png")
        results[target_id] = result
        log.info("%s: %s", target_id, json.dumps(result["brightness"], default=str)[:600])
    args.out.write_text(report(results), encoding="utf-8", newline="\n")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
