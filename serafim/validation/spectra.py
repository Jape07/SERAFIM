"""Spectra and light curves of the targets by forced photometry (task A4, AI & validation).

SPHEREx sees each target at a different wavelength in every frame (its filters vary across
the detector), so the flux at the target's position in one frame is one point of its
spectrum. Grouped by season it is a spectrum; ordered by date, a light curve.

Each bronze frame is measured in its own pixels (no resampling):

1. Position: the target moved to the frame's mean date (targets.propagate()), through the
   frame's sky WCS.
2. Wavelength: the spectral WCS (key W, WAVE-TAB) at that pixel, without the SIP terms
   (docs/real_fits.md).
3. PSF: the archive EPSF of the detector zone where the target falls (WCS key A gives the
   detector pixel), 5x oversampled, evaluated at every native pixel.
4. Fit: weighted linear least squares of target + Gaia neighbours (visibility.neighbours())
   + a constant background, without flagged pixels. Targets that saturate are fitted on
   their wings: saturated pixels and their spill-over (BLOOM) are left out.

    from serafim.validation.spectra import spectra
    rows = spectra(config["targets"], frames=frames)   # columns of contract.SPECTRA

`python -m serafim.validation.spectra [--target ...]` also writes docs/spectra.md with a
plot per target and a check against 2MASS/WISE magnitudes.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import map_coordinates

from serafim import contract
from serafim.validation import targets, visibility

log = logging.getLogger(__name__)

SPECTRA_COLUMNS = list(contract.SPECTRA.names)
RADIUS_PX = 4.0  # fit area around the target
WINGS_RADIUS_PX = 7.0  # for saturated targets, whose core is left out
# The archive EPSF only reaches ~3.3 px from the center. A fit is skipped when less than this
# fraction of the target's light falls in usable pixels, and a wings-only fit of a saturated
# target is flagged poor_fit when the PSF does not describe its pixels (reduced chi2 above
# POOR_FIT_CHI2).
MIN_PSF_SUPPORT = 0.05
POOR_FIT_CHI2 = 25.0
EDGE_PX = 3.0
BLEND_PX = 2.0
BAD_NEAR_PX = 1.5
DEFAULT_OUT = Path("data/validation/spectra.parquet")
REPORT_PATH = targets.ROOT / "docs" / "spectra.md"
SIP_PREFIXES = ("A_", "B_", "AP_", "BP_")
_MJD_ZERO = pd.Timestamp("1858-11-17", tz="UTC")

# Catalog bands for the check against known fluxes: (key in targets.yaml mag, um, Vega zero Jy)
CATALOG_BANDS = (
    ("J", 1.235, 1594.0), ("H", 1.662, 1024.0), ("K", 2.159, 666.7),
    ("W1", 3.353, 309.54), ("W2", 4.603, 171.787),
)  # fmt: skip


def frame_id(target_id: str, obs_id: str) -> str:
    """contract.frame_id() when it exists (PR #49), else the same agreed format."""
    if hasattr(contract, "frame_id"):
        return contract.frame_id(target_id, obs_id)
    return f"{target_id}_{obs_id}"


# --- One frame -----------------------------------------------------------------------------


def without_sip(header: fits.Header) -> fits.Header:
    clean = header.copy()
    for key in list(clean.keys()):
        if key.startswith(SIP_PREFIXES):
            del clean[key]
    return clean


def wavelength_at(hdul: fits.HDUList, x: float, y: float) -> tuple[float, float]:
    """Central wavelength and bandwidth (um) of pixel (x, y), from WCS-WAVE."""
    spectral = WCS(without_sip(hdul["IMAGE"].header), fobj=hdul, key="W")
    spectral.sip = None
    wavelength, bandwidth = spectral.pixel_to_world_values(x, y)
    return float(wavelength), float(bandwidth)


def detector_position(header: fits.Header, x: float, y: float) -> tuple[float, float]:
    """Pixel of the whole detector for pixel (x, y) of the cutout (WCS key A)."""
    linear = WCS(without_sip(header), key="A", naxis=2)
    dx, dy = linear.pixel_to_world_values(x, y)
    return float(dx), float(dy)


def epsf_at(hdu: fits.BinTableHDU, dx: float, dy: float) -> tuple[np.ndarray, int]:
    """The EPSF of the detector zone nearest to (dx, dy), and its oversampling."""
    data = hdu.data
    nearest = int(np.argmin(np.hypot(data["XCENTER"] - dx, data["YCENTER"] - dy)))
    psf = np.asarray(data["EPSF"][nearest], dtype=float)
    side = int(round(np.sqrt(psf.size)))
    return psf.reshape(side, side), int(hdu.header.get("OVSMPX", 1))


def psf_pixels(psf: np.ndarray, over: int, x0: float, y0: float, xs, ys) -> np.ndarray:
    """Fraction of a point source at (x0, y0) that falls in each native pixel (xs, ys)."""
    center = (psf.shape[0] - 1) / 2
    rows = center + (np.asarray(ys, dtype=float) - y0) * over
    cols = center + (np.asarray(xs, dtype=float) - x0) * over
    values = map_coordinates(psf, [rows, cols], order=1, mode="constant", cval=0.0)
    return values / (psf.sum() / over**2)


def pixel_solid_angle_sr(wcs: WCS, x: float, y: float) -> float:
    """Solid angle of pixel (x, y): SIP makes it differ from the nominal 6.15 arcsec."""
    here, right, up = wcs.pixel_to_world([x, x + 1, x], [y, y, y + 1])
    a = here.separation(right).to_value(u.rad)
    b = here.separation(up).to_value(u.rad)
    angle = here.position_angle(right) - here.position_angle(up)
    return float(abs(a * b * np.sin(angle.to_value(u.rad))))


def fit_fluxes(image, variance, use, columns) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted linear least squares: coefficients, their errors and the reduced chi2."""
    weights = 1.0 / variance[use]
    matrix = np.column_stack([column[use] for column in columns]) * np.sqrt(weights)[:, None]
    target = image[use] * np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    covariance = np.linalg.pinv(matrix.T @ matrix)
    residual = target - matrix @ coefficients
    dof = max(int(use.sum()) - len(columns), 1)
    return (
        coefficients,
        np.sqrt(np.clip(np.diag(covariance), 0, None)),
        float(residual @ residual / dof),
    )


def measure_frame(
    path: Path,
    target: dict,
    season: int,
    near: pd.DataFrame | None = None,
    saturates: bool = False,
) -> dict | None:
    """One point of the spectrum: the target's flux in one bronze frame (or None)."""
    from serafim.validation import layers

    with fits.open(path) as hdul, warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.simplefilter("ignore", UserWarning)
        header = hdul["IMAGE"].header
        sky = WCS(header, naxis=2)
        mjd = float(header["MJD-AVG"])
        x0, y0 = (float(v) for v in sky.world_to_pixel(visibility.target_at(target, mjd)))
        image = hdul["IMAGE"].data.astype(float) - hdul["ZODI"].data.astype(float)
        variance = hdul["VARIANCE"].data.astype(float)
        ny, nx = image.shape
        if not (0 <= x0 < nx - 0.5 and 0 <= y0 < ny - 0.5):
            return None
        flags = hdul["FLAGS"].data.astype(np.int64)
        bits = visibility.flag_bits(hdul["FLAGS"].header)
        bad = visibility.any_flag(flags, bits, visibility.BAD_FLAGS)
        bad |= ~np.isfinite(image) | ~(variance > 0)
        unreliable = visibility.any_flag(flags, bits, visibility.UNRELIABLE_FLAGS)
        wavelength, _ = wavelength_at(hdul, x0, y0)
        psf, over = epsf_at(hdul["EPSF"], *detector_position(header, x0, y0))
        omega = pixel_solid_angle_sr(sky, x0, y0)

    others: list[tuple[float, float]] = []
    if near is not None and len(near):
        ra, dec = layers.move(near, Time(mjd, format="mjd"))
        nxs, nys = sky.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        for ox, oy in zip(np.atleast_1d(nxs), np.atleast_1d(nys), strict=True):
            if np.hypot(ox - x0, oy - y0) < RADIUS_PX + 3:
                others.append((float(ox), float(oy)))

    y, x = np.mgrid[:ny, :nx]
    radius = WINGS_RADIUS_PX if saturates else RADIUS_PX
    distance = np.hypot(x - x0, y - y0)
    use = (distance < radius) & ~bad
    if saturates:
        use &= ~unreliable
    if use.sum() < 8:
        return None
    columns = [psf_pixels(psf, over, x0, y0, x, y)]
    if columns[0][use].sum() < MIN_PSF_SUPPORT:
        return None  # almost none of the target's light is in usable pixels: nothing to measure
    columns += [psf_pixels(psf, over, ox, oy, x, y) for ox, oy in others]
    columns.append(np.ones_like(image))
    coefficients, errors, chi2 = fit_fluxes(image, variance, use, columns)
    scale = max(1.0, np.sqrt(chi2))
    to_mjy = 1e9 * omega  # (MJy/sr) x sr -> MJy -> mJy

    near_core = distance < BAD_NEAR_PX
    labels = []
    if saturates and (unreliable & near_core).any():
        labels.append("saturated")
    if saturates and chi2 > POOR_FIT_CHI2:
        # A bright unsaturated star also has a high chi2 (a tiny PSF mismatch at high S/N); its
        # error is already scaled by sqrt(chi2). Only a wings-only fit is unreliable.
        labels.append("poor_fit")
    if any(np.hypot(ox - x0, oy - y0) < BLEND_PX for ox, oy in others):
        labels.append("blend")
    if (bad & near_core).any() and "saturated" not in labels:
        labels.append("bad_pixels")
    if min(x0, y0, nx - 1 - x0, ny - 1 - y0) < EDGE_PX:
        labels.append("edge")
    return {
        "mjd": mjd,
        "season": int(season),
        "wavelength_um": wavelength,
        "flux_mjy": float(coefficients[0] * to_mjy),
        "error_mjy": float(errors[0] * scale * to_mjy),
        "flag": ",".join(labels) or None,
    }


# --- Every frame of every target ----------------------------------------------------------


def spectra(
    target_list: list[dict],
    frames: pd.DataFrame | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Rows of spectra.parquet (contract.SPECTRA) for every bronze frame of these targets.

    Args:
        target_list: targets from config/targets.yaml (id, ra_now, dec_now and, if known,
            ra, dec, pmra_mas_yr, pmdec_mas_yr, parallax_mas). Targets added by coordinates
            work too: without a motion they stay where they are.
        frames: frames.parquet of the run; only rows whose frame_id is in it are kept.
        root: repository root (data/bronze/ lives under it).
    """
    from serafim.bronze.download import read_manifest

    root = root or targets.ROOT
    manifest = pd.DataFrame(read_manifest(root))
    rows = []
    for target in target_list:
        mine = manifest[manifest["target_id"] == target["id"]] if len(manifest) else manifest
        if mine.empty:
            log.warning("%s: no bronze frames, no spectrum", target["id"])
            continue
        season_mjds = mine.groupby("season")["t_mid_mjd"].mean().tolist()
        near = visibility.neighbours(_full(target), season_mjds)
        saturates = visibility.saturates(target)
        measured = 0
        for row in mine.itertuples(index=False):
            path = root / row.path
            if not path.exists():
                continue
            point = measure_frame(path, _full(target), row.season, near, saturates)
            if point is None:
                continue
            rows.append({"target_id": target["id"], "frame_id": frame_id(target["id"], row.obs_id),
                         **point})  # fmt: skip
            measured += 1
        log.info("%s: %d spectrum points (%d neighbours fitted)", target["id"], measured, len(near))
    table = _table(rows)
    if frames is not None:
        table = table[table["frame_id"].isin(set(frames["frame_id"]))].reset_index(drop=True)
    return table


def _full(target: dict) -> dict:
    """A target with every field propagate() needs (added-by-coordinates targets lack some)."""
    return {
        **target,
        "ra": target.get("ra", target["ra_now"]),
        "dec": target.get("dec", target["dec_now"]),
        "pmra_mas_yr": target.get("pmra_mas_yr") or 0.0,
        "pmdec_mas_yr": target.get("pmdec_mas_yr") or 0.0,
        "parallax_mas": target.get("parallax_mas") or 0.0,
    }


def _table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({
            "target_id": pd.Series(dtype="str"), "frame_id": pd.Series(dtype="str"),
            "season": pd.Series(dtype="int32"), "date_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "wavelength_um": pd.Series(dtype="float64"), "flux_mjy": pd.Series(dtype="float64"),
            "error_mjy": pd.Series(dtype="float64"), "flag": pd.Series(dtype="str"),
        })  # fmt: skip
    table = pd.DataFrame(rows)
    table["date_utc"] = (_MJD_ZERO + pd.to_timedelta(table.pop("mjd"), unit="D")).dt.round("s")
    table["season"] = table["season"].astype("int32")
    table = table.sort_values(["target_id", "date_utc", "frame_id"], ignore_index=True)
    return table[SPECTRA_COLUMNS]


# --- Report --------------------------------------------------------------------------------


def doubtful(table: pd.DataFrame) -> pd.Series:
    """Points near an edge, on flagged pixels or badly fitted. `saturated` and `blend` alone
    are still measurements."""
    return table["flag"].fillna("").str.contains("edge|bad_pixels|poor_fit", regex=True)


def binned(table: pd.DataFrame, width_um: float = 0.1) -> pd.DataFrame:
    """Inverse-variance mean flux in wavelength bins of `width_um`, unflagged points only."""
    good = table[~doubtful(table) & (table["error_mjy"] > 0)]
    if good.empty:
        return pd.DataFrame(columns=["wavelength_um", "flux_mjy", "error_mjy", "n"])
    bins = np.floor(good["wavelength_um"] / width_um) * width_um + width_um / 2
    weights = 1 / good["error_mjy"] ** 2
    grouped = pd.DataFrame({"bin": bins, "w": weights, "wf": weights * good["flux_mjy"]})
    sums = grouped.groupby("bin").agg(w=("w", "sum"), wf=("wf", "sum"), n=("w", "size"))
    result = pd.DataFrame(
        {"wavelength_um": sums.index, "flux_mjy": sums["wf"] / sums["w"],
         "error_mjy": 1 / np.sqrt(sums["w"]), "n": sums["n"]}
    )  # fmt: skip
    return result.reset_index(drop=True)


def band_mean(table: pd.DataFrame, low_um: float, high_um: float) -> tuple[float, float] | None:
    """Inverse-variance mean flux (mJy) and its error between two wavelengths."""
    part = table[(table["wavelength_um"] >= low_um) & (table["wavelength_um"] < high_um)]
    part = part[~doubtful(part) & (part["error_mjy"] > 0)]
    if part.empty:
        return None
    weights = 1 / part["error_mjy"] ** 2
    return float((part["flux_mjy"] * weights).sum() / weights.sum()), float(weights.sum() ** -0.5)


def catalog_check(target: dict, table: pd.DataFrame) -> list[dict]:
    """SPHEREx flux near each catalog band against the catalog magnitude (Vega -> mJy)."""
    rows = []
    usable = table[(table["error_mjy"] > 0) & ~doubtful(table)]
    for band, um, zero in CATALOG_BANDS:
        mag = (target.get("mag") or {}).get(band)
        if mag is None:
            continue
        near = usable[(usable["wavelength_um"] - um).abs() < 0.05 * um]
        if near.empty:
            continue
        weights = 1 / near["error_mjy"] ** 2
        flux = float((near["flux_mjy"] * weights).sum() / weights.sum())
        rows.append({"band": band, "um": um, "catalog_mjy": zero * 10 ** (-0.4 * mag) * 1000,
                     "spherex_mjy": flux, "error_mjy": float(1 / np.sqrt(weights.sum())),
                     "n": len(near)})  # fmt: skip
    return rows


def write_plot(target: dict, table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=80)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (season, part) in enumerate(table.groupby("season")):
        color = colors[i % len(colors)]
        fine, flagged = part[~doubtful(part)], part[doubtful(part)]
        ax.errorbar(fine["wavelength_um"], fine["flux_mjy"], fine["error_mjy"], fmt="o", ms=3,
                    color=color, alpha=0.6, label=f"season {season}")  # fmt: skip
        ax.scatter(flagged["wavelength_um"], flagged["flux_mjy"], s=12, facecolors="none",
                   edgecolors=color, alpha=0.6)  # fmt: skip
    bins = binned(table)
    if len(bins):
        ax.plot(bins["wavelength_um"], bins["flux_mjy"], color="black", lw=1.2, label="binned")
    ax.axhline(0, color="grey", lw=0.5)
    # Robust y range: a single absurd fit must not flatten the spectrum
    trusted = table.loc[~doubtful(table), "flux_mjy"]
    values = trusted if len(trusted) >= 5 else table["flux_mjy"]
    low, high = np.percentile(values, [1, 99])
    pad = 0.15 * max(high - low, 1e-3)
    ax.set_ylim(min(low, 0) - pad, high + pad)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux (mJy)")
    ax.set_title(f"{target['name']}: SPHEREx spectrum (hollow points are flagged)", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def report(results: dict) -> str:
    lines = [
        "# SPHEREx spectra of the targets (task A4)",
        "",
        "Generated by `python -m serafim.validation.spectra` from the bronze cutouts.",
        "Each point is one frame: the target's flux at the wavelength SPHEREx saw it, by a PSF",
        "fit (archive EPSF) together with any Gaia star within 20″ of the target. Saturated",
        "targets are fitted on their wings (flag `saturated`); `blend` marks a neighbour closer",
        "than 2 px, fitted too. Points near an edge or on flagged pixels are hollow and left",
        "out of the binned line and of the catalog check.",
        "",
        "**Saturated targets** (Barnard's Star, RW Aurigae, R Draconis): only the PSF wings can",
        "be fitted, and the archive EPSF reaches just ~3.3 px from the center, mostly inside the",
        "saturated core and its spill-over. Most of those fits do not describe the pixels",
        "(reduced chi2 of 100 or more) and are flagged `poor_fit`: their absolute flux is not",
        "reliable, and their brightness change is shown by the images (the size of the glow).",
        "A frame where less than 5 % of the target's light falls in usable pixels gives no point.",
        "",
    ]
    for target_id, (target, table) in results.items():
        flagged = int(table["flag"].notna().sum())
        lines += [f"## {target['name']} (`{target_id}`)", "",
                  f"{len(table)} points, {flagged} flagged.", ""]  # fmt: skip
        check = catalog_check(target, table)
        if check:
            lines += ["| Band | µm | Catalog (mJy) | SPHEREx (mJy) | Points |",
                      "| --- | --- | --- | --- | --- |"]  # fmt: skip
            for row in check:
                lines.append(
                    f"| {row['band']} | {row['um']:.2f} | {row['catalog_mjy']:.3g} | "
                    f"{row['spherex_mjy']:.3g} ± {row['error_mjy']:.2g} | {row['n']} |"
                )
            lines.append("")
        if target.get("kind") == "spectrum":
            dip, peak = band_mean(table, 3.2, 3.5), band_mean(table, 3.9, 4.3)
            if dip and peak:
                lines += [
                    f"**Methane check:** {dip[0]:.3g} mJy at 3.2–3.5 µm against {peak[0]:.3g} mJy",
                    f"at 3.9–4.3 µm, a ratio of {dip[0] / peak[0]:.2f}. Methane absorbs near",
                    "3.3 µm, so a cold, methane-rich atmosphere is dimmer there",
                    "(ratio well below 1).",
                    "",
                ]
        lines += [f"![{target['name']}](img/spectrum_{target_id}.png)", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.spectra")
    parser.add_argument("--target", nargs="+", help="Target ids (default: every target).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    chosen = [t for t in document["targets"] if not args.target or t["id"] in args.target]
    table = spectra(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    results = {}
    for target in chosen:
        mine = table[table["target_id"] == target["id"]]
        if mine.empty:
            continue
        write_plot(target, mine, visibility.IMAGE_DIR / f"spectrum_{target['id']}.png")
        results[target["id"]] = (target, mine)
    args.report.write_text(report(results), encoding="utf-8", newline="\n")
    log.info("wrote %s and %s", args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
