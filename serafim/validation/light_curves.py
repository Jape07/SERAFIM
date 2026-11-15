"""Light curves of saturated targets by ring photometry (task A4, second part).

The archive EPSF reaches only ~3.3 px from the center, so a saturated star cannot be
fitted (spectra.py flags those fits poor_fit). Its brightness can still be compared
between seasons: at a given wavelength, the light in a ring around the saturated core is
a fixed fraction of the star's light. Every frame sees the star at a different wavelength,
so per detector the ring light is modelled as

    ln(ring) = P(wavelength) + offset(season)

with P a smooth polynomial (the star's spectrum times the ring fraction) and one offset per
season, the first season being the reference. exp(offset) is the star's brightness in that
season relative to the first one, at the same wavelength.

- The ring starts RING_GAP_PX beyond the largest saturated or spill-over (BLOOM) pixel seen in
  that detector, so it never touches a saturated pixel; it is RING_WIDTH_PX wide.
- Gaia stars near the ring are masked; the background is the clipped median of a wider ring.
- Barnard's Star, saturated but not variable, is the control: its seasons must give ~1.

    python -m serafim.validation.light_curves [--target r_dra rw_aur barnard]

writes data/validation/light_curves.parquet (one row per frame), docs/light_curves.md,
docs/light_curves.json (per season and detector) and one plot per target in docs/img/.
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
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from serafim.validation import catalogs, layers, spectra, targets, visibility

log = logging.getLogger(__name__)

STAMP_HALF_PX = 24  # 49 x 49 pixels around the target
SATURATION_SEARCH_PX = 12.0  # saturated or BLOOM pixels this close belong to the target
RING_GAP_PX = 1.0
RING_WIDTH_PX = 3.0
MAX_RING_IN_PX = 11.0
BACKGROUND_GAP_PX = 3.0
BACKGROUND_WIDTH_PX = 6.0
SUBPIXELS = 5  # per pixel side, for the ring weights
NEIGHBOUR_MASK_PX = 2.0  # the PSF is ~1.5 px wide
# Only neighbours that matter next to a saturated star are masked: in a crowded field (75
# Gaia stars down to G = 18 around Barnard's Star) masking every one would leave no ring.
NEIGHBOUR_MAX_G = 17.0
# Gaia stars this close to the target can fall in the ring or the background ring
NEIGHBOUR_REACH_PX = MAX_RING_IN_PX + RING_WIDTH_PX + BACKGROUND_GAP_PX + BACKGROUND_WIDTH_PX + 3
MIN_VALID_FRACTION = 0.5  # of the ring area, after masks
MIN_RING_SNR = 5.0
CLIP_SIGMA = 3.0
LOG_ERROR_FLOOR = 0.05  # 5 % per frame, added to the ring's statistical error
MAX_RELATIVE_ERROR = 0.3  # a season ratio less certain than this is shown but not combined
MIN_SEASON_FRAMES = 3  # nor one measured on fewer frames
PERIOD_RANGE_DAYS = (80.0, 600.0)
N_SHUFFLES = 200
MIN_MONTHS_FOR_PERIOD = 8
# Barnard's Star (not variable) scatters by ~10-30 % per detector; only larger changes count
SYSTEMATIC_FLOOR = 0.15
DEFAULT_OUT = Path("data/validation/light_curves.parquet")
REPORT_PATH = targets.ROOT / "docs" / "light_curves.md"
SUMMARY_PATH = targets.ROOT / "docs" / "light_curves.json"
POINT_COLUMNS = [
    "target_id", "frame_id", "detector", "season", "date_utc", "wavelength_um",
    "relative_brightness", "error",
]  # fmt: skip


@dataclass
class RingStamp:
    """The pixels around the target in one frame, ready for ring photometry."""

    frame_id: str
    detector: int
    season: int
    mjd: float
    wavelength_um: float
    image: np.ndarray
    variance: np.ndarray
    bad: np.ndarray
    unreliable: np.ndarray
    x0: float  # target position in the stamp
    y0: float
    saturated_radius: float  # px from the target to its farthest saturated/BLOOM pixel
    neighbours: list[tuple[float, float]]


# --- One frame -------------------------------------------------------------------------------


def read_stamp(path: Path, target: dict, season: int, obs_id: str, near=None) -> RingStamp | None:
    from serafim.validation import layers

    with fits.open(path) as hdul, warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.simplefilter("ignore", UserWarning)
        header = hdul["IMAGE"].header
        sky = WCS(header, naxis=2)
        mjd = float(header["MJD-AVG"])
        x0, y0 = (float(v) for v in sky.world_to_pixel(visibility.target_at(target, mjd)))
        ny, nx = hdul["IMAGE"].data.shape
        if not (0 <= x0 < nx and 0 <= y0 < ny):
            return None
        wavelength, _ = spectra.wavelength_at(hdul, x0, y0)
        xs = slice(max(0, round(x0) - STAMP_HALF_PX), min(nx, round(x0) + STAMP_HALF_PX + 1))
        ys = slice(max(0, round(y0) - STAMP_HALF_PX), min(ny, round(y0) + STAMP_HALF_PX + 1))
        image = (hdul["IMAGE"].data[ys, xs] - hdul["ZODI"].data[ys, xs]).astype(float)
        variance = hdul["VARIANCE"].data[ys, xs].astype(float)
        flags = hdul["FLAGS"].data[ys, xs].astype(np.int64)
        bits = visibility.flag_bits(hdul["FLAGS"].header)
    bad = visibility.any_flag(flags, bits, visibility.BAD_FLAGS)
    bad |= ~np.isfinite(image) | ~(variance > 0)
    unreliable = visibility.any_flag(flags, bits, visibility.UNRELIABLE_FLAGS)
    sx, sy = x0 - xs.start, y0 - ys.start

    others = []
    if near is not None and len(near):
        ra, dec = layers.move(near, Time(mjd, format="mjd"))
        nxs, nys = sky.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        others = [(float(a) - xs.start, float(b) - ys.start)
                  for a, b in zip(np.atleast_1d(nxs), np.atleast_1d(nys), strict=True)]  # fmt: skip
    return RingStamp(
        frame_id=spectra.frame_id(target["id"], obs_id),
        detector=int(header["DETECTOR"]),
        season=int(season),
        mjd=mjd,
        wavelength_um=wavelength,
        image=image,
        variance=variance,
        bad=bad,
        unreliable=unreliable,
        x0=sx,
        y0=sy,
        saturated_radius=saturated_radius(unreliable, sx, sy),
        neighbours=others,
    )


def saturated_radius(unreliable: np.ndarray, x0: float, y0: float) -> float:
    """Distance from the target to its farthest saturated or spill-over pixel (0 if none)."""
    y, x = np.indices(unreliable.shape)
    distance = np.hypot(x - x0, y - y0)
    mine = unreliable & (distance < SATURATION_SEARCH_PX)
    return float(distance[mine].max()) if mine.any() else 0.0


def annulus_weights(shape, x0: float, y0: float, inner: float, outer: float) -> np.ndarray:
    """Fraction of each pixel inside the ring inner <= r < outer (SUBPIXELS x SUBPIXELS).

    A hard-edged ring on the pixel grid over-counts the bright inner edge, by an amount that
    changes with where the star falls inside its pixel, frame to frame. Weighting each pixel by
    the part of it inside the ring removes that.
    """
    offsets = (np.arange(SUBPIXELS) + 0.5) / SUBPIXELS - 0.5
    y, x = np.indices(shape)
    weights = np.zeros(shape)
    for dy in offsets:
        for dx in offsets:
            distance = np.hypot(x + dx - x0, y + dy - y0)
            weights += (distance >= inner) & (distance < outer)
    return weights / SUBPIXELS**2


def ring_flux(stamp: RingStamp, inner: float, outer: float) -> dict | None:
    """Background-subtracted light in the ring inner <= r < outer (MJy/sr x pixel).

    Flagged, saturated and neighbour pixels are left out and the light of the rest is scaled
    up to the whole ring; None if less than MIN_VALID_FRACTION of the ring is usable.
    """
    y, x = np.indices(stamp.image.shape)
    distance = np.hypot(x - stamp.x0, y - stamp.y0)
    usable = ~stamp.bad & ~stamp.unreliable
    for nx, ny in stamp.neighbours:
        usable &= np.hypot(x - nx, y - ny) >= NEIGHBOUR_MASK_PX
    weights = annulus_weights(stamp.image.shape, stamp.x0, stamp.y0, inner, outer)
    area = np.pi * (outer**2 - inner**2)  # the whole ring, also the part off the cutout
    used = np.where(usable, weights, 0.0)
    if used.sum() < MIN_VALID_FRACTION * area:
        return None  # too much of the ring is flagged, masked or off the cutout
    sky_inner = outer + BACKGROUND_GAP_PX
    sky = (distance >= sky_inner) & (distance < sky_inner + BACKGROUND_WIDTH_PX) & usable
    if sky.sum() < 20:
        return None
    _, background, spread = sigma_clipped_stats(stamp.image[sky], sigma=3.0)
    scale = area / used.sum()
    light = np.nansum(used * (stamp.image - background)) * scale
    noise = np.sqrt(np.nansum(used**2 * stamp.variance)) * scale
    error = float(np.hypot(noise, spread / np.sqrt(sky.sum()) * area))
    return {"ring": float(light), "ring_error": error}


# --- The model per detector -----------------------------------------------------------------


def fit_detector(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season offsets and per-frame brightness relative to the first season, one detector.

    `points` needs season, wavelength_um, ring and ring_error. Returns (seasons, frames):
    seasons has season, ratio, error, n_frames, overlap_um; frames adds
    relative_brightness and error to the rows that were used.
    """
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()
    use = points[points["ring"] > MIN_RING_SNR * points["ring_error"]].copy()
    seasons = sorted(use["season"].unique())
    if len(seasons) < 2 or len(use) < 4:
        return pd.DataFrame(), pd.DataFrame()
    reference = seasons[0]
    for attempt in range(2):  # fit, clip the outliers once, fit again
        y = np.log(use["ring"].to_numpy())
        # The frame-to-frame scatter is mostly systematic (PSF changes across the detector):
        # a floor keeps one very bright frame from getting an outsized weight
        sigma = np.hypot((use["ring_error"] / use["ring"]).to_numpy(), LOG_ERROR_FLOOR)
        offset = use["wavelength_um"].to_numpy() - use["wavelength_um"].mean()
        degree = 2 if len(use) >= 10 else 1 if len(use) >= 6 else 0
        columns = [offset**k for k in range(degree + 1)]
        columns += [(use["season"] == s).to_numpy(float) for s in seasons[1:]]
        matrix = np.column_stack(columns) / sigma[:, None]
        coefficients, *_ = np.linalg.lstsq(matrix, y / sigma, rcond=None)
        residual = y / sigma - matrix @ coefficients
        dof = max(len(y) - len(columns), 1)
        scale = max(1.0, float(np.sqrt(residual @ residual / dof)))
        # Clip on a robust spread: an outlier inflates the chi2 scale enough to hide itself
        robust = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        keep = np.abs(residual) < CLIP_SIGMA * max(robust, 1.0)
        if attempt == 1 or keep.all():
            break
        use = use[keep]
        if sorted(use["season"].unique()) != seasons:
            return pd.DataFrame(), pd.DataFrame()  # clipping emptied a season
    covariance = np.linalg.pinv(matrix.T @ matrix) * scale**2
    polynomial = np.column_stack(columns[: degree + 1]) @ coefficients[: degree + 1]
    frames = use.assign(
        relative_brightness=np.exp(y - polynomial),
        error=np.exp(y - polynomial) * sigma * scale,
    )
    ref_range = _range(use, reference)
    rows = [
        {
            "season": reference,
            "ratio": 1.0,
            "error": 0.0,
            "n_frames": int((use["season"] == reference).sum()),
            "overlap_um": ref_range[1] - ref_range[0],
        }
    ]
    for i, season in enumerate(seasons[1:]):
        k = degree + 1 + i
        ratio = float(np.exp(coefficients[k]))
        low, high = _range(use, season)
        rows.append({
            "season": int(season),
            "ratio": ratio,
            "error": ratio * float(np.sqrt(covariance[k, k])),
            "n_frames": int((use["season"] == season).sum()),
            "overlap_um": min(high, ref_range[1]) - max(low, ref_range[0]),
        })  # fmt: skip
    return pd.DataFrame(rows), frames


def _range(table: pd.DataFrame, season: int) -> tuple[float, float]:
    part = table.loc[table["season"] == season, "wavelength_um"]
    return (float(part.min()), float(part.max())) if len(part) else (np.nan, np.nan)


# --- Every frame of a target ------------------------------------------------------------------


def target_light_curve(target: dict, root: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Per-frame relative brightness (POINT_COLUMNS) and per-season summary of one target."""
    from serafim.bronze.download import read_manifest

    root = root or targets.ROOT
    manifest = pd.DataFrame(read_manifest(root))
    mine = manifest[manifest["target_id"] == target["id"]] if len(manifest) else manifest
    if mine.empty:
        return pd.DataFrame(columns=POINT_COLUMNS), []
    full = spectra._full(target)
    near = visibility.neighbours(
        full,
        mine.groupby("season")["t_mid_mjd"].mean().tolist(),
        radius_arcsec=NEIGHBOUR_REACH_PX * targets.PIXEL_SCALE_ARCSEC,
        max_mag=layers.GAIA_MAX_MAG,  # the catalog-layer query of task A2: already cached
        cone_deg=catalogs._region_radius(0.25).deg,
    )
    if len(near):
        near = near[near["mag"] < NEIGHBOUR_MAX_G].reset_index(drop=True)
    stamps = []
    for row in mine.itertuples(index=False):
        if (root / row.path).exists():
            stamp = read_stamp(root / row.path, full, row.season, row.obs_id, near)
            if stamp is not None:
                stamps.append(stamp)
    points, summary = [], []
    for detector in sorted({s.detector for s in stamps}):
        chosen = [s for s in stamps if s.detector == detector]
        inner = min(np.ceil(max(s.saturated_radius for s in chosen)) + RING_GAP_PX, MAX_RING_IN_PX)
        outer = inner + RING_WIDTH_PX
        rows = []
        for s in chosen:
            if s.saturated_radius >= inner:  # this frame spills into the ring
                continue
            measured = ring_flux(s, inner, outer)
            if measured:
                rows.append({"frame_id": s.frame_id, "season": s.season, "mjd": s.mjd,
                             "wavelength_um": s.wavelength_um, **measured})  # fmt: skip
        seasons, frames = fit_detector(pd.DataFrame(rows))
        if seasons.empty:
            continue
        for row in seasons.itertuples(index=False):
            dates = frames.loc[frames["season"] == row.season, "mjd"]
            summary.append({"detector": detector, **row._asdict(), "ring_px": [inner, outer],
                            "mean_date_utc": _utc(float(dates.mean())).isoformat()})  # fmt: skip
        frames = frames.assign(target_id=target["id"], detector=detector,
                               date_utc=frames["mjd"].map(_utc))  # fmt: skip
        points.append(frames)
    table = (
        pd.concat(points, ignore_index=True)[POINT_COLUMNS]
        if points
        else pd.DataFrame(columns=POINT_COLUMNS)
    )
    log.info("%s: %d frames in the light curve, %d (detector, season) values", target["id"],
             len(table), len(summary))  # fmt: skip
    return table, summary


def combine(summary: list[dict]) -> list[dict]:
    """Per season: the detectors' ratios combined (inverse-variance mean of their logs).

    Ratios less certain than MAX_RELATIVE_ERROR are left out. The change can depend on the
    wavelength (RW Aurigae brightens in detector 1 and fades in detectors 4-5), so read the
    detectors first: this mean is only a summary.
    """
    table = pd.DataFrame(summary)
    certain = (table["error"] <= MAX_RELATIVE_ERROR * table["ratio"]) & (
        table.get("n_frames", MIN_SEASON_FRAMES) >= MIN_SEASON_FRAMES
    )
    table = table[(table["error"] == 0) | certain]
    out = []
    for season, part in table.groupby("season"):
        other = part[part["error"] > 0]
        if other.empty:
            out.append({"season": int(season), "ratio": 1.0, "error": 0.0, "detectors": len(part)})
            continue
        logs = np.log(other["ratio"])
        weights = 1 / (other["error"] / other["ratio"]) ** 2
        mean = float((logs * weights).sum() / weights.sum())
        error = float(1 / np.sqrt(weights.sum()))
        spread = float(np.std(logs)) if len(other) > 1 else 0.0
        out.append(
            {
                "season": int(season),
                "ratio": float(np.exp(mean)),
                "error": float(np.exp(mean) * max(error, spread / np.sqrt(len(other)))),
                "detectors": int(len(other)),
                "lowest": float(other["ratio"].min()),
                "highest": float(other["ratio"].max()),
            }
        )
    return out


def periodogram(days, y, dy, periods) -> np.ndarray:
    """Share of the variance a sine wave of each period explains (weighted least squares).

    Written with numpy on purpose: the same as a generalized Lomb-Scargle periodogram,
    without a compiled extension.
    """
    weights = 1 / np.asarray(dy) ** 2
    mean = np.sum(weights * y) / np.sum(weights)
    total = np.sum(weights * (y - mean) ** 2)
    root = np.sqrt(weights)
    power = np.empty(len(periods))
    for i, period in enumerate(periods):
        phase = 2 * np.pi * np.asarray(days) / period
        matrix = (
            np.column_stack([np.ones_like(phase), np.sin(phase), np.cos(phase)]) * root[:, None]
        )
        coefficients, *_ = np.linalg.lstsq(matrix, y * root, rcond=None)
        residual = y * root - matrix @ coefficients
        power[i] = 1 - (residual @ residual) / total
    return power


def best_period(points: pd.DataFrame, shuffles: int = N_SHUFFLES, seed: int = 0) -> dict | None:
    """The period that best explains the per-frame brightness, and how likely it is to be luck.

    Each detector's mean is taken out first (their amplitudes differ). The false-alarm
    probability compares the best power with that of light curves whose values were shuffled
    among the same dates. `periodic` needs it below 1 % and data in at least 8 different
    months: three clumps of seasons can be fitted by many sine waves.
    """
    if len(points) < 20:
        return None
    dates = pd.to_datetime(points["date_utc"], utc=True)
    days = ((dates - dates.min()).dt.total_seconds() / 86400).to_numpy()
    y = np.log(points["relative_brightness"].to_numpy())
    y = y - pd.Series(y).groupby(points["detector"].to_numpy()).transform("mean").to_numpy()
    dy = np.hypot((points["error"] / points["relative_brightness"]).to_numpy(), LOG_ERROR_FLOOR)
    periods = np.linspace(*PERIOD_RANGE_DAYS, 2600)
    power = periodogram(days, y, dy, periods)
    peak = int(np.argmax(power))
    rng = np.random.default_rng(seed)
    coarse = periods[::10]
    luck = [periodogram(days, rng.permutation(y), dy, coarse).max() for _ in range(shuffles)]
    false_alarm = (np.sum(np.array(luck) >= power[peak]) + 1) / (shuffles + 1)
    low = high = peak
    while low > 0 and power[low] > power[peak] / 2:
        low -= 1
    while high < len(power) - 1 and power[high] > power[peak] / 2:
        high += 1
    months = dates.dt.strftime("%Y-%m").nunique()
    return {
        "period_days": round(float(periods[peak]), 1),
        "half_power_days": [round(float(periods[low]), 1), round(float(periods[high]), 1)],
        "power": round(float(power[peak]), 3),
        "false_alarm": round(float(false_alarm), 4),
        "months": int(months),
        "periodic": bool(false_alarm < 0.01 and months >= MIN_MONTHS_FOR_PERIOD),
    }


def _utc(mjd: float) -> pd.Timestamp:
    return (spectra._MJD_ZERO + pd.to_timedelta(mjd, unit="D")).round("s")


# --- Report ------------------------------------------------------------------------------------


def write_plot(target: dict, points: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=80)
    for detector, part in points.groupby("detector"):
        ax.errorbar(pd.to_datetime(part["date_utc"]), part["relative_brightness"], part["error"],
                    fmt="o", ms=3, alpha=0.6, label=f"D{detector}")  # fmt: skip
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Brightness / first season")
    ax.set_title(f"{target['name']}: brightness at the same wavelength, relative to the first "
                 "season", fontsize=9)  # fmt: skip
    ax.legend(fontsize=7, ncol=6, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def report(results: dict) -> str:
    lines = [
        "# Light curves of the saturated targets (task A4)",
        "",
        "Generated by `python -m serafim.validation.light_curves` from the bronze cutouts.",
        "The saturated core cannot be measured, so each frame measures the light in a ring",
        f"around it ({RING_WIDTH_PX:g} px wide, starting {RING_GAP_PX:g} px beyond the farthest",
        "saturated or BLOOM pixel of that detector). Per detector, ln(ring) = P(wavelength) +",
        "offset(season): the ratio is the brightness of a season relative to the first one, at",
        "the same wavelength. Barnard's Star does not vary and is the control: its ratios",
        "should be close to 1.",
        "",
    ]
    for target_id, (target, points, summary, period) in results.items():
        lines += [f"## {target['name']} (`{target_id}`)", "", f"{len(points)} frames used.", ""]
        if period:
            low, high = period["half_power_days"]
            verdict = (
                "a periodic change" if period["periodic"]
                else "no period can be claimed (false alarm too high, or too few months)"
            )  # fmt: skip
            lines += [
                f"**Period search** (sine fit, 80–600 days): best {period['period_days']:.0f} days "
                f"(half power {low:.0f}–{high:.0f}), power {period['power']:.2f}, false alarm "
                f"{period['false_alarm']:.3f}, {period['months']} months with data: {verdict}.",
                "",
            ]
        combined = combine(summary) if summary else []
        if combined:
            lines += ["| Season | Brightness / season 1 (all detectors) | Detectors | Range |",
                      "| --- | --- | --- | --- |"]  # fmt: skip
            for row in combined:
                spread = f"{row['lowest']:.2f}–{row['highest']:.2f}" if "lowest" in row else "—"
                lines.append(
                    f"| {row['season']} | {row['ratio']:.2f} ± {row['error']:.2f} | "
                    f"{row['detectors']} | {spread} |"
                )
            lines += ["", "| Detector | Season | Ratio | Frames | Ring (px) |",
                      "| --- | --- | --- | --- | --- |"]  # fmt: skip
            for row in summary:
                if row["error"] > 0:
                    lines.append(
                        f"| {row['detector']} | {row['season']} | {row['ratio']:.2f} ± "
                        f"{row['error']:.2f} | {row['n_frames']} | "
                        f"{row['ring_px'][0]:g}–{row['ring_px'][1]:g} |"
                    )
            lines.append("")
        lines += [f"![{target['name']}](img/light_curve_{target_id}.png)", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.light_curves")
    parser.add_argument("--target", nargs="+", default=["r_dra", "rw_aur", "barnard"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in document["targets"]}
    results, tables, summaries = {}, [], {}
    for target_id in args.target:
        target = by_id[target_id]
        points, summary = target_light_curve(target)
        if points.empty:
            log.warning("%s: no light curve (no bronze frames?)", target_id)
            continue
        write_plot(target, points, visibility.IMAGE_DIR / f"light_curve_{target_id}.png")
        period = best_period(points)
        results[target_id] = (target, points, summary, period)
        tables.append(points)
        summaries[target_id] = {"by_detector": summary, "combined": combine(summary),
                                "period": period}  # fmt: skip
    if tables:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_parquet(args.out, index=False)
    args.report.write_text(report(results), encoding="utf-8", newline="\n")
    args.summary.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s, %s and %s", args.out, args.report, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
