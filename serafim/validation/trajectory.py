"""Where each target really is in every season, against its catalog motion (task A5).

    from serafim.validation.trajectory import trajectory
    rows = trajectory(config["targets"])   # the columns of contract.TRAJECTORY

For every silver sum of one detector and one season (data engineering's D5, read with
serafim.silver.load_sum()), the target is fitted as a point source whose position is free,
together with its Gaia neighbours and a flat background:

- PSF: a Gaussian as wide as the archive EPSF of that detector (measured in the visibility
  check), widened by the area resampling of silver (reproject_exact).
- Left out: masked pixels and pixels where some frame saturated (n_saturated > 0): their mean
  left out the brightest frames, so it is biased. Saturated targets are fitted on their ring.
- The position error comes from the fit, scaled by its reduced chi2: the variance of the sums
  underestimates the noise 1.5 to 2.9 times in crowded fields (measured in D5).
- A sum is used only where the target is detected (S/N >= DETECTION_SNR at its catalog spot).

Per season, the detectors are combined (inverse-variance, with a floor per detector). The
catalog position of that season is the target moved to the mean date of its sums and shifted by
its parallax as seen from Earth, so measured and catalog positions are directly comparable.

`python -m serafim.validation.trajectory` also writes docs/trajectory.md, docs/trajectory.json
(read by guides.py) and one plot per target in docs/img/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, SkyCoord
from astropy.time import Time
from erfa import ErfaWarning
from scipy.optimize import least_squares

from serafim import contract
from serafim.validation import catalogs, layers, targets, visibility

log = logging.getLogger(__name__)

COLUMNS = list(contract.TRAJECTORY.names)
PIXEL_ARCSEC = targets.PIXEL_SCALE_ARCSEC
# Gaussian sigma (px) of the archive EPSF per detector, from its second moments (visibility check)
PSF_SIGMA_PX = {1: 0.61, 2: 0.61, 3: 0.64, 4: 0.66, 5: 0.69, 6: 0.72}
RESAMPLING_SIGMA_PX = 0.29  # reproject_exact spreads a point over ~1 px: 1 / sqrt(12)
DETECTION_SNR = 5.0
RADIUS_PX = 5.0
RING_RADIUS_PX = 8.0  # saturated targets: the fit uses the ring around the empty core
SEARCH_PX = 2.0  # the fitted position stays within this of the catalog position
DETECTOR_FLOOR_PX = 0.1  # Barnard's Star scatters this much between detectors (visibility check)
DEFAULT_OUT = Path("data/validation/trajectory.parquet")
REPORT_PATH = targets.ROOT / "docs" / "trajectory.md"
SUMMARY_PATH = targets.ROOT / "docs" / "trajectory.json"
_SUM_NAME = re.compile(r"d(\d)_s(\d+)\.npz$")


# --- One sum ------------------------------------------------------------------------------------


def catalog_position(target: dict, when: Time) -> SkyCoord:
    """Where the catalogs put the target at `when`, as seen from Earth (parallax included).

    The parallax shift is the difference between the target at its distance and the same
    direction very far away, both seen from Earth at `when`: aberration cancels out.
    """
    moved = targets.propagate(target, when)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        near = SkyCoord(ra=moved.ra, dec=moved.dec, distance=moved.distance)
        far = SkyCoord(ra=moved.ra, dec=moved.dec, distance=1e9 * u.pc)
        frame = GCRS(obstime=when)
        east, north = far.transform_to(frame).spherical_offsets_to(near.transform_to(frame))
    return SkyCoord(ra=moved.ra, dec=moved.dec).spherical_offsets_by(east, north)


def _psf(xs, ys, x0, y0, sigma):
    return visibility._psf_at(xs, ys, x0, y0, sigma)


def fit_position(
    image: np.ndarray,
    variance: np.ndarray,
    use: np.ndarray,
    guess: tuple[float, float],
    others: list[tuple[float, float]],
    sigma: float,
    free_width: bool = False,
) -> dict | None:
    """Target position (free) + neighbour fluxes + background, with errors from the fit.

    With `free_width` the target's width is fitted too: the ring of a saturated star is much
    wider than the PSF core, which falls to almost nothing beyond 3 pixels.
    """
    y, x = np.nonzero(use)
    values, errors = image[use], np.sqrt(variance[use])
    xs, ys = x.astype(float), y.astype(float)
    n = len(others)

    def model(p):
        width = p[3 + n] if free_width else sigma
        total = p[2] * _psf(xs, ys, p[0], p[1], width) + p[-1]
        for k, (ox, oy) in enumerate(others):
            total = total + p[3 + k] * _psf(xs, ys, ox, oy, sigma)
        return total

    start_width = 2.0 * sigma if free_width else sigma
    peak = float(np.nanmax(values)) * 2 * np.pi * start_width**2
    start = [guess[0], guess[1], peak, *([peak / 10] * n)]
    lower = [guess[0] - SEARCH_PX, guess[1] - SEARCH_PX, -np.inf, *([-np.inf] * n)]
    upper = [guess[0] + SEARCH_PX, guess[1] + SEARCH_PX, np.inf, *([np.inf] * n)]
    if free_width:
        start, lower, upper = start + [start_width], lower + [0.3], upper + [10.0]
    start, lower, upper = start + [float(np.nanmedian(values))], lower + [-np.inf], upper + [np.inf]
    result = least_squares(lambda p: (values - model(p)) / errors, start, bounds=(lower, upper))
    if not result.success:
        return None
    dof = max(len(values) - len(start), 1)
    chi2 = float(2 * result.cost / dof)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * max(chi2, 1.0)
    return {
        "x": float(result.x[0]),
        "y": float(result.x[1]),
        "sx": float(np.sqrt(covariance[0, 0])),
        "sy": float(np.sqrt(covariance[1, 1])),
        "chi2": chi2,
        "at_edge": bool(np.any(np.isclose(result.x[:2], lower[:2])) or np.any(
            np.isclose(result.x[:2], upper[:2]))),
    }  # fmt: skip


def measure_sum(arrays: dict, detector: int, target: dict, near=None, saturates=False) -> dict:
    """The target's position in one sum, or why it was not measured."""
    wcs, mjd = arrays["wcs"], float(arrays["mean_mjd"])
    when = Time(mjd, format="mjd", scale="utc")
    expected = catalog_position(target, when)
    tx, ty = (float(v) for v in wcs.world_to_pixel(expected))
    image = np.asarray(arrays["image"], dtype=float)
    variance = np.asarray(arrays["variance"], dtype=float)
    bad = np.asarray(arrays["mask"], bool) | ~np.isfinite(image) | ~(variance > 0)
    bad |= np.asarray(arrays.get("n_saturated", 0)) > 0
    others = []
    if near is not None and len(near):
        ra, dec = layers.move(near, when)
        nxs, nys = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        for ox, oy in zip(np.atleast_1d(nxs), np.atleast_1d(nys), strict=True):
            if np.hypot(ox - tx, oy - ty) < RING_RADIUS_PX + 2:
                others.append((float(ox), float(oy)))
    sigma = float(np.hypot(PSF_SIGMA_PX.get(detector, 0.65), RESAMPLING_SIGMA_PX))
    radius = RING_RADIUS_PX if saturates else RADIUS_PX
    rows, cols = np.indices(image.shape)
    use = (np.hypot(cols - tx, rows - ty) < radius) & ~bad
    row = {"detector": detector, "mjd": mjd, "expected_x": tx, "expected_y": ty}
    if use.sum() < 12:
        return {**row, "reason": "too few clean pixels"}
    stacked = {"image": image, "variance": variance, "unreliable": bad, "psf_sigma": sigma}
    flux = visibility.blend_fit(stacked, (tx, ty), others, sigma=sigma, radius=radius)
    if flux is None or not flux["snr"] >= DETECTION_SNR:
        return {**row, "reason": "not detected", "snr": flux["snr"] if flux else np.nan}
    fitted = fit_position(image, variance, use, (tx, ty), others, sigma, free_width=saturates)
    if fitted is None or fitted["at_edge"]:
        return {**row, "reason": "fit failed", "snr": flux["snr"]}
    return {**row, **fitted, "snr": flux["snr"], "reason": None}


# --- One target ---------------------------------------------------------------------------------


def sum_cells(
    target_id: str, root: Path | None = None, *, all_seasons: bool = False
) -> list[tuple[int, int, Path]]:
    """(detector, season, path) of every sum of one detector and one season on disk; with
    `all_seasons`, also the sums of every season together (season 0)."""
    from serafim.silver.store import silver_dir

    folder = silver_dir(root) / target_id / "sums"
    cells = []
    for path in sorted(folder.glob("d*_s*.npz")):
        found = _SUM_NAME.search(path.name)
        if found and int(found.group(1)) >= 1 and (int(found.group(2)) >= 1 or all_seasons):
            cells.append((int(found.group(1)), int(found.group(2)), path))
    return cells


def measure_target(
    target: dict, root: Path | None = None, *, all_seasons: bool = False
) -> pd.DataFrame:
    """One row per sum (detector, season) with the fitted position or the reason it has none;
    `all_seasons` adds the sums of every season together (season 0, used by task A6)."""
    from serafim.silver.store import load_sum
    from serafim.validation.spectra import _full

    full = _full(target)
    cells = sum_cells(target["id"], root, all_seasons=all_seasons)
    if not cells:
        return pd.DataFrame()
    mjds = {}
    for detector, season, _ in cells:
        mjds.setdefault(season, []).append(
            float(load_sum(target["id"], detector, season, root)["mean_mjd"])
        )
    near = visibility.neighbours(
        full, [float(np.mean(v)) for v in mjds.values()], radius_arcsec=12 * PIXEL_ARCSEC,
        max_mag=layers.GAIA_MAX_MAG, cone_deg=catalogs._region_radius(0.25).deg,
    )  # fmt: skip
    saturates = visibility.saturates(target)
    rows = []
    for detector, season, _ in cells:
        arrays = load_sum(target["id"], detector, season, root)
        rows.append({"season": season, **measure_sum(arrays, detector, full, near, saturates)})
    return pd.DataFrame(rows)


def combine_season(sums: pd.DataFrame) -> dict | None:
    """Inverse-variance mean of the detectors' positions (px), with a floor per detector."""
    good = sums[sums["reason"].isna()] if "reason" in sums else sums.iloc[0:0]
    if good.empty:
        return None
    sx = np.hypot(good["sx"], DETECTOR_FLOOR_PX)
    sy = np.hypot(good["sy"], DETECTOR_FLOOR_PX)
    wx, wy = 1 / sx**2, 1 / sy**2
    x = float((good["x"] * wx).sum() / wx.sum())
    y = float((good["y"] * wy).sum() / wy.sum())
    ex, ey = float(wx.sum() ** -0.5), float(wy.sum() ** -0.5)
    if len(good) > 1:  # the detectors may disagree more than their errors say
        ex = max(ex, float(np.std(good["x"], ddof=1)) / np.sqrt(len(good)))
        ey = max(ey, float(np.std(good["y"], ddof=1)) / np.sqrt(len(good)))
    return {"x": x, "y": y, "ex": ex, "ey": ey, "detectors": sorted(good["detector"].astype(int))}


def target_trajectory(target: dict, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows with the TRAJECTORY columns plus details, per-sum measurements) for one target."""
    from serafim.silver.store import load_sum
    from serafim.validation.spectra import _full

    full = _full(target)
    sums = measure_target(target, root)
    if sums.empty:
        return pd.DataFrame(columns=COLUMNS), sums
    rows = []
    for season, part in sums.groupby("season"):
        mjd = float(part["mjd"].mean())
        catalog = catalog_position(full, Time(mjd, format="mjd", scale="utc"))
        any_sum = load_sum(target["id"], int(part["detector"].iloc[0]), int(season), root)
        combined = combine_season(part)
        row = {"target_id": target["id"], "season": int(season), "mjd": mjd,
               "ra_catalog": float(catalog.ra.deg), "dec_catalog": float(catalog.dec.deg),
               "ra_measured": None, "dec_measured": None, "error_arcsec": None,
               "detectors": [], "offset_arcsec": None}  # fmt: skip
        if combined:
            measured = any_sum["wcs"].pixel_to_world(combined["x"], combined["y"])
            error = float(np.sqrt((combined["ex"] ** 2 + combined["ey"] ** 2) / 2) * PIXEL_ARCSEC)
            row.update(ra_measured=float(measured.ra.deg), dec_measured=float(measured.dec.deg),
                       error_arcsec=error, detectors=combined["detectors"],
                       offset_arcsec=float(measured.separation(catalog).arcsec))  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows), sums


def trajectory(target_list: list[dict], *, root: Path | None = None) -> pd.DataFrame:
    """Rows of trajectory.parquet (contract.TRAJECTORY): one per target and season with sums."""
    parts = [target_trajectory(target, root)[0] for target in target_list]
    parts = [part for part in parts if len(part)]
    if not parts:
        return _empty()
    table = pd.concat(parts, ignore_index=True)[COLUMNS]
    table["season"] = table["season"].astype("int32")
    for name in ("ra_measured", "dec_measured", "error_arcsec"):
        table[name] = pd.to_numeric(table[name]).astype("float64")
    return table


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "target_id": pd.Series(dtype="str"), "season": pd.Series(dtype="int32"),
        "ra_measured": pd.Series(dtype="float64"), "dec_measured": pd.Series(dtype="float64"),
        "error_arcsec": pd.Series(dtype="float64"), "ra_catalog": pd.Series(dtype="float64"),
        "dec_catalog": pd.Series(dtype="float64"),
    })  # fmt: skip


def proper_motion(rows: pd.DataFrame) -> dict | None:
    """Straight-line motion through the measured positions (mas/yr), weighted by their errors.

    Positions are offsets (east, north) from the first measured season; the catalog positions of
    the same seasons get the same fit, so both carry the same tiny parallax wiggle.
    """
    good = rows.dropna(subset=["ra_measured"])
    if len(good) < 2:
        return None
    years = (good["mjd"].to_numpy() - good["mjd"].iloc[0]) / 365.25
    origin = SkyCoord(good["ra_measured"].iloc[0] * u.deg, good["dec_measured"].iloc[0] * u.deg)

    def fit(ra, dec, sigma):
        east, north = origin.spherical_offsets_to(SkyCoord(ra * u.deg, dec * u.deg))
        out = []
        for values in (east.to_value(u.mas), north.to_value(u.mas)):
            weights = 1 / np.asarray(sigma) ** 2
            matrix = np.column_stack([np.ones_like(years), years])
            normal = matrix.T @ (matrix * weights[:, None])
            slope = np.linalg.solve(normal, matrix.T @ (weights * values))[1]
            error = float(np.sqrt(np.linalg.inv(normal)[1, 1])) if len(years) > 2 else float(
                np.hypot(sigma[0], sigma[-1]) / max(years[-1] - years[0], 1e-6))  # fmt: skip
            out.append((float(slope), error))
        return out

    sigma = good["error_arcsec"].to_numpy() * 1000
    (pmra, epmra), (pmdec, epmdec) = fit(
        good["ra_measured"].to_numpy(), good["dec_measured"].to_numpy(), sigma
    )
    (cra, _), (cdec, _) = fit(good["ra_catalog"].to_numpy(), good["dec_catalog"].to_numpy(), sigma)
    total, catalog_total = float(np.hypot(pmra, pmdec)), float(np.hypot(cra, cdec))
    return {
        "measured_mas_yr": [round(pmra, 1), round(pmdec, 1)],
        "error_mas_yr": [round(epmra, 1), round(epmdec, 1)],
        "measured_total_arcsec_yr": round(total / 1000, 3),
        "catalog_total_arcsec_yr": round(catalog_total / 1000, 3),
        "error_total_arcsec_yr": round(float(np.hypot(epmra, epmdec)) / 1000 / np.sqrt(2), 3),
        "seasons": [int(s) for s in good["season"]],
    }


# --- Report --------------------------------------------------------------------------------------


def write_plot(target: dict, rows: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = rows.dropna(subset=["ra_measured"])
    first = rows.iloc[0]
    origin = SkyCoord(first["ra_catalog"] * u.deg, first["dec_catalog"] * u.deg)
    full = _full_target(target)
    days = np.linspace(rows["mjd"].min() - 30, rows["mjd"].max() + 30, 120)
    track = [catalog_position(full, Time(d, format="mjd")) for d in days]
    offsets = [origin.spherical_offsets_to(p) for p in track]
    te = [e.to_value(u.arcsec) for e, _ in offsets]
    tn = [n.to_value(u.arcsec) for _, n in offsets]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=80)
    ax.plot(np.array(te) / PIXEL_ARCSEC, np.array(tn) / PIXEL_ARCSEC, color="grey", lw=1,
            label="catalog path")  # fmt: skip
    for _, row in rows.iterrows():
        c = SkyCoord(row["ra_catalog"] * u.deg, row["dec_catalog"] * u.deg)
        e, n = origin.spherical_offsets_to(c)
        ax.plot(e.to_value(u.arcsec) / PIXEL_ARCSEC, n.to_value(u.arcsec) / PIXEL_ARCSEC, "o",
                mfc="none", color="grey")  # fmt: skip
    for _, row in good.iterrows():
        m = SkyCoord(row["ra_measured"] * u.deg, row["dec_measured"] * u.deg)
        e, n = origin.spherical_offsets_to(m)
        err = row["error_arcsec"] / PIXEL_ARCSEC
        ax.errorbar(e.to_value(u.arcsec) / PIXEL_ARCSEC, n.to_value(u.arcsec) / PIXEL_ARCSEC,
                    xerr=err, yerr=err, fmt="o", color="tab:orange", ms=4)  # fmt: skip
        ax.annotate(f"s{int(row['season'])}", (e.to_value(u.arcsec) / PIXEL_ARCSEC,
                    n.to_value(u.arcsec) / PIXEL_ARCSEC), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")  # fmt: skip
    ax.invert_xaxis()  # east to the left, as the sky is seen
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east (pixels)")
    ax.set_ylabel("north (pixels)")
    ax.set_title(f"{target['name']}: measured (orange) vs catalog", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _full_target(target: dict) -> dict:
    from serafim.validation.spectra import _full

    return _full(target)


def report(results: dict) -> str:
    lines = [
        "# Measured trajectory of the targets (task A5)",
        "",
        "Generated by `python -m serafim.validation.trajectory` from the silver sums of D5.",
        "In each sum of one detector and one season the target is fitted with a free position",
        "(Gaussian PSF of the EPSF width, Gaia neighbours fitted too, pixels where a frame",
        "saturated left out). The detectors of a season are combined; the catalog position of",
        "that season includes the parallax seen from Earth. 1 pixel = 6.15 arcsec.",
        "",
    ]
    for target_id, (target, rows, sums, motion) in results.items():
        lines += [f"## {target['name']} (`{target_id}`)", ""]
        lines += ["| Season | Detectors used | Measured − catalog (arcsec) | Error (arcsec) |",
                  "| --- | --- | --- | --- |"]  # fmt: skip
        for _, row in rows.iterrows():
            if row["ra_measured"] is None or pd.isna(row["ra_measured"]):
                reasons = sums[sums["season"] == row["season"]]["reason"].dropna().unique()
                lines.append(f"| {row['season']} | none ({', '.join(reasons)}) | — | — |")
            else:
                lines.append(
                    f"| {row['season']} | {', '.join(map(str, row['detectors']))} | "
                    f"{row['offset_arcsec']:.2f} | {row['error_arcsec']:.2f} |"
                )
        lines.append("")
        if motion:
            lines += [
                f"**Motion:** measured {motion['measured_total_arcsec_yr']:.2f} ± "
                f"{motion['error_total_arcsec_yr']:.2f} arcsec/yr against "
                f"{motion['catalog_total_arcsec_yr']:.2f} from the catalog over the same seasons "
                f"({', '.join(map(str, motion['seasons']))}).",
                "",
            ]
        lines += [f"![{target['name']}](img/trajectory_{target_id}.png)", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.trajectory")
    parser.add_argument("--target", nargs="+", help="Target ids (default: every target).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    chosen = [t for t in document["targets"] if not args.target or t["id"] in args.target]
    results, tables, summary = {}, [], {}
    for target in chosen:
        rows, sums = target_trajectory(target)
        if rows.empty:
            log.warning("%s: no silver sums, no trajectory", target["id"])
            continue
        motion = proper_motion(rows)
        results[target["id"]] = (target, rows, sums, motion)
        tables.append(rows[COLUMNS])
        write_plot(target, rows, visibility.IMAGE_DIR / f"trajectory_{target['id']}.png")
        summary[target["id"]] = {"motion": motion, "seasons": [
            {"season": int(r["season"]), "measured": not pd.isna(r["ra_measured"]),
             "offset_arcsec": None if pd.isna(r["offset_arcsec"]) else round(r["offset_arcsec"], 3),
             "error_arcsec": None if pd.isna(r["error_arcsec"]) else round(r["error_arcsec"], 3)}
            for _, r in rows.iterrows()]}  # fmt: skip
    if tables:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_parquet(args.out, index=False)
    args.report.write_text(report(results), encoding="utf-8", newline="\n")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s, %s and %s", args.out, args.report, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
