"""Do the catalog markers fall on what SPHEREx shows? (task A8, review of the MVP)

    python -m serafim.validation.markers [--gold data/gold] [--target X ...]

The viewer draws the rows of gold `catalogs.parquet` on the gold images with the grid of
`targets.parquet` (`wcs_json`). A marker on empty sky, or beside its star, looks wrong to a
visitor. So, for every target, season and detector:

1. each marker of that season is put on the grid with `wcs_json`, as the viewer will;
2. a small forced aperture (APERTURE_PX) on the silver sum says whether something is there
   (S/N >= FOUND_SNR) and, for an isolated star found, how far its light is from the marker;
3. per catalog and detector, the visible limit: the magnitude down to which at least
   VISIBLE_FRACTION of the markers have a source under them beyond chance (the viewer's default
   cut). In a crowded field a small aperture often finds some star anywhere, so the same
   measurement at RANDOM_SPOTS random places of each sum gives the chance rate, and a bin counts
   (found - chance) / (1 - chance);
4. out of place: a marker brighter than that limit by MARGIN_MAG with nothing under it, or an
   isolated star found more than MAX_OFFSET_PX from its marker.

Markers near the target (its own glow, checked in task A6), next to a saturated center or the
edge, or on masked pixels are not judged. SkyBoT asteroids are only counted: they move during a
season, so a sum smears them.

Writes docs/markers.md, docs/markers.json (read by guides.py: the default magnitude cuts of the
layers and reference positions for the viewer's projection) and docs/img/markers_<id>.png, the
image each guide opens with the markers the viewer would draw.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from serafim import contract
from serafim.validation import targets

log = logging.getLogger(__name__)

GOLD_DIR = targets.ROOT / "data" / "gold"
REPORT_PATH = targets.ROOT / "docs" / "markers.md"
SUMMARY_PATH = targets.ROOT / "docs" / "markers.json"
IMAGE_DIR = targets.ROOT / "docs" / "img"
GUIDES_PATH = targets.ROOT / "web" / "texts" / "targets.json"

APERTURE_PX = 1.5
RING_PX = (4.0, 7.0)  # background ring around the aperture
FOUND_SNR = 3.0
CENTROID_SNR = 10.0  # only stars this clear get their position checked
VISIBLE_FRACTION = 0.8
BIN_MAG = 0.5
MIN_IN_BIN = 5
MARGIN_MAG = 1.0
MAX_OFFSET_PX = 1.0
ISOLATION_PX = 4.0  # another star this close pulls the light: the position is not checked
COUNTERPART_PX = 0.5  # the same star in another catalog
BRIGHT_NEIGHBOUR_PX = 8.0  # a star 2 magnitudes brighter this close pulls it too
# "Nothing under it" is judged only near the catalog's own band: a red CatWISE source is
# invisible at 1 um, a hot Gaia star can be faint at 5 um.
BAND_DETECTORS = {"gaia": (1, 2, 3, 4), "catwise": (4, 5, 6), "simbad": (1, 2, 3, 4)}
IMPOSSIBLE_SIGMA = 10.0  # a sum pixel this far below the sky is broken, not sky
TARGET_PX = 6.0  # the target's own glow: checked in task A6
SATURATED_PX = 5.0  # the glow around any saturated center
EDGE_PX = RING_PX[1] + 1
MAX_BAD_FRACTION = 0.3  # of the aperture
CUT_CATALOGS = ("gaia", "catwise")  # the layers with a magnitude cut
JUDGED_CATALOGS = (*CUT_CATALOGS, "simbad")
RANDOM_SPOTS = 400
# The viewer's default cut also keeps the image readable: at most this many markers of one layer
# inside the image (Barnard has ~550 Gaia stars down to its visible limit in 146 x 146 px).
MAX_DEFAULT_MARKERS = 150
CHANCE = "random"  # the catalog name of the random places


# --- One marker on one sum (pure) ---------------------------------------------------------------


def impossible(image: np.ndarray, bad: np.ndarray, noise: float) -> np.ndarray:
    """Pixels far below the sky: no light can be negative, so they are broken, not sky."""
    sky = float(np.median(image[~bad]))
    return ~bad & (image < sky - IMPOSSIBLE_SIGMA * noise)


def sky_noise(image: np.ndarray, bad: np.ndarray) -> float:
    """Robust noise of one pixel of the sum: 1.4826 MAD of the usable pixels."""
    values = image[~bad]
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def measure(image: np.ndarray, bad: np.ndarray, x: float, y: float, noise: float) -> dict:
    """Forced aperture at (x, y) (pixel of the silver array): S/N and the light's offset."""
    ny, nx = image.shape
    out = {"usable": False, "snr": np.nan, "offset_px": np.nan}
    if not (np.isfinite(x) and np.isfinite(y)):
        return out
    ix, iy, half = int(round(x)), int(round(y)), int(RING_PX[1]) + 1
    if not (half <= ix < nx - half and half <= iy < ny - half):  # the ring fits inside
        return out
    cut = image[iy - half : iy + half + 1, ix - half : ix + half + 1]
    cut_bad = bad[iy - half : iy + half + 1, ix - half : ix + half + 1]
    rows, cols = np.mgrid[iy - half : iy + half + 1, ix - half : ix + half + 1]
    distance = np.hypot(cols - x, rows - y)
    aperture, ring = distance <= APERTURE_PX, (distance >= RING_PX[0]) & (distance <= RING_PX[1])
    if cut_bad[aperture].mean() > MAX_BAD_FRACTION or (ring & ~cut_bad).sum() < 10:
        return out
    background = float(np.median(cut[ring & ~cut_bad]))
    use = aperture & ~cut_bad
    flux = float((cut[use] - background).sum()) * aperture.sum() / use.sum()
    out.update(usable=True, snr=flux / (noise * np.sqrt(aperture.sum())))
    if out["snr"] >= FOUND_SNR:
        window = (distance <= 2.0) & ~cut_bad
        weights = np.clip(cut[window] - background, 0, None)
        if weights.sum() > 0:
            cx = float((cols[window] * weights).sum() / weights.sum())
            cy = float((rows[window] * weights).sum() / weights.sum())
            out["offset_px"] = float(np.hypot(cx - x, cy - y))
    return out


def isolated(markers: pd.DataFrame) -> np.ndarray:
    """True where no other star can pull the light of a marker: no marker of the same catalog
    within ISOLATION_PX unless 2 magnitudes fainter, none 2 magnitudes brighter within
    BRIGHT_NEIGHBOUR_PX, and no marker of another catalog within ISOLATION_PX that is not the
    same star (closer than COUNTERPART_PX)."""
    out = np.ones(len(markers), bool)
    x, y = markers["x"].to_numpy(), markers["y"].to_numpy()
    mag = markers["mag"].to_numpy(dtype=float)
    catalog = markers["catalog"].to_numpy()
    for i in range(len(markers)):
        distance = np.hypot(x - x[i], y - y[i])
        same = catalog == catalog[i]
        same[i] = False
        faint = np.nan_to_num(mag, nan=99.0) >= np.nan_to_num(mag[i], nan=-99.0) + 2
        bright = np.nan_to_num(mag, nan=99.0) <= np.nan_to_num(mag[i], nan=99.0) - 2
        crowded = same & (distance < ISOLATION_PX) & ~faint
        crowded |= same & (distance < BRIGHT_NEIGHBOUR_PX) & bright
        crowded |= ~same & (catalog != catalog[i]) & (distance < ISOLATION_PX)
        crowded &= ~((catalog != catalog[i]) & (distance < COUNTERPART_PX))
        out[i] = not crowded.any()
    return out


# --- Limits and judgement (pure) ----------------------------------------------------------------


def visible_limit(mags: np.ndarray, found: np.ndarray, chance: float = 0.0) -> float | None:
    """The faintest magnitude down to which >= VISIBLE_FRACTION of the markers are found beyond
    `chance` (the rate at random places), walking BIN_MAG bins from the brightest (bins with
    fewer than MIN_IN_BIN markers are skipped)."""
    mags, found = np.asarray(mags, float), np.asarray(found, bool)
    keep = np.isfinite(mags)
    mags, found = mags[keep], found[keep]
    if not len(mags):
        return None
    limit = None
    edges = np.arange(np.floor(mags.min() / BIN_MAG) * BIN_MAG, mags.max() + BIN_MAG, BIN_MAG)
    for low in edges:
        inside = (mags >= low) & (mags < low + BIN_MAG)
        if inside.sum() < MIN_IN_BIN:
            continue
        beyond = (found[inside].mean() - chance) / (1 - chance) if chance < 1 else 0.0
        if beyond < VISIBLE_FRACTION:
            break
        limit = float(low + BIN_MAG)
    return limit


def limits(measured: pd.DataFrame) -> dict[str, dict[int, float]]:
    """{catalog: {detector: visible limit}} from the season sums (season >= 1) of one target."""
    out: dict[str, dict[int, float]] = {}
    judged = measured[measured["usable"] & measured["judged"] & (measured["season"] >= 1)]
    spots = judged[judged["catalog"] == CHANCE]
    chance = spots.groupby("detector")["found"].mean().to_dict()
    for (catalog, detector), part in judged.groupby(["catalog", "detector"]):
        if catalog not in CUT_CATALOGS:
            continue
        limit = visible_limit(
            part["mag"].to_numpy(), part["found"].to_numpy(), chance.get(detector, 0.0)
        )
        if limit is not None:
            out.setdefault(catalog, {})[int(detector)] = limit
    return out


def inside(markers: pd.DataFrame, shape: tuple[int, int]) -> pd.Series:
    """Markers that fall on the image (nx, ny): the viewer draws only these."""
    nx, ny = shape
    return markers["x"].between(-0.5, nx - 0.5) & markers["y"].between(-0.5, ny - 0.5)


def default_cuts(
    measured: pd.DataFrame, cuts: dict[str, dict[int, float]], shape: tuple[int, int]
) -> dict[str, float]:
    """{catalog: magnitude}: the viewer's default cut of each layer. The brightest visible limit
    of any detector, made brighter (in BIN_MAG steps) until at most MAX_DEFAULT_MARKERS markers
    of a season fall on the image."""  # fmt: skip
    out = {}
    for catalog, by_detector in cuts.items():
        if not by_detector:
            continue
        limit = min(by_detector.values())
        rows = measured[(measured["catalog"] == catalog) & (measured["detector"] == 1)]
        per_season = rows[inside(rows, shape)].groupby("season")["mag"]
        while limit > 0 and any((m <= limit).sum() > MAX_DEFAULT_MARKERS for _, m in per_season):
            limit -= BIN_MAG
        out[catalog] = float(limit)
    return out


def out_of_place(measured: pd.DataFrame, cuts: dict[str, dict[int, float]]) -> pd.DataFrame:
    """Markers the viewer would draw (within the cut) that do not sit on their star."""
    rows = measured[measured["usable"] & measured["judged"] & (measured["catalog"] != CHANCE)]
    rows = rows.copy()
    if rows.empty:
        return rows.assign(reason=pd.Series(dtype=str))

    def cut(row) -> float:
        catalog = "gaia" if row["catalog"] == "simbad" else row["catalog"]  # V close to G
        return cuts.get(catalog, {}).get(int(row["detector"]), -np.inf)

    rows["cut"] = rows.apply(cut, axis=1)
    bright = rows["mag"].notna() & (rows["mag"] <= rows["cut"] - MARGIN_MAG)
    in_band = [d in BAND_DETECTORS.get(c, ()) for c, d in zip(rows["catalog"], rows["detector"],
                                                              strict=True)]  # fmt: skip
    empty = bright & ~rows["found"] & np.array(in_band, bool)
    shifted = (
        (rows["mag"].isna() | (rows["mag"] <= rows["cut"]))
        & rows["isolated"]
        & (rows["snr"] >= CENTROID_SNR)
        & (rows["offset_px"] > MAX_OFFSET_PX)
    )
    rows["reason"] = np.where(empty, "nothing under it", np.where(shifted, "beside its star", ""))
    return rows[empty | shifted].reset_index(drop=True)


def projection_pairs(wcs) -> list[dict]:
    """Reference positions for the viewer's projection: the center and four points near the
    corners. x, y: silver pixel (row 0 = south); png_x, png_y: the gold PNG pixel (north up)."""
    nx, ny = wcs.pixel_shape
    pairs = []
    for x, y in ((nx / 2 - 0.5, ny / 2 - 0.5), (10, 10), (nx - 11, 10), (10, ny - 11),
                 (nx - 11, ny - 11)):  # fmt: skip
        sky = wcs.pixel_to_world(x, y)
        pairs.append({"ra": round(float(sky.ra.deg), 7), "dec": round(float(sky.dec.deg), 7),
                      "x": float(x), "y": float(y), "png_x": float(x),
                      "png_y": float(ny - 1 - y)})  # fmt: skip
    return pairs


# --- One target (silver and gold on disk) -------------------------------------------------------


def measure_target(target_id: str, gold: dict[str, pd.DataFrame], root: Path | None = None):
    """One row per marker, season and detector with what the silver sum has under it."""
    from serafim.silver.grid import wcs_from_json
    from serafim.silver.store import load_sum

    grid = wcs_from_json(gold["targets"]["wcs_json"].iloc[0])
    layers = gold["catalogs"]
    sums = gold["sums"]
    rows, broken_rows = [], []
    for cell in sums[sums["detector"] >= 1].itertuples():
        markers = layers[layers["season"] == cell.season].reset_index(drop=True)
        if markers.empty:
            continue
        arrays = load_sum(target_id, int(cell.detector), int(cell.season), root)
        image = np.asarray(arrays["image"], float)
        bad = np.asarray(arrays["mask"], bool) | ~np.isfinite(image)
        saturated = np.asarray(arrays.get("n_saturated", 0)) > 0
        noise = sky_noise(image, bad | saturated)
        broken = impossible(image, bad | saturated, noise)
        if broken.any():
            log.warning("%s d%d s%d: %d sum pixels far below the sky (left out)", target_id,
                        cell.detector, cell.season, int(broken.sum()))  # fmt: skip
            broken_rows.append({"detector": int(cell.detector), "season": int(cell.season),
                                "pixels": int(broken.sum())})  # fmt: skip
        bad |= broken
        xs, ys = grid.world_to_pixel_values(markers["ra"].to_numpy(), markers["dec"].to_numpy())
        markers = markers.assign(x=xs, y=ys)
        random = np.random.default_rng(int(cell.detector) * 100 + int(cell.season))
        spots = pd.DataFrame({
            "catalog": CHANCE, "name": "", "mag": np.nan,
            "x": random.uniform(0, grid.pixel_shape[0] - 1, RANDOM_SPOTS),
            "y": random.uniform(0, grid.pixel_shape[1] - 1, RANDOM_SPOTS),
        })  # fmt: skip
        markers = pd.concat([markers, spots], ignore_index=True)
        markers["isolated"] = isolated(markers)
        sat_y, sat_x = np.nonzero(saturated)
        center = (np.array(grid.pixel_shape) - 1) / 2
        for m in markers.itertuples():
            near_saturated = len(sat_x) and np.hypot(sat_x - m.x, sat_y - m.y).min() < SATURATED_PX
            near_target = np.hypot(m.x - center[0], m.y - center[1]) < TARGET_PX
            found = measure(image, bad | saturated, m.x, m.y, noise)
            rows.append({
                "catalog": m.catalog, "name": m.name, "mag": m.mag, "season": int(cell.season),
                "detector": int(cell.detector), "x": m.x, "y": m.y, "isolated": m.isolated,
                "judged": m.catalog in (*JUDGED_CATALOGS, CHANCE) and not near_saturated
                and not near_target,
                **found, "found": bool(found["snr"] >= FOUND_SNR),
            })  # fmt: skip
    return pd.DataFrame(rows), grid, pd.DataFrame(broken_rows, columns=["detector", "season",
                                                                      "pixels"])  # fmt: skip


def guide_view(guide: dict) -> tuple[str, int, int]:
    """(gold PNG path, detector, season of the markers) of the image a guide opens."""
    how = guide["how_to_look"]
    detector, seasons = how["detectors"][0], how["seasons"]
    if how["mode"] == "difference":
        a, b = seasons[0], seasons[-1]
        return contract.difference_png(guide["id"], detector, a, b), detector, b
    if how["mode"] == "all_seasons":
        return contract.sum_png(guide["id"], detector, 0), detector, 0
    return contract.sum_png(guide["id"], detector, seasons[0]), detector, seasons[0]


def write_plot(gold_dir: Path, guide: dict, measured, defaults, wrong, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    png, detector, season = guide_view(guide)
    image = mpimg.imread(gold_dir / png)
    ny = image.shape[0]
    here = measured[(measured["detector"] == detector) & (measured["season"] == season)]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image, interpolation="nearest")
    styles = {"gaia": ("o", "#7fdbff"), "catwise": ("s", "#2ecc40"), "simbad": ("D", "#ffdc00")}
    for catalog, (marker, color) in styles.items():
        part = here[(here["catalog"] == catalog) & inside(here, (image.shape[1], ny))]
        limit = defaults.get(catalog)
        if limit is not None:
            part = part[part["mag"] <= limit]
        ax.scatter(part["x"], ny - 1 - part["y"], s=36, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.8, label=f"{catalog} ({len(part)})")  # fmt: skip
    bad = wrong[(wrong["detector"] == detector) & (wrong["season"] == season)]
    if len(bad):
        ax.scatter(bad["x"], ny - 1 - bad["y"], s=60, marker="x", color="red",
                   label=f"out of place ({len(bad)})")  # fmt: skip
    ax.set_title(f"{guide['name']}: {png.split('/')[-1]} with the default markers", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- Report -------------------------------------------------------------------------------------


def report(results: dict) -> str:
    lines = [
        "# Catalog markers on the real images (task A8)",
        "",
        "Generated by `python -m serafim.validation.markers` from the real gold and its silver",
        "sums. Each marker of `catalogs.parquet` is put on the grid with `wcs_json` (as the viewer",
        f"does) and a {APERTURE_PX} px forced aperture on the sum says whether a source is there",
        f"(S/N ≥ {FOUND_SNR:g}). **Visible limit**: the magnitude down to which "
        f"{VISIBLE_FRACTION:.0%} of the markers have",
        f"a source under them beyond chance (the same measure at {RANDOM_SPOTS} random places of",
        "each sum), the default cut of each layer in the viewer. **Out of place**: a",
        f"marker {MARGIN_MAG:g} mag brighter than that limit with nothing under it, or an isolated "
        f"star more than {MAX_OFFSET_PX:g} px",
        "from its marker. Markers near the target, a saturated center or the edge are not judged.",
        "",
        "| Target | Gaia G limit (D1–D6) | CatWISE W2 limit (D1–D6) | Judged | Out of place |",
        "| --- | --- | --- | --- | --- |",
    ]

    def row_of(cuts: dict[int, float]) -> str:
        return " / ".join(f"{cuts[d]:g}" if d in cuts else "—" for d in range(1, 7))

    for name, measured, cuts, wrong, _ in results.values():
        markers = measured[measured["catalog"] != CHANCE]
        judged = int((markers["usable"] & markers["judged"]).sum())
        lines.append(
            f"| {name} | {row_of(cuts.get('gaia', {}))} | {row_of(cuts.get('catwise', {}))} | "
            f"{judged} | {len(wrong)} |"
        )
    for target_id, (name, measured, _, wrong, broken) in results.items():
        lines += ["", f"## {name} (`{target_id}`)", ""]
        if len(broken):
            cells = ", ".join(
                f"D{r.detector} s{r.season} ({r.pixels})" for r in broken.itertuples()
            )
            lines += [f"Sum pixels far below the sky ({IMPOSSIBLE_SIGMA:g} sigma or more, left "
                      f"out; a silver problem): {cells}.", ""]  # fmt: skip
        found = measured[measured["catalog"] == "skybot"]
        if len(found):
            lines.append(f"Asteroids (SkyBoT, not judged): {len(found)} markers, "
                         f"{int(found['found'].sum())} with light under them.")  # fmt: skip
            lines.append("")
        if wrong.empty:
            lines.append("No marker out of place.")
        else:
            lines += ["| Catalog | Name | Mag | Season | Detector | S/N | Offset (px) | Why |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
            for r in wrong.sort_values(["catalog", "mag"]).itertuples():
                lines.append(
                    f"| {r.catalog} | {r.name} | {r.mag:.1f} | {r.season} | {r.detector} | "
                    f"{r.snr:.1f} | {'' if np.isnan(r.offset_px) else f'{r.offset_px:.2f}'} | "
                    f"{r.reason} |"
                )
        lines += ["", f"![{name}](img/markers_{target_id}.png)"]
    return "\n".join(lines) + "\n"


def summary(results: dict, grids: dict) -> dict:
    out = {}
    for target_id, (_, measured, cuts, wrong, broken) in results.items():
        out[target_id] = {
            "default_max_mag": default_cuts(measured, cuts, grids[target_id].pixel_shape),
            "visible_max_mag": {
                catalog: {str(d): v for d, v in sorted(by_detector.items())}
                for catalog, by_detector in cuts.items()
            },
            "out_of_place": int(len(wrong)),
            "broken_sum_pixels": broken.to_dict("records"),
            "projection_check": projection_pairs(grids[target_id]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    from serafim.validation import verification

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.markers")
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    parser.add_argument("--target", nargs="+", help="Target ids (default: every one in gold).")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tables, _ = verification.read_gold(args.gold)
    written = json.loads(GUIDES_PATH.read_text(encoding="utf-8"))
    guides = {g["id"]: g for g in written["targets"]}
    results, grids = {}, {}
    for target_id in sorted(tables["targets"]["target_id"]):
        if args.target and target_id not in args.target:
            continue
        gold = verification.of_target(tables, target_id)
        measured, grids[target_id], broken = measure_target(target_id, gold)
        cuts = limits(measured)
        wrong = out_of_place(measured, cuts)
        name = gold["targets"]["name"].iloc[0]
        results[target_id] = (name, measured, cuts, wrong, broken)
        if target_id in guides:
            defaults = default_cuts(measured, cuts, grids[target_id].pixel_shape)
            write_plot(args.gold, guides[target_id], measured, defaults, wrong,
                       args.images / f"markers_{target_id}.png")  # fmt: skip
        log.info("%s: limits %s, %d out of place", target_id, cuts, len(wrong))

    args.report.write_text(report(results), encoding="utf-8", newline="\n")
    args.summary.write_text(json.dumps(summary(results, grids), indent=2) + "\n",
                            encoding="utf-8", newline="\n")  # fmt: skip
    log.info("wrote %s and %s", args.report, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
