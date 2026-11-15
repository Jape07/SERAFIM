"""Motion dipoles in the season differences, found by a classic rule (task A9, stage 1).

    python -m serafim.validation.dipoles [--gold data/gold] [--target X ...]

Something that moves between two seasons leaves a dipole in their difference (season b minus a):
a negative blob where it was and a positive one where it went. This is how Backyard Worlds
volunteers find nearby brown dwarfs, and what the CNN of stage 2 learns. The rule:

1. Significance: the silver difference over its measured noise (1.4826 MAD: the propagated error
   underestimates it up to 7 times in crowded fields). Masked and saturated pixels left out.
2. Blobs: `sep.extract` at THRESHOLD_SIGMA, MIN_AREA_PX pixels or more, on the positive and on the
   negative image.
3. Dipole: a positive and a negative blob closer than MAX_SEPARATION_PX with fluxes within
   FLUX_RATIO of each other. Next to a saturated center the core is masked, so only the halo on
   each side is seen: those lobes may be up to MAX_SEPARATION_SATURATED_PX apart and less even
   (FLUX_RATIO_SATURATED).
4. Status, with the catalog layers of gold (each source at the date of each season):
   - Discarded: a lobe at the edge or on masked pixels; or the dipole sits on a catalog source
     that does not move (a residual: each season sees a star at another wavelength, so its PSF
     and brightness change a little).
   - Known: the negative lobe is where a catalog source was in season a and the positive one
     where it was in season b (the target itself is in the SIMBAD layer, with its exact motion).
   - Unidentified candidate: anything else. Never "new object": a person checks it.

Writes docs/dipoles.md, docs/dipoles.json and docs/img/dipoles.png (stamps of the Known and the
Unidentified). The check of the plan: the rule finds Barnard's Star in its last-minus-first
differences without being told where it is, and RW Aurigae and R Draconis (they only change
brightness) give no dipole.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from scipy import ndimage

from serafim.validation import targets

log = logging.getLogger(__name__)

GOLD_DIR = targets.ROOT / "data" / "gold"
REPORT_PATH = targets.ROOT / "docs" / "dipoles.md"
SUMMARY_PATH = targets.ROOT / "docs" / "dipoles.json"
IMAGE_PATH = targets.ROOT / "docs" / "img" / "dipoles.png"
PIXEL_ARCSEC = targets.PIXEL_SCALE_ARCSEC

THRESHOLD_SIGMA = 5.0
MIN_AREA_PX = 3
MAX_SEPARATION_PX = 3.0
FLUX_RATIO = (0.5, 2.0)
SATURATED_NEAR_PX = 2.0  # a lobe this close to a saturated pixel is a halo lobe
MAX_SEPARATION_SATURATED_PX = 6.0
FLUX_RATIO_SATURATED = (1 / 3, 3.0)
EDGE_PX = 5.0
MASK_NEAR_PX = 1.5
MATCH_PX = 2.0  # a lobe within this of a catalog position matches it
MATCH_SATURATED_PX = 4.0  # halo lobes sit around the masked core
MOVING_PX = 0.3  # a catalog source that moves less than this between the seasons is fixed
RESIDUAL_PX = 1.5  # a dipole centered this close to a fixed source is its residual
RESIDUAL_SATURATED_PX = 4.0  # halo lobes around a saturated star that does not move
CATALOG_ORDER = {"simbad": 0, "gaia": 1, "catwise": 2}  # who names a Known dipole first
TARGET_PX = 8.0  # lobes this close to the grid center belong to the target (for the checks)

KNOWN, DISCARDED, CANDIDATE = "Known", "Discarded", "Unidentified candidate"
SYMBOL = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
_DIFFERENCE_NAME = re.compile(r"d(\d)_s(\d+)-s(\d+)\.npz$")


# --- Blobs and dipoles in one difference (pure) -------------------------------------------------


def significance(difference: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, float]:
    """(difference / measured noise with bad pixels at 0, the noise)."""
    values = difference[~bad]
    noise = float(1.4826 * np.median(np.abs(values - np.median(values))))
    return np.where(bad, 0.0, difference / noise).astype(np.float32), noise


def blobs(sigma: np.ndarray, bad: np.ndarray) -> pd.DataFrame:
    """Positive and negative blobs: sign, x, y, flux (sum, in sigma units), peak, npix."""
    rows = []
    for sign in (1, -1):
        found = sep.extract(np.ascontiguousarray(sign * sigma), THRESHOLD_SIGMA,
                            mask=np.ascontiguousarray(bad), minarea=MIN_AREA_PX)  # fmt: skip
        for blob in found:
            rows.append({"sign": sign, "x": float(blob["x"]), "y": float(blob["y"]),
                         "flux": float(blob["flux"]), "peak": float(blob["peak"]),
                         "npix": int(blob["npix"])})  # fmt: skip
    return pd.DataFrame(rows, columns=["sign", "x", "y", "flux", "peak", "npix"])


def core_of(saturated: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """(radius from its area, center x, center y) of the saturated core next to (x, y);
    (0, nan, nan) if none is within SATURATED_NEAR_PX."""
    cores, count = ndimage.label(np.asarray(saturated, bool))
    if not count:
        return 0.0, np.nan, np.nan
    ys, xs = np.nonzero(cores)
    distance = np.hypot(xs - x, ys - y)
    if distance.min() > SATURATED_NEAR_PX:
        return 0.0, np.nan, np.nan
    label = cores[ys[distance.argmin()], xs[distance.argmin()]]
    cy, cx = ndimage.center_of_mass(cores == label)
    return float(np.sqrt((cores == label).sum() / np.pi)), float(cx), float(cy)


def core_radius(saturated: np.ndarray, x: float, y: float) -> float:
    return core_of(saturated, x, y)[0]


def _near(mask: np.ndarray, x: float, y: float, radius: float) -> bool:
    ys, xs = np.nonzero(mask)
    return bool(len(xs)) and float(np.hypot(xs - x, ys - y).min()) <= radius


def merge_halo_blobs(found: pd.DataFrame, saturated: np.ndarray) -> pd.DataFrame:
    """Same-sign blobs next to the same saturated core are one lobe split by the masked core
    (Barnard's Star in D6): one blob each, flux added, flux-weighted position."""
    if found.empty or not np.asarray(saturated).any():
        return found
    cores, _ = ndimage.label(np.asarray(saturated, bool))
    grown = ndimage.grey_dilation(cores, size=int(2 * SATURATED_NEAR_PX) + 1)
    ny, nx = cores.shape
    xi = np.clip(np.round(found["x"]).astype(int), 0, nx - 1)
    yi = np.clip(np.round(found["y"]).astype(int), 0, ny - 1)
    found = found.assign(core=grown[yi, xi])
    alone = found[found["core"] == 0]
    merged = []
    for (sign, _core), part in found[found["core"] > 0].groupby(["sign", "core"]):
        weight = part["flux"].abs()
        merged.append({"sign": sign, "x": float((part["x"] * weight).sum() / weight.sum()),
                       "y": float((part["y"] * weight).sum() / weight.sum()),
                       "flux": float(part["flux"].sum()), "peak": float(part["peak"].max()),
                       "npix": int(part["npix"].sum())})  # fmt: skip
    out = pd.concat([alone.drop(columns="core"), pd.DataFrame(merged)], ignore_index=True)
    return out[["sign", "x", "y", "flux", "peak", "npix"]]


def pair(found: pd.DataFrame, saturated: np.ndarray) -> pd.DataFrame:
    """Dipoles: each negative blob with its closest positive one, each blob used once."""
    found = merge_halo_blobs(found, saturated).reset_index(drop=True)
    negatives, positives = found[found["sign"] < 0], found[found["sign"] > 0]
    options = []
    for n in negatives.itertuples():
        for p in positives.itertuples():
            distance = float(np.hypot(p.x - n.x, p.y - n.y))
            halo = _near(saturated, n.x, n.y, SATURATED_NEAR_PX) or _near(
                saturated, p.x, p.y, SATURATED_NEAR_PX
            )
            core = (
                max(core_radius(saturated, n.x, n.y), core_radius(saturated, p.x, p.y))
                if halo
                else 0.0
            )
            limit = max(MAX_SEPARATION_SATURATED_PX, 2 * core + 2) if halo else MAX_SEPARATION_PX
            low, high = FLUX_RATIO_SATURATED if halo else FLUX_RATIO
            ratio = p.flux / n.flux if n.flux > 0 else np.inf
            if distance <= limit and low <= ratio <= high:
                options.append((distance, n.Index, p.Index, ratio, halo))
    used, rows = set(), []
    for distance, ni, pi, ratio, halo in sorted(options):
        if ni in used or pi in used:
            continue
        used |= {ni, pi}
        n, p = found.loc[ni], found.loc[pi]
        core = max(core_of(saturated, n["x"], n["y"]), core_of(saturated, p["x"], p["y"]),
                   key=lambda c: c[0])  # fmt: skip
        rows.append({"x_from": n["x"], "y_from": n["y"], "x_to": p["x"], "y_to": p["y"],
                     "separation_px": distance, "flux_ratio": ratio, "halo": halo,
                     "core_px": core[0] if halo else 0.0, "core_x": core[1], "core_y": core[2],
                     "snr": float(min(n["peak"], p["peak"]))})  # fmt: skip
    columns = ["x_from", "y_from", "x_to", "y_to", "separation_px", "flux_ratio", "halo",
               "core_px", "core_x", "core_y", "snr"]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)


def find_dipoles(difference: np.ndarray, mask: np.ndarray, saturated: np.ndarray) -> pd.DataFrame:
    """The dipoles of one difference, before the catalogs (pixel positions of the silver array)."""
    difference = np.asarray(difference, float)
    bad = np.asarray(mask, bool) | np.asarray(saturated, bool) | ~np.isfinite(difference)
    sigma, _ = significance(difference, bad)
    return pair(blobs(sigma, bad), np.asarray(saturated, bool))


# --- Status (pure: dipoles and catalog positions in) --------------------------------------------


def catalog_moves(layers: pd.DataFrame, wcs, season_a: int, season_b: int) -> pd.DataFrame:
    """Every catalog source present in both seasons: name, catalog and its pixel at a and b."""
    a = layers[layers["season"] == season_a].drop_duplicates(["catalog", "name"])
    b = layers[layers["season"] == season_b].drop_duplicates(["catalog", "name"])
    both = a.merge(b, on=["catalog", "name"], suffixes=("_a", "_b"))
    if both.empty:
        return pd.DataFrame(columns=["catalog", "name", "mag", "xa", "ya", "xb", "yb",
                                     "moved_px"])  # fmt: skip
    xa, ya = wcs.world_to_pixel_values(both["ra_a"].to_numpy(), both["dec_a"].to_numpy())
    xb, yb = wcs.world_to_pixel_values(both["ra_b"].to_numpy(), both["dec_b"].to_numpy())
    return pd.DataFrame({"catalog": both["catalog"], "name": both["name"], "mag": both["mag_a"],
                         "xa": xa, "ya": ya, "xb": xb, "yb": yb,
                         "moved_px": np.hypot(xb - xa, yb - ya)})  # fmt: skip


def status(dipole: dict, sources: pd.DataFrame, mask: np.ndarray, shape: tuple[int, int]) -> tuple:
    """(status, reason) of one dipole. `mask`: masked pixels that are not a saturated core."""
    ny, nx = shape
    for x, y in ((dipole["x_from"], dipole["y_from"]), (dipole["x_to"], dipole["y_to"])):
        if min(x, y, nx - 1 - x, ny - 1 - y) < EDGE_PX:
            return DISCARDED, "at the edge"
        if _near(mask, x, y, MASK_NEAR_PX):
            return DISCARDED, "on masked pixels"
    core = dipole.get("core_px", 0.0)
    match = max(MATCH_SATURATED_PX, core + SATURATED_NEAR_PX) if dipole["halo"] else MATCH_PX
    moving = sources[sources["moved_px"] >= MOVING_PX]
    if len(moving):
        start = np.hypot(moving["xa"] - dipole["x_from"], moving["ya"] - dipole["y_from"])
        end = np.hypot(moving["xb"] - dipole["x_to"], moving["yb"] - dipole["y_to"])
        hits = moving[(start <= match) & (end <= match)].assign(
            order=lambda t: t["catalog"].map(CATALOG_ORDER).fillna(9),
            brightness=lambda t: t["mag"].fillna(99.0),
        )
        if len(hits):  # the star itself before its planets (SIMBAD lists Barnard's Star b, c...)
            best = hits.sort_values(["order", "brightness"]).iloc[0]
            return KNOWN, f"{best['name']} ({best['catalog']})"
    fixed = sources[sources["moved_px"] < MOVING_PX]
    if len(fixed):
        cx = (dipole["x_from"] + dipole["x_to"]) / 2
        cy = (dipole["y_from"] + dipole["y_to"]) / 2
        distance = np.hypot(fixed["xa"] - cx, fixed["ya"] - cy)
        residual = (
            max(RESIDUAL_SATURATED_PX, core + SATURATED_NEAR_PX)
            if dipole["halo"]
            else (RESIDUAL_PX)
        )
        if distance.min() <= residual:
            nearest = fixed.loc[distance.idxmin()]
            return (
                DISCARDED,
                f"residual of {nearest['name']} ({nearest['catalog']}), which does not move",
            )
    if dipole["halo"] and len(sources):  # the glare of a saturated star that does not match
        distance = np.hypot(sources["xa"] - dipole["core_x"], sources["ya"] - dipole["core_y"])
        if distance.min() <= core + SATURATED_NEAR_PX:
            star = sources.loc[distance.idxmin()]
            return DISCARDED, f"around the saturated core of {star['name']} ({star['catalog']})"
    return CANDIDATE, "No moving source of Gaia, CatWISE or SIMBAD there"


# --- Disk -------------------------------------------------------------------------------------


def difference_cells(target_id: str, root: Path | None = None) -> list[tuple[int, int, int]]:
    from serafim.silver.store import silver_dir

    folder = silver_dir(root) / target_id / "differences"
    cells = []
    for path in sorted(folder.glob("d*_s*-s*.npz")):
        found = _DIFFERENCE_NAME.search(path.name)
        if found:
            cells.append(tuple(int(v) for v in found.groups()))
    return cells


def target_dipoles(target_id: str, layers: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Every dipole of every difference of one target, with its status."""
    from serafim.silver.store import load_difference, load_sum

    rows = []
    for detector, a, b in difference_cells(target_id, root):
        d = load_difference(target_id, detector, a, b, root)
        wcs = d["wcs"]
        mask = np.asarray(d["mask"], bool)
        saturated = np.asarray(d["saturated"], bool)
        found = find_dipoles(d["difference"], mask, saturated)
        sources = catalog_moves(layers, wcs, a, b)
        mjd_a = float(load_sum(target_id, detector, a, root)["mean_mjd"])
        mjd_b = float(load_sum(target_id, detector, b, root)["mean_mjd"])
        years = (mjd_b - mjd_a) / 365.25
        for dipole in found.to_dict("records"):
            state, reason = status(dipole, sources, mask & ~saturated, mask.shape)
            start = wcs.pixel_to_world(dipole["x_from"], dipole["y_from"])
            end = wcs.pixel_to_world(dipole["x_to"], dipole["y_to"])
            rows.append({
                "target_id": target_id, "detector": detector, "season_a": a, "season_b": b,
                **dipole, "ra_from": float(start.ra.deg), "dec_from": float(start.dec.deg),
                "ra_to": float(end.ra.deg), "dec_to": float(end.dec.deg),
                "years": years,
                "status": state, "reason": reason,
            })  # fmt: skip
        log.info("%s d%d s%d-s%d: %d dipoles", target_id, detector, a, b, len(found))
    return pd.DataFrame(rows)


def last_minus_first(target_id: str, root: Path | None = None) -> list[tuple[int, int, int]]:
    """(detector, first season, last season) differences of a target."""
    cells = difference_cells(target_id, root)
    out = []
    for detector in sorted({c[0] for c in cells}):
        mine = [c for c in cells if c[0] == detector]
        first = min(c[1] for c in mine)
        last = max(c[2] for c in mine)
        if (detector, first, last) in mine:
            out.append((detector, first, last))
    return out


# --- Report -------------------------------------------------------------------------------------


def at_target(rows: pd.DataFrame, shape: tuple[int, int] = (146, 146)) -> pd.Series:
    """Dipoles whose lobes are near the grid center, where the target is (it moves < 3 px)."""
    cx, cy = (shape[1] - 1) / 2, (shape[0] - 1) / 2
    return (np.hypot(rows["x_to"] - cx, rows["y_to"] - cy) < TARGET_PX) & (
        np.hypot(rows["x_from"] - cx, rows["y_from"] - cy) < TARGET_PX
    )


def checks(found: dict[str, pd.DataFrame], names: dict[str, str], root: Path | None = None) -> list:
    """The checks of the plan, as (state, text): "pass", "warn" (a known limit) or "fail"."""
    out = []
    for target_id in ("barnard", "eps_indi_b", "wise_0855"):
        if target_id not in found:
            continue
        rows = found[target_id]
        cells = last_minus_first(target_id, root)
        known = rows[(rows["status"] == KNOWN) & at_target(rows)] if len(rows) else rows
        hit = {(r.detector, r.season_a, r.season_b) for r in known.itertuples()}
        seen = [c for c in cells if c in hit]
        text = (
            f"{names[target_id]}: found as Known in {len(seen)} of {len(cells)} last-minus-first "
        )
        text += "differences" + (
            f" (detectors {', '.join(str(c[0]) for c in seen)})" if seen else ""
        )
        if target_id == "wise_0855":  # S/N ~7 with every season added: too faint for 5 sigma
            state = "pass" if seen else "warn"
            text += "" if seen else (": too faint for the rule (S/N 5-7 in one season); the target "
                                     "of the CNN")  # fmt: skip
        else:
            state = "pass" if len(seen) * 2 >= len(cells) else "fail"
        out.append((state, text))
    for target_id in ("rw_aur", "r_dra"):
        if target_id not in found:
            continue
        rows = found[target_id]
        center = rows[at_target(rows) & (rows["status"] != DISCARDED)] if len(rows) else rows
        out.append(("pass" if center.empty else "fail",
                    f"{names[target_id]} (only changes brightness): {len(center)} Known or "
                    "unidentified dipoles at the target"))  # fmt: skip
    return out


def report(found: dict[str, pd.DataFrame], names: dict[str, str], tests: list) -> str:
    lines = [
        "# Motion dipoles in the season differences (task A9, stage 1: the rule)",
        "",
        "Generated by `python -m serafim.validation.dipoles` from the silver differences and",
        "the catalog layers of gold. A dipole is a negative blob (where something was in season a)",
        f"next to a positive one (where it went in season b): blobs at {THRESHOLD_SIGMA:g} sigma "
        f"of the measured noise, at most {MAX_SEPARATION_PX:g} px apart (farther around a",
        "saturated core, whose two halo lobes are what shows) with similar fluxes. Known: it",
        "matches a catalog source that moves. Discarded: edge, masked pixels, the residual of a",
        "star that does not move or the glare of a saturated one. Unidentified candidate: anything",
        "else, for a person to check. Lobes apart is not the shift: for a shift smaller than the",
        "PSF the lobes stay ~2 px apart.",
        "",
        "## Checks",
        "",
        *[f"- {SYMBOL[state]} {text}" for state, text in tests],
        "",
        "## Per target",
        "",
        "| Target | Differences | Dipoles | Known | Discarded | Unidentified candidates |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target_id, rows in found.items():
        count = rows["status"].value_counts() if len(rows) else pd.Series(dtype=int)
        cells = len(difference_cells(target_id))
        lines.append(f"| {names[target_id]} | {cells} | {len(rows)} | {count.get(KNOWN, 0)} | "
                     f"{count.get(DISCARDED, 0)} | {count.get(CANDIDATE, 0)} |")  # fmt: skip
    every = pd.concat([r for r in found.values() if len(r)], ignore_index=True)
    for title, state in (("Known", KNOWN), ("Unidentified candidates", CANDIDATE)):
        part = every[every["status"] == state] if len(every) else every
        lines += ["", f"## {title}", ""]
        if part.empty:
            lines.append("None.")
            continue
        lines += ["| Target | Detector | Seasons | From (x, y) | Lobes apart (px) | S/N | Reason |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        for r in part.sort_values(["target_id", "detector", "season_a", "season_b"]).itertuples():
            lines.append(
                f"| {names[r.target_id]} | {r.detector} | {r.season_b} − {r.season_a} | "
                f"({r.x_from:.1f}, {r.y_from:.1f}) | {r.separation_px:.2f}"
                f"{' (halo)' if r.halo else ''} | {r.snr:.0f} | {r.reason} |"
            )
    discarded = every[every["status"] == DISCARDED] if len(every) else every
    if len(discarded):
        why = discarded["reason"].str.replace(r"residual of .*", "residual of a star that does "
                                              "not move", regex=True).value_counts()  # fmt: skip
        lines += ["", "## Discarded, by reason", ""]
        lines += [f"- {reason}: {count}" for reason, count in why.items()]
    lines += ["", "![Known dipoles and unidentified candidates](img/dipoles.png)"]
    return "\n".join(lines) + "\n"


def write_stamps(found: pd.DataFrame, names: dict[str, str], path: Path, limit: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from serafim.silver.store import load_difference

    known = found[found["status"] == KNOWN].sort_values("snr", ascending=False)
    shown = pd.concat([known.drop_duplicates("reason").head(limit // 2),
                       found[found["status"] == CANDIDATE].head(limit // 2)])  # fmt: skip
    if shown.empty:
        return
    columns = min(len(shown), 4)
    lines = int(np.ceil(len(shown) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(3 * columns, 3.2 * lines), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, shown.itertuples(), strict=False):
        d = load_difference(r.target_id, r.detector, r.season_a, r.season_b)
        cx, cy = int(round((r.x_from + r.x_to) / 2)), int(round((r.y_from + r.y_to) / 2))
        half = 10
        stamp = np.asarray(d["difference"], float)[max(cy - half, 0) : cy + half + 1,
                                                    max(cx - half, 0) : cx + half + 1]  # fmt: skip
        limit_value = np.nanpercentile(np.abs(stamp), 99) or 1.0
        ax.imshow(stamp, origin="lower", cmap="RdBu_r", vmin=-limit_value, vmax=limit_value)
        ox, oy = max(cx - half, 0), max(cy - half, 0)
        ax.annotate("", xy=(r.x_to - ox, r.y_to - oy), xytext=(r.x_from - ox, r.y_from - oy),
                    arrowprops={"arrowstyle": "->", "color": "black"})  # fmt: skip
        who = r.reason.split(" (")[0] if r.status == KNOWN else r.status
        ax.set_title(f"{names[r.target_id]} field, D{r.detector} s{r.season_b}−s{r.season_a}\n"
                     f"{who}", fontsize=8)  # fmt: skip
    fig.suptitle("Dipoles: blue = season a, red = season b, arrow = motion (north up)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from serafim.validation import verification

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.dipoles")
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    parser.add_argument("--target", nargs="+", help="Target ids (default: every one in gold).")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tables, _ = verification.read_gold(args.gold)
    names = dict(zip(tables["targets"]["target_id"], tables["targets"]["name"], strict=True))
    found = {}
    for target_id in sorted(names):
        if args.target and target_id not in args.target:
            continue
        layers = tables["catalogs"][tables["catalogs"]["target_id"] == target_id]
        found[target_id] = target_dipoles(target_id, layers)
    tests = checks(found, names)
    args.report.write_text(report(found, names, tests), encoding="utf-8", newline="\n")
    every = pd.concat([r for r in found.values() if len(r)], ignore_index=True)
    keep = ["target_id", "detector", "season_a", "season_b", "ra_from", "dec_from", "ra_to",
            "dec_to", "separation_px", "halo", "snr", "status", "reason"]  # fmt: skip
    summary = {
        "checks": [{"state": state, "text": text} for state, text in tests],
        "dipoles": json.loads(
            every[every["status"] != DISCARDED][keep].round(6).to_json(orient="records")
        )
        if len(every)
        else [],  # fmt: skip
        "discarded": int((every["status"] == DISCARDED).sum()) if len(every) else 0,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_stamps(every, names, IMAGE_PATH)
    for state, text in tests:
        log.info("%s %s", state.upper(), text)
    return 1 if any(state == "fail" for state, _ in tests) else 0


if __name__ == "__main__":
    raise SystemExit(main())
