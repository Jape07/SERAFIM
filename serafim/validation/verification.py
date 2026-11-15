"""Verification of the five challenge targets on the real gold (task A6).

    python -m serafim.validation.verification [--gold data/gold] [--target X ...]

validate_gold() (D2) says that gold follows the contract. This says whether what gold shows is
right, target by target, as the plan asks: is the target seen, in which detectors, what the
flags say, and whether what SPHEREx measured agrees with the catalogs. It reads:

- the gold the site publishes (`python -m serafim gold`, task D8);
- the silver sums behind it, to know where the target is detected (S/N at its catalog spot);
- config/targets.yaml, the guides (web/texts/targets.json) and docs/light_curves.json (A4).

Each check is `pass`, `warn` (it works, with a limit the site has to say) or `fail` (gold or a
guide says something wrong). Writes docs/verification.md and docs/verification.json; exits 1 if
any check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord

from serafim import contract
from serafim.validation import guides, targets, trajectory

log = logging.getLogger(__name__)

PIXEL_ARCSEC = targets.PIXEL_SCALE_ARCSEC
GOLD_DIR = targets.ROOT / "data" / "gold"
MANIFEST_PATH = targets.ROOT / "data" / "bronze" / "manifest.parquet"
REPORT_PATH = targets.ROOT / "docs" / "verification.md"
SUMMARY_PATH = targets.ROOT / "docs" / "verification.json"

PASS, WARN, FAIL = "pass", "warn", "fail"
ORDER = {PASS: 0, WARN: 1, FAIL: 2}
SYMBOL = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
CHECKS = ("gold", "seen", "saturation", "motion", "spectrum", "brightness", "catalogs")

MIN_SEASONS = 2  # a "video" needs at least two seasons
KEPT_FRACTION = 0.9  # gold keeps at least this share of the downloaded frames
MAX_OFFSET_PX = 0.5  # measured position of a season against the catalog
MOTION_SIGMA = 2.0  # measured motion within this many errors of the catalog: pass (3: warn)
MOTION_MAX_ERROR = 0.2  # a motion error above this share of the motion is only a warning
FLUX_FACTOR = 1.5  # SPHEREx flux against a catalog magnitude, within this factor
METHANE_MAX_RATIO = 0.7  # flux at 3.2-3.5 um over 3.9-4.3 um of a methane brown dwarf
PERIOD_TOLERANCE = 0.1  # period found by SPHEREx alone against the catalog
MARKER_PX = 1.0  # a catalog marker this close to the target's catalog position marks it
DETECTABLE = 3.0  # a catalog band is compared only if its flux is this many SPHEREx errors
# Methane brown dwarfs: WISE W1 (2.8-3.8 um) is wide, while SPHEREx sees 3.3 um in narrow slices,
# right inside the methane absorption, so it is dimmer than W1 by design. W1 is not compared.
METHANE_DWARFS = ("eps_indi_b", "wise_0855")

# What each target has to show in the viewer (the challenge text)
SHOWS = {
    "barnard": "motion",
    "eps_indi_b": "motion and a methane spectrum",
    "wise_0855": "that SPHEREx sees it, and its motion",
    "rw_aur": "a change of brightness",
    "r_dra": "a change of brightness with its period",
}


def check(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


def worst(checks: list[dict]) -> str:
    return max((c["status"] for c in checks), key=ORDER.get, default=PASS)


def _detectors(values) -> str:
    values = sorted(int(v) for v in values)
    return ", ".join(map(str, values)) if values else "none"


def _named(values) -> str:
    """'detector 6' or 'detectors 5, 6'."""
    return ("detector " if len(set(values)) == 1 else "detectors ") + _detectors(set(values))


# --- Gold ---------------------------------------------------------------------------------------


def read_gold(gold_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every gold table (empty if missing) and meta.json."""
    tables = {}
    for filename, schema in contract.GOLD_TABLES.items():
        path = Path(gold_dir) / filename
        table = pq.read_table(path) if path.is_file() else schema.empty_table()
        tables[filename.removesuffix(".parquet")] = table.to_pandas()
    meta_path = Path(gold_dir) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return tables, meta


def of_target(tables: dict[str, pd.DataFrame], target_id: str) -> dict[str, pd.DataFrame]:
    return {
        name: table[table["target_id"] == target_id].reset_index(drop=True)
        for name, table in tables.items()
    }


def season_mjds(sums: pd.DataFrame) -> pd.Series:
    """Mean date (MJD) of each season from the gold sums of one detector and one season."""
    cells = sums[(sums["detector"] >= 1) & (sums["season"] >= 1)]
    days = (cells["mean_date_utc"] - pd.Timestamp("1858-11-17", tz="UTC")) / pd.Timedelta(days=1)
    return days.groupby(cells["season"]).mean()


# --- Checks (pure: tables in, one check out) ----------------------------------------------------


def gold_check(target: dict, gold: dict[str, pd.DataFrame], downloaded: int | None) -> dict:
    """The target is in gold, with frames, sums of every detector, differences and colors."""
    if gold["targets"].empty:
        return check(
            "gold", FAIL, "Not in gold: it has no silver sums (the site says Coming soon)."
        )
    frames, sums = gold["frames"], gold["sums"]
    seasons = sorted(int(s) for s in frames["season"].unique())
    cells = sums[(sums["detector"] >= 1) & (sums["season"] >= 1)]
    observed = set(zip(frames["detector"].astype(int), frames["season"].astype(int), strict=True))
    summed = set(zip(cells["detector"].astype(int), cells["season"].astype(int), strict=True))
    missing = sorted(observed - summed)  # frames without their sum: gold is wrong
    unobserved = sorted({(d, s) for d in range(1, 7) for s in seasons} - observed)
    detail = (
        f"{len(frames)} frames in {len(seasons)} seasons ({target['n_frames']} exist; at most 12 "
        f"per detector and season are used), {len(cells)} sums, {len(gold['differences'])} "
        "differences."
    )
    status = PASS
    if len(seasons) < MIN_SEASONS:
        status, detail = FAIL, detail + " Fewer than two seasons: there is no video."
    if unobserved:
        detail += (
            " SPHEREx did not observe it in (detector, season) "
            + ", ".join(map(str, unobserved)) + ": no image there."
        )  # fmt: skip
    if missing:
        status = FAIL
        detail += " Frames without their sum (detector, season): " + ", ".join(map(str, missing))
        detail += "."
    if downloaded and len(frames) < KEPT_FRACTION * downloaded:
        status = max(status, WARN, key=ORDER.get)
        detail += f" Only {len(frames)} of the {downloaded} downloaded frames are in gold."
    return check("gold", status, detail)


def seen_check(detections: pd.DataFrame, recommended: list[int]) -> dict:
    """Where the target is detected (S/N >= 5) against the detectors its guide recommends.

    `detections`: one row per silver sum with detector, season (0 = every season), snr and
    saturated (a saturated center is seen, whatever its S/N: see with_saturation()).
    """
    if detections.empty:
        return check(
            "seen", WARN, "No silver sums on this machine: run `python -m serafim silver`."
        )
    detected = detections[seen_rows(detections)]
    seen = set(detected["detector"].astype(int))
    blind = sorted(set(recommended) - seen)
    seasons = sorted(int(s) for s in detections["season"].unique() if s >= 1)
    dark = [s for s in seasons if not (detected["season"] == s).any()]
    detail = f"Detected in {_named(seen)}; the guide recommends {_named(recommended)}."
    if blind:
        return check("seen", FAIL, detail + f" Not detected in {_named(blind)}.")
    if dark:
        return check(
            "seen", WARN, detail + " Not detected in season " + _detectors(dark) + " alone."
        )
    return check("seen", PASS, detail)


def painted_saturation(sums: pd.DataFrame) -> tuple[list[int], list[int]]:
    """(detectors whose center gold paints saturated in every season, in only some seasons)."""
    cells = sums[(sums["detector"] >= 1) & (sums["season"] >= 1)]
    painted = cells[cells["saturated"].astype(bool)]
    always = sorted(
        int(d) for d, part in cells.groupby("detector") if part["saturated"].astype(bool).all()
    )
    return always, sorted(set(painted["detector"].astype(int)) - set(always))


def saturation_check(target: dict, gold: dict[str, pd.DataFrame]) -> dict:
    """The detectors gold paints as saturated against what the guide tells the visitor."""
    always, sometimes = painted_saturation(gold["sums"])
    told_always, told_sometimes = guides.saturation_claims(target)
    flags = Counter(
        name for text in gold["frames"]["flags"].fillna("") for name in text.split(",") if name
    )
    common = ", ".join(f"{name} {count}" for name, count in flags.most_common(3)) or "none"
    detail = (
        f"Gold paints the center saturated in every season of detectors {_detectors(always)}"
        + (f" and in some seasons of {_detectors(sometimes)}" if sometimes else "")
        + f"; the guide says always {_detectors(told_always)}, sometimes "
        f"{_detectors(told_sometimes)}. Frame flags at the target: {common}."
    )
    if (always, sometimes) == (sorted(told_always), sorted(told_sometimes)):
        return check("saturation", PASS, detail)
    return check("saturation", WARN, detail)


def motion_check(target: dict, gold: dict[str, pd.DataFrame]) -> dict | None:
    """The measured trajectory (A5) in gold against the catalog motion of a target that moves."""
    if target["kind"] not in ("motion", "spectrum"):
        return None
    rows = gold["trajectory"].copy()
    if rows.empty:
        return check("motion", FAIL, "No trajectory rows in gold.")
    rows["mjd"] = rows["season"].map(season_mjds(gold["sums"]))
    rows = rows.sort_values("season").reset_index(drop=True)
    first, last = rows.iloc[0], rows.iloc[-1]
    shift_px = (
        SkyCoord(first["ra_catalog"] * u.deg, first["dec_catalog"] * u.deg)
        .separation(SkyCoord(last["ra_catalog"] * u.deg, last["dec_catalog"] * u.deg))
        .arcsec
        / PIXEL_ARCSEC
    )
    measured = rows.dropna(subset=["ra_measured"])
    unmeasured = sorted(int(s) for s in rows.loc[rows["ra_measured"].isna(), "season"])
    offsets = SkyCoord(measured["ra_measured"].to_numpy() * u.deg,
                       measured["dec_measured"].to_numpy() * u.deg).separation(
        SkyCoord(measured["ra_catalog"].to_numpy() * u.deg,
                 measured["dec_catalog"].to_numpy() * u.deg)).arcsec / PIXEL_ARCSEC  # fmt: skip
    motion = trajectory.proper_motion(rows)
    detail = (
        f"The catalog moves it {shift_px:.2f} px from season {first['season']} to {last['season']}."
    )
    if motion is None:
        return check("motion", FAIL, detail + " Measured in fewer than two seasons.")
    value, catalog = motion["measured_total_arcsec_yr"], motion["catalog_total_arcsec_yr"]
    error = motion["error_total_arcsec_yr"]
    detail += (
        f" Measured {value:.2f} ± {error:.2f} arcsec/yr against {catalog:.2f} from the catalog "
        f"(seasons {_detectors(motion['seasons'])}); every season within "
        f"{float(np.max(offsets)):.2f} px of its catalog position."
    )
    if unmeasured:
        detail += f" Not measured in season {_detectors(unmeasured)}."
    distance = abs(value - catalog)
    if distance > 3 * error or float(np.max(offsets)) > 2 * MAX_OFFSET_PX:
        return check("motion", FAIL, detail)
    if (distance > MOTION_SIGMA * error or error > MOTION_MAX_ERROR * catalog or unmeasured
            or float(np.max(offsets)) > MAX_OFFSET_PX):  # fmt: skip
        return check("motion", WARN, detail)
    return check("motion", PASS, detail)


def spectrum_check(target: dict, gold: dict[str, pd.DataFrame]) -> dict | None:
    """The spectrum in gold (A4): its points, the catalog bands and, for Epsilon Indi B, methane."""
    from serafim.validation import spectra

    table = gold["spectra"]
    if table.empty:
        return None if target["kind"] == "brightness" else check("spectrum", FAIL, "No points.")
    doubtful = int(spectra.doubtful(table).sum())
    detail = f"{len(table)} points, {doubtful} doubtful (edge, bad pixels or a poor fit)."
    if guides.saturated_detectors(target) and doubtful > len(table) / 2:
        return check(
            "spectrum", WARN,
            detail + " Saturated: its absolute flux is not reliable (docs/spectra.md); the site "
            "should draw those points hollow.",
        )  # fmt: skip
    status = PASS
    for row in spectra.catalog_check(spectra._full(target), table):
        band, ratio = row["band"], row["spherex_mjy"] / row["catalog_mjy"]
        if row["catalog_mjy"] < DETECTABLE * row["error_mjy"]:
            detail += f" {band}: below SPHEREx's noise, as the catalog expects."
            continue
        if band == "W1" and target["id"] in METHANE_DWARFS:
            detail += f" {band}: {ratio:.2f} (inside the methane band, not compared)."
            continue
        consistent = abs(row["spherex_mjy"] - row["catalog_mjy"]) <= DETECTABLE * row["error_mjy"]
        detail += f" {band}: SPHEREx/catalog {ratio:.2f}."
        if not (1 / FLUX_FACTOR <= ratio <= FLUX_FACTOR or consistent):
            status = WARN
    if target["kind"] == "spectrum":
        dip, peak = spectra.band_mean(table, 3.2, 3.5), spectra.band_mean(table, 3.9, 4.3)
        if not (dip and peak):
            return check("spectrum", FAIL, detail + " No points at 3.2-3.5 or 3.9-4.3 um.")
        detail += f" Methane: 3.3 um over 4.1 um is {dip[0] / peak[0]:.2f}."
        if dip[0] / peak[0] > METHANE_MAX_RATIO:
            status = FAIL
    return check("spectrum", status, detail)


def brightness_check(target: dict, light_curves: dict) -> dict | None:
    """The brightness measured on the rings (A4, docs/light_curves.json; not a gold table)."""
    target_id = target["id"]
    curve = light_curves.get(target_id)
    if target["kind"] != "brightness" and not curve:
        return None
    if not curve:
        return check("brightness", FAIL, "No light curve in docs/light_curves.json.")
    rows = curve.get("by_detector") or []
    period = curve.get("period") or {}
    if target_id in guides.PERIOD_DAYS:
        catalog = guides.PERIOD_DAYS[target_id]
        if not period.get("periodic"):
            return check("brightness", FAIL, f"No period found (the catalog says {catalog:.0f} d).")
        found = period["period_days"]
        detail = (
            f"SPHEREx alone finds {found:.0f} days (false alarm {period['false_alarm']}); "
            f"the catalog says {catalog:.0f}."
        )
        ok = abs(found - catalog) <= PERIOD_TOLERANCE * catalog
        return check("brightness", PASS if ok else FAIL, detail)
    change = guides.strongest_change(rows)
    if target["kind"] == "brightness":
        if change is None:
            return check("brightness", WARN, "No change beyond the ~30 % the control allows.")
        return check(
            "brightness", PASS,
            f"Detector {change['detector']}: {change['ratio']:.2f} times the first season in "
            f"season {change['season']} ({guides._change_text(rows)})",
        )  # fmt: skip
    # A target that should not change (Barnard's Star): the control of the method
    largest = max(guides.certain(rows), key=lambda r: abs(r["ratio"] - 1), default=None)
    if largest is None:
        return check("brightness", WARN, "Control: no certain season ratio.")
    detail = (
        f"Control, a star that does not change: its largest certain ratio is "
        f"{largest['ratio']:.2f} (detector {largest['detector']}, season {largest['season']})."
    )
    if change is None:
        return check("brightness", PASS, detail)
    return check(
        "brightness", WARN,
        detail + f" That is at the {guides.CHANGE_THRESHOLD:.0%} the guides require before "
        "telling a change: smaller changes in one detector are not believed.",
    )  # fmt: skip


def catalogs_check(target: dict, gold: dict[str, pd.DataFrame]) -> dict:
    """The overlay layers (A2) have rows, and one marks the target in every season."""
    layers = gold["catalogs"]
    if layers.empty:
        return check("catalogs", FAIL, "No catalog rows.")
    counts = layers["catalog"].value_counts()
    detail = "Rows: " + ", ".join(
        f"{name} {counts.get(name, 0)}" for name in contract.CATALOG_NAMES
    )
    rows = gold["trajectory"]
    if rows.empty:
        return check("catalogs", PASS, detail + ".")
    unmarked = []
    for _, row in rows.iterrows():
        here = layers[layers["season"].isin([0, row["season"]])]
        if here.empty:
            unmarked.append(int(row["season"]))
            continue
        spot = SkyCoord(row["ra_catalog"] * u.deg, row["dec_catalog"] * u.deg)
        nearest = spot.separation(
            SkyCoord(here["ra"].to_numpy() * u.deg, here["dec"].to_numpy() * u.deg)
        ).arcsec.min()
        if nearest > MARKER_PX * PIXEL_ARCSEC:
            unmarked.append(int(row["season"]))
    if unmarked:
        return check(
            "catalogs",
            WARN,
            detail + f"; no marker on the target in season {_detectors(unmarked)}.",
        )
    return check("catalogs", PASS, detail + "; a marker sits on the target in every season.")


# --- One target ---------------------------------------------------------------------------------


def detections(target: dict, root: Path | None = None) -> pd.DataFrame:
    """S/N of the target at its catalog spot in every silver sum (season 0 = every season)."""
    measured = trajectory.measure_target(target, root, all_seasons=True)
    if measured.empty:
        return pd.DataFrame(columns=["detector", "season", "snr"])
    return measured[["detector", "season", "snr"]].astype(
        {"detector": int, "season": int, "snr": float}
    )


def with_saturation(found: pd.DataFrame, sums: pd.DataFrame) -> pd.DataFrame:
    """Add `saturated` from the gold sums. The S/N of a saturated center means nothing (the fit
    has only the ring, where a narrow PSF is ~0), and a saturated target is seen anyway."""
    painted = sums[["detector", "season", "saturated"]].astype({"detector": int, "season": int})
    out = found.merge(painted, on=["detector", "season"], how="left")
    out["saturated"] = out["saturated"].astype("boolean").fillna(False).astype(bool)
    return out


def seen_rows(found: pd.DataFrame) -> pd.Series:
    saturated = found["saturated"] if "saturated" in found else False
    return (found["snr"] >= trajectory.DETECTION_SNR) | saturated


def verify_target(
    target: dict,
    gold: dict[str, pd.DataFrame],
    found: pd.DataFrame,
    *,
    recommended: list[int],
    light_curves: dict,
    downloaded: int | None = None,
) -> list[dict]:
    """Every check of one target, in the order of CHECKS (those that do not apply left out)."""
    first = gold_check(target, gold, downloaded)
    if first["status"] == FAIL and gold["targets"].empty:
        return [first]
    if len(found) and "saturated" not in found:
        found = with_saturation(found, gold["sums"])
    checks = [
        first,
        seen_check(found, recommended),
        saturation_check(target, gold),
        motion_check(target, gold),
        spectrum_check(target, gold),
        brightness_check(target, light_curves),
        catalogs_check(target, gold),
    ]
    return [c for c in checks if c is not None]


# --- Report -------------------------------------------------------------------------------------


def snr_table(found: pd.DataFrame) -> list[str]:
    if found.empty:
        return []
    seasons = sorted(found["season"].unique(), key=lambda s: (s == 0, s))
    head = ["Detector", *[f"Season {s}" if s else "All seasons" for s in seasons]]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for detector, part in found.groupby("detector"):
        by_season = {row.season: row for row in part.itertuples()}
        cells = []
        for season in seasons:
            row = by_season.get(season)
            if row is None:
                cells.append("—")
            elif getattr(row, "saturated", False):
                cells.append("**sat**")
            elif not np.isfinite(row.snr):
                cells.append("no fit")
            elif row.snr >= trajectory.DETECTION_SNR:
                cells.append("**> 999**" if row.snr > 999 else f"**{row.snr:.0f}**")
            else:
                cells.append(f"{row.snr:.1f}")
        lines.append(f"| {detector} | " + " | ".join(cells) + " |")
    return lines


def report(results: dict, meta: dict) -> str:
    lines = [
        "# Verification of the five targets (task A6)",
        "",
        "Generated by `python -m serafim.validation.verification` from the real gold of D8"
        + (f" (commit `{meta['commit'][:7]}`, run {meta['run_date_utc'][:10]})" if meta else "")
        + ".",
        "`validate_gold()` checks that gold follows the contract; this checks that what it shows",
        "is right: ✅ pass, ⚠️ works with a limit the site has to say, ❌ wrong.",
        "",
        "| Target | Shows | " + " | ".join(name.capitalize() for name in CHECKS) + " |",
        "| --- | --- |" + " --- |" * len(CHECKS),
    ]
    for target_id, (target, checks, *_) in results.items():
        by_name = {c["check"]: SYMBOL[c["status"]] for c in checks}
        lines.append(
            f"| {target['name']} | {SHOWS.get(target_id, target['kind'])} | "
            + " | ".join(by_name.get(name, "—") for name in CHECKS)
            + " |"
        )
    lines += [
        "",
        "S/N is the target's flux at its catalog position over its error, in the silver sum of",
        "each detector and season; **bold** is a detection (S/N ≥ 5). **sat**: the center is",
        "saturated, so the target is seen and an S/N means nothing; — : no frames there.",
        "",
    ]
    for target_id, (target, checks, found, _) in results.items():
        lines += [f"## {target['name']} (`{target_id}`)", ""]
        lines += [f"- {SYMBOL[c['status']]} **{c['check']}**: {c['detail']}" for c in checks]
        lines += ["", *snr_table(found), ""]
    return "\n".join(lines).rstrip() + "\n"


def summary(results: dict, meta: dict) -> dict:
    out = {"gold": {key: meta.get(key) for key in ("commit", "run_date_utc", "n_frames")}}
    for target_id, (_, checks, found, painted) in results.items():
        detected = found[seen_rows(found)] if len(found) else found
        out[target_id] = {
            "status": worst(checks),
            "seen_in_detectors": sorted({int(d) for d in detected.get("detector", [])}),
            "saturated": {"always": painted[0], "sometimes": painted[1]},
            "checks": checks,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(prog="python -m serafim.validation.verification")
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    parser.add_argument("--target", nargs="+", help="Target ids (default: every target).")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    contract.validate_gold(args.gold)  # the contract first: a broken gold is not verified
    tables, meta = read_gold(args.gold)
    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    written = json.loads(guides.GUIDES_PATH.read_text(encoding="utf-8"))
    recommended = {g["id"]: g["how_to_look"]["detectors"] for g in written["targets"]}
    light_curves = guides._light_curves()
    manifest = pd.read_parquet(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None

    results = {}
    for target in document["targets"]:
        if args.target and target["id"] not in args.target:
            continue
        downloaded = None
        if manifest is not None:
            downloaded = int((manifest["target_id"] == target["id"]).sum()) or None
        own = of_target(tables, target["id"])
        found = pd.DataFrame(columns=["detector", "season", "snr", "saturated"])
        if not own["targets"].empty:
            found = with_saturation(detections(target), own["sums"])
        checks = verify_target(
            target, own, found, recommended=recommended.get(target["id"], []),
            light_curves=light_curves, downloaded=downloaded,
        )  # fmt: skip
        results[target["id"]] = (target, checks, found, painted_saturation(own["sums"]))
        log.info("%s: %s", target["id"], ", ".join(f"{c['check']} {c['status']}" for c in checks))

    args.report.write_text(report(results, meta), encoding="utf-8", newline="\n")
    args.summary.write_text(
        json.dumps(summary(results, meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )  # fmt: skip
    log.info("wrote %s and %s", args.report, args.summary)
    return 1 if any(worst(checks) == FAIL for _, checks, *_ in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
