"""Guide texts for the 5 challenge targets (tasks A3 and A7, AI & validation).

`python -m serafim.validation.guides` writes web/texts/targets.json from
config/targets.yaml, so every number in the texts comes from the same queries
(SIMBAD, WISE, IRSA) and the texts follow when the targets file is regenerated.
`--check` only tells whether the file is up to date (exit code 1 if not).

The file has, in plain words for anyone:

- targets: per target, what it is, what to look at, which detectors and mode to
  pick, what to watch out for, and a few facts (today's position, motion, seasons).
- detectors, modes and layers: what each choice in the viewer shows.
- flags: what each SPHEREx pixel flag means and what the sums do with it.

Final texts (task A7): every claim was checked against the real gold images
(docs/verification.md, task A6), and the measured numbers come from the AI documents
(docs/light_curves.json, docs/trajectory.json, docs/verification.json).
The site is in English, so the texts are too.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import yaml

from serafim.silver.frames import MASKED_FLAGS
from serafim.validation import targets

log = logging.getLogger(__name__)

GUIDES_PATH = targets.ROOT / "web" / "texts" / "targets.json"
LIGHT_YEARS_PER_PARSEC = 3.26156
MOON_WIDTH_ARCSEC = 1800.0  # the full Moon is about half a degree wide
MOON_WIDTH_ARCMIN = 30.0

# The only number not in config/targets.yaml: R Draconis pulsation period, from
# SIMBAD mesVar (Gaia DR3 long-period variables, 2022yCat.1358), 247.2 days.
PERIOD_DAYS = {"r_dra": 247.2}
PERIOD_SOURCE = "SIMBAD mesVar, Gaia DR3 long-period variables (2022yCat.1358)"

# Where SPHEREx really shows the target, from the visibility check on real frames
# (docs/visibility.md, PR #50) and the verification of the real gold (docs/verification.md,
# task A6). The first detector is the one the viewer opens.
# WISE 0855: S/N 9.6 in D6 and 6.3 in D5 in a stack that follows it, but in the site's images
# (every season added on a grid fixed on the sky, where it smears by 1.3 px) only D6 reaches
# S/N 5; D5 stays at 4. Not seen in D1-D4.
# Barnard's Star: its motion measured on the ring matches the catalog to 0.1 px in D5-D6,
# where its saturated center is smallest.
SEEN_DETECTORS = {"wise_0855": (6,), "barnard": (5, 6)}

# A field star to compare a target with, where the stretch of each detector makes brightness hard
# to judge by itself (each detector has its own color scale). Measured in task A7: flux in 2.5 px
# around each, background from a ring, on the all-seasons silver sums. Epsilon Indi B against the
# star 11 px east and 15 px north (22h04m31.5s -56d46m29s): the pair matches it in detector 4,
# where methane absorbs, and is ~5 times brighter in detector 5.
COMPARISON_STAR = {
    "eps_indi_b": {
        "where": "up and to the left",
        "offset_px": [-11, 15],
        "pair_over_star": {3: 0.60, 4: 1.03, 5: 4.92, 6: 3.37},
    }
}

# Brightness change of the saturated targets, measured on the light around their saturated
# core (python -m serafim.validation.light_curves -> docs/light_curves.json). Barnard's Star,
# not variable, scatters by up to ~30 % in one detector, so only larger changes are told.
LIGHT_CURVES_PATH = targets.ROOT / "docs" / "light_curves.json"
# Motion measured on the SPHEREx sums (python -m serafim.validation.trajectory, task A5); told
# in the text only when its error is under MOTION_MAX_ERROR of the value
TRAJECTORY_PATH = targets.ROOT / "docs" / "trajectory.json"
MOTION_MAX_ERROR = 0.1
# What the real gold paints as saturated (python -m serafim.validation.verification, task A6):
# the saturation sentence follows the site's images when this file exists, else task A1's check
VERIFICATION_PATH = targets.ROOT / "docs" / "verification.json"
CHANGE_THRESHOLD = 0.3
MAX_RATIO_ERROR = 0.3
MIN_FRAMES = 3  # a season ratio from fewer frames is not told
SYSTEMATIC_SCATTER = 0.15
VISIBILITY_SOURCE = "docs/visibility.md (real SPHEREx frames)"

# A background star that a target passes very close to, from Gaia DR3 (epoch 2016.0); found by
# the visibility check, where it faked a detection of WISE 0855 in detectors 1-4. Its distance
# to the target in each season is computed from both motions, not typed.
NEIGHBOURS = {
    "wise_0855": {
        "name": "Gaia DR3 5756956644387831296", "g_mag": 16.31,
        "ra": 133.758183, "dec": -7.241156, "pmra_mas_yr": -6.187835, "pmdec_mas_yr": 1.930855,
        "parallax_mas": 0.285879, "epoch_jyear": 2016.0,
    },
}  # fmt: skip

# Detector edges measured in task D1 (IRSA SIA em_min/em_max, docs/real_fits.md)
DETECTORS = (
    (1, 0.746, 1.117, "Near-infrared just past red light. Stars like the Sun are bright here."),
    (2, 1.103, 1.650, "Near-infrared. Dust hides less here than in visible light."),
    (3, 1.630, 2.435, "Near-infrared. Cool red stars are brightest around here."),
    (4, 2.403, 3.847, "Infrared. Methane gas absorbs light near 3.3 micrometers, inside this "
        "band, so methane-rich brown dwarfs look dimmer here."),
    (5, 3.811, 4.430, "Longer infrared. Cold brown dwarfs give off most of their light around "
        "here."),
    (6, 4.412, 5.010, "The longest infrared SPHEREx sees. The coldest objects stand out most."),
)  # fmt: skip

MODES = (
    ("sum", "One season",
     "The frames of one season in one detector, added together so faint things show up."),
    ("difference", "Difference",
     "A later season minus an earlier one. What did not change fades to white. Red is "
     "brighter in the later season, blue in the earlier one: a star that moved leaves blue "
     "where it was and red where it went. Saturated pixels are white."),
    ("all_seasons", "All seasons",
     "Every season added together: the deepest image, but anything that moves looks "
     "stretched."),
    ("white", "All detectors",
     "The 6 detectors added together for one season: the most light, without color."),
    ("color", "Color",
     "Detectors 1, 3 and 6 shown as blue, green and red."),
)  # fmt: skip

LAYERS = (
    ("gaia", "Gaia", "Stars measured by ESA's Gaia space telescope, moved to the date of "
     "each season."),
    ("catwise", "CatWISE", "Infrared sources seen by NASA's WISE telescope, moved by their "
     "measured motion. Entries along the target's own path are hidden: there the catalog is "
     "unreliable."),
    ("simbad", "SIMBAD", "Named objects from the SIMBAD database, including the target itself "
     "at each season."),
    ("skybot", "Asteroids", "Known asteroids that crossed the field during a visit, only those "
     "bright enough for SPHEREx."),
)  # fmt: skip

# SPHEREx FLAGS bits (MSKN* keys, docs/real_fits.md) and what the sums do with them
LEFT_OUT = "left out"
KEPT = "kept"
FLAGS = (
    (0, "TRANSIENT", "Something hit or changed this pixel during the exposure, such as a "
     "cosmic ray."),
    (1, "OVERFLOW", "The pixel got more light than it can count: it is saturated and its value "
     "is not real. The site paints it with the brightest color."),
    (2, "SUR_ERROR", "The camera could not measure a clean brightness for this pixel during the "
     "exposure."),
    (6, "NONFUNC", "A pixel that does not work."),
    (7, "DICHROIC", "Possible stray light from the part of the telescope that splits the light "
     "between detectors."),
    (9, "MISSING_DATA", "There is no data for this pixel."),
    (10, "HOT", "A pixel that reads too high even without light."),
    (11, "COLD", "A pixel that is less sensitive than the others."),
    (12, "FULLSAMPLE", "A processing note about how this pixel was read. Not a problem by "
     "itself."),
    (14, "PHANMISS", "A correction for an electronic echo of bright stars could not be applied "
     "here."),
    (15, "NONLINEAR", "The pixel was nearly full, so its brightness is not reliable. Treated "
     "like a saturated pixel."),
    (17, "PERSIST", "Afterglow: a bright source seen earlier may leave a faint trace here."),
    (19, "OUTLIER", "This value disagrees strongly with the other readings of the same spot."),
    (20, "CROSSTALK", "A bright star elsewhere on the detector leaks a faint electronic copy "
     "here."),
    (21, "SOURCE", "A star or galaxy covers this pixel. This is normal and never hidden."),
    (22, "GHOST", "A faint reflection of a bright star inside the telescope may fall here."),
    (23, "GHOST_FPA", "A reflection between the detectors and the telescope optics may fall "
     "here."),
    (24, "GHOST_EXT", "A reflection of a bright star outside the image may fall here."),
    (25, "STREAK", "A streak crosses this pixel, for example from a satellite."),
    (26, "BLOOM", "Charge spilled over from a saturated neighbor. It is still light from that "
     "bright star."),
    (27, "SNOWBALL", "A round splash left by a particle hitting the detector."),
    (28, "HALO", "Inside the glow around a very bright star."),
)  # fmt: skip

SATURATED_BADGE = (
    "Saturated center: the camera got more light than it can count, so the brightest pixels "
    "are painted with the top color. Look at the edge of the star instead."
)

# Directions on an image shown with north up and east to the left, as the sky is seen
COMPASS = (
    ("north", "up"), ("north-east", "up and to the left"), ("east", "to the left"),
    ("south-east", "down and to the left"), ("south", "down"),
    ("south-west", "down and to the right"), ("west", "to the right"),
    ("north-west", "up and to the right"),
)  # fmt: skip


# --- Building the file ------------------------------------------------------------------


def build(document: dict) -> dict:
    """The whole guides file from the parsed config/targets.yaml."""
    return {
        "draft": False,
        "note": "Final texts (task A7), checked against the real SPHEREx images of the gold "
        "(docs/verification.md).",
        "generated_from": f"config/targets.yaml, now_utc {document['now_utc']}",
        "saturated_badge": SATURATED_BADGE,
        "targets": [guide(target, document.get("settings")) for target in document["targets"]],
        "detectors": [
            {"detector": number, "from_um": low, "to_um": high, "text": text}
            for number, low, high, text in DETECTORS
        ],
        "modes": [{"id": key, "name": name, "text": text} for key, name, text in MODES],
        "layers": [{"id": key, "name": name, "text": text} for key, name, text in LAYERS],
        "flags": [
            {
                "bit": bit,
                "name": name,
                "text": text,
                "in_sums": LEFT_OUT if name in MASKED_FLAGS else KEPT,
            }
            for bit, name, text in FLAGS
        ],
        "sources": {"period_days": PERIOD_SOURCE},
    }


def guide(target: dict, settings: dict | None = None) -> dict:
    """The guide of one target. Every number comes from `target` (or PERIOD_DAYS)."""
    facts = target_facts(target)
    per_cell = (settings or {}).get("max_frames_per_season_detector")
    if per_cell:
        facts["frames_per_detector_and_season"] = per_cell
    writer = WRITERS[target["id"]]
    texts = writer(target, facts)
    return {
        "id": target["id"],
        "name": target["name"],
        "kind": target["kind"],
        "headline": texts["headline"],
        "what_it_is": texts["what_it_is"],
        "what_to_look_at": texts["what_to_look_at"],
        "how_to_look": {
            "detectors": texts["detectors"],
            "mode": texts["mode"],
            "seasons": texts["seasons"],
            "text": texts["how"],
        },
        "watch_out": texts["watch_out"],
        "facts": facts,
    }


def target_facts(target: dict) -> dict:
    seasons = target["seasons"]
    facts = {
        "ra_now": target["ra_now"],
        "dec_now": target["dec_now"],
        "seasons": len(seasons),
        "frames": target["n_frames"],
        "first_frame": seasons[0]["start_utc"][:7],
        "last_frame": seasons[-1]["end_utc"][:7],
        "saturated_detectors": saturated_detectors(target),
    }
    if target["kind"] in ("motion", "spectrum"):
        facts["distance_light_years"] = round(light_years(target), 2)
        facts["motion_arcsec_per_year"] = target["motion"]["pm_total_arcsec_yr"]
        facts["shift_pixels_first_to_last_season"] = target["motion"][
            "shift_px_first_to_last_season"
        ]
        motion = measured_motion(target["id"])
        if motion:
            facts["measured_motion_arcsec_per_year"] = motion["measured_total_arcsec_yr"]
            facts["measured_motion_error_arcsec_per_year"] = motion["error_total_arcsec_yr"]
    if target["id"] in PERIOD_DAYS:
        facts["period_days"] = PERIOD_DAYS[target["id"]]
    if target["id"] in SEEN_DETECTORS:
        facts["seen_in_detectors"] = list(SEEN_DETECTORS[target["id"]])
    rows = light_curve(target["id"]) if target["kind"] == "brightness" else None
    if rows:
        ratios: dict[str, dict[str, float]] = {}
        for row in certain(rows):
            ratios.setdefault(str(row["detector"]), {})[str(row["season"])] = round(row["ratio"], 2)
        facts["brightness_vs_first_season"] = ratios
        period = measured_period(target["id"])
        if period:
            facts["measured_period_days"] = period["period_days"]
    if target["id"] in NEIGHBOURS:
        star = NEIGHBOURS[target["id"]]
        facts["neighbour_star"] = {
            "name": star["name"],
            "g_mag": star["g_mag"],
            "pixels_by_season": neighbour_pixels(target),
        }
    return facts


def neighbour_pixels(target: dict) -> dict[int, float]:
    """Distance (SPHEREx pixels) between the target and its neighbour star, per season."""
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    from serafim.validation import layers

    star = pd.DataFrame([NEIGHBOURS[target["id"]]])
    out = {}
    for season in target["seasons"]:
        when = Time(season["mean_utc"], scale="utc")
        here = targets.propagate(target, when)
        ra, dec = layers.move(star, when)
        there = SkyCoord(ra[0] * u.deg, dec[0] * u.deg)
        separation = SkyCoord(here.ra, here.dec).separation(there).to_value(u.arcsec)
        out[int(season["season"])] = round(float(separation) / targets.PIXEL_SCALE_ARCSEC, 2)
    return out


def saturated_detectors(target: dict) -> list[int]:
    """Detectors where the target pixel was saturated in at least one checked frame."""
    return [row["detector"] for row in target["saturation"] if row["saturated"] > 0]


def least_saturated(target: dict) -> list[int]:
    """Detectors with the smallest share of saturated frames among those checked."""
    share = {
        row["detector"]: row["saturated"] / row["checked"]
        for row in target["saturation"]
        if row["checked"]
    }
    lowest = min(share.values())
    return sorted(detector for detector, value in share.items() if value == lowest)


def light_years(target: dict) -> float:
    return 1000.0 / target["parallax_mas"] * LIGHT_YEARS_PER_PARSEC


def direction(target: dict) -> tuple[str, str]:
    """Compass direction of the motion and where that is in a north-up image."""
    angle = math.degrees(math.atan2(target["pmra_mas_yr"], target["pmdec_mas_yr"])) % 360
    return COMPASS[round(angle / 45) % 8]


def detectors_seen(target: dict) -> set[int]:
    return {int(detector) for detector in target["frames_by_detector"]}


# --- One writer per target ---------------------------------------------------------------
# Plain words only: no "arcsec", "parallax", "proper motion" and the like in these texts.


def _barnard(target: dict, facts: dict) -> dict:
    compass, where = direction(target)
    moon_years = _round_to(MOON_WIDTH_ARCSEC / facts["motion_arcsec_per_year"], 10)
    since_2000 = target["motion"]["shift_since_j2000_arcmin"] / MOON_WIDTH_ARCMIN
    seasons = facts["seasons"]
    return {
        "headline": "The fastest-moving star in our sky",
        "what_it_is": (
            f"A small red star only {facts['distance_light_years']:.0f} light-years away, one "
            "of the Sun's closest neighbors. It crosses the sky faster than any other star: "
            f"in about {moon_years:.0f} years it moves the width of the full Moon."
        ),
        "what_to_look_at": (
            f"Between {_month(target['seasons'][0]['start_utc'])} and "
            f"{_month(target['seasons'][-1]['end_utc'])} it moved about "
            f"{facts['shift_pixels_first_to_last_season']:.1f} pixels {compass} ({where}). "
            "In the difference image it leaves blue where it was and red where it went."
        ),
        "detectors": list(SEEN_DETECTORS["barnard"]),
        "mode": "difference",
        "seasons": [1, seasons],
        "how": (
            "Detector 5 or 6, where its saturated center is smallest; difference of season "
            f"{seasons} minus season 1.{_motion_text(target['id'])}"
        ),
        "watch_out": [
            _saturated_text(target),
            f"Charts from 2000 put it {_fraction(since_2000)} of the Moon's width from where it "
            "is today.",
        ],
    }


def _eps_indi_b(target: dict, facts: dict) -> dict:
    compass, where = direction(target)
    star = COMPARISON_STAR["eps_indi_b"]
    facts["comparison_star"] = star
    return {
        "headline": "Two failed stars with methane in their air",
        "what_it_is": (
            "Two brown dwarfs orbiting each other, and together orbiting the star Epsilon "
            f"Indi A, {facts['distance_light_years']:.0f} light-years away. Brown dwarfs are "
            "too small to burn like the Sun: they glow in the infrared as they slowly cool."
        ),
        "what_to_look_at": (
            f"Compare the pair with the star {star['where']}. In detector 4 they are "
            f"{_as_bright(star['pair_over_star'][4])}; in detector 5, "
            f"{_as_bright(star['pair_over_star'][5])}. Methane in their air absorbs light "
            "inside detector 4: that dip is its fingerprint."
        ),
        "detectors": [4, 5],
        "mode": "all_seasons",
        "seasons": [0],
        "how": "All seasons added; switch between detectors 4 and 5.",
        "watch_out": [
            "Epsilon Indi A, a much brighter star, is in the same field. The brown dwarfs are "
            "the faint dot at the center.",
            f"They move too: about {facts['shift_pixels_first_to_last_season']:.1f} pixels "
            f"{compass} ({where}) between the first and the last season."
            f"{_motion_text(target['id'])}",
        ],
    }


def _wise_0855(target: dict, facts: dict) -> dict:
    compass, where = direction(target)
    best = SEEN_DETECTORS["wise_0855"][0]
    with_best = [s["season"] for s in target["seasons"] if best in s["detectors"]]
    pixels = facts["neighbour_star"]["pixels_by_season"]
    first, last = target["seasons"][0], target["seasons"][-1]
    return {
        "headline": "The coldest brown dwarf known",
        "what_it_is": (
            f"A brown dwarf only {facts['distance_light_years']:.0f} light-years away, colder "
            "than any other found so far. It is so cold that it gives off almost no visible "
            "light: it only shines in the longest infrared colors."
        ),
        "what_to_look_at": (
            f"Open detector {best} with every season added and zoom in: the faint dot at the "
            "center, barely above the noise, sits where WISE puts it. It moved about "
            f"{facts['shift_pixels_first_to_last_season']:.1f} pixels {compass} ({where}) over "
            "the year, too little to follow season by season."
        ),
        "detectors": list(SEEN_DETECTORS["wise_0855"]),
        "mode": "all_seasons",
        "seasons": [0, *with_best],
        "how": (f"Detector {best} only: in 5 it is too faint, and in 1 to 4 it does not show."),
        "watch_out": [
            f"A background star sits next to its path: {pixels[first['season']]:.1f} pixels "
            f"away in {_month(first['mean_utc'])}, {pixels[last['season']]:.1f} in "
            f"{_month(last['mean_utc'])}. In detectors 1 to 4, the dot at the center is that "
            "star, not WISE 0855.",
            "The challenge's position is from about 2010; this site centers on where it is today.",
        ],
    }


def _light_curves() -> dict:
    if not LIGHT_CURVES_PATH.exists():
        return {}
    return json.loads(LIGHT_CURVES_PATH.read_text(encoding="utf-8"))


def measured_motion(target_id: str) -> dict | None:
    """The motion measured on the SPHEREx sums (docs/trajectory.json, task A5), if any."""
    if not TRAJECTORY_PATH.exists():
        return None
    data = json.loads(TRAJECTORY_PATH.read_text(encoding="utf-8"))
    return (data.get(target_id) or {}).get("motion")


def precise(motion: dict | None) -> bool:
    """A measured motion good enough to be told: its error is under MOTION_MAX_ERROR of it."""
    if not motion:
        return False
    return motion["error_total_arcsec_yr"] <= MOTION_MAX_ERROR * motion["measured_total_arcsec_yr"]


def _motion_text(target_id: str) -> str:
    """' SPHEREx measured X pixels a year; catalogs say Y.' when the measurement is precise."""
    motion = measured_motion(target_id)
    if not precise(motion):
        return ""
    measured = motion["measured_total_arcsec_yr"] / targets.PIXEL_SCALE_ARCSEC
    catalog = motion["catalog_total_arcsec_yr"] / targets.PIXEL_SCALE_ARCSEC
    return f" SPHEREx measured {measured:.2f} pixels a year; catalogs say {catalog:.2f}."


def light_curve(target_id: str) -> list[dict] | None:
    """Per (detector, season) brightness relative to the first season, measured on the rings
    of the saturated targets (docs/light_curves.json, from serafim.validation.light_curves)."""
    return (_light_curves().get(target_id) or {}).get("by_detector") or None


def measured_period(target_id: str) -> dict | None:
    """The period found in the SPHEREx light curve alone, only if it is a real one."""
    period = (_light_curves().get(target_id) or {}).get("period")
    return period if period and period.get("periodic") else None


def certain(rows: list[dict]) -> list[dict]:
    """Season ratios (not the reference) known to better than 30 %, from 3 frames or more."""
    return [
        r
        for r in rows
        if 0 < r["error"] <= MAX_RATIO_ERROR * r["ratio"] and r.get("n_frames", 0) >= MIN_FRAMES
    ]


def strongest_change(rows: list[dict]) -> dict | None:
    """The largest certain change, if it is beyond what the Barnard control allows."""
    real = [r for r in certain(rows) if abs(r["ratio"] - 1) > CHANGE_THRESHOLD]
    if not real:
        return None

    def significance(row: dict) -> float:
        spread = math.hypot(row["error"] / row["ratio"], SYSTEMATIC_SCATTER)
        return abs(math.log(row["ratio"])) / spread

    return max(real, key=significance)


def _change_text(rows: list[dict]) -> str:
    change = strongest_change(rows)
    if change is None:
        return (
            f"SPHEREx shows no change larger than its method can measure (about "
            f"{CHANGE_THRESHOLD:.0%})."
        )
    reference = next(r for r in rows if r["detector"] == change["detector"] and r["error"] == 0)
    when, before = _month(change["mean_date_utc"]), _month(reference["mean_date_utc"])
    if change["ratio"] >= 1:
        how = f"{change['ratio']:.1f} times as bright in {when} as in {before}"
    else:
        how = f"{1 / change['ratio']:.1f} times fainter in {when} than in {before}"
    return f"Measured: in detector {change['detector']} it was {how}."


def _rw_aur(target: dict, facts: dict) -> dict:
    best = least_saturated(target)
    rows = light_curve(target["id"])
    measured = f" {_change_text(rows)}" if rows else ""
    change = strongest_change(rows) if rows else None
    if change:  # the difference shows the change as a red (brighter) or blue (fainter) ring
        detector, season = int(change["detector"]), int(change["season"])
        ring = "red" if change["ratio"] > 1 else "blue"
        look = {
            "detectors": [detector],
            "mode": "difference",
            "seasons": [1, season],
            "how": (
                f"Detector {detector}, difference of season {season} minus season 1: the {ring} "
                "ring around its white center is the light it gained."
                if ring == "red"
                else f"Detector {detector}, difference of season {season} minus season 1: the "
                "blue ring around its white center is the light it lost."
            ),
        }
    else:
        look = {
            "detectors": best,
            "mode": "sum",
            "seasons": list(range(1, facts["seasons"] + 1)),
            "how": f"{_detectors(best).capitalize()} (it saturates the least), one season at a "
            "time.",
        }
    return {
        "headline": "A young star that dims without warning",
        "what_it_is": (
            "A very young star, still surrounded by the disk of gas and dust it formed from. "
            "Clumps of that dust sometimes pass in front of it and dim it for weeks or months."
        ),
        "what_to_look_at": (
            "It does not move, so a difference should erase it. It does not: its brightness "
            f"changes.{measured}"
        ),
        **look,
        "watch_out": [
            _saturated_text(target),
            "In single seasons, compare its glow, not its center: a brighter star looks bigger.",
        ],
    }


def _r_dra(target: dict, facts: dict) -> dict:
    months = facts["period_days"] / 30.44
    rows = light_curve(target["id"])
    period = measured_period(target["id"])
    # With a measurement there is no room (150 words) for the hint about the glow
    glow = (
        []
        if rows
        else ["Compare the size of its glow: when the star is brighter, the glow is bigger."]
    )
    if period:
        measured = (
            " Measured with SPHEREx alone: it brightens and fades every "
            f"{period['period_days']:.0f} days or so; catalogs say {facts['period_days']:.0f}."
        )
    else:
        measured = f" {_change_text(rows)}" if rows else ""
    change = strongest_change(rows) if rows else None
    if change:  # the detector and seasons where the glow changes most
        first = next(r for r in rows if r["detector"] == change["detector"] and r["season"] == 1)
        detector, season = int(change["detector"]), int(change["season"])
        dimmer = change["ratio"] < 1
        how = (
            f"Detector {detector}, one season at a time: its glow is "
            f"{'bigger' if dimmer else 'smaller'} in season 1 ({_month(first['mean_date_utc'])}) "
            f"than in season {season} ({_month(change['mean_date_utc'])})."
        )
        shown = [detector]
    else:
        how, shown = "Any detector, one season at a time.", sorted(detectors_seen(target))
    per_cell = facts.get("frames_per_detector_and_season")
    frames = (
        f"SPHEREx passes over it more than over the others ({facts['frames']} frames)"
        + (f"; this site adds up to {per_cell} per detector and season." if per_cell else ".")
    )  # fmt: skip
    return {
        "headline": f"A giant star that swells and fades every {months:.0f} months",
        "what_it_is": (
            "An old red giant that pulsates: it grows and shrinks, and its brightness changes "
            f"a lot, in a cycle of about {facts['period_days']:.0f} days. Stars like this are "
            "called Mira variables."
        ),
        "what_to_look_at": (
            f"Its {facts['seasons']} seasons fall at different points of its cycle, so it looks "
            f"brighter in some than in others.{measured}"
        ),
        "detectors": shown,
        "mode": "sum",
        "seasons": list(range(1, facts["seasons"] + 1)),
        "how": how,
        "watch_out": [_saturated_text(target), *glow, frames],
    }


WRITERS = {
    "barnard": _barnard,
    "eps_indi_b": _eps_indi_b,
    "wise_0855": _wise_0855,
    "rw_aur": _rw_aur,
    "r_dra": _r_dra,
}


def saturation_claims(target: dict) -> tuple[list[int], list[int]]:
    """(detectors whose center is saturated in every season, in only some) as the guide says.

    From the real gold (docs/verification.json, task A6) when it has the target, so the text
    matches the images; else from the frames checked in task A1 (config/targets.yaml).
    """
    if VERIFICATION_PATH.exists():
        verified = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8")).get(target["id"])
        painted = (verified or {}).get("saturated") or {}
        if painted.get("always") or painted.get("sometimes"):
            return painted["always"], painted["sometimes"]
    rows = [row for row in target["saturation"] if row["checked"]]
    always = [row["detector"] for row in rows if row["saturated"] == row["checked"]]
    sometimes = [row["detector"] for row in rows if 0 < row["saturated"] < row["checked"]]
    return always, sometimes


def _as_bright(ratio: float) -> str:
    """'about as bright as it', 'about 5 times brighter' or 'about 2 times fainter'."""
    if 0.8 <= ratio <= 1.25:
        return "about as bright as it"
    if ratio > 1:
        return f"about {ratio:.0f} times brighter"
    return f"about {1 / ratio:.0f} times fainter"


def _saturated_text(target: dict) -> str:
    """How often the center saturates (saturation_claims())."""
    always, sometimes = saturation_claims(target)
    if len(always) == 6:
        state = "saturated in every detector"
    elif always and sometimes:
        state = f"always saturated in {_detectors(always)}, and sometimes in detector"
        state += ("s " if len(sometimes) > 1 else " ") + _list(sometimes)
    else:
        state = f"saturated in {_detectors(always or sometimes)}"
    return (
        f"It is so bright that its center is {state}: the camera got more light than it can "
        "count. Those pixels are painted with the brightest color."
    )


def _detectors(numbers: list[int]) -> str:
    return ("detector " if len(numbers) == 1 else "detectors ") + _list(numbers)


def _month(stamp: str) -> str:
    """'2025-08-15T20:46' -> 'August 2025'."""
    names = ("January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December")  # fmt: skip
    return f"{names[int(stamp[5:7]) - 1]} {stamp[:4]}"


def _list(numbers: list[int]) -> str:
    words = [str(number) for number in numbers]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _fraction(value: float) -> str:
    """0.155 -> 'a sixth'; plain words for a fraction of the Moon's width."""
    names = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 6: "a sixth",
             7: "a seventh", 8: "an eighth", 10: "a tenth"}  # fmt: skip
    denominator = min(names, key=lambda n: abs(1 / n - value))
    return names[denominator]


def _round_to(value: float, step: float) -> float:
    return round(value / step) * step


# --- Command line --------------------------------------------------------------------------


def render(document: dict) -> str:
    return json.dumps(build(document), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m serafim.validation.guides",
        description="Write web/texts/targets.json from config/targets.yaml.",
    )
    parser.add_argument("--out", type=Path, default=GUIDES_PATH)
    parser.add_argument("--check", action="store_true", help="Only check it is up to date.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    document = yaml.safe_load(targets.TARGETS_PATH.read_text(encoding="utf-8"))
    text = render(document)
    current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
    if args.check:
        if current != text:
            log.error("%s is out of date: run python -m serafim.validation.guides", args.out)
            return 1
        log.info("%s is up to date", args.out)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
