"""Guide texts for the 5 challenge targets (task A3, AI & validation).

`python -m serafim.validation.guides` writes web/texts/targets.json from
config/targets.yaml, so every number in the texts comes from the same queries
(SIMBAD, WISE, IRSA) and the texts follow when the targets file is regenerated.
`--check` only tells whether the file is up to date (exit code 1 if not).

The file has, in plain words for anyone:

- targets: per target, what it is, what to look at, which detectors and mode to
  pick, what to watch out for, and a few facts (today's position, motion, seasons).
- detectors, modes and layers: what each choice in the viewer shows.
- flags: what each SPHEREx pixel flag means and what the sums do with it.

These are drafts: task A7 rewrites them once the real SPHEREx images are checked.
The site is in English, so the texts are too.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import yaml

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
# (docs/visibility.md, PR #50). The first detector is the one the viewer opens.
# WISE 0855: S/N 9.6 in D6 and 6.3 in D5 (every season added), not seen in D1-D4.
# Barnard's Star: its motion measured on the ring matches the catalog to 0.1 px in D5-D6,
# where its saturated center is smallest.
SEEN_DETECTORS = {"wise_0855": (6, 5), "barnard": (5, 6)}

# Brightness change of the saturated targets, measured on the light around their saturated
# core (python -m serafim.validation.light_curves -> docs/light_curves.json). Barnard's Star,
# not variable, scatters by up to ~30 % in one detector, so only larger changes are told.
LIGHT_CURVES_PATH = targets.ROOT / "docs" / "light_curves.json"
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
     "A later season minus an earlier one. What did not change disappears. A star that moved "
     "leaves a dark spot where it was and a bright spot where it went."),
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
UNDECIDED = "decided in task D4"
FLAGS = (
    (0, "TRANSIENT", "Something hit or changed this pixel during the exposure, such as a "
     "cosmic ray.", UNDECIDED),
    (1, "OVERFLOW", "The pixel got more light than it can count: it is saturated and its value "
     "is not real. The site paints it with the brightest color.", LEFT_OUT),
    (2, "SUR_ERROR", "The camera could not measure a clean brightness for this pixel during the "
     "exposure.", UNDECIDED),
    (6, "NONFUNC", "A pixel that does not work.", LEFT_OUT),
    (7, "DICHROIC", "Possible stray light from the part of the telescope that splits the light "
     "between detectors.", UNDECIDED),
    (9, "MISSING_DATA", "There is no data for this pixel.", LEFT_OUT),
    (10, "HOT", "A pixel that reads too high even without light.", LEFT_OUT),
    (11, "COLD", "A pixel that is less sensitive than the others.", LEFT_OUT),
    (12, "FULLSAMPLE", "A processing note about how this pixel was read. Not a problem by "
     "itself.", UNDECIDED),
    (14, "PHANMISS", "A correction for an electronic echo of bright stars could not be applied "
     "here.", UNDECIDED),
    (15, "NONLINEAR", "The pixel was nearly full, so its brightness is not reliable. Treated "
     "like a saturated pixel.", LEFT_OUT),
    (17, "PERSIST", "Afterglow: a bright source seen earlier may leave a faint trace here.",
     UNDECIDED),
    (19, "OUTLIER", "This value disagrees strongly with the other readings of the same spot.",
     UNDECIDED),
    (20, "CROSSTALK", "A bright star elsewhere on the detector leaks a faint electronic copy "
     "here.", UNDECIDED),
    (21, "SOURCE", "A star or galaxy covers this pixel. This is normal and never hidden.", KEPT),
    (22, "GHOST", "A faint reflection of a bright star inside the telescope may fall here.",
     UNDECIDED),
    (23, "GHOST_FPA", "A reflection between the detectors and the telescope optics may fall "
     "here.", UNDECIDED),
    (24, "GHOST_EXT", "A reflection of a bright star outside the image may fall here.",
     UNDECIDED),
    (25, "STREAK", "A streak crosses this pixel, for example from a satellite.", UNDECIDED),
    (26, "BLOOM", "Charge spilled over from a saturated neighbor. It is still light from that "
     "bright star.", KEPT),
    (27, "SNOWBALL", "A round splash left by a particle hitting the detector.", UNDECIDED),
    (28, "HALO", "Inside the glow around a very bright star.", KEPT),
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
        "draft": True,
        "note": "Draft texts (task A3). Task A7 rewrites them after checking the real SPHEREx "
        "images.",
        "generated_from": f"config/targets.yaml, now_utc {document['now_utc']}",
        "saturated_badge": SATURATED_BADGE,
        "targets": [guide(target) for target in document["targets"]],
        "detectors": [
            {"detector": number, "from_um": low, "to_um": high, "text": text}
            for number, low, high, text in DETECTORS
        ],
        "modes": [{"id": key, "name": name, "text": text} for key, name, text in MODES],
        "layers": [{"id": key, "name": name, "text": text} for key, name, text in LAYERS],
        "flags": [
            {"bit": bit, "name": name, "text": text, "in_sums": effect}
            for bit, name, text, effect in FLAGS
        ],
        "sources": {"period_days": PERIOD_SOURCE},
    }


def guide(target: dict) -> dict:
    """The guide of one target. Every number comes from `target` (or PERIOD_DAYS)."""
    facts = target_facts(target)
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
            "In the difference image it leaves a dark spot where it was and a bright spot "
            "where it went."
        ),
        "detectors": list(SEEN_DETECTORS["barnard"]),
        "mode": "difference",
        "seasons": [1, seasons],
        "how": (
            "Detector 5 or 6, where its saturated center is smallest. Choose the difference of "
            f"season {seasons} minus season 1."
        ),
        "watch_out": [
            _saturated_text(target),
            "Star charts from the year 2000 put it about "
            f"{_fraction(since_2000)} of the Moon's width away from where it is today. This "
            "site uses today's position.",
        ],
    }


def _eps_indi_b(target: dict, facts: dict) -> dict:
    compass, where = direction(target)
    return {
        "headline": "Two failed stars with methane in their air",
        "what_it_is": (
            "Two brown dwarfs orbiting each other, and together orbiting the star Epsilon "
            f"Indi A, {facts['distance_light_years']:.0f} light-years away. Brown dwarfs are "
            "too small to burn like the Sun: they glow in the infrared as they slowly cool."
        ),
        "what_to_look_at": (
            "Compare detector 4 with detector 5. Methane in their air absorbs light inside "
            "detector 4, so the pair should look dimmer there than in detector 5. That dip "
            "is the fingerprint of methane in their spectrum."
        ),
        "detectors": [4, 5],
        "mode": "all_seasons",
        "seasons": [0],
        "how": "Use the image of all seasons and switch between detector 4 and detector 5.",
        "watch_out": [
            "Epsilon Indi A, a much brighter star, is in the same field. The brown dwarfs are "
            "the faint dot at the center.",
            f"They move too: about {facts['shift_pixels_first_to_last_season']:.1f} pixels "
            f"{compass} ({where}) between the first and the last season.",
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
            f"Open detector {best} with every season added: the faint dot at the center, "
            "right where WISE puts it. It moved about "
            f"{facts['shift_pixels_first_to_last_season']:.1f} pixels {compass} ({where}) over "
            "the year, a shift too small to follow season by season."
        ),
        "detectors": list(SEEN_DETECTORS["wise_0855"]),
        "mode": "all_seasons",
        "seasons": [0, *with_best],
        "how": (
            f"Detector {best}, or {SEEN_DETECTORS['wise_0855'][1]} with every season added. "
            "It does not show in detectors 1 to 4."
        ),
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
    return {
        "headline": "A young star that dims without warning",
        "what_it_is": (
            "A very young star, still surrounded by the disk of gas and dust it formed from. "
            "Clumps of that dust sometimes pass in front of it and dim it for weeks or months."
        ),
        "what_to_look_at": (
            "It does not move. Keep the same detector and step through the "
            f"{facts['seasons']} seasons: its brightness can change from one season to the "
            f"next.{measured}"
        ),
        "detectors": best,
        "mode": "sum",
        "seasons": list(range(1, facts["seasons"] + 1)),
        "how": f"{_detectors(best).capitalize()} (it saturates the least), one season at a time.",
        "watch_out": [
            _saturated_text(target),
            "Compare the size of its glow rather than its center: a brighter star looks bigger.",
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
    return {
        "headline": f"A giant star that swells and fades every {months:.0f} months",
        "what_it_is": (
            "An old red giant that pulsates: it grows and shrinks, and its brightness changes "
            f"a lot, in a cycle of about {facts['period_days']:.0f} days. Stars like this are "
            "called Mira variables."
        ),
        "what_to_look_at": (
            f"Keep one detector and step through the {facts['seasons']} seasons. They fall at "
            "different points of its cycle, so it should look brighter in some seasons than in "
            f"others.{measured}"
        ),
        "detectors": sorted(detectors_seen(target)),
        "mode": "sum",
        "seasons": list(range(1, facts["seasons"] + 1)),
        "how": "Any detector, one season at a time.",
        "watch_out": [
            _saturated_text(target),
            *glow,
            f"It sits near the spot SPHEREx passes over most often, so it has {facts['frames']} "
            "frames, far more than the others.",
        ],
    }


WRITERS = {
    "barnard": _barnard,
    "eps_indi_b": _eps_indi_b,
    "wise_0855": _wise_0855,
    "rw_aur": _rw_aur,
    "r_dra": _r_dra,
}


def _saturated_text(target: dict) -> str:
    """How often the center saturates, from the frames checked in task A1."""
    rows = [row for row in target["saturation"] if row["checked"]]
    always = [row["detector"] for row in rows if row["saturated"] == row["checked"]]
    sometimes = [row["detector"] for row in rows if 0 < row["saturated"] < row["checked"]]
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
