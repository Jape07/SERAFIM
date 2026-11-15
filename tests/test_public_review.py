"""Review of the published site (task A8): pure parts, no network."""

import pandas as pd

from serafim.validation import public


def tables(frames_by_target: dict, layers: dict) -> dict:
    frames = pd.DataFrame({"target_id": [t for t, n in frames_by_target.items() for _ in range(n)]})
    catalogs = pd.DataFrame(
        [{"target_id": t, "catalog": c} for t, names in layers.items() for c in names]
    )
    empty = pd.DataFrame({"target_id": list(frames_by_target)})
    return {"frames": frames, "sums": empty, "differences": empty, "catalogs": catalogs}


def test_a_layer_missing_on_the_site_is_reported():
    """A publish run while ESA's archive was down left the Gaia layer out (task A8)."""
    published = tables({"barnard": 167}, {"barnard": ["catwise", "simbad", "skybot"]})
    local = tables({"barnard": 165}, {"barnard": ["gaia", "catwise", "simbad"]})
    notes = public.compare(published, local)
    assert "catalogs: barnard has no gaia layer on the site" in notes
    assert "frames: barnard has 167 published, 165 local" in notes


def test_the_same_gold_has_nothing_to_report():
    same = tables({"barnard": 165}, {"barnard": ["gaia", "catwise", "simbad"]})
    assert public.compare(same, same) == []


GUIDE = {
    "id": "rw_aur", "name": "RW Aurigae", "kind": "brightness", "headline": "A young star",
    "what_to_look_at": "Its glow grows.", "facts": {"saturated_detectors": [2, 3]},
    "how_to_look": {"detectors": [1], "mode": "difference", "seasons": [1, 3]},
    "markers": {"default_max_mag": {"gaia": 16.0, "catwise": 13.5}},
}  # fmt: skip


def test_the_link_opens_what_the_guide_says():
    url = public.link("https://example.org/SERAFIM/", GUIDE)
    assert url == ("https://example.org/SERAFIM/index.html?target=rw_aur&detector=1&season=1-3"
                   "&mode=difference&layers=gaia,catwise,simbad,skybot#visor")  # fmt: skip
    sums = {**GUIDE, "how_to_look": {"detectors": [6], "mode": "all_seasons", "seasons": [0, 2]}}
    assert "detector=6&season=0&mode=all_seasons" in public.link("https://e.org", sums)


def test_the_checklist_has_a_section_per_target():
    notes = ["catalogs: rw_aur has no gaia layer on the site"]
    verified, checked = {"rw_aur": {"status": "pass"}}, {"rw_aur": {"out_of_place": 0}}
    text = public.checklist("https://e.org", {"rw_aur": GUIDE}, {"commit": "abcdef123"}, notes,
                            verified, checked)  # fmt: skip
    assert "### RW Aurigae" in text and "commit `abcdef1`" in text
    assert "Gaia down to G 16.0, CatWISE down to W2 13.5" in text
    assert "saturated badge" in text and "spectrum" not in text.split("### RW Aurigae")[1]
    assert "⚠️ catalogs: rw_aur has no gaia layer" in text
