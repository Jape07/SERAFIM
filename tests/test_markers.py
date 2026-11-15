"""Catalog markers on the real images (task A8): every check on synthetic sums, no network."""

import numpy as np
import pandas as pd
import pytest

from serafim.silver.grid import target_wcs
from serafim.validation import markers as mk

NOISE = 0.1


def star_field(stars, shape=(80, 80), seed=1):
    """A sum with Gaussian stars (x, y, flux) of sigma 0.8 px on a flat sky."""
    rows, cols = np.indices(shape)
    image = np.random.default_rng(seed).normal(0.0, NOISE, shape)
    for x, y, flux in stars:
        image += flux / (2 * np.pi * 0.64) * np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / 1.28)
    return image, np.zeros(shape, bool)


def test_a_marker_on_its_star_is_found_and_centered():
    image, bad = star_field([(40.3, 50.0, 50.0)])
    on = mk.measure(image, bad, 40.3, 50.0, NOISE)
    assert on["usable"] and on["snr"] > 20 and on["offset_px"] < 0.3
    assert mk.measure(image, bad, 20.0, 20.0, NOISE)["snr"] < mk.FOUND_SNR  # empty sky
    beside = mk.measure(image, bad, 41.8, 50.0, NOISE)  # 1.5 px away
    assert beside["offset_px"] > 1.0


def test_markers_at_the_edge_or_on_masked_pixels_are_not_judged():
    image, bad = star_field([(40.0, 40.0, 50.0)])
    assert not mk.measure(image, bad, 3.0, 40.0, NOISE)["usable"]
    bad[39:42, 39:42] = True
    assert not mk.measure(image, bad, 40.0, 40.0, NOISE)["usable"]
    assert not mk.measure(image, bad, np.nan, 40.0, NOISE)["usable"]


def test_impossible_pixels_far_below_the_sky():
    image, bad = star_field([])
    image[30, 30] = -6.5  # Epsilon Indi B, D5 season 1
    broken = mk.impossible(image, bad, mk.sky_noise(image, bad))
    assert broken.sum() == 1 and broken[30, 30]


def test_the_visible_limit_walks_from_the_brightest_bin():
    mags = np.repeat(np.arange(10.0, 18.0, 0.5) + 0.25, 10)
    found = mags < 16
    assert mk.visible_limit(mags, found) == 16.0
    assert mk.visible_limit([], []) is None


def test_the_limit_discounts_what_a_random_place_finds():
    """Crowded fields: 50 % of random places already 'find' a star."""
    mags = np.repeat([12.25, 12.75], 20)
    found = np.r_[np.ones(18, bool), np.zeros(2, bool), np.ones(17, bool), np.zeros(3, bool)]
    assert mk.visible_limit(mags, found) == 13.0  # 90 % and 85 %
    assert mk.visible_limit(mags, found, chance=0.5) == 12.5  # (0.85 - 0.5) / 0.5 = 0.7


def markers(*rows):
    names = ("catalog", "x", "y", "mag")
    return pd.DataFrame([dict(zip(names, row, strict=True)) for row in rows])


def test_the_same_star_in_two_catalogs_is_still_isolated():
    table = markers(("gaia", 40.0, 40.0, 14.0), ("catwise", 40.2, 40.0, 12.0),
                    ("gaia", 60.0, 60.0, 14.0), ("simbad", 62.0, 60.0, np.nan),
                    ("gaia", 20.0, 20.0, 15.0), ("gaia", 26.0, 20.0, 12.5))  # fmt: skip
    assert mk.isolated(table).tolist() == [True, True, False, False, False, True]


def measured(**changes):
    row = {"catalog": "gaia", "name": "a star", "mag": 12.0, "season": 1, "detector": 2,
           "x": 40.0, "y": 40.0, "isolated": True, "judged": True, "usable": True,
           "snr": 50.0, "offset_px": 0.2, "found": True}  # fmt: skip
    return {**row, **changes}


def test_a_bright_marker_with_nothing_under_it_is_out_of_place():
    rows = pd.DataFrame([
        measured(),
        measured(name="empty", found=False, snr=0.5, offset_px=np.nan),
        measured(name="red source at 1 um", catalog="catwise", detector=1, found=False),
        measured(name="beside", offset_px=1.6),
        measured(name="faint", mag=16.5, found=False),
        measured(name="", catalog=mk.CHANCE, mag=np.nan, offset_px=3.0),
    ])  # fmt: skip
    cuts = {"gaia": {2: 16.0}, "catwise": {1: 15.0}}
    wrong = mk.out_of_place(rows, cuts).set_index("name")["reason"].to_dict()
    assert wrong == {"empty": "nothing under it", "beside": "beside its star"}


def test_the_default_cut_keeps_the_image_readable():
    shape = (146, 146)
    rng = np.random.default_rng(2)
    rows = pd.DataFrame({"catalog": "gaia", "detector": 1, "season": 1,
                         "mag": rng.uniform(8, 18, 600), "x": rng.uniform(0, 145, 600),
                         "y": rng.uniform(0, 145, 600)})  # fmt: skip
    cut = mk.default_cuts(rows, {"gaia": {1: 17.5, 2: 17.0}}, shape)["gaia"]
    assert (rows["mag"] <= cut).sum() <= mk.MAX_DEFAULT_MARKERS
    assert (rows["mag"] <= cut + mk.BIN_MAG).sum() > mk.MAX_DEFAULT_MARKERS
    few = rows.head(100)
    assert mk.default_cuts(few, {"gaia": {1: 17.5, 2: 17.0}}, shape)["gaia"] == 17.0


def test_the_projection_pairs_are_what_astropy_says():
    wcs = target_wcs(269.446, 4.77, 0.25)
    pairs = mk.projection_pairs(wcs)
    ny = wcs.pixel_shape[1]
    for pair in pairs:
        x, y = wcs.world_to_pixel_values(pair["ra"], pair["dec"])
        assert (float(x), float(y)) == pytest.approx((pair["x"], pair["y"]), abs=1e-3)
        assert pair["png_y"] == ny - 1 - pair["y"] and pair["png_x"] == pair["x"]
    assert pairs[0]["ra"] == pytest.approx(269.446, abs=1e-6)  # the center is the target


def test_the_guide_view_is_the_image_each_guide_opens():
    guide = {"id": "barnard", "how_to_look": {"detectors": [5, 6], "mode": "difference",
                                              "seasons": [1, 3]}}  # fmt: skip
    assert mk.guide_view(guide) == ("png/barnard/diff_d5_s1-s3.png", 5, 3)
    guide["how_to_look"].update(mode="all_seasons", seasons=[0])
    assert mk.guide_view(guide) == ("png/barnard/sum_d5_s0.png", 5, 0)
