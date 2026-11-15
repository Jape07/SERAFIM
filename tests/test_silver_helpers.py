"""What silver shares with gold and with AI: the grid of a target and the differences to make.

Both were agreed with AI & validation (the document "Pipelines y red neuronal"), so the tests
write the agreement down: if someone changes it, they have to talk to AI first.
"""

import json

import numpy as np
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord

from serafim.config import load_config
from serafim.gold.mock import build_mock, write_mock_gold
from serafim.silver.differences import difference_pairs
from serafim.silver.grid import grid_size, target_wcs, wcs_from_json, wcs_to_json

BARNARD = (269.446074, 4.7707026)  # ra_now, dec_now


# --- The grid ------------------------------------------------------------------------------
def test_the_grid_has_one_spherex_pixel_per_pixel():
    assert grid_size(0.25) == 146  # 0.25 deg = 900 arcsec = 146.3 pixels of 6.15 arcsec
    wcs = target_wcs(*BARNARD, 0.25)
    assert wcs.pixel_shape == (146, 146)
    scale = np.abs(wcs.wcs.cdelt) * 3600
    np.testing.assert_allclose(scale, 6.15)


def test_the_center_of_the_grid_is_the_target_today():
    wcs = target_wcs(*BARNARD, 0.25)
    center = (146 - 1) / 2  # 0-based, the middle of the grid
    sky = wcs.pixel_to_world(center, center)
    assert isinstance(sky, SkyCoord)
    np.testing.assert_allclose([sky.ra.deg, sky.dec.deg], BARNARD, atol=1e-9)


def test_array_orientation_is_fits_row_0_is_south():
    """array[y, x]: y grows to the north and x grows to the west, as in a FITS image."""
    wcs = target_wcs(*BARNARD, 0.25)
    south, north = wcs.pixel_to_world(73, 0), wcs.pixel_to_world(73, 145)
    east, west = wcs.pixel_to_world(0, 73), wcs.pixel_to_world(145, 73)
    assert south.dec.deg < north.dec.deg  # row 0 = south
    assert east.ra.deg > west.ra.deg  # RA grows to the east, which is the left (x = 0)


def test_wcs_json_goes_and_comes_back():
    text = wcs_to_json(target_wcs(*BARNARD, 0.25))
    values = json.loads(text)
    assert (values["NAXIS1"], values["NAXIS2"]) == (146, 146)
    back = wcs_from_json(text)
    assert back.pixel_shape == (146, 146)
    original = target_wcs(*BARNARD, 0.25)
    for x, y in [(0, 0), (10, 120), (145, 145)]:
        a, b = original.pixel_to_world(x, y), back.pixel_to_world(x, y)
        assert a.separation(b).arcsec < 1e-6


def test_gold_and_silver_share_the_same_grid(tmp_path):
    """The wcs_json of targets.parquet is the grid that silver arrays use."""
    config = load_config()
    write_mock_gold(tmp_path / "gold", config)
    row = pq.read_table(tmp_path / "gold" / "targets.parquet").to_pylist()[0]
    first = config["targets"][0]
    grid = target_wcs(first["ra_now"], first["dec_now"], config["settings"]["cutout_size_deg"])
    assert row["wcs_json"] == wcs_to_json(grid)


# --- Which differences to make ---------------------------------------------------------------
def cells(**detectors):
    """{"d1": [1, 2, 3]} -> {(1, 1), (1, 2), (1, 3)}"""
    return {(int(name[1:]), season) for name, seasons in detectors.items() for season in seasons}


def test_three_seasons_give_consecutive_and_last_minus_first():
    pairs = difference_pairs(cells(d1=[1, 2, 3]))
    assert pairs == [(1, 1, 2), (1, 2, 3), (1, 1, 3)]


def test_two_seasons_give_one_difference_not_two():
    assert difference_pairs(cells(d6=[2, 3])) == [(6, 2, 3)]


def test_one_season_gives_none():
    assert difference_pairs(cells(d1=[1])) == []


def test_a_missing_season_is_skipped_not_made_up():
    # Detector 5 of Epsilon Indi B has no season 3: 2 -> 4 is the consecutive one it has.
    assert difference_pairs(cells(d5=[1, 2, 4])) == [(5, 1, 2), (5, 2, 4), (5, 1, 4)]


def test_the_white_image_and_all_seasons_are_not_cells():
    assert difference_pairs([(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]) == [(1, 1, 2)]


def test_season_a_is_always_before_season_b():
    pairs = difference_pairs(cells(d1=[3, 1, 2, 4], d2=[4, 2]))
    assert all(a < b for _, a, b in pairs)
    assert len(pairs) == len(set(pairs))


def test_the_five_targets_have_93_differences():
    """The count in AI's document, from the cells of config/targets.yaml."""
    expected = {"barnard": 18, "eps_indi_b": 21, "wise_0855": 16, "rw_aur": 18, "r_dra": 20}
    found = {}
    for target in load_config()["targets"]:
        target_cells = {(d, s["season"]) for s in target["seasons"] for d in s["detectors"]}
        found[target["id"]] = len(difference_pairs(target_cells))
    assert found == expected
    assert sum(found.values()) == 93


def test_the_simulated_gold_makes_the_same_differences():
    """The simulated gold uses the real rule. Its hole (no D6 in season 1) leaves 16 of 18."""
    config = load_config()
    gold = build_mock(config["targets"][0], config["settings"]["cutout_size_deg"])
    differences = gold.tables["differences.parquet"]
    sums = gold.tables["sums.parquet"]
    wanted = difference_pairs(zip(sums.detector, sums.season, strict=True))
    got = list(zip(differences.detector, differences.season_a, differences.season_b, strict=True))
    assert got == wanted
    assert len(differences) == 16
