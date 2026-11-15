"""Measured trajectory against the catalog motion (task A5).

Synthetic sums on the real target grid: no network and no silver files.
"""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from serafim import contract
from serafim.gold.export import to_contract_table
from serafim.silver.grid import target_wcs
from serafim.validation import trajectory as tr
from serafim.validation import visibility

SIGMA = 0.75
STAR = {"id": "star", "ra": 150.0, "dec": 10.0, "ra_now": 150.0, "dec_now": 10.0,
        "pmra_mas_yr": 0.0, "pmdec_mas_yr": 0.0, "parallax_mas": 0.0}  # fmt: skip


def synthetic_sum(offset=(0.3, -0.2), flux=200.0, noise=0.02, core=0.0, mjd=61000.0, seed=0):
    """A sum with a star `offset` px from where STAR's catalog puts it on `mjd`."""
    wcs = target_wcs(STAR["ra_now"], STAR["dec_now"], 0.25)
    ny, nx = wcs.pixel_shape[1], wcs.pixel_shape[0]
    expected = tr.catalog_position(STAR, Time(mjd, format="mjd"))
    tx, ty = (float(v) for v in wcs.world_to_pixel(expected))
    rows, cols = np.indices((ny, nx))
    width = 2.0 if core else SIGMA  # a saturated star's ring is wide
    image = flux * visibility._psf_at(cols, rows, tx + offset[0], ty + offset[1], width)
    image = image + np.random.default_rng(seed).normal(0, noise, (ny, nx))
    n_saturated = np.where(np.hypot(cols - tx - offset[0], rows - ty - offset[1]) < core, 5, 0)
    return {
        "image": image.astype("float32"), "variance": np.full((ny, nx), noise**2, "float32"),
        "mask": np.zeros((ny, nx), bool), "n_saturated": n_saturated.astype("int16"),
        "saturated": n_saturated > 0, "mean_mjd": mjd, "wcs": wcs,
    }, (tx, ty)  # fmt: skip


def test_the_fit_finds_the_star_where_it_really_is():
    arrays, (tx, ty) = synthetic_sum(offset=(0.3, -0.2))
    found = tr.measure_sum(arrays, 6, STAR)
    assert found["reason"] is None
    assert found["x"] - tx == pytest.approx(0.3, abs=0.05)
    assert found["y"] - ty == pytest.approx(-0.2, abs=0.05)
    assert 0 < found["sx"] < 0.1


def test_a_saturated_star_is_found_on_its_ring():
    arrays, (tx, ty) = synthetic_sum(offset=(0.4, 0.1), flux=5000.0, core=2.5)
    found = tr.measure_sum(arrays, 6, STAR, saturates=True)
    assert found["reason"] is None
    assert found["x"] - tx == pytest.approx(0.4, abs=0.1)
    assert found["y"] - ty == pytest.approx(0.1, abs=0.1)


def test_an_undetected_target_has_no_position():
    arrays, _ = synthetic_sum(flux=0.0)
    found = tr.measure_sum(arrays, 6, STAR)
    assert found["reason"] == "not detected"
    assert "x" not in found


def test_detectors_are_combined_with_a_floor():
    sums = pd.DataFrame({
        "detector": [5, 6], "reason": [None, None], "x": [10.0, 10.2], "y": [20.0, 20.0],
        "sx": [0.01, 0.01], "sy": [0.01, 0.01],
    })  # fmt: skip
    combined = tr.combine_season(sums)
    assert combined["x"] == pytest.approx(10.1)
    assert combined["ex"] >= tr.DETECTOR_FLOOR_PX / np.sqrt(2) - 1e-9  # not the tiny 0.01
    assert combined["detectors"] == [5, 6]
    assert tr.combine_season(sums.assign(reason="not detected")) is None


def test_the_parallax_is_seen_from_earth():
    near = {**STAR, "parallax_mas": 1000.0}  # 1 arcsec
    march, september = Time("2026-03-20"), Time("2026-09-22")
    shift = tr.catalog_position(near, march).separation(tr.catalog_position(near, september))
    assert 0.3 < shift.arcsec < 2.0  # opposite sides of the Sun: up to twice the parallax
    far = tr.catalog_position(STAR, march).separation(tr.catalog_position(STAR, september))
    assert far.arcsec < 0.001  # no parallax (10 kpc): nothing moves


def test_proper_motion_through_the_seasons():
    start = SkyCoord(150.0 * u.deg, 10.0 * u.deg)
    rows = []
    for season, years in ((1, 0.0), (2, 0.5), (3, 1.0)):
        here = start.spherical_offsets_by(-0.8 * years * u.arcsec, 10.4 * years * u.arcsec)
        rows.append({"season": season, "mjd": 61000 + 365.25 * years, "error_arcsec": 0.2,
                     "ra_measured": here.ra.deg, "dec_measured": here.dec.deg,
                     "ra_catalog": here.ra.deg, "dec_catalog": here.dec.deg})  # fmt: skip
    motion = tr.proper_motion(pd.DataFrame(rows))
    assert motion["measured_mas_yr"] == pytest.approx([-800.0, 10400.0], abs=5)
    assert motion["measured_total_arcsec_yr"] == pytest.approx(10.43, abs=0.01)
    assert motion["catalog_total_arcsec_yr"] == motion["measured_total_arcsec_yr"]


def test_the_table_follows_the_contract():
    rows = pd.DataFrame([
        {"target_id": "star", "season": 1, "ra_measured": None, "dec_measured": None,
         "error_arcsec": None, "ra_catalog": 150.0, "dec_catalog": 10.0},
        {"target_id": "star", "season": 2, "ra_measured": 150.0001, "dec_measured": 10.0001,
         "error_arcsec": 0.3, "ra_catalog": 150.0, "dec_catalog": 10.0},
    ])  # fmt: skip
    for name in ("ra_measured", "dec_measured", "error_arcsec"):
        rows[name] = pd.to_numeric(rows[name]).astype("float64")
    rows["season"] = rows["season"].astype("int32")
    to_contract_table(rows[tr.COLUMNS], contract.TRAJECTORY)
    to_contract_table(tr._empty(), contract.TRAJECTORY)
