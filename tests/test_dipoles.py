"""Motion dipoles by the classic rule (task A9, stage 1): synthetic differences, no network."""

import numpy as np
import pandas as pd

from serafim.validation import dipoles as dp

SHAPE = (80, 80)


def gaussian(x, y, flux, sigma=0.8):
    rows, cols = np.indices(SHAPE)
    return (
        flux
        / (2 * np.pi * sigma**2)
        * np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * sigma**2))
    )


def difference(*sources, noise=0.05, seed=3):
    """b minus a: each source is (x_a, y_a, x_b, y_b, flux_a, flux_b)."""
    image = np.random.default_rng(seed).normal(0, noise, SHAPE)
    for xa, ya, xb, yb, fa, fb in sources:
        image += gaussian(xb, yb, fb) - gaussian(xa, ya, fa)
    return image


EMPTY = np.zeros(SHAPE, bool)


def test_a_star_that_moved_leaves_one_dipole():
    found = dp.find_dipoles(difference((40, 40, 41.5, 40.5, 30, 30)), EMPTY, EMPTY)
    assert len(found) == 1
    d = found.iloc[0]
    assert d["x_from"] < 40.5 < d["x_to"]  # from where it was to where it went
    assert 1.0 < d["separation_px"] <= dp.MAX_SEPARATION_PX


def test_a_star_that_only_brightens_is_no_dipole():
    """RW Aurigae and R Draconis: one blob of one sign."""
    assert dp.find_dipoles(difference((40, 40, 40, 40, 30, 60)), EMPTY, EMPTY).empty


def test_two_blobs_of_very_different_flux_are_not_paired():
    found = dp.find_dipoles(difference((40, 40, 41.5, 40, 5, 40)), EMPTY, EMPTY)
    assert found.empty


def test_the_halo_lobes_of_a_saturated_star_are_paired_across_its_core():
    """Barnard's Star: the core is masked, only the halo on each side is seen."""
    saturated = np.zeros(SHAPE, bool)
    saturated[38:43, 38:43] = True
    image = difference((40, 37.5, 40, 43.5, 200, 200))  # lobes 6 px apart
    image[saturated] = np.nan
    found = dp.find_dipoles(image, saturated, saturated)
    assert len(found) == 1 and bool(found.iloc[0]["halo"])
    assert found.iloc[0]["core_px"] > 2


def test_a_lobe_split_by_the_core_is_merged():
    saturated = np.zeros(SHAPE, bool)
    saturated[38:43, 38:43] = True
    found = pd.DataFrame({"sign": [-1, -1, 1], "x": [37.0, 43.0, 40.0], "y": [37.0, 37.0, 43.5],
                          "flux": [50.0, 50.0, 100.0], "peak": [9.0, 9.0, 12.0],
                          "npix": [5, 5, 9]})  # fmt: skip
    merged = dp.merge_halo_blobs(found, saturated)
    negative = merged[merged["sign"] < 0]
    assert len(negative) == 1 and negative["flux"].iloc[0] == 100.0
    assert negative["x"].iloc[0] == 40.0


def sources(*rows):
    names = ("catalog", "name", "mag", "xa", "ya", "xb", "yb")
    table = pd.DataFrame([dict(zip(names, row, strict=True)) for row in rows])
    return table.assign(moved_px=np.hypot(table["xb"] - table["xa"], table["yb"] - table["ya"]))


def dipole(**changes):
    row = {"x_from": 40.0, "y_from": 40.0, "x_to": 41.5, "y_to": 40.0, "separation_px": 1.5,
           "flux_ratio": 1.0, "halo": False, "core_px": 0.0, "core_x": np.nan, "core_y": np.nan,
           "snr": 20.0}  # fmt: skip
    return {**row, **changes}


def test_status_known_residual_edge_and_candidate():
    mover = sources(("simbad", "a brown dwarf", 14.0, 40.2, 40.1, 41.4, 40.0))
    fixed = sources(("gaia", "a fixed star", 12.0, 40.8, 40.0, 40.8, 40.0))
    assert dp.status(dipole(), mover, EMPTY, SHAPE) == (dp.KNOWN, "a brown dwarf (simbad)")
    state, reason = dp.status(dipole(), fixed, EMPTY, SHAPE)
    assert state == dp.DISCARDED and reason.startswith("residual of a fixed star")
    assert dp.status(dipole(x_from=2.0), mover, EMPTY, SHAPE)[1] == "at the edge"
    masked = EMPTY.copy()
    masked[40, 41] = True
    assert dp.status(dipole(), mover, masked, SHAPE)[1] == "on masked pixels"
    far = sources(("gaia", "elsewhere", 12.0, 10.0, 10.0, 10.0, 10.0))
    assert dp.status(dipole(), far, EMPTY, SHAPE)[0] == dp.CANDIDATE


def test_the_star_names_a_known_dipole_before_its_planets():
    both = sources(("simbad", "NAME Barnard's Star c", np.nan, 40.0, 40.0, 41.5, 40.0),
                   ("gaia", "Gaia DR3 1", 8.2, 40.0, 40.0, 41.5, 40.0),
                   ("simbad", "NAME Barnard's star", 9.5, 40.0, 40.0, 41.5, 40.0))  # fmt: skip
    assert dp.status(dipole(), both, EMPTY, SHAPE)[1] == "NAME Barnard's star (simbad)"


def test_the_glare_of_a_saturated_star_is_discarded():
    """Epsilon Indi A: a halo dipole far from where the star moved."""
    star = sources(("simbad", "* eps Ind", 4.7, 60.0, 40.0, 60.6, 40.0))
    glare = dipole(x_from=50.0, x_to=52.0, halo=True, core_px=4.0, core_x=60.0, core_y=40.0)
    state, reason = dp.status(glare, star, EMPTY, SHAPE)
    assert state == dp.DISCARDED and "saturated core of * eps Ind" in reason


def test_a_target_dipole_is_near_the_grid_center():
    rows = pd.DataFrame({"x_from": [72.0, 10.0], "y_from": [70.0, 10.0], "x_to": [72.5, 12.0],
                         "y_to": [74.0, 10.0]})  # fmt: skip
    assert dp.at_target(rows).tolist() == [True, False]
