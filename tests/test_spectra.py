"""Spectra by forced photometry (task A4): the PSF model, the fit and the output table.

Synthetic data only: no network and no bronze files.
"""

import numpy as np
import pandas as pd
import pytest

from serafim import contract
from serafim.gold.export import to_contract_table
from serafim.validation import spectra

OVER = 5
SIGMA = 0.7  # native px, like the SPHEREx EPSF
SIZE = 31


def fake_epsf(sigma=SIGMA, over=OVER, side=33):
    """A Gaussian 'EPSF' oversampled like the archive one (33 x 33, 5x)."""
    center = (side - 1) / 2
    y, x = np.mgrid[:side, :side]
    psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma * over) ** 2))
    return psf / psf.sum() * over**2


def scene(sources, noise=0.01, seed=0):
    """An image with point sources [(flux, x, y)], its variance and pixel grids."""
    psf = fake_epsf()
    y, x = np.mgrid[:SIZE, :SIZE]
    image = np.zeros((SIZE, SIZE))
    for flux, x0, y0 in sources:
        image += flux * spectra.psf_pixels(psf, OVER, x0, y0, x, y)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)
    return image, np.full(image.shape, noise**2), x, y, psf


def test_the_psf_model_sums_to_one_wherever_the_source_falls():
    psf = fake_epsf()
    y, x = np.mgrid[:SIZE, :SIZE]
    for x0, y0 in [(15.0, 15.0), (15.3, 14.6), (10.5, 20.5)]:
        assert spectra.psf_pixels(psf, OVER, x0, y0, x, y).sum() == pytest.approx(1.0, abs=0.01)


def test_fit_recovers_a_known_flux():
    image, variance, x, y, psf = scene([(10.0, 15.2, 14.8)])
    use = np.hypot(x - 15.2, y - 14.8) < spectra.RADIUS_PX
    columns = [spectra.psf_pixels(psf, OVER, 15.2, 14.8, x, y), np.ones_like(image)]
    values, errors, chi2 = spectra.fit_fluxes(image, variance, use, columns)
    assert values[0] == pytest.approx(10.0, abs=4 * errors[0])
    assert chi2 == pytest.approx(1.0, abs=0.5)


def test_a_saturated_core_is_fitted_on_its_wings():
    image, variance, x, y, psf = scene([(500.0, 15.0, 15.0)], noise=0.05)
    core = np.hypot(x - 15.0, y - 15.0) < 1.2
    image[core] = np.nan  # IRSA leaves saturated pixels empty
    use = (np.hypot(x - 15.0, y - 15.0) < spectra.WINGS_RADIUS_PX) & ~core
    columns = [spectra.psf_pixels(psf, OVER, 15.0, 15.0, x, y), np.ones_like(image)]
    values, _, _ = spectra.fit_fluxes(image, variance, use, columns)
    assert values[0] == pytest.approx(500.0, rel=0.10)


def test_a_bright_neighbour_is_fitted_together():
    """WISE 0855 passes 1.25 px from a much brighter star in season 2."""
    image, variance, x, y, psf = scene([(1.0, 15.0, 15.0), (4.0, 16.25, 15.0)], noise=0.003)
    use = np.hypot(x - 15.0, y - 15.0) < spectra.RADIUS_PX
    alone = [spectra.psf_pixels(psf, OVER, 15.0, 15.0, x, y), np.ones_like(image)]
    both = [alone[0], spectra.psf_pixels(psf, OVER, 16.25, 15.0, x, y), alone[1]]
    with_neighbour, _, _ = spectra.fit_fluxes(image, variance, use, both)
    without, _, _ = spectra.fit_fluxes(image, variance, use, alone)
    assert with_neighbour[0] == pytest.approx(1.0, abs=0.1)
    assert with_neighbour[1] == pytest.approx(4.0, abs=0.1)
    assert without[0] > 2.0  # ignoring the neighbour steals its light


def rows():
    return [
        {"target_id": "wise_0855", "frame_id": "wise_0855_2025W19_1B_0116_1", "mjd": 60800.7,
         "season": 1, "wavelength_um": 4.6, "flux_mjy": 0.58, "error_mjy": 0.2, "flag": "blend"},
        {"target_id": "wise_0855", "frame_id": "wise_0855_2025W19_1B_0116_2", "mjd": 60800.8,
         "season": 1, "wavelength_um": 1.2, "flux_mjy": 0.01, "error_mjy": 0.05, "flag": None},
    ]  # fmt: skip


def test_the_table_follows_the_contract():
    table = spectra._table(rows())
    assert list(table.columns) == list(contract.SPECTRA.names)
    to_contract_table(table, contract.SPECTRA)  # raises if a type does not fit
    assert str(table["date_utc"].dt.tz) == "UTC"
    assert table["flag"].isna().tolist() == [False, True]  # sorted by date: "blend" first


def test_an_empty_table_follows_the_contract():
    table = spectra._table([])
    assert list(table.columns) == list(contract.SPECTRA.names)
    to_contract_table(table, contract.SPECTRA)


def test_frame_id_is_the_agreed_one():
    assert spectra.frame_id("barnard", "2025W34_1B_0164_3") == "barnard_2025W34_1B_0164_3"


def test_doubtful_points_are_edges_and_flagged_pixels_only():
    flags = [None, "blend", "saturated", "edge", "blend,bad_pixels", "saturated,poor_fit"]
    table = pd.DataFrame({"flag": flags})
    assert spectra.doubtful(table).tolist() == [False, False, False, True, True, True]


def test_binned_spectrum_weights_by_the_errors():
    table = pd.DataFrame({
        "wavelength_um": [4.61, 4.62, 4.63],
        "flux_mjy": [1.0, 1.0, 4.0],
        "error_mjy": [0.1, 0.1, 10.0],
        "flag": [None, "blend", "edge"],
    })  # fmt: skip
    bins = spectra.binned(table)
    assert len(bins) == 1
    assert bins["flux_mjy"].iloc[0] == pytest.approx(1.0)  # the edge point is left out
    assert bins["n"].iloc[0] == 2


def test_band_mean_measures_the_methane_dip():
    table = pd.DataFrame({
        "wavelength_um": [3.3, 3.4, 4.0, 4.1, 4.2],
        "flux_mjy": [10.0, 12.0, 50.0, 48.0, 99.0],
        "error_mjy": [1.0, 1.0, 1.0, 1.0, 1.0],
        "flag": [None, None, None, None, "edge"],
    })  # fmt: skip
    dip, _ = spectra.band_mean(table, 3.2, 3.5)
    peak, _ = spectra.band_mean(table, 3.9, 4.3)
    assert dip == pytest.approx(11.0)
    assert peak == pytest.approx(49.0)  # the edge point is left out
    assert spectra.band_mean(table, 1.0, 2.0) is None


def test_catalog_check_turns_magnitudes_into_mjy():
    target = {"mag": {"W2": 13.704}}
    table = pd.DataFrame({"wavelength_um": [4.6], "flux_mjy": [0.58], "error_mjy": [0.05],
                          "flag": [None]})  # fmt: skip
    (row,) = spectra.catalog_check(target, table)
    assert row["band"] == "W2"
    assert row["catalog_mjy"] == pytest.approx(0.567, abs=0.005)
