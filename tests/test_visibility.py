"""Visibility check (before task A6): forced photometry and position fits.

Synthetic stars on known noise, no network and no bronze files.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from serafim.validation import visibility as vis

SIZE = vis.STAMP_SIZE
SIGMA = 0.7  # px, like the SPHEREx EPSF
CENTER = SkyCoord(100.0 * u.deg, 10.0 * u.deg)


def star_stamp(flux_unit, noise, rng, x0=None, y0=None, sigma=SIGMA, mjd=60000.0):
    """A stamp dict like vis.stamp() returns: a Gaussian star on white noise."""
    image = flux_unit * vis.gaussian(SIZE, sigma, x0, y0) + rng.normal(0, noise, (SIZE, SIZE))
    return {
        "image": image,
        "variance": np.full((SIZE, SIZE), noise**2),
        "bad": np.zeros((SIZE, SIZE), dtype=bool),
        "unreliable": np.zeros((SIZE, SIZE), dtype=bool),
        "mjd": mjd,
        "psf_sigma": sigma,
    }


def test_forced_flux_recovers_the_flux_in_mjy():
    rng = np.random.default_rng(1)
    stacked = vis.stack([star_stamp(5.0, 0.05, rng)])
    result = vis.forced_flux(stacked, sigma=SIGMA)
    assert result["flux_mjy"] == pytest.approx(5.0 * vis.MJY_PER_PIXEL_UNIT, rel=0.05)
    assert result["snr"] > 20


def test_empty_sky_is_not_seen():
    rng = np.random.default_rng(2)
    stacked = vis.stack([star_stamp(0.0, 0.05, rng)])
    result = vis.forced_flux(stacked, sigma=SIGMA)
    assert abs(result["snr"]) < vis.DOUBTFUL
    assert vis.verdict(result["snr"]) == "not seen"


def test_stacking_lowers_the_noise_like_sqrt_n():
    rng = np.random.default_rng(3)
    one = vis.forced_flux(vis.stack([star_stamp(0.5, 0.05, rng)]), sigma=SIGMA)["error_mjy"]
    many = [star_stamp(0.5, 0.05, rng) for _ in range(16)]
    sixteen = vis.forced_flux(vis.stack(many), sigma=SIGMA)["error_mjy"]
    assert one / sixteen == pytest.approx(4.0, rel=0.25)


def test_flagged_pixels_are_left_out_of_the_stack():
    rng = np.random.default_rng(4)
    good = star_stamp(1.0, 0.01, rng)
    broken = star_stamp(1.0, 0.01, rng)
    broken["image"][SIZE // 2, SIZE // 2] = 1e6
    broken["bad"][SIZE // 2, SIZE // 2] = True
    stacked = vis.stack([good, broken])
    assert stacked["image"][SIZE // 2, SIZE // 2] < 1.0
    assert stacked["n_valid"][SIZE // 2, SIZE // 2] == 1


def test_position_fit_ignores_a_saturated_core():
    rng = np.random.default_rng(5)
    x0, y0 = 20.6, 19.3
    s = star_stamp(200.0, 0.02, rng, x0=x0, y0=y0, sigma=1.2)
    core = np.hypot(*np.meshgrid(np.arange(SIZE) - x0, np.arange(SIZE) - y0)) < 1.6
    s["image"][core] = np.nan  # IRSA leaves saturated pixels empty
    s["unreliable"][core] = True
    fitted = vis.fit_position(vis.stack([s]), guess=(20.0, 20.0))
    assert fitted is not None
    assert fitted["x"] == pytest.approx(x0, abs=0.1)
    assert fitted["y"] == pytest.approx(y0, abs=0.1)


def test_blend_fit_separates_a_faint_target_from_a_bright_neighbour():
    """WISE 0855 passes 1.25 px from a G = 16.3 star in season 2: both are fitted together."""
    rng = np.random.default_rng(7)
    sigma, c = 0.8, SIZE // 2
    image = 1.0 * vis.gaussian(SIZE, sigma, c, c) + 4.0 * vis.gaussian(SIZE, sigma, c + 1.25, c)
    noisy = image + rng.normal(0, 0.003, image.shape)
    s = star_stamp(0.0, 0.003, rng, sigma=sigma)
    s["image"] = noisy
    fit = vis.blend_fit(vis.stack([s]), (c, c), [(c + 1.25, c)], sigma=sigma)
    assert fit["flux_mjy"] / vis.MJY_PER_PIXEL_UNIT == pytest.approx(1.0, abs=0.1)
    assert fit["neighbour_mjy"][0] / vis.MJY_PER_PIXEL_UNIT == pytest.approx(4.0, abs=0.1)
    alone = vis.blend_fit(vis.stack([s]), (c, c), [], sigma=sigma)
    assert alone["flux_mjy"] > 1.5 * fit["flux_mjy"]  # without the neighbour model it is inflated


def test_verdict_thresholds():
    assert vis.verdict(7.0) == "seen"
    assert vis.verdict(4.0) == "doubtful"
    assert vis.verdict(1.0) == "not seen"
    assert vis.verdict(float("nan")) == "no data"


def make_frame(star_at: SkyCoord, flux_unit=50.0, mjd=60000.0):
    """A synthetic Frame: a star at `star_at` on a TAN grid of 6.15 arcsec pixels."""
    wcs = vis.grid_wcs(CENTER, size=101)
    x, y = wcs.world_to_pixel(star_at)
    image = flux_unit * vis.gaussian(101, SIGMA, float(x), float(y))
    image += np.random.default_rng(6).normal(0, 0.01, image.shape)
    zeros = np.zeros(image.shape, dtype=bool)
    return vis.Frame("synthetic", 6, 1, mjd, image, np.full(image.shape, 1e-4), zeros,
                     zeros.copy(), wcs, SIGMA)  # fmt: skip


def test_stamp_centers_on_the_given_position():
    """A stamp centered on where the star is that day keeps the star at the center,
    which is what makes the co-moving stacks follow a moving target."""
    there = CENTER.spherical_offsets_by(20 * u.arcsec, -12 * u.arcsec)
    made = vis.stamp(make_frame(there), there)
    peak = np.unravel_index(np.nanargmax(made["image"]), made["image"].shape)
    assert peak == (SIZE // 2, SIZE // 2)


def test_stamp_outside_the_frame_is_none():
    far = CENTER.spherical_offsets_by(1 * u.deg, 0 * u.deg)
    assert vis.stamp(make_frame(CENTER), far) is None


def test_flag_bits_come_from_the_mskn_keys():
    from astropy.io import fits

    header = fits.Header()
    header["MSKN1"] = "OVERFLOW"
    header["MSKN21"] = "SOURCE"
    bits = vis.flag_bits(header)
    assert bits == {"OVERFLOW": 1, "SOURCE": 21}
    flags = np.array([[2, 2**21, 2 + 2**21]])
    assert vis.any_flag(flags, bits, ("OVERFLOW",)).tolist() == [[True, False, True]]
