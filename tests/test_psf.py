"""The PSF measured on the real sums (training/psf.py, CNN version 2)."""

import numpy as np
import pytest

pytest.importorskip("torch")  # training/ needs the ai extra

from training import psf  # noqa: E402


def gaussian_cuts(sigma=0.8, count=40, seed=0):
    """Star cuts (15 x 15) of a Gaussian at random sub-pixel positions, with noise."""
    rng = np.random.default_rng(seed)
    size = 2 * (psf.RADIUS_PX + 1) + 1
    rows, cols = np.indices((size, size))
    out = []
    for _ in range(count):
        x0, y0 = (size - 1) / 2 + rng.uniform(-0.5, 0.5, 2)
        star = np.exp(-((cols - x0) ** 2 + (rows - y0) ** 2) / (2 * sigma**2)) / (
            2 * np.pi * sigma**2
        )
        out.append((100 * star + 3.0 + rng.normal(0, 0.01, star.shape), x0, y0))
    return out


def test_the_measured_psf_has_unit_flux_and_the_right_width():
    measured = psf.Psf(psf.measure(gaussian_cuts()), 40)
    image = measured.image((31, 31), 15.3, 14.8)
    assert image.sum() == pytest.approx(1.0, abs=0.05)
    rows, cols = np.indices(image.shape)
    width = np.sqrt(((cols - 15.3) ** 2 * image).sum() / image.sum())
    assert width == pytest.approx(0.8, abs=0.1)
    assert image[0, 0] == 0  # nothing beyond the taper


def test_the_psf_follows_the_source():
    measured = psf.Psf(psf.measure(gaussian_cuts()), 40)
    image = measured.image((31, 31), 12.0, 18.0)
    y, x = np.unravel_index(np.argmax(image), image.shape)
    assert (x, y) == (12, 18)
