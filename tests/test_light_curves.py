"""Light curves of saturated targets by ring photometry (task A4, second part).

Synthetic stars and points: no network and no bronze files.
"""

import numpy as np
import pandas as pd
import pytest

from serafim.validation import light_curves as lc

SIZE = 49
CENTER = 24.0


def stamp(flux=1000.0, sigma=3.0, background=5.0, saturated_radius=0.0, neighbours=(), seed=0):
    """A RingStamp of a broad star, with an optional saturated core and neighbour stars."""
    y, x = np.indices((SIZE, SIZE))
    distance = np.hypot(x - CENTER, y - CENTER)
    image = flux * np.exp(-(distance**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    for nx, ny, nflux in neighbours:  # a point source as wide as the SPHEREx PSF (0.65 px)
        image += (
            nflux * np.exp(-((x - nx) ** 2 + (y - ny) ** 2) / (2 * 0.65**2)) / (2 * np.pi * 0.65**2)
        )
    image += background + np.random.default_rng(seed).normal(0, 0.01, image.shape)
    unreliable = distance <= saturated_radius if saturated_radius else np.zeros_like(image, bool)
    image = np.where(unreliable, np.nan, image)
    return lc.RingStamp(
        frame_id="f", detector=1, season=1, mjd=60000.0, wavelength_um=1.0, image=image,
        variance=np.full(image.shape, 1e-4), bad=np.isnan(image), unreliable=unreliable,
        x0=CENTER, y0=CENTER, saturated_radius=lc.saturated_radius(unreliable, CENTER, CENTER),
        neighbours=[(nx, ny) for nx, ny, _ in neighbours],
    )  # fmt: skip


def expected_ring(flux, sigma, inner, outer):
    """Light of a 2D Gaussian between two radii."""
    return flux * (np.exp(-(inner**2) / (2 * sigma**2)) - np.exp(-(outer**2) / (2 * sigma**2)))


def test_saturated_radius_is_the_farthest_flagged_pixel():
    s = stamp(saturated_radius=3.0)
    assert s.saturated_radius == pytest.approx(3.0, abs=0.01)
    assert stamp().saturated_radius == 0.0


def test_ring_flux_measures_the_ring_without_the_background():
    s = stamp(flux=1000.0, sigma=3.0, background=5.0, saturated_radius=3.0)
    measured = lc.ring_flux(s, 4.0, 7.0)
    assert measured["ring"] == pytest.approx(expected_ring(1000.0, 3.0, 4.0, 7.0), rel=0.03)
    assert measured["ring_error"] > 0


def test_ring_flux_does_not_depend_on_where_the_star_falls_in_its_pixel():
    """A hard-edged ring over-counts the bright inner edge by an amount that changes with the
    sub-pixel position; the subpixel weights remove that frame-to-frame scatter."""
    rings = []
    for shift in (0.0, 0.25, 0.5):
        s = stamp()
        y, x = np.indices((SIZE, SIZE))
        distance = np.hypot(x - CENTER - shift, y - CENTER)
        s.image = 1000.0 * np.exp(-(distance**2) / 18) / (2 * np.pi * 9) + 5.0
        s.x0 = CENTER + shift
        rings.append(lc.ring_flux(s, 4.0, 7.0)["ring"])
    assert max(rings) / min(rings) == pytest.approx(1.0, abs=0.01)


def test_a_bright_neighbour_in_the_ring_is_masked():
    clean = lc.ring_flux(stamp(), 4.0, 7.0)["ring"]
    crowded = lc.ring_flux(stamp(neighbours=[(CENTER + 5.5, CENTER, 300.0)]), 4.0, 7.0)["ring"]
    assert crowded == pytest.approx(clean, rel=0.05)


def test_a_ring_mostly_masked_gives_nothing():
    s = stamp()
    s.bad[:, : int(CENTER) + 1] = True  # half the image, plus a column
    assert lc.ring_flux(s, 4.0, 7.0) is None


def points(ratios=(1.0, 1.5, 0.7), per_season=10, noise=0.02, seed=1):
    """Ring fluxes of a star whose brightness changes by `ratios`, seen at random wavelengths."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, ratio in enumerate(ratios, start=1):
        for wavelength in rng.uniform(1.0, 1.6, per_season):
            sed = np.exp(2 + 0.5 * (wavelength - 1.3) - 0.8 * (wavelength - 1.3) ** 2)
            ring = sed * ratio * (1 + rng.normal(0, noise))
            rows.append({"frame_id": f"s{season}_{wavelength:.4f}", "season": season,
                         "mjd": 60000.0 + 180 * season, "wavelength_um": wavelength,
                         "ring": ring, "ring_error": noise * ring})  # fmt: skip
    return pd.DataFrame(rows)


def test_fit_recovers_the_season_ratios_at_the_same_wavelength():
    seasons, frames = lc.fit_detector(points())
    found = seasons.set_index("season")
    assert found.loc[1, "ratio"] == 1.0
    assert found.loc[2, "ratio"] == pytest.approx(1.5, abs=3 * found.loc[2, "error"])
    assert found.loc[3, "ratio"] == pytest.approx(0.7, abs=3 * found.loc[3, "error"])
    by_season = frames.groupby("season")["relative_brightness"].median()
    assert by_season.loc[2] == pytest.approx(1.5, rel=0.05)


def test_one_outlier_frame_is_clipped():
    table = points(ratios=(1.0, 1.0))
    table.loc[3, "ring"] *= 3  # a frame hit by something
    seasons, frames = lc.fit_detector(table)
    assert len(frames) == len(table) - 1
    assert seasons.set_index("season").loc[2, "ratio"] == pytest.approx(1.0, abs=0.05)


def test_a_single_season_has_no_light_curve():
    seasons, frames = lc.fit_detector(points(ratios=(1.0,)))
    assert seasons.empty and frames.empty
    assert lc.fit_detector(pd.DataFrame())[0].empty


def pulsating(period_days=247.0, amplitude=0.5, noise=0.08, every_days=20, seed=2):
    """Per-frame brightness of a star pulsating with `period_days`, seen every few weeks."""
    rng = np.random.default_rng(seed)
    days = np.arange(0, 520, every_days) + rng.uniform(0, 5, len(np.arange(0, 520, every_days)))
    brightness = np.exp(amplitude * np.sin(2 * np.pi * days / period_days) + rng.normal(0, noise,
                        len(days)))  # fmt: skip
    dates = pd.Timestamp("2025-04-24", tz="UTC") + pd.to_timedelta(days, unit="D")
    return pd.DataFrame({"date_utc": dates, "detector": 3, "relative_brightness": brightness,
                         "error": 0.05 * brightness})  # fmt: skip


def test_best_period_finds_a_mira_cycle():
    found = lc.best_period(pulsating(), shuffles=120)
    assert found["period_days"] == pytest.approx(247.0, rel=0.08)
    assert found["periodic"]


def test_noise_has_no_period():
    flat = pulsating(amplitude=0.0)
    found = lc.best_period(flat, shuffles=120)
    assert not found["periodic"]


def test_three_clumps_of_seasons_cannot_claim_a_period():
    """Like RW Aurigae: real changes, but only three clumps of dates."""
    clumps = pulsating(every_days=4)
    days = (clumps["date_utc"] - clumps["date_utc"].min()).dt.days
    clumps = clumps[(days < 30) | ((days > 180) & (days < 210)) | (days > 360) & (days < 390)]
    found = lc.best_period(clumps, shuffles=120)
    assert found["months"] < lc.MIN_MONTHS_FOR_PERIOD
    assert not found["periodic"]


def test_combine_leaves_out_uncertain_ratios():
    summary = [
        {"detector": 1, "season": 1, "ratio": 1.0, "error": 0.0, "n_frames": 9},
        {"detector": 2, "season": 1, "ratio": 1.0, "error": 0.0, "n_frames": 9},
        {"detector": 1, "season": 2, "ratio": 1.2, "error": 0.06, "n_frames": 9},
        {"detector": 2, "season": 2, "ratio": 1.2, "error": 0.06, "n_frames": 9},
        {"detector": 3, "season": 2, "ratio": 2.7, "error": 1.6, "n_frames": 9},  # too uncertain
        {"detector": 4, "season": 2, "ratio": 3.0, "error": 0.3, "n_frames": 1},  # one frame
    ]
    combined = {row["season"]: row for row in lc.combine(summary)}
    assert combined[1]["ratio"] == 1.0
    assert combined[2]["ratio"] == pytest.approx(1.2, rel=0.01)
    assert combined[2]["detectors"] == 2
