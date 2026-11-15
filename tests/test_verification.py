"""Verification of the targets on gold (task A6): every check on small hand-made gold tables.

No network and no silver: the detections (S/N per sum) are given as a table.
"""

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from serafim.validation import verification as ver

UTC = "UTC"
START = SkyCoord(150.0 * u.deg, 10.0 * u.deg)
SEASON_DATES = {1: "2025-05-01", 2: "2025-11-01", 3: "2026-05-01"}


def target(kind="motion", saturated=(), **extra):
    return {
        "id": "star", "name": "A star", "kind": kind, "n_frames": 200,
        "ra_now": 150.0, "dec_now": 10.0, "pmra_mas_yr": 0.0, "pmdec_mas_yr": 10000.0,
        "parallax_mas": 0.0, "motion": {"pm_total_arcsec_yr": 10.0},
        "saturation": [{"detector": d, "checked": 2, "saturated": 2 * (d in saturated)}
                       for d in range(1, 7)],
        **extra,
    }  # fmt: skip


def gold(seasons=(1, 2, 3), saturated=(), measured_shift=10.0, error=0.2, missing=(), unseen=()):
    """Gold tables of one target that moves 10 arcsec/yr north (measured: `measured_shift`).

    `missing`: (detector, season) with frames but no sum; `unseen`: with neither.
    """
    frames, sums, trajectory, catalogs = [], [], [], []
    for season in seasons:
        date = pd.Timestamp(SEASON_DATES[season], tz=UTC)
        years = (date - pd.Timestamp(SEASON_DATES[1], tz=UTC)).days / 365.25
        for detector in range(1, 7):
            if (detector, season) in unseen:
                continue
            frames.append({"target_id": "star", "season": season, "detector": detector,
                           "flags": "OVERFLOW" if detector in saturated else ""})  # fmt: skip
            if (detector, season) in missing:
                continue
            sums.append({"target_id": "star", "detector": detector, "season": season,
                         "mean_date_utc": date, "saturated": detector in saturated})  # fmt: skip
        catalog = START.directional_offset_by(0 * u.deg, 10.0 * years * u.arcsec)
        here = START.directional_offset_by(0 * u.deg, measured_shift * years * u.arcsec)
        trajectory.append({"target_id": "star", "season": season,
                           "ra_measured": here.ra.deg, "dec_measured": here.dec.deg,
                           "error_arcsec": error, "ra_catalog": catalog.ra.deg,
                           "dec_catalog": catalog.dec.deg})  # fmt: skip
        catalogs.append({"target_id": "star", "catalog": "gaia", "season": season,
                         "ra": catalog.ra.deg, "dec": catalog.dec.deg})  # fmt: skip
    return {
        "targets": pd.DataFrame([{"target_id": "star"}]),
        "frames": pd.DataFrame(frames),
        "sums": pd.DataFrame(sums),
        "differences": pd.DataFrame({"target_id": ["star"] * 6}),
        "catalogs": pd.DataFrame(catalogs),
        "spectra": pd.DataFrame(columns=["wavelength_um", "flux_mjy", "error_mjy", "flag"]),
        "trajectory": pd.DataFrame(trajectory),
    }


def found(snr_by_cell: dict) -> pd.DataFrame:
    rows = [{"detector": d, "season": s, "snr": snr} for (d, s), snr in snr_by_cell.items()]
    return pd.DataFrame(rows)


def test_a_target_with_gold_and_a_video_passes():
    result = ver.gold_check(target(), gold(), downloaded=18)
    assert result["status"] == ver.PASS
    assert "18 frames in 3 seasons" in result["detail"]


def test_one_season_is_no_video_and_a_missing_target_is_coming_soon():
    assert ver.gold_check(target(), gold(seasons=(1,)), None)["status"] == ver.FAIL
    empty = {**gold(), "targets": pd.DataFrame(columns=["target_id"])}
    result = ver.gold_check(target(), empty, None)
    assert result["status"] == ver.FAIL and "Coming soon" in result["detail"]


def test_frames_without_their_sum_are_wrong_gold():
    result = ver.gold_check(target(), gold(missing={(6, 1)}), downloaded=18)
    assert result["status"] == ver.FAIL and "Frames without their sum" in result["detail"]
    assert ver.gold_check(target(), gold(), downloaded=40)["status"] == ver.WARN  # frames dropped


def test_a_detector_spherex_did_not_observe_is_only_told():
    # WISE 0855 has no detector 6 frames in season 1
    result = ver.gold_check(target(), gold(unseen={(6, 1)}), downloaded=17)
    assert result["status"] == ver.PASS
    assert "did not observe it in (detector, season) (6, 1)" in result["detail"]


def test_a_guide_must_not_recommend_a_detector_where_nothing_is_seen():
    # WISE 0855: seen in D6, in D5 only with every season added
    seen = found({(6, 1): 7.0, (6, 2): 9.0, (5, 1): 3.0, (5, 2): 2.0, (5, 0): 6.3, (1, 0): 1.0})
    assert ver.seen_check(seen, [6, 5])["status"] == ver.PASS
    result = ver.seen_check(seen, [6, 1])
    assert result["status"] == ver.FAIL and "Not detected in detector 1" in result["detail"]


def test_a_season_where_the_target_is_not_seen_is_a_warning():
    seen = found({(6, 1): 2.0, (6, 2): 9.0, (6, 3): 8.0, (6, 0): 10.0})
    result = ver.seen_check(seen, [6])
    assert result["status"] == ver.WARN and "season 1 alone" in result["detail"]
    assert ver.seen_check(found({}), [6])["status"] == ver.WARN  # no silver here


def test_saturation_painted_in_gold_agrees_with_the_guide():
    data = gold(saturated=(5, 6))
    assert ver.saturation_check(target(saturated=(5, 6)), data)["status"] == ver.PASS
    result = ver.saturation_check(target(saturated=(4, 5, 6)), data)
    assert result["status"] == ver.WARN
    assert (
        "detectors 5, 6" in result["detail"] and "the guide says always 4, 5, 6" in result["detail"]
    )
    assert "OVERFLOW 6" in result["detail"]  # 3 seasons x 2 detectors


def test_saturated_in_only_some_seasons_is_told_apart():
    data = gold(saturated=(5,))
    data["sums"].loc[
        (data["sums"]["detector"] == 5) & (data["sums"]["season"] == 1), "saturated"
    ] = False
    detail = ver.saturation_check(target(saturated=(5,)), data)["detail"]
    assert "every season of detectors none and in some seasons of 5" in detail
    assert "the guide says always 5, sometimes none" in detail


def test_the_measured_motion_agrees_with_the_catalog():
    result = ver.motion_check(target(), gold())
    assert result["status"] == ver.PASS
    assert "Measured 10.00" in result["detail"] and "against 10.00" in result["detail"]


def test_a_target_that_does_not_move_in_gold_fails():
    assert ver.motion_check(target(), gold(measured_shift=0.0))["status"] == ver.FAIL


def test_an_imprecise_or_partial_motion_is_a_warning():
    assert ver.motion_check(target(), gold(error=3.0))["status"] == ver.WARN  # like WISE 0855
    data = gold()
    data["trajectory"].loc[0, ["ra_measured", "dec_measured", "error_arcsec"]] = None
    result = ver.motion_check(target(), data)
    assert result["status"] == ver.WARN and "Not measured in season 1" in result["detail"]
    assert ver.motion_check(target(kind="brightness"), data) is None


def spectrum(dip, peak):
    rows = [{"wavelength_um": um, "flux_mjy": dip if um < 3.6 else peak, "error_mjy": 0.05,
             "flag": None} for um in (3.25, 3.3, 3.4, 3.95, 4.1, 4.2)]  # fmt: skip
    return pd.DataFrame(rows)


def test_a_methane_brown_dwarf_is_dim_at_3_3_microns():
    data = {**gold(), "spectra": spectrum(0.25, 1.0)}
    result = ver.spectrum_check(target(kind="spectrum"), data)
    assert (
        result["status"] == ver.PASS and "Methane: 3.3 um over 4.1 um is 0.25" in result["detail"]
    )
    flat = {**gold(), "spectra": spectrum(1.0, 1.0)}
    assert ver.spectrum_check(target(kind="spectrum"), flat)["status"] == ver.FAIL


def test_a_saturated_spectrum_is_a_warning_the_site_has_to_show():
    table = spectrum(1.0, 1.0).assign(flag="saturated,poor_fit")
    result = ver.spectrum_check(target(saturated=(1,)), {**gold(), "spectra": table})
    assert result["status"] == ver.WARN and "hollow" in result["detail"]


def test_the_catalog_band_is_checked_against_spherex():
    table = pd.DataFrame({"wavelength_um": [4.55, 4.6, 4.65], "flux_mjy": [0.6, 0.62, 0.58],
                          "error_mjy": [0.1] * 3, "flag": [None] * 3})  # fmt: skip
    bright = target(kind="motion", mag={"W2": 14.0})
    detail = ver.spectrum_check(bright, {**gold(), "spectra": table})["detail"]
    assert "W2: SPHEREx/catalog" in detail


def curve(period=None, rows=()):
    return {"star": {"by_detector": list(rows), "period": period}}


def test_the_period_found_by_spherex_alone_matches_the_catalog():
    r_dra = target(kind="brightness", id="r_dra")
    found_period = {"periodic": True, "period_days": 243.5, "false_alarm": 0.005}
    assert ver.brightness_check(r_dra, {"r_dra": {"period": found_period}})["status"] == ver.PASS
    wrong = {"periodic": True, "period_days": 300.0, "false_alarm": 0.005}
    assert ver.brightness_check(r_dra, {"r_dra": {"period": wrong}})["status"] == ver.FAIL
    assert ver.brightness_check(r_dra, {})["status"] == ver.FAIL


def ratio(detector, season, value, error=0.08, n_frames=10):
    return {"detector": detector, "season": season, "ratio": value, "error": error,
            "n_frames": n_frames, "mean_date_utc": f"2026-0{season}-01T00:00:00"}  # fmt: skip


def test_a_change_of_brightness_beyond_the_control_passes():
    rows = [ratio(1, 1, 1.0, 0.0), ratio(1, 2, 1.5)]
    result = ver.brightness_check(target(kind="brightness"), curve(rows=rows))
    assert result["status"] == ver.PASS and "Detector 1: 1.50" in result["detail"]
    still = [ratio(1, 1, 1.0, 0.0), ratio(1, 2, 1.1)]
    assert ver.brightness_check(target(kind="brightness"), curve(rows=still))["status"] == ver.WARN


def test_barnard_is_the_control_that_must_not_change():
    rows = [ratio(5, 1, 1.0, 0.0), ratio(5, 2, 1.03)]
    result = ver.brightness_check(target(), curve(rows=rows))
    assert result["status"] == ver.PASS and "largest certain ratio is 1.03" in result["detail"]
    shaky = [ratio(2, 1, 1.0, 0.0), ratio(2, 3, 1.31)]  # Barnard reached 1.30 in D2
    result = ver.brightness_check(target(), curve(rows=shaky))
    assert result["status"] == ver.WARN and "(detector 2, season 3)" in result["detail"]
    assert ver.brightness_check(target(), {}) is None  # a moving target without a light curve


def test_a_catalog_marker_sits_on_the_target():
    assert ver.catalogs_check(target(), gold())["status"] == ver.PASS
    data = gold()
    data["catalogs"]["dec"] += 30 / 3600  # 30 arcsec off: about 5 pixels
    assert ver.catalogs_check(target(), data)["status"] == ver.WARN
    assert ver.catalogs_check(target(), {**data, "catalogs": data["catalogs"][:0]})["status"] == (
        ver.FAIL
    )


def test_the_report_and_summary_list_every_check():
    seen = found({(6, 1): 7.0, (6, 2): 9.0, (6, 3): 8.0, (6, 0): 12.0, (1, 0): 1.0})
    checks = ver.verify_target(target(), gold(), seen, recommended=[6], light_curves={})
    names = [c["check"] for c in checks]
    assert names == ["gold", "seen", "saturation", "motion", "spectrum", "catalogs"]
    results = {"star": (target(), checks, ver.with_saturation(seen, gold()["sums"]), ([], []))}
    text = ver.report(results, {"commit": "abcdef123", "run_date_utc": "2026-11-15T06:00:00+00:00"})
    assert "| A star | motion |" in text and "commit `abcdef1`" in text
    assert "| 6 | **7** | **9** | **8** | **12** |" in text
    summary = ver.summary(results, {})
    assert summary["star"]["seen_in_detectors"] == [6]
    assert summary["star"]["status"] == ver.FAIL  # no spectrum points


@pytest.mark.parametrize("statuses, expected", [((ver.PASS, ver.WARN), ver.WARN),
                                                ((ver.PASS, ver.FAIL, ver.WARN), ver.FAIL),
                                                ((), ver.PASS)])  # fmt: skip
def test_the_worst_check_decides(statuses, expected):
    assert ver.worst([{"status": s} for s in statuses]) == expected


def test_a_saturated_center_is_seen_whatever_its_s_n():
    """R Draconis: the fit on the ring of a saturated center gives absurd S/N values."""
    data = gold(saturated=(1,))
    seen = ver.with_saturation(found({(1, 1): float("nan"), (1, 2): 0.3, (1, 3): 1e20}),
                               data["sums"])  # fmt: skip
    assert seen["saturated"].all()
    assert ver.seen_check(seen, [1])["status"] == ver.PASS
    assert "| 1 | **sat** | **sat** | **sat** |" in ver.snr_table(seen)


def test_bands_below_the_noise_or_inside_methane_are_not_compared():
    faint = target(kind="motion", id="wise_0855", mag={"J": 25.0, "W1": 13.0, "W2": 14.0})
    table = pd.DataFrame({"wavelength_um": [1.24, 1.25, 3.35, 3.4, 4.55, 4.6],
                          "flux_mjy": [-0.01, 0.02, 0.001, 0.002, 0.42, 0.44],
                          "error_mjy": [0.05] * 6, "flag": [None] * 6})  # fmt: skip
    result = ver.spectrum_check(faint, {**gold(), "spectra": table})
    assert result["status"] == ver.PASS
    assert "J: below SPHEREx's noise" in result["detail"]
    assert "(inside the methane band, not compared)" in result["detail"]
