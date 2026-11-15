"""Catalog layers per season (task A2).

Offline tests replace the catalogs with fakes and run in CI. Tests marked `network`
query the real services and run by hand: pytest -m network -k layers
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.coordinates import SkyCoord
from astropy.time import Time

from serafim.validation import catalogs, layers, targets

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "cutout_size_deg": 0.25,
    "collections": ["spherex_dr1"],
    "seasons": {"gap_days": 30, "max_days": 183},
}

# A star near the ecliptic moving 1 arcsec/yr to the north
STAR = {
    "id": "star",
    "ra": 100.0,
    "dec": 20.0,
    "pmra_mas_yr": 0.0,
    "pmdec_mas_yr": 1000.0,
    "parallax_mas": 100.0,
}
NOW = targets.propagate(STAR, Time("2026-01-01", scale="utc"))
STAR = {**STAR, "ra_now": float(NOW.ra.deg), "dec_now": float(NOW.dec.deg)}

SEASON_1 = Time("2025-03-01T00:00:00", scale="utc").mjd
SEASON_2 = Time("2026-03-01T00:00:00", scale="utc").mjd
# Season 1: two visits (frames 1 h apart, then a day later). Season 2: one visit.
FRAMES = pd.DataFrame(
    {
        "target_id": "star",
        "season": [1, 1, 1, 2, 2],
        "t_mid_mjd": [SEASON_1, SEASON_1 + 1 / 24, SEASON_1 + 1, SEASON_2, SEASON_2 + 0.5 / 24],
    }
)


def sources(catalog, rows):
    """A catalog answer (catalogs.COLUMNS) from a few dicts; missing values are NaN/None."""
    fixed = {"catalog": catalog, "type": "source", "is_variable": False}
    columns = {name: [row.get(name) for row in rows] for name in catalogs.COLUMNS}
    return catalogs._frame(**{**columns, **fixed})


FAR = {"name": "far star", "ra": STAR["ra_now"] + 0.05, "dec": STAR["dec_now"], "mag": 12.0}


class Fakes:
    """Stands in for catalogs.gaia/catwise/simbad/skybot and counts the calls."""

    def __init__(self, monkeypatch, *, gaia=None, catwise=None, simbad=None, skybot=None):
        self.calls = {"gaia": 0, "catwise": 0, "simbad": 0, "skybot": []}
        answers = {
            "gaia": gaia if gaia is not None else sources("Gaia", [FAR]),
            "catwise": catwise if catwise is not None else sources("CatWISE", [FAR]),
            "simbad": simbad if simbad is not None else sources("SIMBAD", [FAR]),
        }
        for name, answer in answers.items():
            monkeypatch.setattr(catalogs, name, self._fake(name, answer))
        self.skybot_answer = skybot if skybot is not None else sources("SkyBoT", [])

        def fake_skybot(center, radius, date):
            self.calls["skybot"].append(date)
            return self.skybot_answer

        monkeypatch.setattr(catalogs, "skybot", fake_skybot)

    def _fake(self, name, answer):
        def fake(center, radius, *args, **kwargs):
            self.calls[name] += 1
            if isinstance(answer, Exception):
                raise answer
            return answer

        return fake


def run(**kwargs):
    kwargs.setdefault("settings", SETTINGS)
    return layers.catalog_layers([STAR], FRAMES, **kwargs)


# --- move() -------------------------------------------------------------------------------


def test_move_follows_the_proper_motion():
    table = sources(
        "Gaia",
        [
            {"name": "mover", "ra": 10.0, "dec": 0.0, "pmra_mas_yr": 0.0, "pmdec_mas_yr": 1000.0,
             "parallax_mas": -0.3, "epoch_jyear": 2000.0},  # negative parallax: far away
            {"name": "fixed", "ra": 11.0, "dec": 1.0},
        ],
    )  # fmt: skip
    ra, dec = layers.move(table, Time(2010.0, format="jyear"))
    assert dec[0] * 3600 == pytest.approx(10.0, abs=0.01)
    assert ra[0] == pytest.approx(10.0, abs=1e-6)
    assert (ra[1], dec[1]) == (11.0, 1.0)


def test_move_uses_each_catalog_epoch():
    rows = [
        {"name": n, "ra": 50.0, "dec": 0.0, "pmra_mas_yr": 0.0, "pmdec_mas_yr": 1000.0,
         "parallax_mas": 10.0, "epoch_jyear": epoch}
        for n, epoch in (("simbad-like", 2000.0), ("gaia-like", 2016.0))
    ]  # fmt: skip
    _, dec = layers.move(sources("x", rows), Time(2026.0, format="jyear"))
    assert dec * 3600 == pytest.approx([26.0, 10.0], abs=0.01)


# --- Seasons and visits ---------------------------------------------------------------------


def test_season_dates_add_season_0_for_all_seasons():
    dates = layers.season_dates(FRAMES).set_index("season")["mjd"]
    assert list(dates.index) == [0, 1, 2]
    assert dates[0] == pytest.approx(FRAMES["t_mid_mjd"].mean())
    assert dates[2] == pytest.approx(SEASON_2 + 0.25 / 24)


def test_season_dates_from_date_utc_like_frames_parquet():
    stamps = pd.to_datetime([Time(mjd, format="mjd").isot for mjd in FRAMES["t_mid_mjd"]], utc=True)
    with_dates = FRAMES.drop(columns="t_mid_mjd").assign(date_utc=stamps)
    a, b = layers.season_dates(FRAMES), layers.season_dates(with_dates)
    assert np.allclose(a["mjd"], b["mjd"], atol=1e-6)


def test_visits_are_frames_less_than_2_hours_apart():
    found = layers.visits(FRAMES)
    assert found["season"].tolist() == [1, 1, 2]
    assert found["mjd"].iloc[0] == pytest.approx(SEASON_1 + 0.5 / 24)


# --- catalog_layers() -----------------------------------------------------------------------


def test_every_source_at_every_season(monkeypatch):
    Fakes(monkeypatch)
    rows = run()
    assert list(rows.columns) == layers.LAYER_COLUMNS
    assert sorted(rows["catalog"].unique()) == ["catwise", "gaia", "simbad"]
    assert len(rows) == 3 * 3  # 3 catalogs x 1 source x (2 seasons + season 0)
    assert sorted(rows["season"].unique()) == [0, 1, 2]
    bands = rows.groupby("catalog")["mag_band"].first().to_dict()
    assert bands == {"catwise": "W2", "gaia": "G", "simbad": "V"}
    assert str(rows["date_utc"].dt.tz) == "UTC"


def test_markers_move_between_seasons(monkeypatch):
    star_itself = {"name": "the star", **{k: STAR[k] for k in ("ra", "dec")},
                   "pmra_mas_yr": 0.0, "pmdec_mas_yr": 1000.0, "parallax_mas": 100.0,
                   "epoch_jyear": 2000.0, "mag": 9.0}  # fmt: skip
    Fakes(monkeypatch, simbad=sources("SIMBAD", [star_itself]))
    rows = run().query("catalog == 'simbad'").set_index("season")
    first = SkyCoord(rows.loc[1, "ra"] * u.deg, rows.loc[1, "dec"] * u.deg)
    second = SkyCoord(rows.loc[2, "ra"] * u.deg, rows.loc[2, "dec"] * u.deg)
    years = (SEASON_2 - SEASON_1) / 365.25
    assert first.separation(second).arcsec == pytest.approx(1.0 * years, abs=0.02)
    expected = targets.propagate(STAR, catalogs._to_time(rows.loc[2, "date_utc"]))
    assert second.separation(SkyCoord(expected.ra, expected.dec)).arcsec < 0.01


def test_catwise_next_to_the_target_is_left_out(monkeypatch):
    at_epoch = targets.propagate(STAR, Time(catalogs.CATWISE_EPOCH_JYEAR, format="jyear"))
    blend = {"name": "blend", "ra": at_epoch.ra.deg, "dec": at_epoch.dec.deg + 5 / 3600}
    Fakes(monkeypatch, catwise=sources("CatWISE", [blend, FAR]))
    names = set(run().query("catalog == 'catwise'")["name"])
    assert names == {"far star"}


def test_catwise_along_a_fast_target_path_is_left_out(monkeypatch):
    fast = {**STAR, "pmdec_mas_yr": 10000.0}  # 10 arcsec/yr, like Barnard's Star
    in_2010 = targets.propagate(fast, Time(2010.5, format="jyear"))  # 49" from its 2015.4 place
    trail = {"name": "trail", "ra": in_2010.ra.deg, "dec": in_2010.dec.deg}
    Fakes(monkeypatch, catwise=sources("CatWISE", [trail, FAR]))
    rows = layers.catalog_layers([fast], FRAMES, settings=SETTINGS, skybot=False)
    assert set(rows.query("catalog == 'catwise'")["name"]) == {"far star"}


def test_catwise_in_the_halo_of_a_star_that_saturates_wise_is_left_out(monkeypatch):
    """Task A8: false CatWISE entries 67 to 126 arcsec around Barnard's Star (W2 = 3.7)."""
    bright = {**STAR, "mag": {"W2": 3.7}}  # halo radius ~170 arcsec
    at_epoch = targets.propagate(bright, Time(catalogs.CATWISE_EPOCH_JYEAR, format="jyear"))
    ghost = {"name": "ghost", "ra": at_epoch.ra.deg, "dec": at_epoch.dec.deg + 80 / 3600}
    real = {"name": "real star", "ra": at_epoch.ra.deg, "dec": at_epoch.dec.deg - 80 / 3600}
    gaia_star = {"name": "Gaia star", "ra": real["ra"], "dec": real["dec"] + 1 / 3600}
    Fakes(monkeypatch, catwise=sources("CatWISE", [ghost, real, FAR]),
          gaia=sources("Gaia", [gaia_star, FAR]))  # fmt: skip
    rows = layers.catalog_layers([bright], FRAMES, settings=SETTINGS, skybot=False)
    assert set(rows.query("catalog == 'catwise'")["name"]) == {"real star", "far star"}
    assert layers.halo_radius(3.7) == pytest.approx(172, abs=1)
    assert layers.halo_radius(-1.0) == layers.HALO_MAX_ARCSEC  # R Draconis: W2 = 0.1


def test_a_bright_catwise_entry_has_a_halo_too(monkeypatch):
    star = {"name": "bright", "ra": STAR["ra_now"] - 0.04, "dec": STAR["dec_now"], "mag": 5.0}
    ghost = {"name": "ghost", "ra": star["ra"], "dec": star["dec"] + 60 / 3600, "mag": 13.0}
    Fakes(monkeypatch, catwise=sources("CatWISE", [star, ghost]))
    names = set(run(skybot=False).query("catalog == 'catwise'")["name"])
    assert names == {"bright"}  # a star is not inside its own halo


def test_skybot_once_per_visit(monkeypatch):
    asteroid = {"name": "Ceres", "ra": STAR["ra_now"], "dec": STAR["dec_now"], "mag": 9.0,
                "url": "https://ssp.imcce.fr/x"}  # fmt: skip
    fakes = Fakes(monkeypatch, skybot=sources("SkyBoT", [asteroid]))
    rows = run(workers=1).query("catalog == 'skybot'")
    assert len(fakes.calls["skybot"]) == 3  # 3 visits
    assert rows["season"].tolist() == [1, 1, 2]
    assert rows["date_utc"].is_unique
    assert (rows["mag_band"] == "V").all()


def test_skybot_leaves_out_asteroids_too_faint_for_spherex(monkeypatch):
    here = {"ra": STAR["ra_now"], "dec": STAR["dec_now"]}
    answer = sources("SkyBoT", [{"name": "bright", "mag": 17.5, **here},
                                {"name": "faint", "mag": 22.4, **here},
                                {"name": "comet without V", **here}])  # fmt: skip
    Fakes(monkeypatch, skybot=answer)
    names = set(run(workers=1).query("catalog == 'skybot'")["name"])
    assert names == {"bright", "comet without V"}


def test_skybot_skipped_far_from_the_ecliptic(monkeypatch):
    fakes = Fakes(monkeypatch)
    pole = {**STAR, "ra_now": 270.0, "dec_now": 66.56}  # north ecliptic pole
    layers.catalog_layers([pole], FRAMES, settings=SETTINGS)
    assert fakes.calls["skybot"] == []


def test_no_skybot_when_asked(monkeypatch):
    fakes = Fakes(monkeypatch)
    run(skybot=False)
    assert fakes.calls["skybot"] == []


def test_a_catalog_that_is_down_is_left_out(monkeypatch):
    Fakes(monkeypatch, gaia=catalogs.CatalogUnavailable("gaia did not answer"))
    rows = run()
    assert sorted(rows["catalog"].unique()) == ["catwise", "simbad"]


def test_a_target_without_frames_has_no_layers(monkeypatch):
    fakes = Fakes(monkeypatch)
    other = {**STAR, "id": "other"}
    rows = layers.catalog_layers([other], FRAMES, settings=SETTINGS)
    assert rows.empty
    assert list(rows.columns) == layers.LAYER_COLUMNS
    assert fakes.calls["gaia"] == 0


# --- Real services (pytest -m network -k layers) ----------------------------------------------


@pytest.mark.network
def test_barnard_gaia_marker_follows_the_star():
    text = (ROOT / "config" / "targets.yaml").read_text(encoding="utf-8")
    barnard = next(t for t in yaml.safe_load(text)["targets"] if t["id"] == "barnard")
    rows = layers.catalog_layers([barnard], settings=targets.read_settings(text), skybot=False)
    gaia = rows[rows["catalog"] == "gaia"]
    nearest = {}
    for season, group in gaia.groupby("season"):
        star = targets.propagate(barnard, catalogs._to_time(group["date_utc"].iloc[0]))
        here = SkyCoord(star.ra, star.dec)
        positions = SkyCoord(group["ra"].to_numpy() * u.deg, group["dec"].to_numpy() * u.deg)
        separation = here.separation(positions)
        assert separation.arcsec.min() < 1.0, f"season {season}"
        nearest[season] = group.iloc[int(np.argmin(separation))]
    first, last = nearest[1], nearest[max(nearest)]
    shift = SkyCoord(first["ra"] * u.deg, first["dec"] * u.deg).separation(
        SkyCoord(last["ra"] * u.deg, last["dec"] * u.deg)
    )
    shift_px = shift.arcsec / targets.PIXEL_SCALE_ARCSEC
    assert shift_px == pytest.approx(barnard["motion"]["shift_px_first_to_last_season"], abs=0.1)
