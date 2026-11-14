"""Catalog queries (task A1).

Offline tests replace the network calls with fakes and run in CI. Tests marked
`network` query the real services and run by hand: pytest -m network
"""

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time

from serafim.validation import catalogs

POLE = SkyCoord(270.0, 66.56, unit="deg")
DATE = Time("2025-06-01T00:00:00", scale="utc")

# Geocentric position of (1) Ceres at DATE, from JPL Horizons
# (Horizons(id="Ceres", id_type="smallbody", location="500")). SkyBoT puts it 0.13" away.
CERES = SkyCoord(7.82642, -7.20073, unit="deg")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Each test gets an empty cache and no waiting between attempts."""
    cache = tmp_path / "catalog_cache"
    monkeypatch.setattr(catalogs, "CACHE_DIR", cache)
    monkeypatch.setattr(catalogs, "BACKOFF_S", 0)
    return cache


# --- Fake answers shaped like the real services --------------------------------------


def skybot_table():
    return QTable(
        {
            "Number": [1],
            "Name": ["Ceres"],
            "RA": [7.82642] * u.deg,
            "DEC": [-7.20073] * u.deg,
            "Type": ["MB>Middle"],
            "V": [9.24] * u.mag,
            "externallink": ["https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/Ceres"],
        }
    )


def simbad_table():
    return Table(
        {
            "main_id": ["V* XX Dra", "NGC 6543", "HD 999"],
            "ra": [270.1, 269.6, 270.3],
            "dec": [66.5, 66.63, 66.4],
            "otype": ["RR*", "PN", "HV*"],
            "V": [12.5, 8.8, float("nan")],
        }
    )


def gaia_table():
    return Table(
        {
            "source_id": [1441146387496625536, 1441146628014797824],
            "ra": [269.68, 269.79],
            "dec": [66.23, 66.26],
            "phot_g_mean_mag": [18.7, 15.2],
            "phot_variable_flag": ["NOT_AVAILABLE", "VARIABLE"],
        }
    )


class FakeService:
    """Stands in for a network call: counts calls and can fail the first ones."""

    def __init__(self, answer, failures=0):
        self.answer = answer
        self.failures = failures
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("service down")
        return self.answer()


FAKES = {
    "skybot": ("_query_skybot", skybot_table, lambda: catalogs.skybot(CERES, 0.1 * u.deg, DATE)),
    "simbad": (
        "_query_simbad",
        lambda: (simbad_table(), catalogs.SIMBAD_TAP_SERVERS[1]),
        lambda: catalogs.simbad(POLE, 0.36 * u.deg),
    ),
    "gaia": ("_query_gaia", gaia_table, lambda: catalogs.gaia(POLE, 0.36 * u.deg)),
}


def install_fake(monkeypatch, catalog, failures=0, answer=None):
    attribute, default_answer, _ = FAKES[catalog]
    fake = FakeService(answer or default_answer, failures)
    monkeypatch.setattr(catalogs, attribute, fake)
    return fake


def query(catalog):
    return FAKES[catalog][2]()


# --- Cache ---------------------------------------------------------------------------


@pytest.mark.parametrize("catalog", sorted(FAKES))
def test_second_call_comes_from_the_cache(monkeypatch, catalog):
    fake = install_fake(monkeypatch, catalog)
    first = query(catalog)
    second = query(catalog)
    assert fake.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_empty_answer_is_cached_too(monkeypatch):
    fake = install_fake(monkeypatch, "skybot", answer=QTable)
    assert len(query("skybot")) == 0
    result = query("skybot")
    assert fake.calls == 1
    assert list(result.columns) == catalogs.COLUMNS


def test_another_date_is_another_query(monkeypatch):
    fake = install_fake(monkeypatch, "skybot")
    catalogs.skybot(CERES, 0.1 * u.deg, DATE)
    catalogs.skybot(CERES, 0.1 * u.deg, DATE + 1 * u.day)
    assert fake.calls == 2


# --- Retries -------------------------------------------------------------------------


def test_retries_until_the_service_answers(monkeypatch):
    fake = install_fake(monkeypatch, "gaia", failures=2)
    assert len(query("gaia")) == 2
    assert fake.calls == 3


def test_gives_up_after_three_attempts_and_caches_nothing(monkeypatch, isolated_cache):
    fake = install_fake(monkeypatch, "gaia", failures=99)
    with pytest.raises(catalogs.CatalogUnavailable):
        query("gaia")
    assert fake.calls == catalogs.RETRIES == 3
    assert not list(isolated_cache.rglob("*.parquet"))


def test_simbad_tries_the_mirrors_in_order(monkeypatch):
    tried = []

    def run_tap(server, adql):
        tried.append(server)
        if server != catalogs.SIMBAD_TAP_SERVERS[-1]:
            raise ConnectionError("down")
        return simbad_table()

    monkeypatch.setattr(catalogs, "_run_tap", run_tap)
    result = catalogs.simbad(POLE, 0.36 * u.deg)
    assert tried == list(catalogs.SIMBAD_TAP_SERVERS)
    assert result["url"].str.startswith("https://simbad.idia.ac.za/").all()


# --- Same columns for every catalog -----------------------------------------------------


@pytest.mark.parametrize("catalog", sorted(FAKES))
def test_every_catalog_returns_the_same_columns(monkeypatch, catalog):
    install_fake(monkeypatch, catalog)
    result = query(catalog)
    assert list(result.columns) == catalogs.COLUMNS
    assert result["is_variable"].dtype == bool
    expected = {"skybot": "SkyBoT", "simbad": "SIMBAD", "gaia": "Gaia"}[catalog]
    assert (result["catalog"] == expected).all()


def test_skybot_rows(monkeypatch):
    install_fake(monkeypatch, "skybot")
    row = query("skybot").iloc[0]
    assert row["name"] == "Ceres"
    assert row["ra"] == pytest.approx(7.82642)
    assert row["mag"] == pytest.approx(9.24)
    assert row["is_variable"]  # moving objects always change between frames


def test_simbad_variable_types(monkeypatch):
    install_fake(monkeypatch, "simbad")
    result = query("simbad").set_index("name")
    assert result.loc["V* XX Dra", "is_variable"]  # RR Lyrae, outside the V* branch
    assert not result.loc["NGC 6543", "is_variable"]
    assert not result.loc["HD 999", "is_variable"]  # HV* = high velocity, not variable
    assert pd.isna(result.loc["HD 999", "mag"])


def test_gaia_variable_flag(monkeypatch):
    install_fake(monkeypatch, "gaia")
    result = query("gaia")
    assert result["is_variable"].tolist() == [False, True]
    assert result["name"].iloc[0] == "Gaia DR3 1441146387496625536"


# --- fetch_for_frames -------------------------------------------------------------------

REGIONS = pd.DataFrame({"region_id": ["pole"], "ra": [270.0], "dec": [66.56], "size_deg": [0.5]})
FRAMES = pd.DataFrame(
    {
        "frame_id": ["f1", "f2"],
        "region_id": ["pole", "pole"],
        "date_utc": pd.to_datetime(["2025-06-01T00:00:00Z", "2025-06-02T12:00:00Z"]),
    }
)


def test_fetch_for_frames_queries_skybot_per_frame(monkeypatch):
    fakes = {catalog: install_fake(monkeypatch, catalog) for catalog in FAKES}
    result = catalogs.fetch_for_frames(FRAMES, REGIONS)
    assert fakes["skybot"].calls == 2
    assert fakes["simbad"].calls == fakes["gaia"].calls == 1
    assert (result["region_id"] == "pole").all()
    skybot_rows = result[result["catalog"] == "SkyBoT"]
    assert sorted(skybot_rows["frame_id"]) == ["f1", "f2"]
    assert result.loc[result["catalog"] != "SkyBoT", "frame_id"].isna().all()


def test_fetch_for_frames_is_none_when_a_catalog_is_down(monkeypatch):
    install_fake(monkeypatch, "skybot")
    install_fake(monkeypatch, "simbad")
    install_fake(monkeypatch, "gaia", failures=99)
    assert catalogs.fetch_for_frames(FRAMES, REGIONS) is None


def test_region_cone_covers_the_corners():
    # Half the diagonal of a 0.5 deg square is 0.354 deg
    assert catalogs._region_radius(0.5).deg > 0.354


# --- Real services (pytest -m network) ----------------------------------------------------


@pytest.mark.network
def test_skybot_finds_ceres_and_the_second_call_is_cached(monkeypatch):
    first = catalogs.skybot(CERES, 0.1 * u.deg, DATE)
    assert "Ceres" in set(first["name"])

    def no_network(*args, **kwargs):
        raise AssertionError("the second call went to the network")

    monkeypatch.setattr(catalogs, "_query_skybot", no_network)
    second = catalogs.skybot(CERES, 0.1 * u.deg, DATE)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.network
def test_simbad_answers_on_the_pole():
    result = catalogs.simbad(POLE, 0.1 * u.deg)
    assert len(result) > 0
    assert list(result.columns) == catalogs.COLUMNS


@pytest.mark.network
def test_gaia_answers_on_the_pole():
    result = catalogs.gaia(POLE, 0.1 * u.deg)
    assert len(result) > 0
    assert result["mag"].max() < catalogs.GAIA_MAX_MAG
