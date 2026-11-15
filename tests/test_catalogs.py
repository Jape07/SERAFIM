"""Catalog queries (task A1).

Offline tests replace the network calls with fakes and run in CI. Tests marked
`network` query the real services and run by hand: pytest -m network
"""

import io

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import ascii
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


NAN = float("nan")


def simbad_table():
    return Table(
        {
            "main_id": ["V* XX Dra", "NGC 6543", "HD 999"],
            "ra": [270.1, 269.6, 270.3],
            "dec": [66.5, 66.63, 66.4],
            "otype": ["RR*", "PN", "HV*"],
            "V": [12.5, 8.8, NAN],
            "pmra": [3.1, NAN, -250.0],
            "pmdec": [-1.2, NAN, 410.0],
            "plx_value": [0.5, NAN, 12.0],
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
            "pmra": [NAN, 12.5],  # a two-parameter source has no proper motion
            "pmdec": [NAN, -3.0],
            "parallax": [NAN, -0.2],
            "ref_epoch": [2016.0, 2016.0],
        }
    )


def catwise_table():
    """Shaped like the Gator answer: arcsec/yr, fixed and moving fits, artifact flags."""
    return Table(
        {
            "source_name": [
                "J085508.34-071439.2",  # fast mover: faint in the fixed fit, bright moving
                "J085513.00-071500.0",  # fixed star: motion below 5 sigma
                "J085520.00-071600.0",  # too faint in both fits
                "J085525.00-071700.0",  # upper-case flag: confirmed artifact
                "J085530.00-071800.0",  # lower-case flag: real source, only contaminated
            ],
            "ra": [133.7845, 133.8042, 133.8333, 133.8542, 133.8750],
            "dec": [-7.2442, -7.2500, -7.2667, -7.2833, -7.3000],
            "ra_pm": [133.7848, 133.8042, 133.8333, 133.8542, 133.8750],
            "dec_pm": [-7.2443, -7.2500, -7.2667, -7.2833, -7.3000],
            "pmra": [-8.01, 0.01, 0.0, 2.0, 0.0],
            "pmdec": [0.81, -0.01, 0.0, 2.0, 0.0],
            "sigpmra": [0.19, 0.01, 0.1, 0.1, 0.1],
            "sigpmdec": [0.22, 0.01, 0.1, 0.1, 0.1],
            "w2mpro": [15.72, 12.0, 16.0, 13.0, 14.0],
            "w2mpro_pm": [13.82, 12.1, 16.2, 13.0, 14.1],
            "cc_flags": ["0000", "0000", "0000", "H000", "h000"],
            "ab_flags": ["00", "00", "00", "00", "00"],
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
    "catwise": ("_query_catwise", catwise_table, lambda: catalogs.catwise(POLE, 0.36 * u.deg)),
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


def test_a_cache_file_with_old_columns_is_asked_again(monkeypatch, isolated_cache):
    fake = install_fake(monkeypatch, "gaia")
    query("gaia")
    (path,) = isolated_cache.rglob("*.parquet")
    old = pd.read_parquet(path).drop(columns=catalogs.MOTION_COLUMNS)  # before task A2
    old.to_parquet(path, index=False)
    result = query("gaia")
    assert fake.calls == 2
    assert list(result.columns) == catalogs.COLUMNS


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
    expected = {"skybot": "SkyBoT", "simbad": "SIMBAD", "gaia": "Gaia", "catwise": "CatWISE"}
    assert (result["catalog"] == expected[catalog]).all()
    for name in catalogs.MOTION_COLUMNS:
        assert result[name].dtype == "float64"


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


# --- Motion columns ---------------------------------------------------------------------


def test_skybot_has_no_motion(monkeypatch):
    install_fake(monkeypatch, "skybot")
    assert query("skybot")[catalogs.MOTION_COLUMNS].isna().all().all()


def test_simbad_motion_is_at_j2000(monkeypatch):
    install_fake(monkeypatch, "simbad")
    result = query("simbad").set_index("name")
    assert result.loc["HD 999", "pmdec_mas_yr"] == pytest.approx(410.0)
    assert result.loc["HD 999", "parallax_mas"] == pytest.approx(12.0)
    assert result.loc["HD 999", "epoch_jyear"] == 2000.0
    assert pd.isna(result.loc["NGC 6543", "epoch_jyear"])  # no motion, no epoch


def test_gaia_motion_is_at_its_reference_epoch(monkeypatch):
    install_fake(monkeypatch, "gaia")
    result = query("gaia")
    assert pd.isna(result["epoch_jyear"].iloc[0])  # two-parameter source
    assert result["epoch_jyear"].iloc[1] == 2016.0
    assert result["pmra_mas_yr"].iloc[1] == pytest.approx(12.5)


def test_catwise_rules(monkeypatch):
    install_fake(monkeypatch, "catwise")
    result = query("catwise").set_index("name")
    # Too faint in both fits, and the confirmed artifact, are left out
    assert sorted(result.index) == [
        "CWISE J085508.34-071439.2", "CWISE J085513.00-071500.0", "CWISE J085530.00-071800.0",
    ]  # fmt: skip
    mover = result.loc["CWISE J085508.34-071439.2"]
    assert mover["mag"] == pytest.approx(13.82)  # the brighter, moving fit
    assert mover["pmra_mas_yr"] == pytest.approx(-8010.0)  # arcsec/yr -> mas/yr
    assert mover["ra"] == pytest.approx(133.7848)  # moving-fit position ...
    assert mover["epoch_jyear"] == pytest.approx(2015.40, abs=0.01)  # ... at MJD 57170
    fixed = result.loc["CWISE J085513.00-071500.0"]
    assert pd.isna(fixed["pmra_mas_yr"]) and pd.isna(fixed["epoch_jyear"])  # 1.4 sigma


def test_catwise_empty_answer(monkeypatch):
    install_fake(monkeypatch, "catwise", answer=lambda: Table(names=catalogs.CATWISE_COLUMNS))
    result = query("catwise")
    assert len(result) == 0
    assert list(result.columns) == catalogs.COLUMNS


class GatorAnswer:
    """Stands in for the requests response of IRSA Gator."""

    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ConnectionError(f"HTTP {self.status_code}")


def ipac_text(table):
    buffer = io.StringIO()
    ascii.write(table, buffer, format="ipac")
    return buffer.getvalue()


def test_catwise_reads_the_gator_table(monkeypatch):
    answers = [GatorAnswer("", 502), GatorAnswer(ipac_text(catwise_table()[:1]))]
    sent = []

    def get(self, url, params):
        sent.append(params)
        return answers.pop(0)

    monkeypatch.setattr(catalogs._TimeoutSession, "get", get)
    monkeypatch.setattr(catalogs, "IRSA_WAITS_S", (0, 0, 0))
    table = catalogs._query_catwise(POLE, 0.1 * u.deg)
    assert len(sent) == 2  # the 502 was tried again
    assert sent[0]["selcols"].split(",") == list(catalogs.CATWISE_COLUMNS)
    assert len(table) == 1
    assert table["source_name"][0] == "J085508.34-071439.2"
    assert float(table["pmra"][0]) == pytest.approx(-8.01)


def test_catwise_with_no_rows(monkeypatch):
    answer = GatorAnswer("\\RowsRetrieved =                 0\n")
    monkeypatch.setattr(catalogs._TimeoutSession, "get", lambda self, url, params: answer)
    assert len(catalogs._query_catwise(POLE, 0.1 * u.deg)) == 0


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
    assert (result["epoch_jyear"].dropna() == catalogs.GAIA_EPOCH_JYEAR).all()


@pytest.mark.network
def test_catwise_finds_wise_0855_moving_at_8_arcsec_per_year():
    # WISE 0855 at the CatWISE epoch (2015.4), from its SIMBAD position and motion
    there = SkyCoord(133.7843, -7.2441, unit="deg")
    result = catalogs.catwise(there, 20 * u.arcsec)
    movers = result[result["pmra_mas_yr"] < -7000]
    assert len(movers) >= 1
    total = (movers["pmra_mas_yr"] ** 2 + movers["pmdec_mas_yr"] ** 2) ** 0.5
    assert total.between(7500, 8700).all()  # SIMBAD: 8.15 arcsec/yr
