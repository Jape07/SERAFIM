"""The gold contract must match the table the team agreed on (issue #18, task D2).

Also: validate_gold() accepts gold that follows the contract and reports every
problem in gold that does not, and `python -m serafim mock` writes valid gold.
"""

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from serafim import contract
from serafim.cli import main
from serafim.config import load_config
from serafim.gold.mock import HOLE, write_mock_gold

# The simulated gold uses the first target in config/targets.yaml.
TARGET = load_config()["targets"][0]["id"]

AGREED_COLUMNS = {
    "targets.parquet": [
        "target_id", "name", "kind", "ra_now", "dec_now", "pmra_mas_yr", "pmdec_mas_yr",
        "size_deg", "wcs_json",
    ],
    "frames.parquet": [
        "frame_id", "target_id", "obs_id", "collection", "date_utc", "season", "detector",
        "wavelength_um", "bandwidth_um", "flags", "saturated",
    ],
    "sums.parquet": [
        "sum_id", "target_id", "detector", "season", "mean_date_utc", "n_frames",
        "background_noise", "flags", "saturated", "png",
    ],
    "differences.parquet": [
        "difference_id", "target_id", "detector", "season_a", "season_b", "png",
    ],
    # Same order as AI's catalog_layers() (serafim.validation.layers.LAYER_COLUMNS).
    "catalogs.parquet": [
        "target_id", "catalog", "name", "season", "date_utc", "ra", "dec", "mag", "mag_band",
        "url",
    ],
    "spectra.parquet": [
        "target_id", "frame_id", "season", "date_utc", "wavelength_um", "flux_mjy", "error_mjy",
        "flag",
    ],
    "trajectory.parquet": [
        "target_id", "season", "ra_measured", "dec_measured", "error_arcsec", "ra_catalog",
        "dec_catalog",
    ],
}


def test_the_contract_has_exactly_the_agreed_tables():
    assert sorted(contract.GOLD_TABLES) == sorted(AGREED_COLUMNS)


@pytest.mark.parametrize("filename", sorted(AGREED_COLUMNS))
def test_columns_match_the_agreed_table(filename):
    assert contract.GOLD_TABLES[filename].names == AGREED_COLUMNS[filename]


@pytest.mark.parametrize("filename", sorted(AGREED_COLUMNS))
def test_ids_are_required(filename):
    for field in contract.GOLD_TABLES[filename]:
        if field.name.endswith("_id"):
            assert not field.nullable, f"{filename}: {field.name} must not be null"


def test_removed_from_the_contract():
    assert "regions.parquet" not in contract.GOLD_TABLES
    assert "p_real" not in {c for names in AGREED_COLUMNS.values() for c in names}
    assert "png_diff" not in {c for names in AGREED_COLUMNS.values() for c in names}


def test_allowed_values():
    assert contract.KINDS == ("motion", "brightness", "spectrum", "custom")
    assert contract.CATALOG_NAMES == ("gaia", "catwise", "simbad", "skybot")
    assert contract.MAG_BANDS == {"gaia": "G", "catwise": "W2", "simbad": "V", "skybot": "V"}
    assert contract.META_KEYS == (
        "run_date_utc", "commit", "collections", "last_frame_date_utc", "n_frames",
    )
    assert contract.COLOR_DETECTORS == (1, 3, 6)


def test_detector_bands_are_the_measured_ones():
    assert contract.DETECTOR_BANDS_UM == {
        1: (0.746, 1.117), 2: (1.103, 1.650), 3: (1.630, 2.435),
        4: (2.403, 3.847), 5: (3.811, 4.430), 6: (4.412, 5.010),
    }
    assert contract.WAVELENGTH_RANGE_UM == (0.746, 5.010)


def test_png_names_are_the_ones_web_builds():
    assert contract.sum_png("barnard", 3, 2) == "png/barnard/sum_d3_s2.png"
    assert contract.sum_png("barnard", 3, 0) == "png/barnard/sum_d3_s0.png"
    assert contract.sum_png("barnard", 0, 1) == "png/barnard/sum_d0_s1.png"
    assert contract.difference_png("barnard", 5, 1, 3) == "png/barnard/diff_d5_s1-s3.png"
    assert contract.color_png("barnard", 2) == "png/barnard/color_s2.png"


def test_frame_id_is_the_target_and_the_obs_id():
    assert contract.frame_id("barnard", "2025W33_2C_0321_1") == "barnard_2025W33_2C_0321_1"


# --- Simulated gold ------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_template(tmp_path_factory):
    """Simulated gold written once per test module."""
    gold = tmp_path_factory.mktemp("mock") / "gold"
    write_mock_gold(gold, load_config())
    return gold


@pytest.fixture
def gold(mock_template, tmp_path):
    """A fresh copy of the simulated gold that each test can break."""
    return shutil.copytree(mock_template, tmp_path / "gold")


def read(gold, filename):
    return pq.read_table(gold / filename).to_pandas()


def set_value(gold, filename, column, row, value):
    """Overwrite one cell of a gold table, keeping the column type."""
    path = gold / filename
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    values = table.column(column).to_pylist()
    values[row] = value
    # pyarrow refuses nulls in a required field, so mark it nullable to simulate gold
    # written without the contract schema.
    field = table.schema.field(index)
    field = field.with_nullable(value is None or field.nullable)
    pq.write_table(table.set_column(index, field, pa.array(values, field.type)), path)


def problems(gold):
    with pytest.raises(contract.GoldValidationError) as error:
        contract.validate_gold(gold)
    return error.value.errors


def test_mock_gold_follows_the_contract(gold):
    contract.validate_gold(gold)  # Raises if anything is wrong.


def test_mock_gold_has_what_the_web_needs(gold):
    frames = read(gold, "frames.parquet")
    sums = read(gold, "sums.parquet")
    differences = read(gold, "differences.parquet")
    meta = json.loads((gold / "meta.json").read_text())

    # 6 detectors and 3 seasons, with the white image (detector 0) and all seasons (season 0).
    assert set(frames.detector) == {1, 2, 3, 4, 5, 6}
    assert set(frames.season) == {1, 2, 3}
    assert set(sums.detector) == {0, 1, 2, 3, 4, 5, 6}
    assert set(sums.season) == {0, 1, 2, 3}

    # The hole: no frames for one detector in one season, so no sum or difference there.
    detector, season = HOLE
    assert not ((frames.detector == detector) & (frames.season == season)).any()
    assert not ((sums.detector == detector) & (sums.season == season)).any()
    in_hole = (differences.detector == detector) & (
        (differences.season_a == season) | (differences.season_b == season)
    )
    assert not in_hole.any()
    assert len(differences) > 0

    # Every frame_id is built the way silver and AI's spectra build it.
    assert (frames.frame_id == frames.target_id + "_" + frames.obs_id).all()

    # Saturated center: both cases, and the flags say why.
    assert sums.saturated.any() and not sums.saturated.all()
    assert "BLOOM,OVERFLOW" in set(frames["flags"]) and "" in set(frames["flags"])

    # The color image exists exactly where D1, D3 and D6 exist: not in season 1 (no D6).
    for season, expected in {1: False, 2: True, 3: True}.items():
        assert (gold / contract.color_png(TARGET, season)).is_file() is expected

    # Example rows from AI & validation, including the nullable columns.
    catalogs = read(gold, "catalogs.parquet")
    spectra = read(gold, "spectra.parquet")
    assert set(catalogs.catalog) == set(contract.CATALOG_NAMES)
    assert catalogs.mag.isna().any() and catalogs.url.isna().any()
    assert (catalogs.mag_band == catalogs.catalog.map(contract.MAG_BANDS)).all()
    # Season 0 (all seasons) for Gaia, CatWISE and SIMBAD; SkyBoT is one row per visit, so an
    # asteroid has several rows in a season and each one has its own date.
    layers = catalogs[catalogs.catalog != "skybot"]
    assert set(layers.season) == {0, 1, 2, 3}
    asteroids = catalogs[catalogs.catalog == "skybot"]
    assert 0 not in set(asteroids.season)
    assert (asteroids.groupby("season").date_utc.nunique() > 1).all()
    assert spectra.flag.isna().any() and spectra.flag.notna().any()
    trajectory = read(gold, "trajectory.parquet")
    assert len(trajectory) == 3
    # One season without a measurement: all three measured columns are null together.
    unmeasured = trajectory[trajectory.season == 2]
    assert unmeasured[["ra_measured", "dec_measured", "error_arcsec"]].isna().all().all()
    assert trajectory[trajectory.season != 2][["ra_measured", "dec_measured"]].notna().all().all()

    assert meta["n_frames"] == len(frames) == 51
    assert meta["collections"] == load_config()["settings"]["collections"]


def test_trajectory_sits_where_the_pngs_show_the_target(gold):
    """ra_catalog, dec_catalog and the PNG grid describe the same point."""
    target = read(gold, "targets.parquet").iloc[0]
    trajectory = read(gold, "trajectory.parquet").sort_values("season")
    wcs = WCS(fits.Header(json.loads(target.wcs_json)))
    n_rows = json.loads(target.wcs_json)["NAXIS2"]

    # Season 1 is the grid center, the same position the target has today.
    first = trajectory.iloc[0]
    np.testing.assert_allclose([first.ra_catalog, first.dec_catalog],
                               [target.ra_now, target.dec_now], atol=1e-5)

    # A `motion` target moves north (up in the PNG, y smaller) by 5 px per season.
    assert target.kind == "motion"
    x, y_fits = wcs.wcs_world2pix(trajectory.ra_catalog, trajectory.dec_catalog, 0)
    center = (n_rows - 1) / 2
    np.testing.assert_allclose(x, center, atol=1e-3)
    np.testing.assert_allclose(n_rows - 1 - y_fits, [center, center - 5, center - 10], atol=1e-3)


def test_mock_is_reproducible(gold, tmp_path):
    again = tmp_path / "again"
    write_mock_gold(again, load_config())
    for filename in contract.GOLD_TABLES:
        assert pq.read_table(gold / filename).equals(pq.read_table(again / filename)), filename
    pngs = sorted((gold / "png").rglob("*.png"))
    assert pngs
    for png in pngs:
        assert png.read_bytes() == (again / png.relative_to(gold)).read_bytes(), png.name


def test_cli_mock_writes_valid_gold(tmp_path):
    out = tmp_path / "gold"
    assert main(["mock", "--out", str(out)]) == 0
    contract.validate_gold(out)


def test_empty_ai_tables_are_valid(gold):
    """Until AI delivers a table, it is written empty with its schema and the site must not fail."""
    for filename in ("catalogs.parquet", "spectra.parquet", "trajectory.parquet"):
        pq.write_table(contract.GOLD_TABLES[filename].empty_table(), gold / filename)
    contract.validate_gold(gold)


# --- validate_gold() catches broken gold ---------------------------------------
@pytest.mark.parametrize(
    ("filename", "column", "value", "expected"),
    [
        ("targets.parquet", "kind", "planet", "kind must be one of"),
        ("targets.parquet", "size_deg", 0.0, "size_deg must be > 0"),
        # 4.9 um is inside detector 6, but this is a detector 1 frame.
        ("frames.parquet", "wavelength_um", 4.9, "wavelength_um outside the band of its detector"),
        # Row 0 is detector 1 (0.746 to 1.117 um): 0.72 is beyond the 0.02 um margin.
        ("frames.parquet", "wavelength_um", 0.72, "wavelength_um outside the band of its detector"),
        ("frames.parquet", "detector", 0, "detector must be 1 to 6"),
        ("frames.parquet", "detector", 7, "detector must be 1 to 6"),
        ("frames.parquet", "season", 0, "season must be 1 or more"),
        ("frames.parquet", "flags", "BLOOM,SOURCE", "flags must not include SOURCE"),
        ("frames.parquet", "flags", "bloom, overflow", "flags must be names separated by commas"),
        ("frames.parquet", "frame_id", None, "frame_id has 1 empty values"),
        ("frames.parquet", "frame_id", "nope", "frame_id must be <target_id>_<obs_id>"),
        ("sums.parquet", "detector", 7, "detector must be 0 to 6"),
        ("sums.parquet", "n_frames", 99, "n_frames does not match"),
        ("sums.parquet", "target_id", "nope", "target_id not found in targets.parquet"),
        ("sums.parquet", "png", "png/barnard/other.png", "png must be png/<target>/sum_d"),
        ("sums.parquet", "png", f"/{contract.sum_png(TARGET, 0, 1)}", "must be a relative path"),
        ("differences.parquet", "season_a", 3, "season_a must be less than season_b"),
        ("differences.parquet", "season_b", 9, "season_b has no sum of that detector"),
        ("differences.parquet", "detector", 0, "detector must be 1 to 6"),
        ("catalogs.parquet", "catalog", "tycho", "catalog must be one of"),
        ("catalogs.parquet", "season", -1, "season must be 0 or more"),
        ("catalogs.parquet", "date_utc", None, "date_utc has 1 empty values"),
        ("catalogs.parquet", "mag_band", None, "mag_band has 1 empty values"),
        # Row 0 is a Gaia source: its magnitude is G, not W2.
        ("catalogs.parquet", "mag_band", "W2", "mag_band must match its catalog"),
        ("catalogs.parquet", "ra", 361.0, "ra outside [0, 360)"),
        ("spectra.parquet", "frame_id", "nope", "frame_id not found in frames.parquet"),
        ("spectra.parquet", "wavelength_um", 7.5, "wavelength_um outside"),
        ("spectra.parquet", "error_mjy", -1.0, "error_mjy must be >= 0"),
        ("trajectory.parquet", "error_arcsec", -1.0, "error_arcsec must be >= 0"),
        ("trajectory.parquet", "dec_catalog", 95.0, "dec_catalog outside [-90, 90]"),
    ],
)
def test_rejects_bad_values(gold, filename, column, value, expected):
    set_value(gold, filename, column, 0, value)
    assert any(expected in problem for problem in problems(gold)), problems(gold)


def test_a_wavelength_at_the_edge_of_the_detector_is_accepted(gold):
    # A real Barnard frame at the edge of detector 1 measured 0.7414 um (band from 0.746).
    set_value(gold, "frames.parquet", "wavelength_um", 0, 0.7414)
    contract.validate_gold(gold)


def test_a_spectrum_point_at_the_edge_of_the_range_is_accepted(gold):
    """A spectrum point takes the wavelength of its frame, so it gets the same margin.

    Real gold (D8) had RW Aurigae at 5.0116 um in detector 6 and 0.7409 um in detector 1.
    """
    for value in (0.7409, 5.0116):
        set_value(gold, "spectra.parquet", "wavelength_um", 0, value)
        contract.validate_gold(gold)
    set_value(gold, "spectra.parquet", "wavelength_um", 0, 0.72)  # beyond the 0.02 um margin
    assert any("wavelength_um outside" in problem for problem in problems(gold))


def test_rejects_repeated_keys(gold):
    first = read(gold, "frames.parquet").frame_id[0]
    set_value(gold, "frames.parquet", "frame_id", 1, first)
    assert any("repeated frame_id" in p for p in problems(gold))


def test_rejects_a_missing_png(gold):
    (gold / contract.sum_png(TARGET, 0, 1)).unlink()
    assert any("png file not found in gold" in p for p in problems(gold))


def test_rejects_a_missing_color_image(gold):
    (gold / contract.color_png(TARGET, 2)).unlink()
    assert any("color image not found" in p for p in problems(gold))


def test_rejects_missing_column(gold):
    path = gold / "frames.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["saturated"]), path)
    assert any("missing columns ['saturated']" in p for p in problems(gold))


def test_rejects_wrong_type(gold):
    path = gold / "sums.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index("detector")
    table = table.set_column(index, "detector", table.column("detector").cast(pa.int64()))
    pq.write_table(table, path)
    assert any("detector is int64, the contract says int8" in p for p in problems(gold))


@pytest.mark.parametrize("filename", ["sums.parquet", "meta.json"])
def test_rejects_missing_file(gold, filename):
    (gold / filename).unlink()
    assert f"{filename}: file not found" in problems(gold)


def test_rejects_meta_that_does_not_match(gold):
    meta = json.loads((gold / "meta.json").read_text())
    meta["n_frames"] = 3
    meta["last_frame_date_utc"] = "2020-01-01T00:00:00+00:00"
    meta["collections"] = ["spherex_dr1"]  # frames also use spherex_qr3
    del meta["commit"]
    (gold / "meta.json").write_text(json.dumps(meta))
    found = problems(gold)
    assert any("missing keys ['commit']" in p for p in found)
    assert any("n_frames is 3" in p for p in found)
    assert any("last_frame_date_utc is 2020-01-01" in p for p in found)
    assert any("collections does not list ['spherex_qr3']" in p for p in found)


def test_rejects_meta_dates_without_utc_offset(gold):
    meta = json.loads((gold / "meta.json").read_text())
    meta["run_date_utc"] = "2026-11-14T10:00:00"
    (gold / "meta.json").write_text(json.dumps(meta))
    assert any("run_date_utc" in p and "UTC offset" in p for p in problems(gold))


def test_reports_every_problem_at_once(gold):
    set_value(gold, "targets.parquet", "kind", 0, "planet")
    set_value(gold, "frames.parquet", "wavelength_um", 0, 4.9)
    (gold / contract.sum_png(TARGET, 0, 1)).unlink()
    assert len(problems(gold)) >= 3
