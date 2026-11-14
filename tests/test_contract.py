"""The gold contract must match the table the team agreed on at H0.

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
from serafim.gold.mock import write_mock_gold

AGREED_COLUMNS = {
    "regions.parquet": ["region_id", "name", "ra", "dec", "size_deg", "wcs_json"],
    "frames.parquet": [
        "frame_id",
        "region_id",
        "obs_id",
        "date_utc",
        "wavelength_um",
        "bandwidth_um",
        "png",
        "png_diff",
    ],
    "detections.parquet": [
        "detection_id",
        "frame_id",
        "object_id",
        "ra",
        "dec",
        "x_px",
        "y_px",
        "snr",
        "sign",
        "p_real",
    ],
    "objects.parquet": [
        "object_id",
        "region_id",
        "name",
        "type",
        "status",
        "reason",
        "catalog",
        "url",
        "n_detections",
        "first_date_utc",
        "last_date_utc",
    ],
}


@pytest.mark.parametrize("filename", sorted(AGREED_COLUMNS))
def test_columns_match_the_agreed_table(filename):
    assert contract.GOLD_TABLES[filename].names == AGREED_COLUMNS[filename]


@pytest.mark.parametrize("filename", sorted(AGREED_COLUMNS))
def test_ids_are_required(filename):
    schema = contract.GOLD_TABLES[filename]
    for field in schema:
        if field.name.endswith("_id"):
            assert not field.nullable, f"{filename}: {field.name} must not be null"


def test_only_three_statuses():
    assert contract.STATUSES == ("Known", "Unidentified candidate", "Discarded")


def test_meta_keys():
    assert contract.META_KEYS == ("run_date_utc", "commit", "n_frames")


# --- Simulated gold (task D1) -------------------------------------------------
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
    detections = read(gold, "detections.parquet")
    objects = read(gold, "objects.parquet")

    assert (len(frames), len(detections), len(objects)) == (10, 20, 5)
    assert set(objects.status) == set(contract.STATUSES)
    # Both cases the web has to handle: frames with and without a difference image,
    # detections with and without p_real.
    assert frames.png_diff.isna().any() and frames.png_diff.notna().any()
    assert detections.p_real.isna().any() and detections.p_real.notna().any()
    assert (gold / "objects.csv").is_file()
    assert json.loads((gold / "meta.json").read_text())["n_frames"] == 10


def test_markers_and_sky_positions_agree(gold):
    """x_px, y_px (PNG, origin top-left) and ra, dec describe the same point."""
    region = read(gold, "regions.parquet").iloc[0]
    detections = read(gold, "detections.parquet")
    wcs = WCS(fits.Header(json.loads(region.wcs_json)))
    n_rows = json.loads(region.wcs_json)["NAXIS2"]

    ra, dec = wcs.wcs_pix2world(detections.x_px, n_rows - 1 - detections.y_px, 0)
    np.testing.assert_allclose(ra, detections.ra, atol=1e-5)
    np.testing.assert_allclose(dec, detections.dec, atol=1e-5)


def test_mock_is_reproducible(gold, tmp_path):
    again = tmp_path / "again"
    write_mock_gold(again, load_config())
    for filename in contract.GOLD_TABLES:
        assert pq.read_table(gold / filename).equals(pq.read_table(again / filename)), filename
    for png in (gold / "png").iterdir():
        assert png.read_bytes() == (again / "png" / png.name).read_bytes(), png.name


def test_cli_mock_writes_valid_gold(tmp_path):
    out = tmp_path / "gold"
    assert main(["mock", "--out", str(out)]) == 0
    contract.validate_gold(out)


# --- validate_gold() catches broken gold --------------------------------------
@pytest.mark.parametrize(
    ("filename", "column", "value", "expected"),
    [
        ("objects.parquet", "status", "New object", "status must be one of"),
        ("frames.parquet", "wavelength_um", 7.5, "wavelength_um outside"),
        ("detections.parquet", "frame_id", "nope", "frame_id not found in frames.parquet"),
        ("detections.parquet", "object_id", "nope", "object_id not found in objects.parquet"),
        ("detections.parquet", "p_real", 1.4, "p_real outside [0, 1]"),
        ("detections.parquet", "sign", 0, "sign must be +1 or -1"),
        ("frames.parquet", "png", "/png/pole_f000.png", "must be a relative path"),
        ("frames.parquet", "png", "png/missing.png", "file not found in gold"),
        ("frames.parquet", "frame_id", None, "frame_id has 1 empty values"),
        ("objects.parquet", "n_detections", 99, "n_detections does not match"),
    ],
)
def test_rejects_bad_values(gold, filename, column, value, expected):
    set_value(gold, filename, column, 0, value)
    assert any(expected in problem for problem in problems(gold)), problems(gold)


def test_rejects_missing_column(gold):
    path = gold / "frames.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["png"]), path)
    assert any("missing columns ['png']" in p for p in problems(gold))


def test_rejects_wrong_type(gold):
    path = gold / "detections.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index("sign")
    table = table.set_column(index, "sign", table.column("sign").cast(pa.float64()))
    pq.write_table(table, path)
    assert any("sign is double, the contract says int8" in p for p in problems(gold))


@pytest.mark.parametrize("filename", ["objects.parquet", "meta.json"])
def test_rejects_missing_file(gold, filename):
    (gold / filename).unlink()
    assert f"{filename}: file not found" in problems(gold)


def test_rejects_meta_that_does_not_match(gold):
    meta = json.loads((gold / "meta.json").read_text())
    meta["n_frames"] = 3
    del meta["commit"]
    (gold / "meta.json").write_text(json.dumps(meta))
    found = problems(gold)
    assert any("missing keys ['commit']" in p for p in found)
    assert any("n_frames is 3" in p for p in found)


def test_reports_every_problem_at_once(gold):
    set_value(gold, "objects.parquet", "status", 0, "New object")
    set_value(gold, "frames.parquet", "wavelength_um", 0, 9.0)
    (gold / "png" / "pole_f000.png").unlink()
    assert len(problems(gold)) >= 3
