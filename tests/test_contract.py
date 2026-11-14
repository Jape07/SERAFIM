"""The gold contract must match the table the team agreed on at H0."""

import pytest

from serafim import contract

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
