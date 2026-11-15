"""Silver 1 (task D4) on a real SPHEREx cutout of Barnard's Star (detector 3, season 2).

tests/fixtures/barnard_2026W12_1A_0598_1.fits is that bronze file without EPSF (0.4 MB).
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from serafim import contract
from serafim.bronze.download import write_manifest
from serafim.cli import build_parser
from serafim.config import ROOT, load_config
from serafim.silver import load_frame
from serafim.silver.frames import (
    build_frames,
    flag_bits,
    frame_arrays,
    frame_row,
    has_flags,
)
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import frame_path, frames_table_path

FIXTURE = Path(__file__).parent / "fixtures" / "barnard_2026W12_1A_0598_1.fits"
OBS_ID = "2026W12_1A_0598_1"
BARNARD = next(t for t in load_config()["targets"] if t["id"] == "barnard")
FRAME = {
    "target_id": "barnard", "obs_id": OBS_ID, "collection": "spherex_dr1", "detector": 3,
    "season": 2, "t_mid_mjd": 61118.3098975, "ra": BARNARD["ra_now"], "dec": BARNARD["dec_now"],
    "size_deg": 0.25, "path": f"data/bronze/barnard/{OBS_ID}.fits",
}


def grid():
    return target_wcs(BARNARD["ra_now"], BARNARD["dec_now"], 0.25)


@pytest.fixture(scope="module")
def arrays():
    with fits.open(FIXTURE) as hdul:
        return frame_arrays(hdul, grid())


@pytest.fixture(scope="module")
def row():
    with fits.open(FIXTURE) as hdul:
        return frame_row(hdul, BARNARD, FRAME, 2.0)


def test_the_fixture_is_small_and_has_what_silver_needs():
    assert FIXTURE.stat().st_size < 1_000_000
    with fits.open(FIXTURE) as hdul:
        names = {hdu.name for hdu in hdul}
    assert {"IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE"} <= names


def test_flag_names_come_from_the_header():
    with fits.open(FIXTURE) as hdul:
        bits = flag_bits(hdul["FLAGS"].header)
    assert (bits["OVERFLOW"], bits["NONLINEAR"], bits["SOURCE"], bits["HOT"]) == (1, 15, 21, 10)


# --- The row of frames.parquet ----------------------------------------------------------------
def test_the_row_has_the_contract_columns(row):
    assert list(row) == contract.FRAMES.names
    assert row["frame_id"] == contract.frame_id("barnard", OBS_ID)
    assert (row["detector"], row["season"], row["collection"]) == (3, 2, "spherex_dr1")


def test_the_wavelength_at_the_target_is_inside_detector_3(row):
    low, high = contract.DETECTOR_BANDS_UM[3]
    assert low <= row["wavelength_um"] <= high
    assert 0 < row["bandwidth_um"] < 0.2


def test_the_date_is_the_mean_date_of_the_exposure(row):
    with fits.open(FIXTURE) as hdul:
        assert row["date_utc"] == pd.Timestamp(hdul["IMAGE"].header["DATE-AVG"], tz="UTC")


def test_barnard_is_saturated_and_source_is_not_reported(row):
    names = row["flags"].split(",")
    assert "OVERFLOW" in names and "SOURCE" not in names
    assert row["saturated"] is True


def test_a_detector_that_does_not_match_the_manifest_is_an_error():
    with fits.open(FIXTURE) as hdul, pytest.raises(ValueError, match="DETECTOR 3"):
        frame_row(hdul, BARNARD, {**FRAME, "detector": 5}, 2.0)


def test_the_row_passes_the_checks_of_validate_gold(row):
    errors = []
    contract._check_values({"frames.parquet": pd.DataFrame([row])}, errors)
    assert errors == []


# --- The arrays of the .npz --------------------------------------------------------------------
def test_arrays_are_on_the_grid_with_the_agreed_types(arrays):
    for name in ("image", "variance", "mask", "saturated", "footprint"):
        assert arrays[name].shape == (146, 146), name
    assert arrays["image"].dtype == arrays["variance"].dtype == np.float32
    assert arrays["footprint"].dtype == np.float32
    assert arrays["mask"].dtype == arrays["saturated"].dtype == bool


def test_nothing_unusable_is_left_unmasked(arrays):
    usable = ~arrays["mask"]
    assert np.isfinite(arrays["image"][usable]).all()
    assert np.isfinite(arrays["variance"][usable]).all()
    assert (arrays["footprint"][usable] >= 0.5).all()
    assert arrays["footprint"].min() >= 0 and arrays["footprint"].max() <= 1


def test_the_saturated_center_of_barnard_is_marked(arrays):
    center = 145 / 2
    rows, cols = np.nonzero(arrays["saturated"])
    distance = np.hypot(cols - center, rows - center)
    assert distance.min() < 2  # Barnard is at the center of its grid


def test_a_bad_pixel_does_not_leak_into_its_neighbors(tmp_path):
    """A HOT pixel with a huge value must be masked on the grid, not averaged into neighbors."""
    copy = tmp_path / "hot.fits"
    with fits.open(FIXTURE) as hdul:
        bits = flag_bits(hdul["FLAGS"].header)
        hdul["FLAGS"].data[30, 30] |= 1 << bits["HOT"]
        hdul["IMAGE"].data[30, 30] = 1e6
        hdul.writeto(copy)
    with fits.open(copy) as hdul:
        out = frame_arrays(hdul, grid())
    assert np.nanmax(np.where(out["mask"], np.nan, out["image"])) < 1e4


def test_a_star_lands_where_the_sky_says(tmp_path):
    """Orientation and alignment: a blob at a native pixel ends at the right grid pixel."""
    copy = tmp_path / "blob.fits"
    with fits.open(FIXTURE) as hdul:
        sky = WCS(hdul["IMAGE"].header)
        ny, nx = hdul["IMAGE"].data.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        hdul["IMAGE"].data = (100 * np.exp(-((xx - 40.3) ** 2 + (yy - 100.6) ** 2) / 4.5)).astype(
            np.float32)
        hdul["ZODI"].data = np.zeros((ny, nx), np.float32)
        hdul["VARIANCE"].data = np.ones((ny, nx), np.float32)
        hdul["FLAGS"].data = np.zeros((ny, nx), np.int32)
        hdul.writeto(copy)
    with fits.open(copy) as hdul:
        out = frame_arrays(hdul, grid())
    want_x, want_y = grid().world_to_pixel(sky.pixel_to_world(40.3, 100.6))
    image = np.where(out["mask"], 0, out["image"])
    rows, cols = np.mgrid[0:146, 0:146]
    near = np.hypot(cols - want_x, rows - want_y) < 7
    weight = image * near
    got_x, got_y = (weight * cols).sum() / weight.sum(), (weight * rows).sum() / weight.sum()
    assert abs(got_x - want_x) < 0.1 and abs(got_y - want_y) < 0.1


def test_has_flags_ignores_names_the_header_does_not_have():
    flags = np.array([[2, 0]])
    assert has_flags(flags, ["OVERFLOW", "NOT_A_FLAG"], {"OVERFLOW": 1}).tolist() == [[True, False]]


# --- Disk: build_frames, the .npz and load_frame ------------------------------------------------
@pytest.fixture
def root(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "targets.yaml", tmp_path / "config" / "targets.yaml")
    bronze = tmp_path / FRAME["path"]
    bronze.parent.mkdir(parents=True)
    shutil.copy(FIXTURE, bronze)
    write_manifest([FRAME], tmp_path)
    return tmp_path


def test_build_writes_a_portable_npz_and_the_frames_table(root):
    assert build_frames("barnard", root=root) == 1
    path = frame_path("barnard", contract.frame_id("barnard", OBS_ID), root)
    with np.load(path) as npz:  # numpy only, no pickle
        assert set(npz.files) == {"image", "variance", "mask", "saturated", "footprint", "wcs_json"}
        assert str(npz["wcs_json"]) == wcs_to_json(grid())

    frame = load_frame("barnard", contract.frame_id("barnard", OBS_ID), root)
    assert frame["wcs"].pixel_shape == (146, 146)
    assert frame["image"].shape == (146, 146)

    table = pq.read_table(frames_table_path(root))
    assert table.schema.names == contract.FRAMES.names
    assert table.schema.equals(contract.FRAMES)
    assert table.num_rows == 1


def test_a_second_run_reprojects_nothing_and_force_does(root):
    build_frames("barnard", root=root)
    assert build_frames("barnard", root=root) == 0
    assert build_frames("barnard", root=root, force=True) == 1
    assert pq.read_table(frames_table_path(root)).num_rows == 1  # rows are replaced, not added


def test_other_targets_stay_in_the_frames_table(root):
    build_frames("barnard", root=root)
    table = pq.read_table(frames_table_path(root)).to_pandas()
    other = table.assign(target_id="wise_0855", frame_id="wise_0855_x", obs_id="x")
    pq.write_table(pq.read_table(frames_table_path(root)).from_pandas(
        pd.concat([table, other]), schema=contract.FRAMES), frames_table_path(root))
    build_frames("barnard", root=root)
    assert set(pq.read_table(frames_table_path(root)).to_pandas().target_id) == {
        "barnard", "wise_0855"}


def test_an_empty_bronze_is_an_error(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "targets.yaml", tmp_path / "config" / "targets.yaml")
    with pytest.raises(FileNotFoundError, match="bronze"):
        build_frames(root=tmp_path)


def test_the_silver_command_is_in_the_cli():
    args = build_parser().parse_args(["silver", "--target", "barnard", "--force"])
    assert (args.command, args.target, args.force) == ("silver", "barnard", True)
