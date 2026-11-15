"""Silver 3 (task D6): one season minus another, per detector.

Synthetic silver sums in the D5 format, written to a temporary folder. No network.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from serafim import contract
from serafim.config import load_config
from serafim.gold.export import to_contract_table
from serafim.silver import load_difference
from serafim.silver.differences import build_differences, subtract
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import (
    difference_path,
    differences_table_path,
    save_npz,
    sum_path,
    sums_table_path,
)

BARNARD = next(t for t in load_config()["targets"] if t["id"] == "barnard")
GRID = target_wcs(BARNARD["ra_now"], BARNARD["dec_now"], 0.25)
N = 146


def a_sum(value=0.0, variance=1.0, masked=None, n_saturated=None, n_valid=3):
    shape = (N, N)
    mask = np.zeros(shape, bool) if masked is None else masked
    n_sat = np.zeros(shape, np.int16) if n_saturated is None else n_saturated.astype(np.int16)
    image = np.full(shape, value, np.float32)
    image[mask] = np.nan
    return {"image": image, "variance": np.full(shape, variance, np.float32),
            "n_valid": np.where(mask, 0, n_valid).astype(np.int16), "mask": mask,
            "saturated": n_sat >= n_valid, "n_saturated": n_sat}


def pixel(y, x):
    out = np.zeros((N, N), bool)
    out[y, x] = True
    return out


# --- subtract -----------------------------------------------------------------------------------
def test_the_difference_is_b_minus_a_with_its_error():
    out = subtract(a_sum(2.0, 1.0), a_sum(5.0, 3.0))
    assert out["difference"][0, 0] == pytest.approx(3.0)
    assert out["error"][0, 0] == pytest.approx(2.0)  # sqrt(1 + 3)
    assert out["difference"].dtype == out["error"].dtype == np.float32


def test_the_mask_is_the_union_and_masked_pixels_stay_empty():
    out = subtract(a_sum(masked=pixel(1, 1)), a_sum(masked=pixel(2, 2)))
    assert out["mask"][1, 1] and out["mask"][2, 2] and not out["mask"][3, 3]
    assert np.isnan(out["difference"][1, 1]) and np.isnan(out["error"][2, 2])
    assert np.isfinite(out["difference"][3, 3])


def test_saturated_follows_the_paint_rule_of_either_sum():
    out = subtract(a_sum(n_saturated=pixel(4, 4) * 3), a_sum(n_saturated=pixel(5, 5) * 1))
    assert out["saturated"][4, 4]  # every frame of a saturated
    assert not out["saturated"][5, 5]  # 1 of 3 frames saturated in b: valid, not painted


def test_a_star_that_moves_leaves_a_dipole():
    yy, xx = np.mgrid[0:N, 0:N]

    def star(y):
        s = a_sum()
        s["image"] = (100 * np.exp(-((xx - 70) ** 2 + (yy - y) ** 2) / 2)).astype(np.float32)
        return s

    out = subtract(star(70), star(71))  # moved 1 px north (+y)
    assert out["difference"][72, 70] > 0 > out["difference"][69, 70]
    assert out["difference"].max() == pytest.approx(-out["difference"].min(), rel=0.01)


# --- Disk ---------------------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path):
    return tmp_path


def write_sums(root, cells):
    """cells: (detector, season) with a sum; also writes sums.parquet like D5 does."""
    rows = []
    for detector, season in cells:
        value = float(10 * detector + season)
        save_npz(sum_path("barnard", detector, season, root),
                 {**a_sum(value, 0.5), "mean_mjd": np.float64(60900 + 200 * season),
                  "wcs_json": np.array(wcs_to_json(GRID))})
        rows.append({
            "sum_id": f"barnard_d{detector}_s{season}", "target_id": "barnard",
            "detector": detector, "season": season,
            "mean_date_utc": pd.Timestamp("2025-08-15", tz="UTC"), "n_frames": 3,
            "background_noise": 0.1, "flags": "", "saturated": False,
            "png": contract.sum_png("barnard", detector, season),
        })
    sums_table_path(root).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(pd.DataFrame(rows), contract.SUMS), sums_table_path(root))
    return pd.DataFrame(rows)


# Detector 1 has 3 seasons (3 differences), detector 6 has 2 (1); the white image and all
# seasons are not cells.
CELLS = [(1, 1), (1, 2), (1, 3), (6, 2), (6, 3), (0, 1), (0, 2), (0, 3), (1, 0), (6, 0)]


def test_build_makes_the_agreed_pairs_and_a_table_that_follows_the_contract(root):
    sums = write_sums(root, CELLS)
    assert build_differences("barnard", root=root) == 4
    table = pq.read_table(differences_table_path(root))
    assert table.schema.equals(contract.DIFFERENCES)
    rows = table.to_pandas()
    assert list(zip(rows.detector, rows.season_a, rows.season_b, strict=True)) == [
        (1, 1, 2), (1, 1, 3), (1, 2, 3), (6, 2, 3)]
    errors = []
    contract._check_values({"differences.parquet": rows}, errors)
    contract._check_links({"sums.parquet": sums, "differences.parquet": rows}, errors)
    assert errors == []


def test_load_difference_gives_the_agreed_arrays(root):
    write_sums(root, CELLS)
    build_differences("barnard", root=root)
    with np.load(difference_path("barnard", 1, 1, 3, root)) as npz:  # numpy only, no pickle
        assert set(npz.files) == {"difference", "error", "mask", "saturated", "season_a",
                                  "season_b", "wcs_json"}
    d = load_difference("barnard", 1, 1, 3, root)
    assert (d["season_a"], d["season_b"]) == (1, 3)
    assert d["difference"][0, 0] == pytest.approx(13.0 - 11.0)
    assert d["error"][0, 0] == pytest.approx(1.0)  # sqrt(0.5 + 0.5)
    assert d["wcs"].pixel_shape == (N, N)


def test_a_second_run_gives_the_same_table_and_old_pairs_go(root):
    write_sums(root, CELLS)
    build_differences("barnard", root=root)
    first = pq.read_table(differences_table_path(root))
    stale = difference_path("barnard", 6, 1, 3, root)
    save_npz(stale, {"difference": np.zeros(1)})
    build_differences("barnard", root=root)
    assert pq.read_table(differences_table_path(root)).equals(first)
    assert not stale.exists()


def test_other_targets_stay_in_the_differences_table(root):
    write_sums(root, CELLS)
    build_differences("barnard", root=root)
    rows = pq.read_table(differences_table_path(root)).to_pandas()
    other = rows.assign(target_id="wise_0855", difference_id="wise_0855_" + rows.difference_id)
    pq.write_table(to_contract_table(pd.concat([rows, other]), contract.DIFFERENCES),
                   differences_table_path(root))
    build_differences("barnard", root=root)
    assert set(pq.read_table(differences_table_path(root)).to_pandas().target_id) == {
        "barnard", "wise_0855"}


def test_without_sums_it_is_an_error(root):
    with pytest.raises(FileNotFoundError, match="sums"):
        build_differences(root=root)
