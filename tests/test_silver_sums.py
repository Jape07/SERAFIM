"""Silver 2 (task D5): sums per detector and season, and their alignment with the sky.

Synthetic silver frames in the D4 format, written to a temporary folder. No network.
"""

import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from serafim import contract
from serafim.cli import build_parser
from serafim.config import ROOT, load_config
from serafim.gold.export import to_contract_table
from serafim.silver import load_sum
from serafim.silver.alignment import measure_alignment, measure_offsets, write_section
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import frame_path, frames_table_path, save_npz, sum_path, sums_table_path
from serafim.silver.sums import (
    background_noise,
    build_sums,
    center_saturated,
    combine,
    painted_saturated,
    sum_cells,
)

BARNARD = next(t for t in load_config()["targets"] if t["id"] == "barnard")
GRID = target_wcs(BARNARD["ra_now"], BARNARD["dec_now"], 0.25)
N = 146


def frame(value=0.0, variance=1.0, masked=None, saturated=None):
    image = np.full((N, N), value, np.float32)
    mask = np.zeros((N, N), bool) if masked is None else masked
    sat = np.zeros((N, N), bool) if saturated is None else saturated
    return {"image": image, "variance": np.full((N, N), variance, np.float32), "mask": mask,
            "saturated": sat, "footprint": np.ones((N, N), np.float32)}


def pixel(y, x):
    out = np.zeros((N, N), bool)
    out[y, x] = True
    return out


# --- combine -------------------------------------------------------------------------------------
def test_the_sum_is_the_inverse_variance_weighted_mean():
    out = combine([frame(1.0, 1.0), frame(4.0, 4.0)])
    np.testing.assert_allclose(out["image"][0, 0], (1 / 1 + 4 / 4) / (1 + 1 / 4))  # 1.6
    np.testing.assert_allclose(out["variance"][0, 0], 1 / (1 + 1 / 4))  # 0.8
    assert out["n_valid"][0, 0] == 2


def test_masked_pixels_are_left_out_and_an_empty_pixel_stays_empty():
    out = combine([frame(1.0, masked=pixel(5, 5) | pixel(9, 9)), frame(3.0, masked=pixel(5, 5))])
    assert out["n_valid"][9, 9] == 1 and out["image"][9, 9] == pytest.approx(3.0)
    assert out["n_valid"][5, 5] == 0 and out["mask"][5, 5] and np.isnan(out["image"][5, 5])
    assert not out["mask"][0, 0]


def test_saturated_means_every_frame_and_n_saturated_counts_them():
    out = combine([frame(saturated=pixel(1, 1) | pixel(2, 2)), frame(saturated=pixel(2, 2))])
    assert (out["n_saturated"][1, 1], out["n_saturated"][2, 2]) == (1, 2)
    assert not out["saturated"][1, 1] and out["saturated"][2, 2]


def test_an_absurd_value_of_one_frame_is_dropped():
    # Like eps_indi_b d5 s1: one frame at -85 with an error of 0.04, the others at 0.1.
    frames = [frame(0.1, 0.04**2) for _ in range(12)]
    frames[0]["image"][7, 7] = -85.0
    out = combine(frames)
    assert out["image"][7, 7] == pytest.approx(0.1, abs=1e-6)
    assert out["n_valid"][7, 7] == 11 and out["n_valid"][0, 0] == 12


def test_with_three_frames_the_odd_one_goes():
    # Like wise_0855 d5 s1: 0.02, -52.3 and 0.04.
    frames = [frame(0.02, 0.033**2), frame(-52.3, 0.032**2), frame(0.04, 0.034**2)]
    out = combine(frames)
    assert out["image"][0, 0] == pytest.approx(0.03, abs=0.002) and out["n_valid"][0, 0] == 2


def test_a_star_that_changes_between_frames_is_kept():
    # Each frame sees another wavelength: a star pixel at 10, 12, 8, 11, 9 is real, not clipped.
    frames = [frame(v, 0.01) for v in (10.0, 12.0, 8.0, 11.0, 9.0)]
    out = combine(frames)
    assert out["n_valid"][0, 0] == 5


def test_two_frames_are_never_clipped():
    out = combine([frame(0.1, 0.0016), frame(-50.0, 0.0016)])
    assert out["n_valid"][0, 0] == 2


def test_types_are_the_agreed_ones():
    out = combine([frame(), frame()])
    assert out["image"].dtype == out["variance"].dtype == np.float32
    assert out["n_valid"].dtype == out["n_saturated"].dtype == np.int16
    assert out["mask"].dtype == out["saturated"].dtype == bool


def test_the_paint_rule():
    arrays = {"saturated": np.array([True, False, False, False]),
              "n_saturated": np.array([3, 2, 1, 0]), "n_valid": np.array([0, 2, 3, 0])}
    assert painted_saturated(arrays).tolist() == [True, True, False, False]


def test_the_center_is_saturated_only_if_painted_near_the_target():
    out = combine([frame(saturated=pixel(73, 73)), frame(saturated=pixel(73, 73))])
    assert center_saturated(out, GRID, 72.5, 72.5, 2.0)
    assert not center_saturated(out, GRID, 20.0, 20.0, 2.0)


def test_background_noise_falls_with_the_square_root_of_n():
    rng = np.random.default_rng(1)
    frames = [frame() for _ in range(9)]
    for f in frames:
        f["image"] = rng.normal(0, 1, (N, N)).astype(np.float32)
    single = background_noise(frames[0]["image"], frames[0]["mask"])
    summed = combine(frames)
    assert single == pytest.approx(1.0, rel=0.1)
    assert background_noise(summed["image"], summed["mask"]) == pytest.approx(1 / 3, rel=0.15)


# --- Which sums -----------------------------------------------------------------------------------
def cells_table(cells):
    return pd.DataFrame([{"detector": d, "season": s, "frame_id": f"f{d}{s}"} for d, s in cells])


def test_barnard_has_27_sums():
    found = sum_cells(cells_table([(d, s) for d in range(1, 7) for s in range(1, 4)]))
    pairs = [(d, s) for d, s, _ in found]
    assert len(pairs) == 27
    assert (0, 1) in pairs and (3, 0) in pairs and (0, 0) not in pairs


def test_a_cell_without_frames_has_no_sum():
    cells = [(d, s) for d in range(1, 7) for s in range(1, 4) if (d, s) != (6, 1)]
    pairs = [(d, s) for d, s, _ in sum_cells(cells_table(cells))]
    assert (6, 1) not in pairs and (6, 0) in pairs and len(pairs) == 26


# --- Disk ---------------------------------------------------------------------------------------
def write_frames(root, specs, *, stars=None, seed=0):
    """specs: (detector, season, n). Writes frame .npz and frames.parquet like D4 does."""
    rng = np.random.default_rng(seed)
    rows = []
    for detector, season, n in specs:
        for k in range(n):
            obs = f"S{season}D{detector}K{k}"
            f = frame()
            f["image"] = rng.normal(0, 0.5, (N, N)).astype(np.float32)
            if stars is not None:
                f["image"] += stars
            f["variance"][:] = 0.25
            fid = contract.frame_id("barnard", obs)
            wcs_text = np.array(wcs_to_json(GRID))
            save_npz(frame_path("barnard", fid, root), {**f, "wcs_json": wcs_text})
            low, high = contract.DETECTOR_BANDS_UM[detector]
            start = pd.Timestamp("2025-08-15", tz="UTC")
            rows.append({
                "frame_id": fid, "target_id": "barnard", "obs_id": obs,
                "collection": "spherex_dr1",
                "date_utc": start + pd.Timedelta(days=200 * season + k),
                "season": season, "detector": detector, "wavelength_um": (low + high) / 2,
                "bandwidth_um": 0.05, "flags": "BLOOM" if k == 0 else "OVERFLOW",
                "saturated": False,
            })
    table = to_contract_table(pd.DataFrame(rows), contract.FRAMES)
    frames_table_path(root).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, frames_table_path(root))
    return pd.DataFrame(rows)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "targets.yaml", tmp_path / "config" / "targets.yaml")
    return tmp_path


SPECS = [(1, 1, 2), (1, 2, 3), (3, 1, 2), (3, 2, 2)]  # 4 cells + 2 white + 2 all seasons


def test_build_writes_the_sums_and_a_table_that_follows_the_contract(root):
    frames = write_frames(root, SPECS)
    assert build_sums("barnard", root=root) == 8
    table = pq.read_table(sums_table_path(root))
    assert table.schema.equals(contract.SUMS)
    sums = table.to_pandas()
    assert sorted(zip(sums.detector, sums.season, strict=True)) == [
        (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2)]
    errors = []
    tables = {"targets.parquet": pd.DataFrame({"target_id": ["barnard"]}),
              "frames.parquet": frames, "sums.parquet": sums}
    contract._check_links(tables, errors)
    contract._check_values({"sums.parquet": sums}, errors)
    assert errors == []  # n_frames matches frames.parquet, png names, flags
    white = sums[(sums.detector == 0) & (sums.season == 2)].iloc[0]
    assert white.n_frames == 5 and white["flags"] == "BLOOM,OVERFLOW"


def test_load_sum_gives_the_agreed_arrays(root):
    write_frames(root, SPECS)
    build_sums("barnard", root=root)
    with np.load(sum_path("barnard", 1, 2, root)) as npz:  # numpy only, no pickle
        assert set(npz.files) == {"image", "variance", "n_valid", "mask", "saturated",
                                  "n_saturated", "mean_mjd", "wcs_json"}
    arrays = load_sum("barnard", 1, 2, root)
    assert arrays["n_valid"].dtype == arrays["n_saturated"].dtype == np.int16
    assert arrays["n_valid"].max() == 3
    assert isinstance(arrays["mean_mjd"], float)
    assert arrays["wcs"].pixel_shape == (N, N)


def test_noise_of_the_sums_is_lower_than_one_frame(root):
    write_frames(root, SPECS)
    build_sums("barnard", root=root)
    sums = pq.read_table(sums_table_path(root)).to_pandas().set_index(["detector", "season"])
    assert sums.loc[(1, 2), "background_noise"] == pytest.approx(0.5 / np.sqrt(3), rel=0.15)
    assert sums.loc[(0, 2), "background_noise"] == pytest.approx(0.5 / np.sqrt(5), rel=0.15)


def test_a_second_run_gives_the_same_table_and_stale_sums_go(root):
    write_frames(root, SPECS)
    build_sums("barnard", root=root)
    first = pq.read_table(sums_table_path(root))
    stale = sum_path("barnard", 6, 9, root)
    save_npz(stale, {"image": np.zeros(1)})
    build_sums("barnard", root=root)
    assert pq.read_table(sums_table_path(root)).equals(first)
    assert not stale.exists()


def test_other_targets_stay_in_the_sums_table(root):
    write_frames(root, SPECS)
    build_sums("barnard", root=root)
    sums = pq.read_table(sums_table_path(root)).to_pandas()
    other = sums.assign(target_id="wise_0855", sum_id="wise_0855_" + sums.sum_id)
    pq.write_table(to_contract_table(pd.concat([sums, other]), contract.SUMS),
                   sums_table_path(root))
    build_sums("barnard", root=root)
    assert set(pq.read_table(sums_table_path(root)).to_pandas().target_id) == {
        "barnard", "wise_0855"}


def test_without_frames_it_is_an_error(root):
    with pytest.raises(FileNotFoundError, match="frames"):
        build_sums(root=root)


# --- Alignment ------------------------------------------------------------------------------------
def star_field(shift=(0.0, 0.0), n=40, seed=3):
    """Gaussian stars at catalog pixels, drawn `shift` away. Returns image and catalog pixels."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(12, N - 12, (n, 2))
    xy = xy[np.hypot(xy[:, 0] - 72.5, xy[:, 1] - 72.5) > 15]
    yy, xx = np.mgrid[0:N, 0:N]
    image = np.zeros((N, N))
    for x, y in xy:
        image += 50 * np.exp(-((xx - x - shift[0]) ** 2 + (yy - y - shift[1]) ** 2) / (2 * 0.9**2))
    return image, xy


def test_a_known_shift_is_measured():
    image, xy = star_field(shift=(0.3, -0.2))
    image += np.random.default_rng(4).normal(0, 0.2, image.shape)
    sky = GRID.pixel_to_world(xy[:, 0], xy[:, 1])
    found = measure_offsets(image, np.zeros(image.shape, bool), GRID, sky.ra.deg, sky.dec.deg,
                            (72.5, 72.5))
    assert found["n_stars"] >= 25
    assert found["dx_px"] == pytest.approx(0.3, abs=0.05)
    assert found["dy_px"] == pytest.approx(-0.2, abs=0.05)


def test_alignment_of_built_sums_against_a_catalog(root):
    stars, xy = star_field()
    write_frames(root, SPECS, stars=stars)
    build_sums("barnard", root=root)
    sky = GRID.pixel_to_world(xy[:, 0], xy[:, 1])
    sources = pd.DataFrame({"name": [f"s{i}" for i in range(len(xy))], "ra": sky.ra.deg,
                            "dec": sky.dec.deg, "pmra_mas_yr": np.nan, "pmdec_mas_yr": np.nan,
                            "parallax_mas": np.nan, "epoch_jyear": np.nan, "mag": 12.0,
                            "url": None})
    table = measure_alignment("barnard", root=root, sources=sources)
    assert sorted(zip(table.detector, table.season, strict=True)) == [
        (1, 1), (1, 2), (3, 1), (3, 2)]
    assert (table.n_stars >= 20).all()
    assert np.abs(table[["dx_px", "dy_px"]].to_numpy()).max() < 0.05


def test_the_doc_keeps_the_sections_of_other_targets(tmp_path):
    path = tmp_path / "alignment.md"
    table = pd.DataFrame([{"detector": 1, "season": 1, "n_stars": 30, "dx_px": 0.01,
                           "dy_px": -0.02}])
    write_section("wise_0855", table, path)
    write_section("barnard", table, path)
    write_section("barnard", table.assign(dx_px=0.2), path)
    text = path.read_text(encoding="utf-8")
    assert text.count("## barnard") == 1 and "## wise_0855" in text
    assert "DOES NOT meet" in text.split("## wise_0855")[0]  # barnard's section now has 0.2 px


def write_two_sums(root, shift):
    """Detector 1, seasons 1 and 2: the same stars, season 2 drawn `shift` px away."""
    rows = []
    for season, offset in ((1, (0.0, 0.0)), (2, shift)):
        image, _ = star_field(shift=offset)
        image += np.random.default_rng(season).normal(0, 0.2, image.shape)
        save_npz(sum_path("barnard", 1, season, root), {
            "image": image.astype(np.float32), "variance": np.full((N, N), 0.04, np.float32),
            "n_valid": np.full((N, N), 3, np.int16), "mask": np.zeros((N, N), bool),
            "saturated": np.zeros((N, N), bool), "n_saturated": np.zeros((N, N), np.int16),
            "mean_mjd": np.float64(61000 + 180 * season), "wcs_json": np.array(wcs_to_json(GRID)),
        })
        rows.append({"sum_id": f"barnard_d1_s{season}", "target_id": "barnard", "detector": 1,
                     "season": season, "mean_date_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                     "n_frames": 3, "background_noise": 0.2, "flags": "", "saturated": False,
                     "png": contract.sum_png("barnard", 1, season)})
    sums_table_path(root).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(pd.DataFrame(rows), contract.SUMS), sums_table_path(root))


def test_a_shift_between_seasons_is_measured_without_a_catalog(root):
    from serafim.silver.alignment import measure_between_seasons

    write_two_sums(root, (0.3, -0.2))
    table = measure_between_seasons("barnard", root=root)
    row = table.iloc[0]
    assert (row.detector, row.season) == (1, "1-2") and row.n_stars >= 20
    assert row.dx_px == pytest.approx(0.3, abs=0.05)
    assert row.dy_px == pytest.approx(-0.2, abs=0.05)


def test_without_gaia_the_doc_says_it_measured_between_seasons(root, monkeypatch):
    from serafim.silver import alignment
    from serafim.validation.catalogs import CatalogUnavailable

    def no_gaia(*args, **kwargs):
        raise CatalogUnavailable("gaia did not answer")

    write_two_sums(root, (0.0, 0.0))
    monkeypatch.setattr(alignment, "measure_alignment", no_gaia)
    worst = alignment.run("barnard", root=root)
    text = (root / alignment.DOC_PATH).read_text(encoding="utf-8")
    assert "Gaia did not answer" in text and "| 1 | 1-2 |" in text
    assert worst < 0.05


def test_the_alignment_command_is_in_the_cli():
    args = build_parser().parse_args(["alignment", "--target", "barnard"])
    assert (args.command, args.target) == ("alignment", "barnard")
