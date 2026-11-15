"""Real gold (task D8) from synthetic silver; AI's functions are replaced by small fakes."""

import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from serafim import contract
from serafim.cli import build_parser
from serafim.config import ROOT, load_config
from serafim.gold.build import ai_trajectory, build_gold, empty, targets_table
from serafim.gold.export import to_contract_table
from serafim.silver.differences import build_differences
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import frame_path, frames_table_path, save_npz
from serafim.silver.sums import build_sums

BARNARD = next(t for t in load_config()["targets"] if t["id"] == "barnard")
GRID = target_wcs(BARNARD["ra_now"], BARNARD["dec_now"], 0.25)
N = 146
# D1, D3 and D6 in two seasons: sums, differences and color images.
SPECS = [(d, s, 2) for d in (1, 3, 6) for s in (1, 2)]


@pytest.fixture
def root(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "targets.yaml", tmp_path / "config" / "targets.yaml")
    rng = np.random.default_rng(0)
    rows = []
    for detector, season, n in SPECS:
        for k in range(n):
            obs = f"S{season}D{detector}K{k}"
            fid = contract.frame_id("barnard", obs)
            save_npz(frame_path("barnard", fid, tmp_path), {
                "image": rng.normal(0, 0.5, (N, N)).astype(np.float32),
                "variance": np.full((N, N), 0.25, np.float32),
                "mask": np.zeros((N, N), bool), "saturated": np.zeros((N, N), bool),
                "footprint": np.ones((N, N), np.float32),
                "wcs_json": np.array(wcs_to_json(GRID)),
            })
            low, high = contract.DETECTOR_BANDS_UM[detector]
            rows.append({
                "frame_id": fid, "target_id": "barnard", "obs_id": obs,
                "collection": "spherex_dr1",
                "date_utc": pd.Timestamp("2025-08-15", tz="UTC")
                + pd.Timedelta(days=200 * season + k),
                "season": season, "detector": detector, "wavelength_um": (low + high) / 2,
                "bandwidth_um": 0.05, "flags": "", "saturated": False,
            })
    frames_table_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(pd.DataFrame(rows), contract.FRAMES),
                   frames_table_path(tmp_path))
    build_sums("barnard", root=tmp_path)
    build_differences("barnard", root=tmp_path)
    return tmp_path


def no_ai(**overrides):
    return {"catalogs": None, "spectra": None, "trajectory": None, **overrides}


def test_gold_has_the_seven_tables_meta_and_pngs_and_passes_validation(root):
    gold = root / "gold"
    counts = build_gold(root=root, gold_dir=gold, **no_ai())
    contract.validate_gold(gold)  # build_gold already checked it; once more on disk
    assert {p.name for p in gold.glob("*.parquet")} == set(contract.GOLD_TABLES)
    assert counts["frames.parquet"] == 12 and counts["catalogs.parquet"] == 0
    assert (gold / "png" / "barnard" / "color_s2.png").is_file()
    meta = json.loads((gold / "meta.json").read_text())
    assert meta["n_frames"] == 12
    targets = pq.read_table(gold / "targets.parquet").to_pandas()
    assert list(targets.target_id) == ["barnard"]  # only targets with real data


def test_the_grid_of_gold_is_the_grid_of_silver(root):
    gold = root / "gold"
    build_gold(root=root, gold_dir=gold, **no_ai())
    targets = pq.read_table(gold / "targets.parquet").to_pandas()
    with np.load(frame_path("barnard", contract.frame_id("barnard", "S1D1K0"), root)) as npz:
        assert targets.wcs_json[0] == str(npz["wcs_json"])


def test_rows_from_ai_go_into_gold(root):
    def catalogs(targets, frames):
        return pd.DataFrame([{
            "target_id": "barnard", "catalog": "gaia", "name": "Gaia DR3 1", "season": 0,
            "date_utc": pd.Timestamp("2026-01-01", tz="UTC"), "ra": BARNARD["ra_now"],
            "dec": BARNARD["dec_now"], "mag": 9.5, "mag_band": "G", "url": None}])

    def spectra(targets, frames):
        first = frames.iloc[0]
        return pd.DataFrame([{
            "target_id": "barnard", "frame_id": first.frame_id, "season": first.season,
            "date_utc": first.date_utc, "wavelength_um": first.wavelength_um, "flux_mjy": 3.0,
            "error_mjy": 0.1, "flag": None}])

    gold = root / "gold"
    counts = build_gold(root=root, gold_dir=gold, catalogs=catalogs, spectra=spectra,
                        trajectory=None)
    assert counts["catalogs.parquet"] == 1 and counts["spectra.parquet"] == 1


def test_a_failed_check_keeps_the_previous_gold(root):
    gold = root / "gold"
    build_gold(root=root, gold_dir=gold, **no_ai())
    before = (gold / "meta.json").read_text()

    def bad_catalogs(targets, frames):
        return pd.DataFrame([{
            "target_id": "barnard", "catalog": "tycho", "name": "x", "season": 1,
            "date_utc": pd.Timestamp("2026-01-01", tz="UTC"), "ra": 1.0, "dec": 1.0,
            "mag": None, "mag_band": "V", "url": None}])

    with pytest.raises(contract.GoldValidationError, match="catalog must be one of"):
        build_gold(root=root, gold_dir=gold, **no_ai(catalogs=bad_catalogs))
    assert (gold / "meta.json").read_text() == before
    assert (root / "gold.new").is_dir()  # left for inspection


def test_old_files_in_gold_are_gone(root):
    gold = root / "gold"
    gold.mkdir()
    (gold / "regions.parquet").write_bytes(b"old contract")
    build_gold(root=root, gold_dir=gold, **no_ai())
    assert not (gold / "regions.parquet").exists()


def test_a_target_without_silver_is_an_error(root):
    with pytest.raises(ValueError, match="wise_0855"):
        build_gold("wise_0855", root=root, gold_dir=root / "gold", **no_ai())


def test_without_silver_it_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="silver"):
        build_gold(root=tmp_path, gold_dir=tmp_path / "gold", **no_ai())


def test_a_target_added_by_coordinates_has_no_motion():
    table = targets_table([{"id": "field_1", "ra_now": 10.0, "dec_now": -5.0}], 0.25)
    row = table.iloc[0]
    assert (row.kind, row.pmra_mas_yr, row.pmdec_mas_yr, row["name"]) == ("custom", 0, 0, "field_1")


def test_ai_tables_have_the_contract_columns():
    # trajectory() reads data/silver of the repo: rows if there are sums, none in CI.
    table = ai_trajectory([BARNARD], pd.DataFrame())
    assert list(table.columns) == contract.TRAJECTORY.names
    assert list(empty("spectra.parquet").columns) == contract.SPECTRA.names


def test_the_gold_command_is_in_the_cli():
    args = build_parser().parse_args(["gold", "--no-catalogs", "--no-spectra"])
    assert (args.command, args.no_catalogs, args.no_spectra, args.out) == (
        "gold", True, True, "data/gold")
