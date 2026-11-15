"""Gold PNGs (task D7) from synthetic silver sums and their differences. No network."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from matplotlib import colormaps
from matplotlib import image as mpimg

from serafim import contract
from serafim.cli import build_parser
from serafim.config import load_config
from serafim.gold.export import to_contract_table
from serafim.gold.png import NEUTRAL, build_pngs, color_rgba
from serafim.silver.differences import build_differences
from serafim.silver.grid import target_wcs, wcs_to_json
from serafim.silver.store import save_npz, sum_path, sums_table_path

BARNARD = next(t for t in load_config()["targets"] if t["id"] == "barnard")
GRID = target_wcs(BARNARD["ra_now"], BARNARD["dec_now"], 0.25)
N = 146
MAGMA_TOP = colormaps["magma"](1.0, bytes=True)
DIVERGING_MIDDLE = colormaps["RdBu_r"](0.5, bytes=True)


def a_sum(seed, back=0.0, star=None, saturated=None, empty=None):
    rng = np.random.default_rng(seed)
    image = rng.normal(back, 0.1, (N, N))
    if star is not None:
        yy, xx = np.mgrid[0:N, 0:N]
        image += 20 * np.exp(-((xx - star[0]) ** 2 + (yy - star[1]) ** 2) / 2)
    n_saturated = np.zeros((N, N), np.int16)
    mask = np.zeros((N, N), bool)
    if saturated is not None:
        n_saturated[saturated] = 3
    if empty is not None:
        mask[empty] = True
    n_valid = np.where(mask | (n_saturated == 3), 0, 3).astype(np.int16)
    mask |= n_valid == 0
    image[mask] = np.nan
    return {"image": image.astype(np.float32), "variance": np.full((N, N), 0.01, np.float32),
            "n_valid": n_valid, "mask": mask, "saturated": n_saturated == 3,
            "n_saturated": n_saturated, "mean_mjd": np.float64(61000),
            "wcs_json": np.array(wcs_to_json(GRID))}


def box(y, x, size=2):
    out = np.zeros((N, N), bool)
    out[y - size : y + size + 1, x - size : x + size + 1] = True
    return out


def write_silver(root, arrays):
    """arrays: {(detector, season): sum dict}. Writes the sums, the table and the differences."""
    rows = []
    for (detector, season), s in arrays.items():
        save_npz(sum_path("barnard", detector, season, root), s)
        rows.append({
            "sum_id": f"barnard_d{detector}_s{season}", "target_id": "barnard",
            "detector": detector, "season": season,
            "mean_date_utc": pd.Timestamp("2025-08-15", tz="UTC"), "n_frames": 3,
            "background_noise": 0.1, "flags": "", "saturated": False,
            "png": contract.sum_png("barnard", detector, season),
        })
    sums_table_path(root).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_contract_table(pd.DataFrame(rows), contract.SUMS), sums_table_path(root))
    build_differences("barnard", root=root)


def read(gold, relative):
    return (mpimg.imread(gold / relative) * 255).round().astype(np.uint8)


# Detector 6 has no season 1, so season 1 has no color image.
CELLS = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3), (6, 2), (6, 3)]


@pytest.fixture
def built(tmp_path):
    arrays = {cell: a_sum(i) for i, cell in enumerate(CELLS)}
    arrays[(1, 2)] = a_sum(50, back=5.0)  # a different background level in season 2
    arrays[(6, 3)] = a_sum(51, star=(30, 130))  # a star near the north edge, only in D6
    arrays[(1, 3)] = a_sum(52, saturated=box(73, 73), empty=box(10, 120))
    write_silver(tmp_path, arrays)
    gold = tmp_path / "gold"
    count = build_pngs("barnard", root=tmp_path, gold_dir=gold)
    return tmp_path, gold, count


def test_every_sum_difference_and_color_gets_its_png(built):
    _, gold, count = built
    names = sorted(p.name for p in (gold / "png" / "barnard").glob("*.png"))
    sums = [f"sum_d{d}_s{s}.png" for d, s in CELLS]
    differences = ["diff_d1_s1-s2.png", "diff_d1_s1-s3.png", "diff_d1_s2-s3.png",
                   "diff_d3_s1-s2.png", "diff_d3_s1-s3.png", "diff_d3_s2-s3.png",
                   "diff_d6_s2-s3.png"]
    colors = ["color_s2.png", "color_s3.png"]  # not s1: detector 6 has no season 1
    assert names == sorted(sums + differences + colors)
    assert count == len(names)
    assert read(gold, contract.sum_png("barnard", 1, 1)).shape == (N, N, 4)


def window(image, row, col, size=1):
    return image[row - size : row + size + 1, col - size : col + size + 1, :3].astype(float)


def test_pngs_are_north_up(built):
    _, gold, _ = built
    image = read(gold, contract.sum_png("barnard", 6, 3))
    top = window(image, N - 1 - 130, 30).sum(axis=2).mean()  # silver row 130 = north
    unflipped = window(image, 130, 30).sum(axis=2).mean()
    assert top > 2 * unflipped


def test_a_saturated_center_is_the_brightest_color_and_no_data_is_gray(built):
    _, gold, _ = built
    image = read(gold, contract.sum_png("barnard", 1, 3))
    assert tuple(image[N - 1 - 73, 73]) == tuple(MAGMA_TOP)
    assert tuple(image[N - 1 - 10, 120]) == tuple(NEUTRAL)


def test_the_seasons_of_one_detector_share_the_same_background(built):
    _, gold, _ = built
    medians = [np.median(read(gold, contract.sum_png("barnard", 1, s))[..., :3], axis=(0, 1))
               for s in (1, 2)]
    np.testing.assert_allclose(medians[0], medians[1], atol=3)  # season 2 had +5 MJy/sr


def test_masked_or_saturated_pixels_of_a_difference_are_the_middle_color(built):
    _, gold, _ = built
    image = read(gold, contract.difference_png("barnard", 1, 2, 3))
    assert tuple(image[N - 1 - 73, 73]) == tuple(DIVERGING_MIDDLE)  # saturated in season 3
    assert tuple(image[N - 1 - 10, 120]) == tuple(DIVERGING_MIDDLE)  # no data in season 3


def test_a_star_that_appears_in_season_b_is_red_in_the_difference(built):
    _, gold, _ = built
    image = read(gold, contract.difference_png("barnard", 6, 2, 3))
    # The red end of RdBu_r is a dark red: the star sits at the top of the scale.
    assert tuple(image[N - 1 - 130, 30]) == tuple(colormaps["RdBu_r"](1.0, bytes=True))


def test_a_star_only_in_detector_6_is_red_in_the_color_image(built):
    _, gold, _ = built
    image = read(gold, contract.color_png("barnard", 3))
    r, g, b = window(image, N - 1 - 130, 30).mean(axis=(0, 1))
    assert r > 250 and r > g + 100 and r > b + 100


def test_color_of_missing_data_is_gray_and_saturated_is_white():
    blue = np.array([[0.2, 1.0, np.nan]])
    out = color_rgba(blue, np.array([[0.2, 1.0, 0.3]]), np.array([[0.2, 1.0, 0.3]]))
    assert tuple(out[0, 1]) == (255, 255, 255, 255)
    assert tuple(out[0, 2]) == tuple(NEUTRAL)


def test_old_pngs_of_the_target_go(built):
    root, gold, _ = built
    stale = gold / "png" / "barnard" / "sum_d5_s9.png"
    stale.write_bytes(b"old")
    build_pngs("barnard", root=root, gold_dir=gold)
    assert not stale.exists()


def test_without_silver_it_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="silver"):
        build_pngs(root=tmp_path, gold_dir=tmp_path / "gold")


def test_the_png_command_is_in_the_cli():
    args = build_parser().parse_args(["png", "--target", "barnard"])
    assert (args.command, args.target, args.out) == ("png", "barnard", "data/gold")
