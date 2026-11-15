"""Bronze per target (task D3): the season comes from the complete list, at most 12 frames per
season and detector spread in time, a failed download is retried, a second run downloads nothing.

No network: IRSA is replaced by a function that returns the list of frames.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from astropy.io import fits

from serafim.bronze import download as bronze
from serafim.bronze.download import (
    MANIFEST_SCHEMA,
    ArchiveError,
    collections_for,
    cutout_size_deg,
    download,
    fits_has_products,
    parallel_workers,
    read_manifest,
    retry_waits,
    select_frames,
    spread_in_time,
)
from serafim.cli import build_parser, main
from serafim.config import ROOT, load_config
from serafim.validation.targets import FRAME_COLUMNS

CAP_3 = {"max_frames_per_season_detector: 12": "max_frames_per_season_detector: 3"}
CAP_2 = {"max_frames_per_season_detector: 12": "max_frames_per_season_detector: 2"}


def _config_root(tmp_path: Path, **replace: str) -> Path:
    """A folder with a copy of config/targets.yaml, with some text replaced."""
    config = tmp_path / "config"
    config.mkdir()
    text = (ROOT / "config" / "targets.yaml").read_text(encoding="utf-8")
    for old, new in replace.items():
        assert old in text, old
        text = text.replace(old, new)
    (config / "targets.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def _mef(*dropped: str) -> bytes:
    extensions = ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE", "EPSF")
    names = [name for name in extensions if name not in dropped]
    hdus = [fits.PrimaryHDU(), *[fits.ImageHDU(data=[[0]], name=name) for name in names]]
    buffer = BytesIO()
    fits.HDUList(hdus).writeto(buffer)
    return buffer.getvalue()


def _listing(*cells: tuple[int, list[float]], collection: str = "spherex_dr1") -> pd.DataFrame:
    """What spherex_frames() returns: one row per exposure, for (detector, [t_mid_mjd, ...])."""
    rows = [
        {
            "obs_id": f"obs-d{detector}-{i}-{int(t)}",
            "collection": collection,
            "t_mid_mjd": float(t),
            "detector": detector,
            "bandpass": f"SPHEREx-D{detector}",
            "em_min_um": 1.0,
            "em_max_um": 2.0,
            "access_url": f"https://example.test/cutout/{detector}/{i}",
        }
        for detector, times in cells
        for i, t in enumerate(times)
    ]
    return pd.DataFrame(rows, columns=FRAME_COLUMNS).sort_values("t_mid_mjd", ignore_index=True)


def _fetch_ok(job: dict) -> None:
    job["destination"].parent.mkdir(parents=True, exist_ok=True)
    job["destination"].write_bytes(_mef())


def _nothing_to_fetch(job: dict) -> None:
    raise AssertionError(f"downloaded {job['obs_id']}")


# Two seasons of Barnard: 100 days without frames in between (more than the 30 of the settings).
SEASON_1 = [60000 + n for n in range(10)]
SEASON_2 = [60200 + n for n in range(2)]
BARNARD = _listing((1, SEASON_1 + SEASON_2), (2, SEASON_1))


def _only(listing: pd.DataFrame):
    return lambda center, collections: listing


# --- The files and settings ---------------------------------------------------------------
def test_a_real_mef_has_the_required_extensions():
    assert fits_has_products(_mef())
    assert not fits_has_products(_mef("WCS-WAVE"))
    assert not fits_has_products(b"<html>502 Bad Gateway</html>")


def test_retries_wait_5_10_and_15_seconds():
    assert retry_waits(3) == (5, 10, 15)
    assert retry_waits(0) == ()


def test_never_more_than_eight_downloads_at_once():
    assert parallel_workers({"parallel_downloads": 50}) == 8
    assert parallel_workers({"parallel_downloads": 3}) == 3
    assert parallel_workers({}) == 8


def test_a_target_can_set_its_own_cutout_size():
    settings = {"cutout_size_deg": 0.25}
    assert cutout_size_deg({"id": "a"}, settings) == 0.25
    assert cutout_size_deg({"id": "b", "cutout_size_deg": 0.5}, settings) == 0.5


def test_the_settings_have_the_cap_and_the_collections():
    settings = load_config()["settings"]
    assert settings["max_frames_per_season_detector"] == 12
    assert collections_for(settings)[0] == "spherex_dr1"


# --- Choosing frames ----------------------------------------------------------------------
def test_all_frames_are_kept_when_there_are_few():
    assert list(spread_in_time([5.0, 1.0, 3.0], 12)) == [0, 1, 2]


def test_the_cap_spreads_frames_over_the_whole_season():
    times = np.arange(100.0)
    chosen = spread_in_time(times, 12)
    assert len(chosen) == len(set(chosen)) == 12
    assert chosen[0] == 0 and chosen[-1] == 99  # the oldest and the newest are in
    assert np.diff(chosen).max() <= 10  # no big gap: 99 / 11 = 9 days apart


def test_a_visit_with_many_frames_does_not_use_up_the_cap():
    # Three visits of 20 frames each, a month apart: one frame from each visit at least.
    times = np.concatenate([np.arange(20) / 24 + start for start in (0, 30, 60)])
    chosen = spread_in_time(times, 6)
    visits = {int(times[k] // 30) for k in chosen}
    assert visits == {0, 1, 2}


def test_a_single_frame_is_the_middle_one():
    assert list(spread_in_time(np.arange(11.0), 1)) == [5]


def test_the_cap_is_per_season_and_detector():
    listing = _listing((1, SEASON_1 + SEASON_2), (2, SEASON_1))
    listing["season"] = np.where(listing.t_mid_mjd < 60100, 1, 2)
    chosen = select_frames(listing, 3)
    counts = chosen.groupby(["season", "detector"]).size().to_dict()
    assert counts == {(1, 1): 3, (1, 2): 3, (2, 1): 2}  # season 2 has only 2 frames: all kept
    assert chosen.t_mid_mjd.is_monotonic_increasing
    assert set(chosen.obs_id) <= set(listing.obs_id)
    assert select_frames(listing, 3).equals(chosen)  # the same list gives the same frames


# --- Download, per target -----------------------------------------------------------------
def test_download_per_target_and_a_second_run_downloads_nothing(tmp_path):
    root = _config_root(tmp_path, **CAP_3)
    barnard = next(t for t in load_config()["targets"] if t["id"] == "barnard")
    asked = []

    def frames(center, collections):
        asked.append((center.ra.deg, center.dec.deg, list(collections)))
        return BARNARD

    fetched = []

    def fetch(job):
        fetched.append(job["obs_id"])
        _fetch_ok(job)

    assert download("barnard", root=root, frames=frames, fetch=fetch) == 8
    # IRSA is asked once, at the position today (corrected for proper motion), in every collection.
    assert len(asked) == 1
    assert asked[0][:2] == pytest.approx((barnard["ra_now"], barnard["dec_now"]))
    assert asked[0][2] == collections_for(load_config()["settings"])

    manifest = pd.DataFrame(read_manifest(root))
    assert list(manifest.columns) == MANIFEST_SCHEMA.names
    assert len(manifest) == len(fetched) == 8
    assert set(manifest.target_id) == {"barnard"} and set(manifest.collection) == {"spherex_dr1"}
    assert set(manifest.season) == {1, 2} and set(manifest.detector) == {1, 2}
    assert manifest.groupby(["season", "detector"]).size().max() == 3
    assert (manifest.size_deg == 0.25).all()
    assert (manifest.ra == barnard["ra_now"]).all() and (manifest.dec == barnard["dec_now"]).all()
    for row in manifest.itertuples():
        assert row.path == f"data/bronze/barnard/{row.obs_id}.fits"
        assert (root / row.path).is_file()

    assert download("barnard", root=root, frames=frames, fetch=_nothing_to_fetch) == 0
    assert len(read_manifest(root)) == 8


def test_the_season_comes_from_the_complete_list_before_the_cap(tmp_path):
    # Frames every 10 days for 60 days: one season. With a cap of 2 only the first and the last
    # are kept, 60 days apart; numbering seasons after the cap would split them in two.
    root = _config_root(tmp_path, **CAP_2)
    listing = _listing((2, [60000 + 10 * n for n in range(7)]))
    assert download("barnard", root=root, frames=_only(listing), fetch=_fetch_ok) == 2
    assert [row["season"] for row in read_manifest(root)] == [1, 1]


def test_seasons_in_the_manifest_follow_the_complete_list_on_every_run(tmp_path):
    root = _config_root(tmp_path)
    assert download("barnard", root=root, frames=_only(BARNARD), fetch=_fetch_ok) == 22
    assert {row["season"] for row in read_manifest(root)} == {1, 2}

    # A new rule puts everything in one season (the 201 days are also under the new max_days):
    # the manifest follows, nothing is downloaded.
    config = root / "config" / "targets.yaml"
    rules = config.read_text().replace("gap_days: 30", "gap_days: 200")
    config.write_text(rules.replace("max_days: 183", "max_days: 400"), encoding="utf-8")
    assert download("barnard", root=root, frames=_only(BARNARD), fetch=_nothing_to_fetch) == 0
    assert {row["season"] for row in read_manifest(root)} == {1}


def test_a_failed_download_is_skipped_and_the_next_run_tries_it_again(tmp_path):
    root = _config_root(tmp_path)
    bad = BARNARD.obs_id.iloc[3]

    def flaky(job):
        if job["obs_id"] == bad:
            raise ArchiveError("502 Bad Gateway")
        _fetch_ok(job)

    assert download("barnard", root=root, frames=_only(BARNARD), fetch=flaky) == 21
    assert bad not in {row["obs_id"] for row in read_manifest(root)}
    assert not (root / "data" / "bronze" / "barnard" / f"{bad}.fits").exists()

    again = []
    assert download("barnard", root=root, frames=_only(BARNARD),
                    fetch=lambda job: (again.append(job["obs_id"]), _fetch_ok(job))) == 1
    assert again == [bad]
    assert len(read_manifest(root)) == 22


def test_a_file_that_disappeared_is_downloaded_again(tmp_path):
    root = _config_root(tmp_path)
    download("barnard", root=root, frames=_only(BARNARD), fetch=_fetch_ok)
    gone = read_manifest(root)[0]
    (root / gone["path"]).unlink()
    assert download("barnard", root=root, frames=_only(BARNARD), fetch=_fetch_ok) == 1
    assert (root / gone["path"]).is_file()


def test_a_manifest_from_before_this_task_is_rebuilt_without_downloading(tmp_path):
    root = _config_root(tmp_path)
    first = BARNARD.iloc[0]
    path = f"data/bronze/barnard/{first.obs_id}.fits"
    (root / path).parent.mkdir(parents=True)
    (root / path).write_bytes(_mef())
    legacy = pa.table({
        "target_id": ["barnard"], "obs_id": [first.obs_id], "collection": ["spherex_dr1"],
        "band": ["SPHEREx-D1"], "t_min": [60000.0], "ra": [1.0], "dec": [2.0],
        "size_deg": [0.25], "path": [path],
    })
    (root / "data" / "bronze").mkdir(exist_ok=True)
    pq.write_table(legacy, bronze.manifest_path(root))

    fetched = []
    download("barnard", root=root, frames=_only(BARNARD),
             fetch=lambda job: (fetched.append(job["obs_id"]), _fetch_ok(job)))
    assert first.obs_id not in fetched
    row = next(r for r in read_manifest(root) if r["obs_id"] == first.obs_id)
    assert (row["season"], row["detector"]) == (1, int(first.detector))


def test_other_targets_stay_in_the_manifest(tmp_path):
    root = _config_root(tmp_path)
    download("barnard", root=root, frames=_only(BARNARD), fetch=_fetch_ok)
    wise = _listing((5, [61000, 61001]))
    download("wise_0855", root=root, frames=_only(wise), fetch=_fetch_ok)
    assert {row["target_id"] for row in read_manifest(root)} == {"barnard", "wise_0855"}


def test_a_failed_search_does_not_stop_the_other_targets(tmp_path):
    root = _config_root(tmp_path)
    calls = []

    def frames(center, collections):
        calls.append(center)
        if len(calls) == 2:
            raise RuntimeError("IRSA is down")
        return BARNARD

    with pytest.raises(ArchiveError, match="eps_indi_b"):
        download(root=root, frames=frames, fetch=_fetch_ok)
    assert len(calls) == len(load_config()["targets"])  # the search went on to every target
    assert {row["target_id"] for row in read_manifest(root)} == {
        t["id"] for t in load_config()["targets"] if t["id"] != "eps_indi_b"
    }


def test_no_frames_anywhere_is_an_error(tmp_path):
    root = _config_root(tmp_path)
    empty = pd.DataFrame({name: [] for name in FRAME_COLUMNS})
    with pytest.raises(ArchiveError, match="bronze is empty"):
        download("barnard", root=root, frames=_only(empty), fetch=_nothing_to_fetch)


def test_unknown_target_is_rejected(tmp_path):
    root = _config_root(tmp_path)
    with pytest.raises(ArchiveError, match="Unknown target"):
        download("not-a-target", root=root, frames=_only(BARNARD), fetch=_fetch_ok)
    assert main(["bronze", "--target", "not-a-target"]) == 1


# --- One request ----------------------------------------------------------------------------
class _Response:
    def __init__(self, content: bytes = b"", status: int = 200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Server Error")


def test_a_request_waits_5_10_and_15_seconds_between_tries(monkeypatch):
    waits = []
    monkeypatch.setattr(bronze.time, "sleep", waits.append)
    answers = iter([_Response(status=502), _Response(status=503), _Response(b"ok")])
    monkeypatch.setattr(bronze.requests, "get", lambda *args, **kwargs: next(answers))
    assert bronze._get_bytes("https://example.test", retry_waits(3)) == b"ok"
    assert waits == [5, 10]


def test_a_request_gives_up_after_the_last_retry(monkeypatch):
    waits, tries = [], []
    monkeypatch.setattr(bronze.time, "sleep", waits.append)
    monkeypatch.setattr(
        bronze.requests, "get", lambda *args, **kwargs: tries.append(1) or _Response(status=502)
    )
    with pytest.raises(ArchiveError, match="502"):
        bronze._get_bytes("https://example.test", retry_waits(3))
    assert waits == [5, 10, 15] and len(tries) == 4


def test_fetch_cutout_keeps_the_file_exactly_as_it_arrives(tmp_path, monkeypatch):
    payload = _mef()
    seen = []
    monkeypatch.setattr(bronze, "_get_bytes", lambda url, waits: seen.append(url) or payload)
    job = {"obs_id": "x", "access_url": "https://example.test/c?a=1", "ra": 1.5, "dec": -2.5,
           "size_deg": 0.25, "waits": (5,), "destination": tmp_path / "barnard" / "x.fits"}
    bronze.fetch_cutout(job)
    assert job["destination"].read_bytes() == payload  # EPSF and everything else
    assert not list(tmp_path.rglob("*.part"))
    assert seen == ["https://example.test/c?a=1&center=1.500000,-2.500000&size=0.2500"]


def test_fetch_cutout_rejects_a_file_without_the_spherex_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(bronze, "_get_bytes", lambda url, waits: _mef("ZODI"))
    job = {"obs_id": "x", "access_url": "https://example.test/c", "ra": 1.0, "dec": 2.0,
           "size_deg": 0.25, "waits": (), "destination": tmp_path / "x.fits"}
    with pytest.raises(ArchiveError, match="missing IMAGE"):
        bronze.fetch_cutout(job)
    assert not job["destination"].exists()


def test_full_pipeline_command_is_still_to_do():
    assert main(["run", "--target", "barnard"]) == 1


def test_the_bronze_command_is_in_the_cli():
    args = build_parser().parse_args(["bronze", "--target", "barnard"])
    assert (args.command, args.target) == ("bronze", "barnard")
