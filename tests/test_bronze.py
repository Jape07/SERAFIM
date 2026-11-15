"""Bronze download (task D2): DR1 wins, a second run fetches nothing new."""

import shutil
from io import BytesIO
from pathlib import Path

import pytest
from astropy.io import fits

from serafim.bronze.download import (
    ArchiveError,
    choose_exposures,
    download,
    fits_has_products,
    read_manifest,
)
from serafim.cli import main
from serafim.config import ROOT


def _config_root(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(ROOT / "config" / "regions.yaml", config / "regions.yaml")
    return tmp_path


def _mef(*extra: str) -> bytes:
    hdus = [fits.PrimaryHDU()]
    for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE", *extra):
        hdus.append(fits.ImageHDU(data=[[0]], name=name))
    buffer = BytesIO()
    fits.HDUList(hdus).writeto(buffer)
    return buffer.getvalue()


def _row(collection: str, obs_id: str = "visit-1", t_min: float = 60000.0) -> dict:
    return {
        "obs_id": obs_id,
        "collection": collection,
        "access_url": "https://example.test/cutout",
        "band": "SPHEREx-D2",
        "t_min": t_min,
    }


def test_a_real_mef_has_the_required_extensions():
    assert fits_has_products(_mef())
    assert not fits_has_products(_mef_without("WCS-WAVE"))


def _mef_without(dropped: str) -> bytes:
    names = [name for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE") if name != dropped]
    hdus = [fits.PrimaryHDU(), *[fits.ImageHDU(data=[[0]], name=name) for name in names]]
    buffer = BytesIO()
    fits.HDUList(hdus).writeto(buffer)
    return buffer.getvalue()


def test_dr1_wins_when_the_same_exposure_is_in_qr3():
    rows = [
        _row("spherex_qr3", t_min=60001),
        _row("spherex_dr1", t_min=60000),
        _row("spherex_dr1_deep"),
    ]
    chosen = choose_exposures(rows, limit=10)
    assert len(chosen) == 1
    assert chosen[0]["collection"] == "spherex_dr1"


def test_download_keeps_dr1_and_skips_it_the_second_time(tmp_path):
    root = _config_root(tmp_path)
    calls = {"fetch": 0, "collections": []}

    def query(collection, ra, dec, limit, attempts):
        calls["collections"].append(collection)
        return [_row(collection)]

    def fetch(job):
        calls["fetch"] += 1
        job["destination"].parent.mkdir(parents=True, exist_ok=True)
        job["destination"].write_bytes(_mef())

    assert download("pole", root=root, query=query, fetch=fetch) == 1
    manifest = read_manifest(root)
    assert manifest[0]["collection"] == "spherex_dr1"
    assert manifest[0]["path"] == "data/bronze/pole/visit-1.fits"
    assert (root / manifest[0]["path"]).is_file()
    assert "spherex_dr1_deep" in calls["collections"]
    assert "spherex_qr3" in calls["collections"]

    def fetch_again(job):
        raise AssertionError(f"second run downloaded {job['obs_id']}")

    assert download("pole", root=root, query=query, fetch=fetch_again) == 0


def test_unknown_region_is_rejected(tmp_path):
    root = _config_root(tmp_path)
    with pytest.raises(ArchiveError, match="Unknown region"):
        download("not-a-region", root=root, query=lambda *args: [], fetch=lambda job: None)


def test_full_pipeline_command_is_still_task_d8():
    assert main(["run", "--region", "pole"]) == 1
