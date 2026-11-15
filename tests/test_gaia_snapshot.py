"""The fixed copy of Gaia DR3 in config/gaia_snapshot/: what it covers, how it is cut and
written, and that alignment uses it before asking ESA. No network."""

import shutil

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from serafim.bronze import gaia_snapshot
from serafim.bronze.gaia_snapshot import (
    SNAPSHOT_MAX_MAG,
    SNAPSHOT_RADIUS_DEG,
    from_snapshot,
    read_metadata,
    save_snapshots,
    snapshot_path,
    vizier_rows,
    write,
)
from serafim.config import ROOT, load_config
from serafim.validation import catalogs, layers

CENTER = SkyCoord(100.0 * u.deg, 20.0 * u.deg)


def stars(center=CENTER):
    """Four stars: at the center (G 12), 0.1 deg north (G 18.5), 0.19 deg east (G 15) and
    0.1 deg south (G 9, variable)."""
    north = center.directional_offset_by(0 * u.deg, 0.1 * u.deg)
    east = center.directional_offset_by(90 * u.deg, 0.19 * u.deg)
    south = center.directional_offset_by(180 * u.deg, 0.1 * u.deg)
    points = [center, north, east, south]
    return catalogs._frame(
        catalog="Gaia", name=[f"Gaia DR3 {i}" for i in range(4)],
        ra=np.array([p.ra.deg for p in points]), dec=np.array([p.dec.deg for p in points]),
        type=np.array(["source", "source", "source", "VARIABLE"]),
        is_variable=np.array([False, False, False, True]),
        mag=np.array([12.0, 18.5, 15.0, 9.0]), url=None,
        pmra_mas_yr=np.ones(4), pmdec_mas_yr=np.ones(4), parallax_mas=np.ones(4),
        epoch_jyear=np.full(4, 2016.0),
    )  # fmt: skip


def save(root, target_id="field", center=CENTER, table=None):
    meta = {"target_id": target_id, "ra": center.ra.deg, "dec": center.dec.deg,
            "radius_deg": SNAPSHOT_RADIUS_DEG, "max_mag": SNAPSHOT_MAX_MAG, "source": "test"}
    write(stars(center) if table is None else table, meta, snapshot_path(target_id, root))


def test_a_cone_inside_the_copy_is_cut_by_distance_and_magnitude(tmp_path):
    save(tmp_path)
    found = from_snapshot(CENTER, 0.15 * u.deg, 18.0, root=tmp_path)
    assert sorted(found["name"]) == ["Gaia DR3 0", "Gaia DR3 3"]  # north is too faint
    assert list(found.columns) == catalogs.COLUMNS
    found = from_snapshot(CENTER, 0.15 * u.deg, 19.0, root=tmp_path)
    assert sorted(found["name"]) == ["Gaia DR3 0", "Gaia DR3 1", "Gaia DR3 3"]


def test_a_cone_off_center_but_inside_the_copy_is_served(tmp_path):
    save(tmp_path)
    near = CENTER.directional_offset_by(90 * u.deg, 0.05 * u.deg)
    found = from_snapshot(near, 0.15 * u.deg, 19.0, root=tmp_path)
    assert "Gaia DR3 2" in set(found["name"])  # the east star, 0.14 deg from `near`


def test_a_cone_the_copy_does_not_hold_returns_none(tmp_path):
    save(tmp_path)
    assert from_snapshot(CENTER, 0.25 * u.deg, 18.0, root=tmp_path) is None  # too wide
    assert from_snapshot(CENTER, 0.1 * u.deg, 20.0, root=tmp_path) is None  # too faint
    elsewhere = SkyCoord(200.0 * u.deg, -30.0 * u.deg)
    assert from_snapshot(elsewhere, 0.1 * u.deg, 18.0, root=tmp_path) is None
    assert from_snapshot(CENTER, 0.1 * u.deg, 18.0, root=tmp_path / "nothing") is None


def test_an_empty_answer_inside_the_copy_is_an_empty_table_not_none(tmp_path):
    save(tmp_path)
    found = from_snapshot(CENTER.directional_offset_by(45 * u.deg, 0.05 * u.deg),
                          0.01 * u.deg, 18.0, root=tmp_path)
    assert found is not None and found.empty


def test_the_file_keeps_its_metadata_and_is_sorted_by_magnitude(tmp_path):
    save(tmp_path)
    path = snapshot_path("field", tmp_path)
    assert read_metadata(path)["radius_deg"] == SNAPSHOT_RADIUS_DEG
    assert list(pd.read_parquet(path)["mag"]) == [9.0, 12.0, 15.0, 18.5]


def test_a_parquet_without_metadata_is_refused(tmp_path):
    path = snapshot_path("field", tmp_path)
    path.parent.mkdir(parents=True)
    stars().to_parquet(path)
    with pytest.raises(ValueError, match="not a Gaia snapshot"):
        read_metadata(path)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "targets.yaml", tmp_path / "config" / "targets.yaml")
    return tmp_path


def test_save_writes_one_copy_per_target_and_skips_the_one_that_fails(project):
    asked = []

    def fetch(center, radius, max_mag):
        asked.append((round(radius.deg, 3), max_mag))
        if len(asked) == 2:
            raise catalogs.CatalogUnavailable("no mirror answered")
        return stars(center), "vizier.test"

    written = save_snapshots(root=project, fetch=fetch)
    ids = [t["id"] for t in load_config(project / "config" / "targets.yaml")["targets"]]
    assert [p.stem for p in written] == [i for n, i in enumerate(ids) if n != 1]
    assert set(asked) == {(SNAPSHOT_RADIUS_DEG, SNAPSHOT_MAX_MAG)}
    meta = read_metadata(written[0])
    assert meta["target_id"] == ids[0] and meta["source"] == "vizier.test"


def test_an_empty_answer_is_not_saved(project):
    written = save_snapshots("barnard", root=project,
                             fetch=lambda c, r, m: (catalogs._empty(), "vizier.test"))
    assert written == [] and not snapshot_path("barnard", project).exists()


def test_vizier_columns_become_the_columns_of_esa():
    table = pd.DataFrame({
        "Source": [11, 22], "RA_ICRS": [10.0, 10.1], "DE_ICRS": [-5.0, -5.1],
        "pmRA": [3.0, np.nan], "pmDE": [-4.0, np.nan], "Plx": [1.5, np.nan],
        "Gmag": [12.5, 17.0], "VarFlag": ["VARIABLE", "NOT_AVAILABLE"],
    })  # fmt: skip
    rows = vizier_rows(table)
    assert list(rows.columns) == catalogs.COLUMNS
    assert list(rows["name"]) == ["Gaia DR3 11", "Gaia DR3 22"]
    assert list(rows["is_variable"]) == [True, False]
    assert list(rows["type"]) == ["VARIABLE", "source"]
    assert rows["epoch_jyear"].iloc[0] == 2016.0 and np.isnan(rows["epoch_jyear"].iloc[1])
    assert set(rows["catalog"]) == {"Gaia"}


def test_alignment_takes_gaia_from_the_copy_without_asking_esa(project, monkeypatch):
    from serafim.silver.alignment import gaia_sources

    target = next(t for t in load_config(project / "config" / "targets.yaml")["targets"]
                  if t["id"] == "barnard")
    center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
    save(project, "barnard", center)

    def no_network(*args, **kwargs):
        raise AssertionError("ESA or VizieR was asked")

    monkeypatch.setattr(catalogs, "gaia", no_network)
    monkeypatch.setattr(gaia_snapshot, "from_vizier", no_network)
    found = gaia_sources(target, 0.25, root=project)
    # The cone of a 0.25 deg cutout is 0.187 deg: the east star (0.19 deg) is outside, and the
    # north one is fainter than the G 18 of the catalog layers.
    assert sorted(found["name"]) == ["Gaia DR3 0", "Gaia DR3 3"]


# --- The committed copies ---------------------------------------------------------------------
@pytest.mark.parametrize("target", load_config()["targets"], ids=lambda t: t["id"])
def test_the_committed_copy_covers_every_gaia_query_of_the_pipeline(target):
    center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
    layer_cone = from_snapshot(center, catalogs._region_radius(0.25), layers.GAIA_MAX_MAG)
    fit_cone = from_snapshot(center, (20 + 90) * u.arcsec, 19.0)  # visibility.neighbours()
    assert layer_cone is not None and len(layer_cone) > 0
    assert fit_cone is not None
    assert read_metadata(snapshot_path(target["id"]))["target_id"] == target["id"]
