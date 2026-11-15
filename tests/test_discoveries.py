"""web/texts/discoveries.json: the candidates the site shows (no network, no silver)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from serafim.validation import dipoles, discoveries

ROOT = Path(__file__).resolve().parents[1]


def row(status, p_dipole=np.nan, halo=False, detector=2, a=1, b=2, reason="x"):
    return {"detector": detector, "season_a": a, "season_b": b, "status": status,
            "reason": reason, "ra_from": 10.0, "dec_from": 5.0, "ra_to": 10.001, "dec_to": 5.0,
            "snr": 12.34, "halo": halo, "p_dipole": p_dipole}  # fmt: skip


def test_the_ai_opinion_follows_the_official_threshold():
    assert discoveries.ai_says(0.9, 0.731) == "moving"
    assert discoveries.ai_says(0.3, 0.731) == "not_moving"
    assert discoveries.ai_says(np.nan, 0.731) == "none"
    assert discoveries.ai_says(None, 0.731) == "none"
    assert discoveries.ai_says(0.9, None) == "none"  # no evaluated model: no opinion


def test_discarded_are_only_counted_and_saturated_have_no_opinion():
    found = {
        "star": pd.DataFrame([
            row(dipoles.KNOWN, 0.95, reason="Gaia DR3 1 (gaia)"),
            row(dipoles.KNOWN, np.nan, halo=True, detector=5, reason="NAME star (simbad)"),
            row(dipoles.CANDIDATE, 0.08, detector=2, a=1, b=3),
            row(dipoles.DISCARDED, 0.99, reason="at the edge"),
            row(dipoles.DISCARDED, np.nan, reason="on masked pixels"),
        ]),
        "quiet": pd.DataFrame(),
    }  # fmt: skip
    content = discoveries.build(found, {"commit": "abc", "run_date_utc": "2026-11-15"}, 0.731)
    assert content["counts"]["star"] == {"known": 2, "unidentified": 1, "discarded": 2}
    assert content["counts"]["quiet"] == {"known": 0, "unidentified": 0, "discarded": 0}
    kinds = [c["status"] for c in content["candidates"]]
    assert kinds == [dipoles.CANDIDATE, dipoles.KNOWN, dipoles.KNOWN]  # to check first
    by_id = {c["id"]: c for c in content["candidates"]}
    assert by_id["star-d2-s1-s2-1"]["ai_says"] == "moving"
    assert by_id["star-d2-s1-s3-1"]["ai_says"] == "not_moving"
    saturated = by_id["star-d5-s1-s2-1"]
    assert saturated["saturated"] and saturated["p_dipole"] is None
    assert saturated["ai_says"] == "none"
    assert content["model"]["status"] == "experimental" and content["gold_commit"] == "abc"
    assert "never changes a status" in content["texts"]["footnote"]


def test_the_committed_file_matches_the_dipole_report():
    """docs/dipoles.json (the report) and web/texts/discoveries.json (for the site) come from
    the same rule: same candidates, same opinions."""
    site_path = ROOT / "web" / "texts" / "discoveries.json"
    report_path = ROOT / "docs" / "dipoles.json"
    if not (site_path.exists() and report_path.exists()):
        pytest.skip("no generated files")
    site = json.loads(site_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))["dipoles"]
    assert len(site["candidates"]) == len(report)
    threshold = site["model"]["threshold"]
    for c in site["candidates"]:
        assert c["status"] in (dipoles.KNOWN, dipoles.CANDIDATE)
        assert c["ai_says"] == discoveries.ai_says(c["p_dipole"], threshold)
        if c["saturated"]:
            assert c["p_dipole"] is None and c["ai_says"] == "none"
    to_check = [c for c in site["candidates"] if c["status"] == dipoles.CANDIDATE]
    assert all(c["ai_says"] == "not_moving" for c in to_check)  # the two single-pixel pairs
    assert set(site["counts"]) == {"barnard", "eps_indi_b", "r_dra", "rw_aur", "wise_0855"}
    assert (
        site["counts"]["rw_aur"]["unidentified"] + site["counts"]["wise_0855"]["unidentified"] == 2
    )
    assert site["counts"]["r_dra"]["known"] == 0 and site["counts"]["barnard"]["known"] >= 6
