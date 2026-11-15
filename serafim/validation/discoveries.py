"""The moving-object candidates for the site: web/texts/discoveries.json (task A9 for web).

    python -m serafim.validation.discoveries [--gold data/gold]

The site cannot run the dipole rule or the CNN (they need the silver sums), so this file carries
their results, ready to show:

- the texts a visitor reads (plain words, no jargon);
- per target, how many dipoles are Known, to check (Unidentified candidate) or Discarded;
- every Known and Unidentified dipole: where it was and where it went, its S/N, whether it
  touches a saturated star, and the CNN's second opinion already decided with the official
  threshold (`ai_says`: "moving", "not_moving" or "none"). Discarded dipoles are only counted.

The CNN never changes a status and never announces a new object (docs/model_card.md); the site
shows its opinion next to each candidate as "experimental". Regenerate this file after every
pipeline run, like web/texts/targets.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from serafim.validation import dipoles, targets

log = logging.getLogger(__name__)

OUT_PATH = targets.ROOT / "web" / "texts" / "discoveries.json"
MODEL_CARD_URL = "https://github.com/Jape07/SERAFIM/blob/main/docs/model_card.md"
MOVING, NOT_MOVING, NONE = "moving", "not_moving", "none"

TEXTS = {
    "title": "Moving-object candidates",
    "lead": (
        "When something moves between two seasons, their difference shows a blue spot where it "
        "was and a red spot where it went: a dipole. A classic rule finds these pairs and checks "
        "them against Gaia, CatWISE and SIMBAD."
    ),
    "known": "Known: a catalog object that moves.",
    "unidentified": "To check: nothing in the catalogs moves here. A person has to look.",
    "discarded": "Discarded: an edge, bad pixels or the residual of a star that does not move.",
    "ai_moving": "AI second opinion (experimental): looks like something that moved",
    "ai_not_moving": "AI second opinion (experimental): does not look like something that moved",
    "ai_none": "AI second opinion: none (saturated star)",
    "footnote": (
        "The AI never changes a status and never announces a new object. How it was measured: "
        "docs/dipole_cnn.md."
    ),
    "empty_rw_aur": (
        "RW Aurigae only changes brightness: no motion was expected, and none was found."
    ),
    "empty_r_dra": (
        "R Draconis only changes brightness: no motion was expected, and none was found."
    ),
    "empty_default": "No candidates in this field.",
    "unavailable": "Candidates are not available right now.",
}


def ai_says(p_dipole, threshold: float | None) -> str:
    """The CNN's opinion, decided once here so the site never applies a threshold itself."""
    if threshold is None or p_dipole is None or pd.isna(p_dipole):
        return NONE
    return MOVING if float(p_dipole) >= threshold else NOT_MOVING


def build(found: dict[str, pd.DataFrame], meta: dict, threshold: float | None) -> dict:
    """The file's content from the rows of dipoles.target_dipoles(), one DataFrame per target."""
    counts, candidates = {}, []
    for target_id, rows in found.items():
        status = rows["status"] if len(rows) else pd.Series(dtype=str)
        counts[target_id] = {
            "known": int((status == dipoles.KNOWN).sum()),
            "unidentified": int((status == dipoles.CANDIDATE).sum()),
            "discarded": int((status == dipoles.DISCARDED).sum()),
        }
        if not len(rows):
            continue
        kept = rows[rows["status"] != dipoles.DISCARDED]
        numbers: dict[tuple, int] = {}
        for r in kept.itertuples():
            cell = (int(r.detector), int(r.season_a), int(r.season_b))
            numbers[cell] = numbers.get(cell, 0) + 1
            p = None if pd.isna(r.p_dipole) else round(float(r.p_dipole), 3)
            candidates.append({
                "id": f"{target_id}-d{cell[0]}-s{cell[1]}-s{cell[2]}-{numbers[cell]}",
                "target_id": target_id, "detector": cell[0],
                "season_a": cell[1], "season_b": cell[2],
                "status": r.status, "reason": r.reason,
                "ra_from": round(float(r.ra_from), 6), "dec_from": round(float(r.dec_from), 6),
                "ra_to": round(float(r.ra_to), 6), "dec_to": round(float(r.dec_to), 6),
                "snr": round(float(r.snr), 1), "saturated": bool(r.halo),
                "p_dipole": p, "ai_says": ai_says(p, threshold),
            })  # fmt: skip

    # To check first, then Known; within each, the CNN's most confident first (no opinion last)
    def order(c: dict) -> tuple:
        confidence = c["p_dipole"] if c["p_dipole"] is not None else -np.inf
        return (c["status"] != dipoles.CANDIDATE, -confidence, c["target_id"], c["detector"],
                c["season_a"], c["season_b"])  # fmt: skip

    candidates.sort(key=order)
    return {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gold_run_utc": meta.get("run_date_utc"),
        "gold_commit": meta.get("commit"),
        "model": {"status": "experimental", "threshold": threshold, "card": MODEL_CARD_URL},
        "texts": TEXTS,
        "counts": counts,
        "candidates": candidates,
    }


def read_gold(gold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(targets, catalogs, meta) of a gold folder, read with DuckDB: the same engine the site
    uses, and it needs no pyarrow (which a signing policy may block on Windows)."""
    import duckdb

    gold_dir = Path(gold_dir)
    targets_table = duckdb.sql(f"SELECT * FROM '{gold_dir / 'targets.parquet'}'").df()
    catalogs = duckdb.sql(f"SELECT * FROM '{gold_dir / 'catalogs.parquet'}'").df()
    meta_path = gold_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return targets_table, catalogs, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m serafim.validation.discoveries")
    parser.add_argument("--gold", type=Path, default=dipoles.GOLD_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    targets_table, catalogs, meta = read_gold(args.gold)
    found = {}
    for target_id in sorted(targets_table["target_id"]):
        layers = catalogs[catalogs["target_id"] == target_id]
        found[target_id] = dipoles.target_dipoles(target_id, layers)
    content = build(found, meta, dipoles.cnn_threshold())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")  # fmt: skip
    log.info("%s: %d candidates, counts %s", args.out, len(content["candidates"]),
             content["counts"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
