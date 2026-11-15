"""The classic rule of stage 1, applied to the CNN's stamps (task A9, stage 2).

Its decisions are stored in the dataset (labels.parquet `rule`, real.parquet `rule` and `stage1`),
so the CNN can be compared with the rule anywhere, a Colab notebook included, without silver.

- rule_says(stamp): the rule on one stamp alone (a positive and a negative 5-sigma blob pairing up
  within RULE_CENTER_PX of the center), as on the injected stamps;
- stage_one(rows): the full rule of the pipeline (whole difference, halo lobes, catalog check)
  at each real target: does it keep a dipole there? Needs gold and silver on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from serafim.validation import dipoles

RULE_CENTER_PX = 4.0
_EMPTY = np.zeros((32, 32), bool)


def rule_says(stamp: np.ndarray) -> bool:
    """The stage-1 rule on one stamp: a dipole near the center."""
    sigma = np.sinh(np.asarray(stamp[2], float))  # back from asinh to sigma units
    found = dipoles.find_dipoles(sigma, _EMPTY, _EMPTY)
    if found.empty:
        return False
    cx = (found["x_from"] + found["x_to"]) / 2 - 16
    cy = (found["y_from"] + found["y_to"]) / 2 - 16
    return bool((np.hypot(cx, cy) <= RULE_CENTER_PX).any())


def stage_one(real_rows: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.Series:
    """For each real difference (target_id, detector, season_a, season_b): whether the full
    stage-1 rule keeps a Known or unidentified dipole at the target."""
    out = pd.Series(False, index=real_rows.index)
    for target_id, part in real_rows.groupby("target_id"):
        layers = tables["catalogs"][tables["catalogs"]["target_id"] == target_id]
        rows = dipoles.target_dipoles(target_id, layers)
        if rows.empty:
            continue
        kept = rows[dipoles.at_target(rows) & (rows["status"] != dipoles.DISCARDED)]
        cells = {(r.detector, r.season_a, r.season_b) for r in kept.itertuples()}
        for r in part.itertuples():
            out[r.Index] = (r.detector, r.season_a, r.season_b) in cells
    return out
