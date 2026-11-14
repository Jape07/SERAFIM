"""Status rules (task A3, AI & validation).

Pure functions: no network calls in here. They receive catalog results that
were already fetched with `serafim.validation.catalogs`, so they can be tested
with simulated detections (tests/test_status.py).

Rules are checked in this order: discard reasons first, then Known, then
Unidentified candidate.

Discarded
    Touches flagged pixels or the edge; seen in a single exposure; sits on a
    non-variable Gaia star (subtraction residual); months-long change in
    something not catalogued; p_real < 0.5 if the CNN is in; or a catalog did
    not answer. Reason: the exact cause, e.g. "not validated".
Known
    Within 2 px (~12 arcsec) of a SkyBoT object at that date, a SIMBAD variable
    (type V*) or a Gaia source with phot_variable_flag == "VARIABLE".
    Reason: name and catalog.
Unidentified candidate
    Anything else seen in at least 2 exposures at the same position within one
    day. Reason: "No match in SkyBoT, SIMBAD or Gaia".
"""

from __future__ import annotations

import pandas as pd

PIXEL_SCALE_ARCSEC = 6.15  # SPHEREx pixel, from the IRSA data products docs
MATCH_RADIUS_PX = 2  # ~12 arcsec
MIN_EXPOSURES = 2
MIN_P_REAL = 0.5

REASON_NO_MATCH = "No match in SkyBoT, SIMBAD or Gaia"
REASON_NOT_VALIDATED = "not validated"


def classify(
    detections: pd.DataFrame,
    frames: pd.DataFrame,
    catalogs: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group detections into objects and give each object a status and a reason.

    Args:
        detections: silver detections (x, y, ra, dec, flux, snr, sign, frame_id).
        frames: frames table (frame_id, region_id, date_utc, ...).
        catalogs: catalog sources already fetched for these frames (SkyBoT per
            date, SIMBAD, Gaia). None means a catalog did not answer and there
            was no cache: everything becomes Discarded with REASON_NOT_VALIDATED.

    Returns:
        (detections with object_id filled in, objects following contract.OBJECTS)
    """
    raise NotImplementedError("Task A3")
