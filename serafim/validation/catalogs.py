"""Catalog queries with an on-disk cache (task A1, AI & validation).

- skybot(center, radius, date): known asteroids and comets at that exact date
  (astroquery.imcce.Skybot.cone_search).
- simbad(center, radius): stars and galaxies; object type V* marks variables.
- gaia(center, radius): Gaia sources, including phot_variable_flag.

Every query is cached in data/catalog_cache/, has a timeout and 3 retries.
If a service does not answer and there is no cache, nothing becomes a
candidate: those objects end up Discarded with reason "not validated".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

CACHE_DIR = Path("data/catalog_cache")
TIMEOUT_S = 60
RETRIES = 3


def skybot(center: SkyCoord, radius: Angle, date: Time) -> pd.DataFrame:
    """Solar system objects inside the cone at that date."""
    raise NotImplementedError("Task A1")


def simbad(center: SkyCoord, radius: Angle) -> pd.DataFrame:
    """SIMBAD objects inside the cone, with their object type."""
    raise NotImplementedError("Task A1")


def gaia(center: SkyCoord, radius: Angle) -> pd.DataFrame:
    """Gaia sources inside the cone, with phot_variable_flag."""
    raise NotImplementedError("Task A1")
