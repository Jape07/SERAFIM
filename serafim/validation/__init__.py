"""Validation (AI & validation): the challenge targets and what is known about them.

- targets.py: the 5 challenge targets in config/targets.yaml, plus `propagate()` and
  `split_seasons()`, shared with the pipeline (task A1).
- catalogs.py: SkyBoT, SIMBAD, Gaia and CatWISE queries with an on-disk cache.
- layers.py: `catalog_layers()`, the rows of gold catalogs.parquet: every catalog
  source in each target's field, moved to every season (task A2).
- status.py: `classify()`, only for the optional candidate layer (extra task A9).
"""
