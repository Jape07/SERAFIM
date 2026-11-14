"""Gold layer (data engineering).

PNG per frame and per difference image (tasks D4 and D5), then the final tables
following serafim/contract.py (task D7), after calling
`serafim.validation.status.classify()`.

Output: data/gold/ (Parquet, PNG, meta.json, objects.csv). This is the only
layer that gets published.
"""
