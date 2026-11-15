"""Bronze layer (data engineering).

Search IRSA for SPHEREx images covering each target in config/targets.yaml and
download a cutout (settings.cutout_size_deg) of each one, exactly as it arrives
(task D3).

Output: data/bronze/<target>/<obs_id>.fits and data/bronze/manifest.parquet.
Bronze is never modified; a second run downloads nothing new.

``download.py`` runs the search. From the command line: ``python -m serafim bronze``.
"""
