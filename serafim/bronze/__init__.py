"""Bronze layer (data engineering).

Search IRSA for SPHEREx images covering each region and download a 0.5 deg
cutout of each one, exactly as it arrives (task D2).

Output: data/bronze/<region>/<obs_id>.fits and data/bronze/manifest.parquet.
Bronze is never modified; a second run downloads nothing new.

``download.py`` runs the search. From the command line: ``python -m serafim bronze``.
"""
