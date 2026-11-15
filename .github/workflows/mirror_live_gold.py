"""Copy the gold already served by GitHub Pages into web/gold.

The full publish rebuilds gold before it can replace the site. A web-only
change can go out with the tables and PNGs that are already public.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

LIVE = "https://jape07.github.io/SERAFIM/gold/"
OUT = Path("web/gold")
TABLES = (
    "meta.json",
    "targets.parquet",
    "frames.parquet",
    "sums.parquet",
    "differences.parquet",
    "catalogs.parquet",
    "spectra.parquet",
    "trajectory.parquet",
)


def fetch(name: str) -> bytes:
    request = urllib.request.Request(LIVE + name, headers={"User-Agent": "serafim-deploy-web"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(name))
        print("table", name, target.stat().st_size)

    sums = pd.read_parquet(OUT / "sums.parquet")
    differences = pd.read_parquet(OUT / "differences.parquet")
    pngs = set(sums["png"].dropna()) | set(differences["png"].dropna())
    for (target_id, season), group in sums.groupby(["target_id", "season"]):
        detectors = set(int(value) for value in group["detector"])
        if {1, 3, 6} <= detectors:
            pngs.add(f"png/{target_id}/color_s{int(season)}.png")

    missing = []
    for name in sorted(pngs):
        try:
            payload = fetch(name)
        except urllib.error.HTTPError as error:
            # A missing color composite must not block the HTML deploy.
            # The last run died on color_s0.png even when Pages answered 404.
            if "/color_" in name:
                print("skip", name, error.code)
                continue
            missing.append(f"{name} ({error.code})")
            continue
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print("pngs", len(pngs) - len(missing))
    if missing:
        raise SystemExit("missing pngs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
