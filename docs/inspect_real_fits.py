"""Measure what IRSA really serves for the 5 challenge targets (task D1).

Every number in docs/real_fits.md comes from this script. It uses the network
(IRSA SIA, the cutout service and TAP), so it is not part of the test suite:

    python docs/inspect_real_fits.py

It writes data/inspect/frames.parquet (one row per SPHEREx frame of every target)
and the WISE 0855 cutouts it inspects; data/ is git-ignored.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from serafim.bronze.download import USER_AGENT, ArchiveError, fetch_cutout
from serafim.config import ROOT, load_config
from serafim.validation.targets import spherex_frames, split_seasons

OUT = ROOT / "data" / "inspect"
SITE_ORIGIN = "https://jape07.github.io"
SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
INSPECTED_TARGET = "wise_0855"
CAPS_PER_CELL = [None, 30, 20, 12, 8]  # frames per (season, detector)
WAITS_S = (5, 10, 15)  # IRSA answers 502/503 now and then: retry with growing waits


def main() -> None:
    warnings.simplefilter("ignore", FITSFixedWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    frames = frames_per_target(config)
    target = next(t for t in config["targets"] if t["id"] == INSPECTED_TARGET)
    first = frames[frames.target == INSPECTED_TARGET].iloc[0]
    report_cors(target, first.access_url)
    report_cutout(target, first)


# --- Frames per target ------------------------------------------------------
def frames_per_target(config: dict) -> pd.DataFrame:
    settings = config["settings"]
    parts = []
    print("## Frames per target (SIA, one query per collection)\n")
    for target in config["targets"]:
        center = SkyCoord(target["ra_now"], target["dec_now"], unit="deg")
        found = spherex_frames(center, settings["collections"])
        found["season"] = split_seasons(found["t_mid_mjd"].to_numpy(), **settings["seasons"])
        found["target"] = target["id"]
        parts.append(found)
        print(f"- {target['id']}: {len(found)} frames, {found.season.nunique()} seasons, "
              f"by collection {found.collection.value_counts().to_dict()}")
    frames = pd.concat(parts, ignore_index=True)
    frames.to_parquet(OUT / "frames.parquet")

    print("\nSIA em_min/em_max per detector (um), identical in every frame of a detector:")
    edges = frames.groupby("detector")[["em_min_um", "em_max_um"]].agg(["min", "max"]).round(3)
    print(edges.to_string())

    cells = frames.groupby(["target", "season", "detector"]).size()
    print("\nFrames per (season, detector) cell:")
    print(cells.groupby("target").agg(["min", "median", "max", "count"]).to_string())
    return frames


def cap_table(cells: pd.Series, mb_per_cutout: float, seconds_per_cutout: float) -> None:
    print("\n| Cap per (season, detector) | Frames | GB | Minutes with 8 in parallel |")
    print("| --- | --- | --- | --- |")
    for cap in CAPS_PER_CELL:
        n = int(cells.sum() if cap is None else cells.clip(upper=cap).sum())
        minutes = n * seconds_per_cutout / 8 / 60
        label = "none" if cap is None else str(cap)
        print(f"| {label} | {n} | {n * mb_per_cutout / 1000:.2f} | {minutes:.0f} |")


# --- CORS -------------------------------------------------------------------
def report_cors(target: dict, access_url: str) -> None:
    ra, dec = target["ra_now"], target["dec_now"]
    urls = {
        "SIA": f"{SIA_URL}?POS=CIRCLE+{ra}+{dec}+0.0003&COLLECTION=spherex_dr1&MAXREC=1",
        "Cutout": f"{access_url}?center={ra},{dec}d&size=0.25",
        "TAP": f"{TAP_URL}?LANG=ADQL&FORMAT=csv&QUERY=SELECT+1+FROM+TAP_SCHEMA.tables",
    }
    print(f"\n## CORS (Origin: {SITE_ORIGIN})\n")
    print("| Service | GET | Allow-Origin on GET | OPTIONS (preflight) | Allow-Origin on OPTIONS |")
    print("| --- | --- | --- | --- | --- |")
    for name, url in urls.items():
        get = _request("GET", url)
        options = _request("OPTIONS", url, {"Access-Control-Request-Method": "GET"})
        cells = [name, get.status_code, _allow(get), options.status_code, _allow(options)]
        print("| " + " | ".join(str(cell) for cell in cells) + " |")


def _allow(response: requests.Response) -> str:
    return response.headers.get("Access-Control-Allow-Origin", "none")


def _request(method: str, url: str, extra: dict | None = None) -> requests.Response:
    headers = {"Origin": SITE_ORIGIN, "User-Agent": USER_AGENT, **(extra or {})}
    response = None
    for wait in (*WAITS_S, None):
        response = requests.request(method, url, headers=headers, timeout=120)
        if response.status_code not in (502, 503) or wait is None:
            break
        time.sleep(wait)
    return response


# --- One real cutout --------------------------------------------------------
def report_cutout(target: dict, frame: pd.Series) -> None:
    print(f"\n## Cutout of {target['id']}, frame {frame.obs_id} ({frame.bandpass})\n")
    sizes = {}
    for size_deg in (0.25, 0.1):
        destination = OUT / f"{target['id']}_{size_deg}.fits"
        job = {
            "obs_id": frame.obs_id,
            "access_url": frame.access_url,
            "ra": float(target["ra_now"]),
            "dec": float(target["dec_now"]),
            "size_deg": size_deg,
            "attempts": len(WAITS_S),
            "destination": destination,
        }
        seconds = _fetch_with_waits(job)
        sizes[size_deg] = (destination.stat().st_size / 1e6, seconds)
        print(f"- {size_deg} deg: {sizes[size_deg][0]:.2f} MB in {seconds:.1f} s")

    path = OUT / f"{target['id']}_0.25.fits"
    with fits.open(path) as hdul:
        total = path.stat().st_size
        print("\n| HDU | Type | Shape | MB | Share |")
        print("| --- | --- | --- | --- | --- |")
        for hdu in hdul:
            shape = getattr(hdu.data, "shape", None)
            share = 100 * hdu.filebytes() / total
            print(f"| {hdu.name} | {type(hdu).__name__} | {shape} | "
                  f"{hdu.filebytes() / 1e6:.2f} | {share:.1f} % |")

        image = hdul["IMAGE"].header
        keys = ["BUNIT", "DETECTOR", "OBSID", "DATE-AVG", "MJD-AVG", "CTYPE1", "CTYPE1W"]
        print("\nIMAGE header:", {key: image.get(key) for key in keys})
        bits = _flag_bits(hdul["FLAGS"].header)
        print("FLAGS bits (MSKN*):", bits)
        print("WCS-WAVE columns:", [(c.name, c.format, c.dim) for c in hdul["WCS-WAVE"].columns])

        x, y = WCS(image).world_to_pixel(SkyCoord(target["ra_now"], target["dec_now"], unit="deg"))
        wavelength, bandwidth = _spectral_wcs(hdul).pixel_to_world_values(x, y)
        flags = int(hdul["FLAGS"].data[int(round(float(y))), int(round(float(x)))])
        names = [name for bit, name in bits.items() if flags & (1 << bit)]
        print(f"Target pixel ({float(x):.1f}, {float(y):.1f}): "
              f"wavelength {float(wavelength):.4f} um, bandwidth {float(bandwidth):.4f} um, "
              f"FLAGS {flags} = {names}")

    cells = pd.read_parquet(OUT / "frames.parquet").groupby(["target", "season", "detector"]).size()
    mb, seconds = sizes[0.25]
    cap_table(cells, mb, seconds)


def _fetch_with_waits(job: dict) -> float:
    """fetch_cutout() with growing waits between tries; returns the seconds of the good try."""
    for wait in (*WAITS_S, None):
        start = time.time()
        try:
            fetch_cutout(job)
            return time.time() - start
        except ArchiveError as error:
            if wait is None:
                raise
            print(f"  (IRSA: {str(error)[:40]}... waiting {wait} s)")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _flag_bits(header: fits.Header) -> dict[int, str]:
    bits = {int(key[4:]): str(header[key]) for key in header if key.startswith("MSKN")}
    return dict(sorted(bits.items()))


def _spectral_wcs(hdul: fits.HDUList) -> WCS:
    """The WAVE-TAB WCS (key W). The SIP keys belong to the sky WCS, so they are left out."""
    header = hdul["IMAGE"].header.copy()
    for key in [k for k in header if k.split("_")[0] in ("A", "B", "AP", "BP") and "_" in k]:
        del header[key]
    spectral = WCS(header, fobj=hdul, key="W")
    spectral.sip = None
    return spectral


if __name__ == "__main__":
    main()
