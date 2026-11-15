"""Bronze layer: SPHEREx cutouts from IRSA, exactly as they arrive.

For each target in config/targets.yaml, search with ``Irsa.query_sia`` around
its ``ra_now``/``dec_now`` (position corrected for proper motion) in
``settings.collections``. When the same exposure is in several collections, the
first one in that list wins (DR1 over QR3). Each page asks for a few rows: a wide
cone or a large MAXREC makes the IRSA proxy return HTTP 502. The cutout itself
is ``settings.cutout_size_deg``.

A file already on disk is not downloaded again.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io import fits

from serafim.config import ROOT, load_config

log = logging.getLogger("serafim.bronze")

# Images that cover this point. A wider POS is what makes the proxy return 502.
SEARCH_RADIUS_ARCSEC = 1.0
# Four rows per page. maxrec of 8 or more often comes back as HTTP 502.
PAGE_SIZE = 4
QUERY_TIMEOUT_S = 40
DOWNLOAD_TIMEOUT_S = 120
USER_AGENT = "Mozilla/5.0"
REQUIRED_HDUS = ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE")

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.string()),
        pa.field("obs_id", pa.string()),
        pa.field("collection", pa.string()),
        pa.field("band", pa.string()),
        pa.field("t_min", pa.float64()),
        pa.field("ra", pa.float64()),
        pa.field("dec", pa.float64()),
        pa.field("size_deg", pa.float64()),
        pa.field("path", pa.string()),
    ]
)


class ArchiveError(RuntimeError):
    """IRSA did not deliver a usable cutout."""


def manifest_path(root: Path | None = None) -> Path:
    return (root or ROOT) / "data" / "bronze" / "manifest.parquet"


def read_manifest(root: Path | None = None) -> list[dict]:
    path = manifest_path(root)
    if not path.is_file():
        return []
    return pq.read_table(path).to_pylist()


def write_manifest(rows: list[dict], root: Path | None = None) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), path)


def fits_has_products(payload: bytes) -> bool:
    """True when the cutout is a SPHEREx MEF with the extensions D2 requires."""
    if not payload.startswith(b"SIMPLE"):
        return False
    try:
        with fits.open(BytesIO(payload)) as hdul:
            names = {hdu.name.strip() for hdu in hdul}
    except OSError:
        return False
    return all(name in names for name in REQUIRED_HDUS)


def choose_exposures(
    rows: list[dict], limit: int, priority: list[str] | None = None
) -> list[dict]:
    """One row per obs_id, oldest first, cut at ``limit``.

    When an exposure is in several collections, the one listed first in
    ``priority`` wins; without a list, DR1 wins over QR3.
    """

    def rank(collection: str) -> int:
        if priority and collection in priority:
            return priority.index(collection)
        return _dr1_rank(collection) + (len(priority) if priority else 0)

    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["obs_id"])
        if current is None or rank(row["collection"]) < rank(current["collection"]):
            best[row["obs_id"]] = row
    ordered = sorted(best.values(), key=lambda row: (float(row.get("t_min") or 0), row["obs_id"]))
    return ordered[:limit]


def collections_for(settings: dict) -> list[str]:
    """SIA collections in priority order, as listed in config/targets.yaml."""
    return [str(name) for name in settings.get("collections") or []]


def download(
    target_id: str | None = None,
    *,
    root: Path | None = None,
    query=None,
    fetch=None,
) -> int:
    """Download missing cutouts. Return how many new files were written.

    ``query`` and ``fetch`` are injection points for tests. The real ones call
    IRSA. An exposure that is already a FITS file under ``data/bronze`` is skipped.
    """
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    settings = config.get("settings") or {}
    limit = int(settings.get("max_frames_per_target") or 60)
    attempts = int(settings.get("retries") or 3)
    workers = int(settings.get("parallel_downloads") or 8)
    size_deg = float(settings.get("cutout_size_deg") or 0.25)
    collections = collections_for(settings)
    query = query or query_collection
    fetch = fetch or fetch_cutout

    previous = read_manifest(root)
    by_key = {(row["target_id"], row["obs_id"]): row for row in previous}
    new_rows: list[dict] = []
    kept = 0

    for target in _targets(config, target_id):
        name = str(target["id"])
        ra, dec = float(target["ra_now"]), float(target["dec_now"])
        on_disk = {
            obs_id
            for (target_name, obs_id), row in by_key.items()
            if target_name == name and (root / str(row["path"])).is_file()
        }
        kept += len(on_disk)
        slots = max(0, limit - len(on_disk))
        log.info(
            "target=%s: RA %.4f, Dec %.4f, cutout %s deg, %s already on disk.",
            name,
            ra,
            dec,
            size_deg,
            len(on_disk),
        )
        if slots == 0:
            continue
        found: list[dict] = []
        for collection in collections:
            try:
                found.extend(query(collection, ra, dec, limit, attempts))
            except ArchiveError as error:
                log.warning("target=%s: %s did not answer: %s", name, collection, error)
        chosen = [
            row
            for row in choose_exposures(found, limit, collections)
            if row["obs_id"] not in on_disk
        ][:slots]
        jobs = [_job(name, ra, dec, row, size_deg, attempts, root) for row in chosen]
        for job in _download_all(jobs, workers, fetch):
            record = _record(name, job)
            by_key[(name, job["obs_id"])] = record
            new_rows.append(record)
            on_disk.add(job["obs_id"])

    if new_rows:
        write_manifest(list(by_key.values()), root)
    if kept == 0 and not new_rows:
        raise ArchiveError("IRSA returned no SPHEREx cutouts and bronze is empty.")
    log.info("Bronze: %s new cutouts, %s already on disk.", len(new_rows), kept)
    return len(new_rows)


def query_collection(
    collection: str,
    ra: float,
    dec: float,
    limit: int,
    attempts: int,
) -> list[dict]:
    """SIA2 rows for one collection, oldest first, a few at a time."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astroquery.ipac.irsa import Irsa

    # The proxy answers 502 unless the client looks like a browser.
    Irsa._session.headers["User-Agent"] = USER_AGENT
    coord = SkyCoord(ra, dec, unit="deg")
    radius = SEARCH_RADIUS_ARCSEC * u.arcsec
    found: list[dict] = []
    seen: set[str] = set()
    start_mjd: float | None = None
    pages = max(1, (limit + PAGE_SIZE - 1) // PAGE_SIZE + 1)
    for _ in range(pages):
        if len(found) >= limit:
            break
        kwargs: dict = {"pos": (coord, radius), "collection": collection, "maxrec": PAGE_SIZE}
        if start_mjd is not None:
            kwargs["time"] = (Time(start_mjd, format="mjd").isot, "2035-01-01")
        table = _query_sia(Irsa, kwargs, attempts)
        if table is None or len(table) == 0:
            break
        fresh = 0
        page_max = start_mjd or 0.0
        for row in table:
            obs_id = _text(row, "obs_id")
            access_url = _text(row, "access_url")
            if not obs_id or not access_url or obs_id in seen:
                continue
            seen.add(obs_id)
            fresh += 1
            t_min = _number(row, "t_min")
            t_max = _number(row, "t_max") or t_min
            page_max = max(page_max, t_max)
            found.append(
                {
                    "obs_id": obs_id,
                    "collection": collection,
                    "access_url": access_url,
                    "band": _text(row, "energy_bandpassname"),
                    "t_min": t_min,
                }
            )
            if len(found) >= limit:
                break
        if fresh == 0 or page_max <= (start_mjd or 0.0):
            break
        start_mjd = page_max + 0.001
    found.sort(key=lambda item: (item["t_min"], item["obs_id"]))
    return found


def fetch_cutout(job: dict) -> None:
    """Download one cutout and keep it only if the SPHEREx extensions are there."""
    separator = "&" if "?" in job["access_url"] else "?"
    url = (
        f"{job['access_url']}{separator}center={job['ra']:.6f},{job['dec']:.6f}"
        f"&size={job['size_deg']:.4f}"
    )
    payload = _get_bytes(url, job["attempts"])
    if payload.startswith(b"\x1f\x8b"):
        import gzip

        payload = gzip.decompress(payload)
    if not fits_has_products(payload):
        raise ArchiveError(
            f"{job['obs_id']} is missing IMAGE, FLAGS, VARIANCE, ZODI or WCS-WAVE."
        )
    destination = job["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.write_bytes(payload)
    partial.replace(destination)


def _download_all(jobs: list[dict], workers: int, fetch) -> list[dict]:
    if not jobs:
        return []
    saved: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except (ArchiveError, OSError, requests.RequestException) as error:
                log.warning("Skipping %s: %s", job["obs_id"], error)
                destination = job["destination"]
                if destination.exists() and not fits_has_products(destination.read_bytes()):
                    destination.unlink()
            else:
                saved.append(job)
    return saved


def _query_sia(irsa, kwargs: dict, attempts: int):
    last = "network error"
    for attempt in range(1, attempts + 1):
        try:
            return irsa.query_sia(**kwargs)
        except Exception as error:
            last = str(error)
            log.warning("SIA %s failed (%s).", kwargs.get("collection"), last)
            if attempt < attempts:
                time.sleep(2)
    raise ArchiveError(f"{kwargs.get('collection')}: {last}")


def _get_bytes(url: str, attempts: int) -> bytes:
    last = "network error"
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=DOWNLOAD_TIMEOUT_S,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last = str(error)
            if attempt < attempts:
                time.sleep(2)
    raise ArchiveError(last)


def _targets(config: dict, target_id: str | None) -> list[dict]:
    targets = list(config.get("targets") or [])
    if target_id is None:
        return targets
    chosen = [target for target in targets if target.get("id") == target_id]
    if not chosen:
        known = ", ".join(str(target.get("id")) for target in targets) or "(none)"
        raise ArchiveError(f"Unknown target {target_id}. Config has: {known}.")
    return chosen


def _job(
    target_id: str, ra: float, dec: float, row: dict, size_deg: float, attempts: int, root: Path
) -> dict:
    filename = f"{_safe(row['obs_id'])}.fits"
    relative = f"data/bronze/{target_id}/{filename}"
    return {
        "obs_id": row["obs_id"],
        "collection": row["collection"],
        "band": row.get("band") or "",
        "t_min": float(row.get("t_min") or 0.0),
        "access_url": row["access_url"],
        "ra": ra,
        "dec": dec,
        "size_deg": size_deg,
        "attempts": attempts,
        "path": relative,
        "destination": root / relative,
    }


def _record(target_id: str, job: dict) -> dict:
    return {
        "target_id": target_id,
        "obs_id": job["obs_id"],
        "collection": job["collection"],
        "band": job["band"],
        "t_min": float(job["t_min"]),
        "ra": float(job["ra"]),
        "dec": float(job["dec"]),
        "size_deg": float(job["size_deg"]),
        "path": job["path"],
    }


def _dr1_rank(collection: str) -> int:
    return 0 if "dr1" in collection.lower() else 1


def _safe(obs_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "-" for char in obs_id)
    return cleaned.strip("-.") or "frame"


def _text(row, name: str) -> str:
    if name not in getattr(row, "colnames", ()):
        return ""
    value = row[name]
    if value is None or value is np.ma.masked:
        return ""
    return str(value).strip()


def _number(row, name: str) -> float:
    text = _text(row, name)
    if not text:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number if np.isfinite(number) else 0.0
