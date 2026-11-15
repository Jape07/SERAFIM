"""Bronze layer: SPHEREx cutouts from IRSA, exactly as they arrive (task D3).

For each target in config/targets.yaml:

1. `spherex_frames()` (AI & validation) lists every SPHEREx exposure that covers the target's
   position today (`ra_now`, `dec_now`): one SIA query per collection in `settings.collections`,
   and when an exposure is in several, the first collection wins (DR1 over QR3).
2. `split_seasons()` numbers the season of every frame on that COMPLETE list, before any cap,
   so the numbers are the same as the ones AI wrote in config/targets.yaml.
3. `select_frames()` keeps at most `settings.max_frames_per_season_detector` frames in each
   (season, detector) cell, spread in time (not the oldest ones).
4. The cutout of each kept frame (`settings.cutout_size_deg`, or the target's own
   `cutout_size_deg`) is saved to `data/bronze/<target>/<obs_id>.fits` exactly as IRSA sends it,
   EPSF included. At most 8 at a time; a failed request is retried after 5, 10 and 15 s.
5. `data/bronze/manifest.parquet` says what is on disk, one row per cutout:

       target_id, obs_id, collection  as in IRSA
       detector      1 to 6
       season        1 or more, from step 2
       t_mid_mjd     middle of the exposure
       ra, dec       center of the cutout (the target today)
       size_deg      side of the cutout
       path          relative to the repo root, e.g. data/bronze/barnard/<obs_id>.fits

A file already on disk is not downloaded again, so a second run downloads nothing. The
manifest is rebuilt from the IRSA list every run (seasons can move when new frames arrive)
and only lists files that exist.

When IRSA's search (SIA) does not answer, the frame list saved in
config/spherex_frames/<target>.parquet is used instead (`python -m serafim bronze
--save-frame-lists` writes them; the file service usually still answers). Without a saved list,
bronze goes on with the cutouts already on disk; with nothing at all, that target fails.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from serafim.config import ROOT, load_config
from serafim.validation.targets import FRAME_COLUMNS, spherex_frames, split_seasons

log = logging.getLogger("serafim.bronze")

MAX_PARALLEL = 8  # never more than 8 downloads at once: be nice to IRSA
FRAME_LISTS_DIR = Path("config") / "spherex_frames"  # saved frame lists, committed
DEFAULT_CAP = 12
DEFAULT_SIZE_DEG = 0.25
RETRY_STEP_S = 5  # waits 5, 10, 15 s: IRSA answers 502/503 now and then
DOWNLOAD_TIMEOUT_S = 120
USER_AGENT = "Mozilla/5.0"
REQUIRED_HDUS = ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE")

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.string()),
        pa.field("obs_id", pa.string()),
        pa.field("collection", pa.string()),
        pa.field("detector", pa.int32()),
        pa.field("season", pa.int32()),
        pa.field("t_mid_mjd", pa.float64()),
        pa.field("ra", pa.float64()),
        pa.field("dec", pa.float64()),
        pa.field("size_deg", pa.float64()),
        pa.field("path", pa.string()),
    ]
)


class ArchiveError(RuntimeError):
    """IRSA did not deliver a usable answer."""


# --- Manifest -----------------------------------------------------------------------------
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
    ordered = sorted(rows, key=lambda r: (r["target_id"], r.get("t_mid_mjd") or 0.0, r["obs_id"]))
    pq.write_table(pa.Table.from_pylist(ordered, schema=MANIFEST_SCHEMA), path)


# --- Settings -----------------------------------------------------------------------------
def collections_for(settings: dict) -> list[str]:
    """SIA collections in priority order, as listed in config/targets.yaml."""
    return [str(name) for name in settings.get("collections") or []]


def retry_waits(retries: int) -> tuple[int, ...]:
    """Seconds to wait before each retry: 3 retries wait 5, 10 and 15 s."""
    return tuple(RETRY_STEP_S * (n + 1) for n in range(max(0, retries)))


def parallel_workers(settings: dict) -> int:
    """Downloads at once: `settings.parallel_downloads`, but never more than MAX_PARALLEL."""
    wanted = int(settings.get("parallel_downloads") or MAX_PARALLEL)
    return max(1, min(wanted, MAX_PARALLEL))


def cutout_size_deg(target: dict, settings: dict) -> float:
    """Side of the cutout: the target's own `cutout_size_deg`, else the one in `settings`."""
    return float(
        target.get("cutout_size_deg") or settings.get("cutout_size_deg") or DEFAULT_SIZE_DEG
    )


# --- Choosing frames: pure functions, no network and no disk ------------------------------
def spread_in_time(times, count: int) -> np.ndarray:
    """Positions (sorted) of `count` frames spread over the time span.

    For each of `count` equally spaced instants between the first and the last frame, the
    nearest frame that is still free is taken, so the ends are always in and a visit with many
    frames does not use up the whole cap. With `count` or fewer frames, all of them.
    """
    t = np.asarray(times, dtype=float)
    if count >= t.size:
        return np.arange(t.size)
    low, high = float(t.min()), float(t.max())
    instants = np.linspace(low, high, count) if count > 1 else np.array([(low + high) / 2])
    free = np.ones(t.size, dtype=bool)
    chosen = []
    for instant in instants:
        nearest = int(np.argmin(np.where(free, np.abs(t - instant), np.inf)))
        chosen.append(nearest)
        free[nearest] = False
    return np.sort(chosen)


def select_frames(frames: pd.DataFrame, cap: int) -> pd.DataFrame:
    """At most `cap` frames per (season, detector), spread in time, in time order.

    `frames` needs `season`, `detector`, `t_mid_mjd` and `obs_id`. The choice depends only on
    the frames, so the same list always gives the same frames.
    """
    cap = max(1, int(cap))
    kept = []
    for _, cell in frames.groupby(["season", "detector"], sort=True):
        cell = cell.sort_values(["t_mid_mjd", "obs_id"])
        kept.append(cell.iloc[spread_in_time(cell["t_mid_mjd"].to_numpy(), cap)])
    if not kept:
        return frames.iloc[0:0].reset_index(drop=True)
    return pd.concat(kept).sort_values(["t_mid_mjd", "obs_id"], ignore_index=True)


# --- Download -----------------------------------------------------------------------------
def download(
    target_id: str | None = None,
    *,
    root: Path | None = None,
    frames=None,
    fetch=None,
) -> int:
    """Download the missing cutouts and rewrite the manifest. Return how many files are new.

    `frames(center, collections)` and `fetch(job)` are injection points for tests; the real
    ones are `spherex_frames()` and `fetch_cutout()`, which call IRSA.

    Raises:
        ArchiveError: if a search failed (after the other targets were done) or if there is
            not a single cutout on disk for the targets asked for.
    """
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    settings = config.get("settings") or {}
    context = {
        "root": root,
        "settings": settings,
        "cap": int(settings.get("max_frames_per_season_detector") or DEFAULT_CAP),
        "waits": retry_waits(int(settings.get("retries", 3))),
        "workers": parallel_workers(settings),
        "find": frames or spherex_frames,
        "fetch": fetch or fetch_cutout,
    }
    rows = {(row["target_id"], row["obs_id"]): row for row in read_manifest(root)}
    downloaded = on_disk = 0
    failed: list[str] = []

    for target in _targets(config, target_id):
        name = str(target["id"])
        try:
            new, kept = _bronze_target(target, rows, context)
        except ArchiveError as error:  # the other targets go on
            log.error("target=%s: %s", name, error)
            failed.append(name)
            continue
        downloaded += new
        on_disk += kept
        write_manifest(list(rows.values()), root)  # after each target: nothing is lost

    if failed:
        raise ArchiveError(f"The IRSA search failed for: {', '.join(failed)}.")
    if on_disk == 0:
        raise ArchiveError("IRSA returned no SPHEREx cutouts and bronze is empty.")
    log.info("Bronze: %d new cutouts, %d on disk in total.", downloaded, on_disk)
    return downloaded


def frame_list_path(target_id: str, root: Path | None = None) -> Path:
    """The saved list of SPHEREx frames of a target, used when IRSA's search does not answer."""
    return (root or ROOT) / FRAME_LISTS_DIR / f"{target_id}.parquet"


def save_frame_lists(target_id: str | None = None, *, root: Path | None = None,
                     frames=None) -> list[Path]:
    """Write config/spherex_frames/<target>.parquet from the IRSA search (or its cache).

    Commit them: when IRSA's search (SIA) is down, bronze downloads from these lists, because
    the file service (IBE) usually still answers. Run it again when SIA answers to refresh them.
    """
    root = root or ROOT
    config = load_config(root / "config" / "targets.yaml")
    settings = config.get("settings") or {}
    find = frames or spherex_frames
    written = []
    for target in _targets(config, target_id):
        center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
        try:
            listing = find(center, collections_for(settings))
        except Exception as error:  # any failure of a remote service
            log.error("target=%s: no frame list saved, the IRSA search failed: %s", target["id"],
                      error)
            continue
        path = frame_list_path(str(target["id"]), root)
        path.parent.mkdir(parents=True, exist_ok=True)
        listing[FRAME_COLUMNS].reset_index(drop=True).to_parquet(path, index=False)
        log.info("target=%s: %d frames saved in %s", target["id"], len(listing), path)
        written.append(path)
    return written


def _search(target: dict, rows: dict, context: dict) -> pd.DataFrame | None:
    """The frames of a target from IRSA; if the search fails, from the saved list.

    None means: the search failed, there is no saved list, but cutouts of this target are on
    disk, so bronze goes on with them. With nothing at all it is an ArchiveError.
    """
    root, settings = context["root"], context["settings"]
    name = str(target["id"])
    center = SkyCoord(float(target["ra_now"]) * u.deg, float(target["dec_now"]) * u.deg)
    try:
        return context["find"](center, collections_for(settings))
    except Exception as error:  # any failure of a remote service counts as a failed search
        failure = error
    saved = frame_list_path(name, root)
    if saved.is_file():
        log.warning("target=%s: the IRSA search failed (%s); using the frame list saved in %s",
                    name, failure, saved.relative_to(root))
        return pd.read_parquet(saved)
    on_disk = [row for key, row in rows.items()
               if key[0] == name and (root / str(row["path"])).is_file()]
    if on_disk:
        log.warning("target=%s: the IRSA search failed (%s) and there is no saved frame list; "
                    "going on with the %d cutouts on disk", name, failure, len(on_disk))
        return None
    raise ArchiveError(f"the IRSA search failed and there is no saved frame list: {failure}")


def _bronze_target(target: dict, rows: dict, context: dict) -> tuple[int, int]:
    """Bronze for one target. Updates `rows`; returns (new files, files on disk)."""
    root, settings = context["root"], context["settings"]
    name = str(target["id"])
    ra, dec = float(target["ra_now"]), float(target["dec_now"])
    size_deg = cutout_size_deg(target, settings)

    listing = _search(target, rows, context)
    if listing is None:  # no search and no saved list: keep the cutouts on disk as they are
        for key in [k for k, row in rows.items() if k[0] == name]:
            if not (root / str(rows[key]["path"])).is_file():
                del rows[key]
        return 0, sum(1 for key in rows if key[0] == name)
    if listing.empty:
        log.warning("target=%s: IRSA lists no SPHEREx frames at RA %.4f, Dec %.4f", name, ra, dec)
        return 0, 0
    # Seasons come from the complete list, before the cap, as in AI's config/targets.yaml.
    seasons = split_seasons(listing["t_mid_mjd"], **(settings.get("seasons") or {}))
    listing = listing.assign(season=seasons)
    chosen = select_frames(listing, context["cap"])

    def record(frame) -> dict:
        return {
            "target_id": name,
            "obs_id": frame.obs_id,
            "collection": frame.collection,
            "detector": int(frame.detector),
            "season": int(frame.season),
            "t_mid_mjd": float(frame.t_mid_mjd),
            "ra": ra,
            "dec": dec,
            "size_deg": size_deg,
            "path": f"data/bronze/{name}/{_safe(frame.obs_id)}.fits",
        }

    # The manifest only lists files that exist; a file on disk that is no longer chosen stays.
    for key in [k for k, row in rows.items() if k[0] == name]:
        if not (root / str(rows[key]["path"])).is_file():
            del rows[key]
    by_obs = {frame.obs_id: frame for frame in listing.itertuples()}
    for target_name, obs_id in list(rows):
        if target_name == name and obs_id in by_obs:
            rows[(name, obs_id)] = record(by_obs[obs_id])  # the season may have moved

    jobs = []
    for frame in chosen.itertuples():
        if (name, frame.obs_id) not in rows:
            entry = record(frame)
            jobs.append(
                {
                    **entry,
                    "access_url": frame.access_url,
                    "waits": context["waits"],
                    "destination": root / entry["path"],
                }
            )
    saved = _download_all(jobs, context["workers"], context["fetch"])
    for job in saved:
        rows[(name, job["obs_id"])] = record(by_obs[job["obs_id"]])

    mine = [row for key, row in rows.items() if key[0] == name]
    log.info(
        "target=%s: %d frames in IRSA (%d seasons); kept %d, at most %d per season and detector; "
        "%d on disk, %d new, %d failed",
        name,
        len(listing),
        listing["season"].nunique(),
        len(chosen),
        context["cap"],
        len(mine),
        len(saved),
        len(jobs) - len(saved),
    )
    if len({row["season"] for row in mine}) < 2:
        log.warning("target=%s: bronze has frames in fewer than 2 seasons", name)
    return len(saved), len(mine)


def fetch_cutout(job: dict) -> None:
    """Download one cutout and keep it only if the SPHEREx extensions are there."""
    separator = "&" if "?" in job["access_url"] else "?"
    url = (
        f"{job['access_url']}{separator}center={job['ra']:.6f},{job['dec']:.6f}"
        f"&size={job['size_deg']:.4f}"
    )
    payload = _get_bytes(url, job["waits"])
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


def fits_has_products(payload: bytes) -> bool:
    """True when the cutout is a SPHEREx MEF with the extensions the silver layer needs."""
    if not payload.startswith(b"SIMPLE"):
        return False
    try:
        with fits.open(BytesIO(payload)) as hdul:
            names = {hdu.name.strip() for hdu in hdul}
    except OSError:
        return False
    return all(name in names for name in REQUIRED_HDUS)


def _download_all(jobs: list[dict], workers: int, fetch) -> list[dict]:
    """Run `fetch` on every job, `workers` at a time. Return the jobs that worked."""
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
                log.warning(
                    "target=%s obs_id=%s: skipped: %s", job["target_id"], job["obs_id"], error
                )
                destination = job["destination"]
                if destination.exists() and not fits_has_products(destination.read_bytes()):
                    destination.unlink()
            else:
                saved.append(job)
    return saved


def _get_bytes(url: str, waits: tuple[int, ...]) -> bytes:
    """GET with one try plus one retry after each wait in `waits`."""
    last = "network error"
    for attempt in range(len(waits) + 1):
        try:
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT_S
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last = str(error)
            if attempt < len(waits):
                log.warning("GET failed (%s): retry %d in %d s", last, attempt + 1, waits[attempt])
                time.sleep(waits[attempt])
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


def _safe(obs_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "-" for char in obs_id)
    return cleaned.strip("-.") or "frame"
