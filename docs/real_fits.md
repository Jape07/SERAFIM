# What IRSA really serves for SPHEREx (task D1)

Measured on 2026-11-15 (UTC) for the 5 challenge targets in `config/targets.yaml`.
Every number here comes from `python docs/inspect_real_fits.py` (it uses the network; run it again
to refresh). Plan change and English names: issue #18.

## Answers

| Question | Answer |
| --- | --- |
| Rows (frames) per target | barnard **181**, eps_indi_b **238**, wise_0855 **129**, rw_aur **168**, r_dra **1259** (total 1975) |
| MB per cutout | **~4.4 MB at 0.25°** (4.08 MB at 0.1°, ~5.3 MB at 0.5°). 90 % of it is the EPSF extension, which does not shrink with the cutout |
| `energy_bandpassname` | `SPHEREx-D1` … `SPHEREx-D6`: the detector number is the last digit |
| FLAGS bit names | From the `MSKN*` keys of the FLAGS header; OVERFLOW = 1, NONLINEAR = 15, SOURCE = 21 (full list below) |
| CORS on SIA, cutouts and TAP | **No.** None of them sends `Access-Control-Allow-Origin` |

## CORS: the browser cannot read IRSA directly

Requests sent with `Origin: https://jape07.github.io`, as the published site would:

| Service | GET | Allow-Origin on GET | OPTIONS (preflight) | Allow-Origin on OPTIONS |
| --- | --- | --- | --- | --- |
| SIA (`/SIA?POS=…`) | 200 | none | 200 | none |
| Cutout (`<access_url>?center=…&size=…`) | 200 | none | 405 | none |
| TAP (`/TAP/sync?QUERY=…`) | 200 | none | 200 | none |

The data comes back, but without that header the browser does not let JavaScript read it. So
**there is no live mode in the browser**. The "type any coordinates" path is the
`workflow_dispatch` of task D10: GitHub Actions has no CORS, downloads the frames, builds gold and
republishes.

## SIA

- Collections, in priority order (the first one wins when an exposure is in several):
  `spherex_dr1`, `spherex_dr1_deep`, `spherex_qr3`, `spherex_qr3_deep` (`settings.collections`).
  `spherex_dr1_cube` exists but has no dates, so it is not used for the video.
- Useful columns: `obs_id`, `t_min`, `t_max`, `em_min`, `em_max`, `energy_bandpassname`,
  `access_url`, `s_region`.
- `em_min`/`em_max` are the edges of the **whole detector**, the same in every frame of that
  detector. They are not the wavelength at the target: that comes from WCS-WAVE (below).

| Detector | `em_min` (µm) | `em_max` (µm) |
| --- | --- | --- |
| 1 | 0.746 | 1.117 |
| 2 | 1.103 | 1.650 |
| 3 | 1.630 | 2.435 |
| 4 | 2.403 | 3.847 |
| 5 | 3.811 | 4.430 |
| 6 | 4.412 | 5.010 |

Neighbouring detectors overlap a little; the gold contract (D2) checks wavelengths against these
edges.

## Frames per target

One SIA query per collection around `ra_now`/`dec_now`; seasons from `split_seasons()` (30-day
gaps, runs longer than 183 days cut into ~6-month passes).

| Target | Frames | Seasons | By collection | Frames per (season, detector): min / median / max |
| --- | --- | --- | --- | --- |
| barnard | 181 | 3 | dr1 121, qr3 60 | 4 / 9 / 20 |
| eps_indi_b | 238 | 4 | dr1 178, qr3 60 | 1 / 12 / 20 |
| wise_0855 | 129 | 3 | dr1 129 | 1 / 7 / 21 |
| rw_aur | 168 | 3 | dr1 113, qr3 55 | 4 / 9 / 15 |
| r_dra | 1259 | 4 | dr1_deep 618, qr3_deep 367, dr1 210, qr3 64 | 7 / 18 / **267** |

R Draconis is ~9° from the north ecliptic pole, so it is observed every month and one
(season, detector) cell has 267 frames.

## Anatomy of a cutout

WISE 0855, frame `2025W19_1B_0116_1` (detector 3), 0.25°:

| HDU | Type | Shape | MB | Share |
| --- | --- | --- | --- | --- |
| PRIMARY | PrimaryHDU | — | 0.00 | 0.1 % |
| IMAGE | ImageHDU | 146 × 146, float32 | 0.11 | 2.5 % |
| FLAGS | ImageHDU | 146 × 146, int32 | 0.10 | 2.4 % |
| VARIANCE | ImageHDU | 146 × 146, float32 | 0.10 | 2.2 % |
| ZODI | ImageHDU | 146 × 146, float32 | 0.10 | 2.2 % |
| EPSF | BinTableHDU | 451 rows | **3.95** | **90.4 %** |
| WCS-WAVE | BinTableHDU | 1 row: `X` (9), `Y` (14), `VALUES` (2 × 9 × 14) | 0.01 | 0.1 % |

- The IMAGE header has what silver needs per frame: `DETECTOR` (1–6), `OBSID`, `DATE-AVG` /
  `MJD-AVG` (mean time), `BUNIT = MJy / sr` (zodiacal light **not** subtracted), a sky WCS
  `RA---TAN-SIP` and a spectral WCS with key `W` (`CTYPE1W = WAVE-TAB`).
- A cutout at the edge of the detector comes back smaller (one came as 117 × 146 px).
- The cutout service only takes `center` and `size`; there is no way to leave EPSF out
  (IRSA cutouts tutorial).

## Wavelength at the target position

```python
x, y = WCS(image_header).world_to_pixel(target_position)          # sky WCS, with SIP
spectral = WCS(image_header_without_sip_keys, fobj=hdul, key="W")  # WAVE-TAB from WCS-WAVE
spectral.sip = None
wavelength_um, bandwidth_um = spectral.pixel_to_world_values(x, y)
```

The SIP coefficients belong to the sky WCS; leaving them in the spectral WCS makes astropy warn
that the result may be wrong. For WISE 0855 in that frame: pixel (72.8, 72.8), **2.3633 µm**,
bandwidth **0.0585 µm** (inside detector 3), FLAGS = `SOURCE` only.

## FLAGS bits (`MSKN*` keys)

| Bit | Name | Bit | Name | Bit | Name |
| --- | --- | --- | --- | --- | --- |
| 0 | TRANSIENT | 11 | COLD | 22 | GHOST |
| 1 | **OVERFLOW** | 12 | FULLSAMPLE | 23 | GHOST_FPA |
| 2 | SUR_ERROR | 14 | PHANMISS | 24 | GHOST_EXT |
| 6 | NONFUNC | 15 | **NONLINEAR** | 25 | STREAK |
| 7 | DICHROIC | 17 | PERSIST | 26 | BLOOM |
| 9 | MISSING_DATA | 19 | OUTLIER | 27 | SNOWBALL |
| 10 | HOT | 20 | CROSSTALK | 28 | HALO |
| | | 21 | **SOURCE** | | |

- Never mask SOURCE: almost every star pixel has it.
- Sums (D5) leave out pixels with OVERFLOW or NONLINEAR. A1 found the target pixel saturated in
  all 6 detectors for **barnard, rw_aur and r_dra** (OVERFLOW, BLOOM…), so their centers will be
  masked in the sums; talk with AI before D5.
- Always read the names from the header, not from this table: the numbering is the archive's.

## How much to download (input for D3)

At 4.37 MB and ~6–7 s per 0.25° cutout (the time depends on IRSA; this run took 7.2 s):

| Cap per (season, detector) | Frames | GB | Minutes with 8 in parallel |
| --- | --- | --- | --- |
| none | 1975 | 8.63 | 30 |
| 30 | 1107 | 4.84 | 17 |
| 20 | 1022 | 4.47 | 15 |
| **12** | **874** | **3.82** | **13** |
| 8 | 681 | 2.98 | 10 |

**Proposal for D3: cap 12 frames per (season, detector)**, as a setting in `config/targets.yaml`
(it replaces the temporary `max_frames_per_target`). Per target that keeps barnard 167/181,
eps_indi_b 204/238, wise_0855 120/129, rw_aur 163/168 and r_dra 220/1259. It fits the GitHub
Actions cache (10 GB per repository) with room to spare, and only R Draconis, which saturates
anyway, loses many frames. Within a cell, take frames spread in time rather than the oldest.

Alternative, needs the team's OK: drop EPSF when saving bronze (~0.4 MB per frame, ~0.8 GB for
all 1975 frames). It breaks "bronze is exactly what arrived" and AI's forced photometry (A4) may
want the PSF, and the download is still the full 8.6 GB.

## Being nice to IRSA

- During D0 and D1 IRSA answered **502** and **503** several times; a later try always worked.
  Retry with growing waits (5, 10, 15 s) and never more than 8 downloads at once. The bronze code
  from PR #17 waits a fixed 2 s and gave up once on a 503; D3 changes that.
- SIA pages of 4 rows (PR #17) are slow (~211 s for 8 cutouts at the pole). One SIA query per
  collection without `MAXREC`, as `serafim.validation.targets.spherex_frames()` does, answered in
  3–11 s per target.
