# Handoff: AI & validation, for whoever continues it

Written 2026-11-15 at commit `477274c` (tag `v1.0.1`), when AI & validation (Jape07) left the
hackathon. Data engineering (Isralejo117) inherits these tasks. Everything below can be done
without the previous owner: the code, the reports and the decisions are all in this repository,
and every command runs on GitHub Actions (Linux) exactly as it did on the owner's machine.

Rules that still apply: everything committed in English; one task, one branch, one small PR with
CI green (0 approvals needed, squash merge); `web/` belongs to web except `web/texts/`, which is
AI & validation's; `serafim/bronze`, `silver`, `gold`, `cli.py` and `publish.yml` are data's.

## 1. Where things stand

- **The public site is `main`.** https://jape07.github.io/SERAFIM/ was deployed from `477274c`
  at 18:08 UTC. `publish.yml` rebuilds gold and the site on every push to `main` and every Monday
  (D10 part 1, #77), and rewrites `web/texts/discoveries.json` after gold (#82). If someone sees an
  older site, it is the browser cache: reload with Ctrl+F5.
- **Nothing of AI & validation is waiting in a branch**, except the CNN version 2 experiment on
  `ai/A9-cnn-phase1` (see section 5). Do not merge it.
- **All AI & validation tasks of the plan are done.** Evidence per task:

| Task | What | Evidence |
| --- | --- | --- |
| A1 | `config/targets.yaml`, the 5 challenge targets | `python -m serafim.validation.targets` |
| A2 | catalog layers Gaia, CatWISE, SIMBAD, SkyBoT per season | `serafim/validation/layers.py`, `catalogs.py` |
| A3 | visibility of each target in real cutouts | [visibility.md](visibility.md) |
| A4 | spectra by EPSF forced photometry; light curves by ring photometry | [spectra.md](spectra.md), [light_curves.md](light_curves.md) |
| A5 | measured trajectory against the catalog | [trajectory.md](trajectory.md) |
| A6 | verification of what gold shows (pass / warn / fail) | [verification.md](verification.md) |
| A7 | the guides of the site, `web/texts/targets.json` | `python -m serafim.validation.guides` |
| A8 | markers checked on gold; review of the published site | [markers.md](markers.md), [mvp_review.md](mvp_review.md) |
| A9 | motion dipoles: classic rule (the detector) and experimental CNN | [dipoles.md](dipoles.md), [dipole_cnn.md](dipole_cnn.md), [model_card.md](model_card.md) |
| A10 | validation cards in the README, live demo script | README "How each target was validated", [demo.md](demo.md) |
| extra | `web/texts/discoveries.json`, the candidates for the site | `python -m serafim.validation.discoveries` (#81), shown by #86 |

## 2. What AI & validation owns

- `serafim/validation/`: `targets`, `catalogs`, `layers`, `guides`, `visibility`, `spectra`,
  `light_curves`, `trajectory`, `verification`, `markers`, `dipoles`, `discoveries`, `public`.
- `serafim/ai/`: `stamps.py` (the 3×32×32 stamps) and `predict.py` (ONNX inference).
  `models/dipole_cnn.onnx` (103 kB, version 1).
- `training/`: dataset, training, evaluation, metrics, learning curve, `RESULTS.md`,
  `experiments.csv`.
- `web/texts/targets.json` and `web/texts/discoveries.json`.
- `docs/*.md` and `docs/*.json` listed in section 1, and `docs/img/`.
- Tests: `tests/test_targets.py`, `test_catalogs.py`, `test_layers.py`, `test_guides.py`,
  `test_spectra.py`, `test_light_curves.py`, `test_trajectory.py`, `test_verification.py`,
  `test_markers.py`, `test_dipoles.py`, `test_discoveries.py`, `test_public.py`,
  `test_docs_links.py`, `tests/ai/`, `tests/training/`.

## 3. The commands, and when to run them

`python -m serafim run` (data's pipeline) already calls AI & validation's `catalogs`, `spectra`
and `trajectory` while building gold, and `discoveries` right after. The reports below are
run by hand when something changes, from the repository root with silver and gold on disk
(`pip install -e ".[dev,ai]"`).

| Command | Writes | Needs | Run it when |
| --- | --- | --- | --- |
| `python -m serafim.validation.targets` | the generated table of `config/targets.yaml` | SIMBAD, IRSA | a target is added or its data changes (rare) |
| `python -m serafim.validation.layers` | the catalog cache under `data/catalog_cache/` | Gaia (fixed copy first), VizieR, SIMBAD, SkyBoT | catalogs fail or a season is added |
| `python -m serafim.validation.guides` | `web/texts/targets.json` | the `docs/*.json` of trajectory, light curves, verification, markers | any report changes, or a season is added |
| `python -m serafim.validation.discoveries` | `web/texts/discoveries.json` | gold + silver | automatic in `serafim run`; by hand to refresh the committed copy |
| `python -m serafim.validation.verification` | `docs/verification.md/json`; exit 1 on a failure | gold + silver | after every gold rebuild; it is the acceptance test of the data |
| `python -m serafim.validation.trajectory` | `docs/trajectory.md/json`, `docs/img/trajectory_*.png` | silver | after a silver rebuild or a new season |
| `python -m serafim.validation.light_curves` | `docs/light_curves.md/json` | bronze + silver | same |
| `python -m serafim.validation.spectra` | `docs/spectra.md` | bronze | same |
| `python -m serafim.validation.markers` | `docs/markers.md/json` | gold + silver | after layers or gold change |
| `python -m serafim.validation.dipoles` | `docs/dipoles.md/json`, `docs/img/dipoles_*.png` | silver + gold catalogs | after silver changes; then rerun `discoveries` so both files agree (a test checks it) |
| `python -m serafim.validation.public` | `docs/mvp_review.md` | the live site | after a deploy, to compare served gold with local gold |
| `python -m training.evaluate` | `docs/dipole_cnn.md/json` | the `training/` outputs | only if the CNN is retrained |

Order after a new season: `serafim run` (data), then `verification`, then `trajectory`,
`light_curves` and `spectra`, then `markers`, `dipoles`, `discoveries`, `guides` and `public`.
Commit `docs/` and `web/texts/` in one PR, and run `pytest`.

## 4. The numbers the reports defend today

The committed reports come from the local gold of commit `5c4609c` (167 frames for Barnard,
3 seasons; R Draconis has 4). The site's gold is rebuilt on Actions with whatever IRSA gave that
day, so small differences are normal: the live `discoveries.json` has 33 candidates (Barnard 17
Known, Epsilon Indi B 14, RW Aurigae 2 to check) against 30 in the committed file. Same rule,
more frames. The tests compare the committed pair (`docs/dipoles.json` and
`web/texts/discoveries.json`), not the live one.

- **Verification** ([verification.md](verification.md)): no failure. The warnings are limits the
  site states: Barnard's spectrum doubtful (saturated), WISE 0855 only seen in detector 6 of the
  fixed-sky sums, RW Aurigae's spectrum doubtful.
- **Trajectory** ([trajectory.md](trajectory.md)): Barnard 10.43 ± 0.37 ″/yr vs 10.40 catalog;
  Epsilon Indi B 4.56 ± 0.28 vs 4.62; WISE 0855 8.66 ± 3.18 (only two seasons in detector 6).
  On the published gold: Barnard 10.45 ± 0.38.
- **Light curves** ([light_curves.md](light_curves.md)): R Draconis 244 d from SPHEREx alone
  (catalog 247); RW Aurigae 1.5× brighter in detector 1 between seasons; Barnard as control,
  1.02–1.04.
- **Spectra** ([spectra.md](spectra.md)): Epsilon Indi B methane dip, ratio 0.23 between 3.3 and
  4.1 µm; WISE 0855 W2 0.53 mJy vs 0.57 catalog. Saturated targets have no reliable absolute flux.
- **Markers** ([markers.md](markers.md)): 0 markers out of place on gold; default cuts keep at most
  150 markers per layer; CatWISE halo ghosts around bright stars are filtered.
- **Dipoles** ([dipoles.md](dipoles.md)): Barnard found as Known in 6 of 6 last-minus-first
  differences without being told where; Epsilon Indi B in 4 of 6; WISE 0855 too faint for the
  rule (S/N 5–7); RW Aurigae and R Draconis give 0 at the target, as they should.
- **CNN** ([dipole_cnn.md](dipole_cnn.md)): 3 of 6 acceptance criteria, so **experimental**.
  Out-of-fold ROC AUC 0.91 on injected stamps, 0.52 on real movers against still saturated stars.
  Threshold 0.731 (the rule's false-alarm rate). It agrees with 10 of the 12 non-saturated Known
  dipoles and calls both "to check" candidates not moving, which is right (single-pixel pairs).

## 5. The CNN: what it is and what it is not

- **The rule in `dipoles.py` is the detector.** The CNN (`DipoleCNN`, 25,921 parameters,
  3×32×32 asinh stamps of sum a, sum b and their difference) only gives a second opinion on
  dipoles the rule already found and that touch no saturated pixel. It never changes a status and
  never announces a new object ([model_card.md](model_card.md)).
- **The opinion is decided in the pipeline, not on the site:** `discoveries.ai_says()` applies the
  threshold from `docs/dipole_cnn.json` and writes `moving`, `not_moving` or `none`. The site only
  shows the text (#86). Keep it that way: a threshold in JavaScript would make the site and the
  reports disagree.
- **Version 2** (branch `ai/A9-cnn-phase1`: 4-channel stamps with a no-data mask, measured PSF,
  real bright stars as negatives, 3-seed ensemble) was not better on real data (real AUC 0.45,
  Barnard 0 of 18). It stays on the branch as a record. Do not merge it, do not delete it.
- **To call it more than experimental** it needs real movers in the training set, not injections:
  more fields with known proper-motion stars (Gaia pmra/pmdec above 0.5 ″/yr inside SPHEREx
  coverage), retrained with leave-one-target-out and judged with the six criteria of
  `training/evaluate.py`, unchanged. The training notebook and the v2 dataset are on the previous
  owner's desktop, not in the repository; `training/train.py` and `training/dataset.py` rebuild
  both from silver.
- **If IRSA stays down**, nothing here needs it: training reads silver on disk.

## 6. What is left, in order

1. **Walk [mvp_review.md](mvp_review.md) on the live site** (30 min, a browser). Tick the checklist
   per target. If a guide text no longer matches what you see, change it in `guides.py`, not in
   the JSON, and regenerate. If an image is wrong, that is data's gold or web's `app.js`.
2. **Rehearse [demo.md](demo.md)** on the live site with a timer (4 min). It is written for the
   jury in English, in the order the challenge asks.
3. **Check the site against the challenge**, now that #85–#87 restored the controls:
   - RA/Dec search: paste the challenge's WISE 0855 coordinate (`08h55m10.831s −07d14m42.53s`,
     a 2010 position). It must open `wise_0855` with the cross about 2.2′ from today's dot; a far
     point must give a clear "not published" message, not an error.
   - `?ra=&dec=&fov=` and `?target=…&detector=…&season=…&mode=difference&dipole=…` survive a reload.
   - Flagged spectrum points are hollow also when the flag is combined (`saturated,poor_fit`):
     `app.js` tests `row.flag === "poor_fit"` in one place and "any flag" in another.
   - The "Observation summary" section still comes from the retired objects/detections design;
     if it shows dashes, ask web to drop it or feed it `trajectory` (the measured shift).
   - Optional extras the challenge mentions and the site lacks: an Aladin Lite sky map, `?embed=1`,
     and an evidence section (`web/texts/evidence.json` built from `verification.json`,
     `trajectory.json` and `light_curves.json`; nothing exists yet, it is a nice-to-have).
4. **Wrap-up (D12, #35):** code frozen at `v1.0.1`; the README "Getting started" is data's. From
   AI & validation nothing else is required for the deliverable.
5. **Next season (Barnard and RW Aurigae not before about February 2027):** "last minus first"
   becomes `1-4`; `guides.py` derives seasons from gold, so run section 3's order and read the new
   texts. The last test of `test_discoveries.py` pins today's counts: update the expected numbers
   after regenerating both files.

## 7. Limits to state honestly (the jury will ask)

- Arbitrary coordinates are not live: IRSA has no CORS and a field is 4.4 MB of cutouts. The
  search works only inside the five published fields and says so. New fields need a pipeline run
  (`config/targets.yaml` + `serafim run`); a visitor cannot trigger it.
- 3–4 frames per detector in the "video": one sum per season, because a single SPHEREx frame sees
  one wavelength and consecutive frames are not comparable.
- WISE 0855 is at the limit: S/N 5–7, seen only in detector 6 (and 5 in some sums); its motion is
  0.65 px between the seasons SPHEREx has, so no dipole is visible.
- Saturated centers (Barnard, RW Aurigae, R Draconis): absolute flux is not reliable; brightness
  changes come from ring photometry, with per-detector systematics up to about 30 %.
- The CNN is experimental and is shown as such; the rule is what finds things.
- External services fail often (IRSA 502, ESA Gaia timeouts): bronze and the Gaia fixed copy
  (#73, #74) let a publish succeed without them, serving the previous frames.

## 8. Machine notes

- **CI (Linux) is the source of truth.** `ci.yml` runs `ruff`, the full `pytest` on simulated gold
  and the headless site check (`tests/web/check_site.py`).
- The previous owner's Windows PC runs Smart App Control, which blocks unsigned DLLs
  (`pyarrow`, `pytest.exe`, parts of astropy). That is why `discoveries.py` reads Parquet with
  DuckDB and why tests were run as `python -m pytest`. None of it affects Linux or Actions.
- Network caches live in `data/catalog_cache/`; `publish.yml` restores and saves them.

## 9. Decisions already taken (so nobody re-opens them)

- The CNN is out of the critical path (#18). It is a second opinion, never a filter.
- No opinion on dipoles touching saturated pixels: the stamps there are halo lobes, not sources.
- Thresholds and statuses are decided in Python, written to JSON, and only displayed by the site.
- Guides stay under 150 words, say what the visitor sees, and quote measured numbers only when the
  error is under 10 %.
- Coordinate search covers the published fields only; outside them the site says so.
- Saturated centers are painted brightest in gold (data's choice with AI's agreement); the guides
  say "saturated" wherever gold does (`verification.json` drives that sentence).
- Light curves of saturated targets by relative ring photometry; no absolute flux is claimed.
