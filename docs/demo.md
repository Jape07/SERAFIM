# Live demo script (4 minutes)

For whoever presents SERAFIM to the jury. The five targets take **2 minutes**: rehearse that part
with a timer. Every number here comes from the reports on the published site
([verification](verification.md), [review of the site](mvp_review.md)).

## Before you start

- Open these six tabs and hard-refresh each one (Ctrl+F5):
  1. [Home](https://jape07.github.io/SERAFIM/)
  2. [Barnard's Star](https://jape07.github.io/SERAFIM/index.html?target=barnard&detector=5&season=1-3&mode=difference&layers=gaia,catwise,simbad,skybot#visor)
  3. [Epsilon Indi B](https://jape07.github.io/SERAFIM/index.html?target=eps_indi_b&detector=4&season=0&mode=all_seasons&layers=gaia,catwise,simbad,skybot#visor)
  4. [WISE 0855](https://jape07.github.io/SERAFIM/index.html?target=wise_0855&detector=6&season=0&mode=all_seasons&layers=gaia,catwise,simbad,skybot#visor)
  5. [RW Aurigae](https://jape07.github.io/SERAFIM/index.html?target=rw_aur&detector=1&season=1-3&mode=difference&layers=gaia,catwise,simbad,skybot#visor)
  6. [R Draconis](https://jape07.github.io/SERAFIM/index.html?target=r_dra&detector=6&season=1&mode=sum&layers=gaia,catwise,simbad,skybot#visor)
- Wait until each tab shows its image (the tables load once, in a few seconds).
- **Plan B without a network:** run the site from a local copy of gold
  (`cp -r data/gold web/gold`, then `python -m http.server 8000 -d web`), or show the images in
  [docs/img](img).

## 0:00–0:30 · What SERAFIM is (Home tab)

> "SPHEREx maps the whole sky in 102 infrared colors every six months. SERAFIM turns those public
> images into a time-lapse you can explore in the browser: pick a target, a detector and a season,
> and see what moved or changed. Everything runs as a static site; a Python pipeline prepares the
> images and the tables. We tested it on the five targets of the challenge."

## 0:30–2:30 · The five targets (about 25 seconds each)

**Barnard's Star** (tab 2, difference of season 3 minus season 1, detector 5)

> "The fastest star in our sky. Blue is where it was, red is where it went: about 1.7 pixels in a
> year. We measured 10.45 arcseconds per year; the catalogs say 10.40."

**Epsilon Indi B** (tab 3, all seasons, detector 4; then switch to detector 5)

> "Two brown dwarfs. Compare them with the star up and to the left: in detector 4 they are about as
> bright as it, in detector 5 about five times brighter. That dip is methane in their atmosphere,
> and SPHEREx sees it."

**WISE 0855** (tab 4, all seasons, detector 6; zoom in on the center)

> "The coldest brown dwarf known. It only shows in detector 6, as a faint dot right where its
> catalog motion puts it. In the bluer detectors the dot you see is a background star it is
> passing, and we model both."

**RW Aurigae** (tab 5, difference of season 3 minus season 1, detector 1)

> "A young star that dims and brightens. It does not move, so a difference should erase it; instead
> you see a red ring: it was 1.5 times brighter in September 2026 than a year before."

**R Draconis** (tab 6, detector 6, step through the four seasons with the slider)

> "A pulsating giant. Watch its glow shrink from season 1 to season 4. From SPHEREx alone we find a
> period of 244 days; the catalogs say 247."

## 2:30–3:30 · How we know it is right

> "We do not trust the site because it looks good. A verification runs on the gold the site
> actually serves: the data contract, whether each target is seen, saturation, the measured motion
> against the catalogs, the spectra and the light curves. It has no failures. We also put every
> catalog marker on the real images: none is out of place. And we say what does not work: saturated
> stars give no reliable spectrum, so those points are hollow, and WISE 0855 is too faint to follow
> season by season."

Show the markers if asked: the Gaia, CatWISE and SIMBAD layers in any tab.

## 3:30–4:00 · The AI

> "SPHEREx can also discover things. A classic rule looks for motion dipoles in every difference
> and checks them against the catalogs: without being told where to look, it finds Barnard's Star
> in all six detectors. We also trained a small neural network and measured it honestly: on
> simulated movers it finds more than twice as many as the rule, but on real saturated stars it is
> not reliable yet, so it only gives a second opinion. Its next step is training on real moving
> stars."

## Likely questions

- **Why only three or four images per detector?** SPHEREx passes over each point about every six
  months. We add the frames of each season, which is also what makes faint objects visible.
- **Why not any coordinate, live?** IRSA does not let a browser download its images directly (no
  CORS), so the pipeline prepares each target beforehand. Adding a new target by its coordinates
  through the pipeline is our next step.
- **What is a dipole?** In a difference, a moving object leaves a negative spot where it was and a
  positive one where it went.
- **How do you know the motion is real?** We fit the target's position in every season and compare
  it with the catalog motion, parallax included: Barnard's Star 10.45 ± 0.38 against 10.40
  arcseconds per year, every season within 0.17 pixels of its catalog position.
- **Is the neural network reliable?** Not yet, and we say so: it meets 3 of its 6 acceptance
  criteria. See [the model card](model_card.md).
- **What would you do next?** Train the network on real moving stars, and publish any coordinate a
  visitor asks for through the pipeline.
