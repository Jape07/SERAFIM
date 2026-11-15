# Model card: SERAFIM motion-dipole CNN

**Status: experimental.** It meets 3 of its 6 acceptance criteria, so it never changes a status in
the pipeline. The classic rule ([docs/dipoles.md](dipoles.md)) decides; the CNN gives a second
opinion. Full evaluation: [docs/dipole_cnn.md](dipole_cnn.md).

## What it does

Given two season sums of SPHEREx around one place, it returns `p_dipole`, the probability that
something moved there between the seasons. A mover leaves a dipole in the difference: negative where
it was, positive where it went.

| | |
| --- | --- |
| Input | 3 × 32 × 32 stamp: sum of season A, sum of season B and B − A, each over its measured noise through asinh ([serafim/ai/stamps.py](../serafim/ai/stamps.py)) |
| Output | `p_dipole` in [0, 1]; decision threshold 0.731, the same false-alarm rate as the rule (2.0 %) |
| Network | 3 convolution blocks (16, 32, 64 filters) with BatchNorm, global pooling, dense 32, dropout 0.3: 25,921 parameters |
| File | [models/dipole_cnn.onnx](../models/dipole_cnn.onnx), 103 kB, run with onnxruntime ([serafim/ai/predict.py](../serafim/ai/predict.py)) |

## Intended use, and what is out of scope

- **For:** scoring dipoles the rule already found in sources that are **not saturated**, as an extra
  opinion next to them ([docs/dipoles.md](dipoles.md)).
- **Not for:**
  - **saturated stars**: it cannot tell one that moves from one that does not;
  - **scanning images blindly**: at a realistic prevalence most of its alarms would be false;
  - **announcing new objects**: anything it flags is at most an "Unidentified candidate" for a
    person to check.

## Training data

- 8,346 stamps cut from the real silver sums of the five challenge targets (93 differences between
  seasons).
- **Positives (3,720) are injected:** a point source moved 0.3–3 px between the seasons, at peak S/N
  3–300, on real sky. Some are bright enough to saturate. Only three objects in these fields really
  move.
- **Negatives (4,626), several of them real:**
  - real sky and real fixed stars, untouched;
  - injected brightness changes;
  - single-pixel artifacts;
  - saturated stars that do not move;
  - the cores of RW Aurigae and R Draconis.
- Split by target, never at random: each model is judged on the field it never saw.

## Results

| | Injected stamps (out of fold) | Real targets |
| --- | --- | --- |
| ROC AUC | 0.910 | 0.52 (movers vs still saturated stars) |
| Sensitivity at 2 % false alarms | 62 % (rule: 26 %) | 21 of 49 (rule: 23) |
| False alarms on still saturated stars | — | 9 of 38 = 24 % (rule: 0) |
| Calibration (ECE) | 0.022 | too few to measure |

- **Where it helps:**
  - Epsilon Indi B (not saturated) is found in 12 of 15 differences, against 8 for the rule.
  - It gives low scores to the two single-pixel candidates the rule left.
- **Where it fails:**
  - Barnard's Star is found in 9 of 18 differences (the rule finds it in 15);
  - RW Aurigae and R Draconis are called movers in 9 of 38 differences;
  - WISE 0855 is too faint for it, and for the rule.

## Limitations

- **Its positives are simulated:** a point source on real sky is not a real mover.
- **The real test is small:** 3 moving objects and 2 still ones, so the intervals are wide.
- **More data would still help:** its learning curve has not flattened.
- **Version 2 is not better on real data.** It adds a no-data channel, the PSF measured on real
  stars, 560 real saturated stars as negatives and an ensemble of three networks. It cut the
  false alarms on still saturated stars to 16 %, but stopped finding Barnard's Star (real AUC
  0.45). It stays on the branch `ai/A9-cnn-phase1`.

## How to reproduce

- **With the scripts:** see [training/RESULTS.md](../training/RESULTS.md). It runs on CPU in about
  15 minutes and needs silver on disk.
- **With the notebook:** "Notebook 1" uses the same code and the same seed, runs on CPU or a Colab
  GPU, and fills a results board at the top.

## Next step

Train on **real** movers: dozens of SPHEREx fields around stars with a proper motion of at least
1.2 arcseconds per year, and bright still stars as negatives. Then judge the model again on sealed
fields against the same acceptance criteria.
