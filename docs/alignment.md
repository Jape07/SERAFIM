# Alignment of the silver sums

Written by `python -m serafim alignment`; do not edit by hand.

For each sum of one detector and one season, the stars found in the sum (sep,
8 times the measured background rms, windowed centroids) are matched within
1.5 px to Gaia stars with G between 10 and 15, moved to the
mean date of the sum with
`serafim.validation.layers.move()`. dx and dy are the median of (found - Gaia) in pixels of the
target grid (6.15 arcsec), x to the west and y to the north. Stars within
8 px of the target are left out. The goal (agreed with AI & validation) is
under 0.1 px: above that, differences between seasons fake dipoles on fixed stars.

## barnard

Largest median offset: **0.031 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 77 | +0.004 | +0.005 |
| 1 | 2 | 79 | -0.019 | -0.006 |
| 1 | 3 | 77 | +0.018 | -0.001 |
| 2 | 1 | 80 | +0.013 | -0.005 |
| 2 | 2 | 80 | -0.029 | +0.012 |
| 2 | 3 | 76 | +0.022 | -0.009 |
| 3 | 1 | 75 | +0.014 | +0.011 |
| 3 | 2 | 76 | -0.023 | +0.015 |
| 3 | 3 | 74 | +0.009 | +0.006 |
| 4 | 1 | 74 | +0.010 | -0.009 |
| 4 | 2 | 71 | -0.016 | +0.004 |
| 4 | 3 | 75 | +0.015 | +0.002 |
| 5 | 1 | 68 | +0.012 | -0.018 |
| 5 | 2 | 66 | -0.017 | +0.013 |
| 5 | 3 | 66 | +0.006 | -0.012 |
| 6 | 1 | 61 | +0.017 | -0.004 |
| 6 | 2 | 57 | -0.031 | +0.006 |
| 6 | 3 | 61 | +0.021 | -0.001 |

## wise_0855

Gaia did not answer (ESA and the VizieR mirrors), so this is measured without a catalog: for every difference, the stars of the season b sum against the same stars in the season a sum (Season = a-b, offsets of b minus a). That is what a difference needs; the comparison with Gaia is still to do.

Largest median offset: **0.066 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1-2 | 72 | -0.036 | -0.028 |
| 1 | 2-3 | 91 | +0.037 | -0.003 |
| 1 | 1-3 | 72 | +0.006 | -0.018 |
| 2 | 1-2 | 103 | -0.032 | +0.005 |
| 2 | 2-3 | 102 | +0.037 | -0.001 |
| 2 | 1-3 | 104 | -0.003 | -0.002 |
| 3 | 1-2 | 78 | -0.037 | +0.022 |
| 3 | 2-3 | 101 | +0.034 | +0.026 |
| 3 | 1-3 | 80 | -0.004 | +0.046 |
| 4 | 1-2 | 42 | -0.021 | -0.022 |
| 4 | 2-3 | 69 | +0.022 | -0.010 |
| 4 | 1-3 | 42 | +0.001 | -0.021 |
| 5 | 1-2 | 27 | -0.051 | +0.028 |
| 5 | 2-3 | 27 | +0.040 | -0.020 |
| 5 | 1-3 | 27 | -0.011 | -0.003 |
| 6 | 2-3 | 24 | +0.066 | +0.023 |
