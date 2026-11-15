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

Largest median offset: **0.033 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 77 | +0.005 | +0.004 |
| 1 | 2 | 80 | -0.019 | -0.007 |
| 1 | 3 | 77 | +0.019 | -0.001 |
| 2 | 1 | 80 | +0.014 | -0.005 |
| 2 | 2 | 80 | -0.029 | +0.012 |
| 2 | 3 | 75 | +0.023 | -0.007 |
| 3 | 1 | 75 | +0.013 | +0.010 |
| 3 | 2 | 76 | -0.023 | +0.015 |
| 3 | 3 | 74 | +0.010 | +0.003 |
| 4 | 1 | 73 | +0.013 | -0.008 |
| 4 | 2 | 71 | -0.013 | +0.007 |
| 4 | 3 | 75 | +0.013 | +0.003 |
| 5 | 1 | 69 | +0.010 | -0.016 |
| 5 | 2 | 64 | -0.017 | +0.013 |
| 5 | 3 | 67 | +0.007 | -0.012 |
| 6 | 1 | 62 | +0.018 | -0.004 |
| 6 | 2 | 58 | -0.033 | +0.008 |
| 6 | 3 | 63 | +0.017 | -0.001 |

## eps_indi_b

Largest median offset: **0.053 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 9 | +0.005 | -0.011 |
| 1 | 2 | 12 | +0.031 | -0.020 |
| 1 | 3 | 11 | -0.006 | +0.009 |
| 2 | 1 | 11 | -0.002 | -0.006 |
| 2 | 2 | 11 | +0.024 | -0.009 |
| 2 | 3 | 11 | -0.015 | -0.007 |
| 2 | 4 | 11 | +0.021 | -0.026 |
| 3 | 1 | 11 | +0.002 | +0.024 |
| 3 | 2 | 10 | +0.032 | -0.017 |
| 3 | 3 | 11 | +0.011 | -0.018 |
| 3 | 4 | 10 | +0.021 | -0.007 |
| 4 | 1 | 10 | -0.018 | +0.053 |
| 4 | 2 | 11 | +0.009 | -0.009 |
| 4 | 3 | 10 | -0.009 | +0.008 |
| 5 | 1 | 11 | -0.001 | +0.003 |
| 5 | 2 | 12 | +0.024 | -0.030 |
| 5 | 4 | 11 | +0.042 | +0.000 |
| 6 | 1 | 11 | -0.012 | +0.035 |
| 6 | 2 | 11 | +0.026 | -0.014 |
| 6 | 3 | 12 | -0.014 | +0.002 |
| 6 | 4 | 10 | +0.001 | -0.027 |

## r_dra

Largest median offset: **0.043 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 6 | +0.022 | +0.024 |
| 1 | 2 | 10 | +0.025 | -0.012 |
| 1 | 4 | 10 | +0.012 | -0.010 |
| 2 | 1 | 9 | -0.001 | +0.011 |
| 2 | 2 | 10 | -0.013 | -0.006 |
| 2 | 4 | 9 | +0.004 | +0.018 |
| 3 | 1 | 8 | -0.019 | +0.031 |
| 3 | 2 | 7 | -0.029 | -0.013 |
| 3 | 3 | 9 | -0.005 | -0.005 |
| 3 | 4 | 7 | +0.005 | +0.043 |
| 4 | 1 | 7 | +0.001 | +0.037 |
| 4 | 2 | 8 | +0.005 | -0.025 |
| 4 | 4 | 9 | +0.017 | +0.004 |
| 5 | 1 | 9 | +0.012 | +0.013 |
| 5 | 2 | 10 | -0.027 | -0.005 |
| 5 | 4 | 10 | +0.003 | +0.029 |
| 6 | 1 | 7 | -0.042 | +0.018 |
| 6 | 2 | 7 | -0.025 | +0.005 |
| 6 | 3 | 8 | -0.015 | -0.002 |
| 6 | 4 | 8 | +0.020 | +0.027 |

## rw_aur

Largest median offset: **0.089 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 64 | -0.012 | +0.005 |
| 1 | 2 | 64 | +0.008 | +0.013 |
| 1 | 3 | 64 | -0.019 | +0.009 |
| 2 | 1 | 63 | -0.010 | +0.013 |
| 2 | 2 | 59 | +0.026 | +0.000 |
| 2 | 3 | 62 | -0.012 | +0.009 |
| 3 | 1 | 65 | -0.020 | +0.003 |
| 3 | 2 | 65 | +0.025 | +0.007 |
| 3 | 3 | 62 | -0.027 | +0.002 |
| 4 | 1 | 61 | -0.015 | +0.005 |
| 4 | 2 | 61 | +0.016 | -0.011 |
| 4 | 3 | 64 | -0.005 | +0.003 |
| 5 | 1 | 56 | -0.013 | +0.007 |
| 5 | 2 | 60 | +0.018 | -0.019 |
| 5 | 3 | 59 | +0.002 | +0.027 |
| 6 | 1 | 52 | -0.034 | +0.003 |
| 6 | 2 | 51 | +0.022 | +0.002 |
| 6 | 3 | 51 | -0.089 | -0.008 |

## wise_0855

Gaia did not answer (ESA and the VizieR mirrors), so this is measured without a catalog: for every difference, the stars of the season b sum against the same stars in the season a sum (Season = a-b, offsets of b minus a). That is what a difference needs; the comparison with Gaia is still to do.

Largest median offset: **0.059 px**, which meets the goal of 0.1 px.

| Detector | Season | Stars | dx (px) | dy (px) |
| --- | --- | --- | --- | --- |
| 1 | 1-2 | 78 | -0.034 | -0.013 |
| 1 | 2-3 | 94 | +0.036 | -0.001 |
| 1 | 1-3 | 78 | -0.004 | -0.018 |
| 2 | 1-2 | 106 | -0.030 | +0.007 |
| 2 | 2-3 | 110 | +0.037 | -0.006 |
| 2 | 1-3 | 107 | -0.001 | -0.003 |
| 3 | 1-2 | 76 | -0.035 | +0.028 |
| 3 | 2-3 | 103 | +0.035 | +0.026 |
| 3 | 1-3 | 79 | -0.006 | +0.049 |
| 4 | 1-2 | 42 | -0.024 | -0.024 |
| 4 | 2-3 | 70 | +0.022 | -0.011 |
| 4 | 1-3 | 42 | +0.001 | -0.023 |
| 5 | 1-2 | 27 | -0.047 | +0.018 |
| 5 | 2-3 | 27 | +0.037 | -0.018 |
| 5 | 1-3 | 27 | -0.001 | -0.003 |
| 6 | 2-3 | 23 | +0.059 | +0.016 |
