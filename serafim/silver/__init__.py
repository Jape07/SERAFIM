"""Silver layer (data engineering).

Physical values (MJy/sr), never stretched: the stretches only go in the PNGs. Built from
bronze in three steps, each one reading the one before:

- Frames (task D4): IMAGE - ZODI, a mask from the FLAGS bit names, and the frame reprojected to
  the grid of its target (grid.py), with detector, wavelength at the target, mean date and season.
- Sums (task D5): per (detector, season), plus the white image and all seasons, weighted by
  inverse variance. The alignment between seasons is measured here.
- Differences (task D6): season b minus season a (differences.py).

Layout (agreed with AI & validation, who reads it for the spectra, the trajectory and the CNN):

    data/silver/<target>/frames/<frame_id>.npz        frame_id = contract.frame_id(...)
    data/silver/<target>/sums/d<detector>_s<season>.npz
    data/silver/<target>/differences/d<detector>_s<a>-s<b>.npz

Every `.npz` has only numpy arrays (float32, bool, int16) and the WCS as text in `wcs_json`,
so it opens with `numpy.load(path)` anywhere, without `serafim` and without pickle. Arrays are
`array[y, x]` with row 0 = south (see grid.py).
"""
