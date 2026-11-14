"""Silver layer (data engineering).

- Clean and align (task D3): subtract ZODI, build the mask from FLAGS, compute
  wavelength and bandwidth, reproject to the region grid.
- Difference images (task D5): new minus a median reference. Exposes
  `subtract()` and `cutout(frame_id, x, y, size=32)`, which AI & validation uses
  to build the training dataset.
- Detection (task D6): `sep.extract` on difference / error, 5 sigma, both signs.

Output: data/silver/<region>/<frame_id>.npz, data/silver/frames.parquet,
data/silver/detections.parquet.
"""
