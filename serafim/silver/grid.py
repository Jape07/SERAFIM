"""The pixel grid of a target: where every silver array lives.

There is one grid per target, centered on its position today (`ra_now`, `dec_now`), with one
SPHEREx pixel (6.15 arcsec) per pixel and `settings.cutout_size_deg` on a side (146 x 146 px
for 0.25 deg). Every frame, sum and difference of the target is reprojected to it, and its
WCS is what `targets.parquet` calls `wcs_json`, so silver and gold share one grid.

Orientation of every silver array (agreed with AI & validation):

    array[y, x], indices from 0, row 0 = south (the FITS convention), so
    WCS(json.loads(wcs_json)).pixel_to_world(x, y) is the position of pixel (x, y).

North is up when the array is drawn with the origin at the bottom; the PNGs are flipped when
they are written, to have north up and the origin at the top-left (see serafim/contract.py).
"""

from __future__ import annotations

import json

from astropy.io import fits
from astropy.wcs import WCS

from serafim.validation.targets import PIXEL_SCALE_ARCSEC


def grid_size(size_deg: float) -> int:
    """Pixels on a side of the grid for a cutout of `size_deg`."""
    return int(round(size_deg * 3600 / PIXEL_SCALE_ARCSEC))


def target_wcs(ra: float, dec: float, size_deg: float) -> WCS:
    """TAN grid centered on (ra, dec) in degrees, one SPHEREx pixel per pixel, north up."""
    n = grid_size(size_deg)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600, PIXEL_SCALE_ARCSEC / 3600]
    wcs.pixel_shape = (n, n)
    return wcs


def wcs_to_json(wcs: WCS) -> str:
    """The text that goes in `targets.parquet` (`wcs_json`) and in every silver `.npz`."""
    header = wcs.to_header()
    values = {key: header[key] for key in header}
    values["NAXIS1"], values["NAXIS2"] = wcs.pixel_shape
    return json.dumps(values, sort_keys=True)


def wcs_from_json(text: str) -> WCS:
    """The WCS back from `wcs_json`, with its `pixel_shape`."""
    return WCS(fits.Header(json.loads(text)))
