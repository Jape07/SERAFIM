"""Build the cutout dataset (task A4).

Inject fake sources with the archive PSF (EPSF extension) into silver frames,
redo the subtraction with silver's `subtract()` and cut 32x32x3 stamps
(new, reference, difference).

Positives: injected sources and detections confirmed by SkyBoT.
Negatives: detections on flagged pixels, residuals of bright Gaia stars, edges.
Train / validation / test are split by sky area, never at random.

Output: data/dataset/cutouts.npz and data/dataset/labels.parquet.
"""


def main() -> None:
    raise NotImplementedError("Task A4")


if __name__ == "__main__":
    main()
