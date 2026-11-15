"""The dipole CNN (task A9, stage 2): its stamps, its ONNX model and, if torch is installed, its
training code. CI has no torch: the training tests are skipped there; the stamps and the ONNX
inference (onnxruntime, a base dependency) always run."""

import numpy as np
import pytest

from serafim.ai import predict, stamps


def sums(shift=(1.5, 0.0), flux=60.0, shape=(64, 64), seed=4):
    """Two season sums (silver dicts) with one star that moved by `shift`."""
    rng = np.random.default_rng(seed)
    rows, cols = np.indices(shape)

    def star(x, y):
        return flux / (2 * np.pi * 0.64) * np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / 1.28)

    out = []
    for x, y in ((31.0, 32.0), (31.0 + shift[0], 32.0 + shift[1])):
        image = 5.0 + rng.normal(0, 0.1, shape) + star(x, y)
        out.append({"image": image, "mask": np.zeros(shape, bool),
                    "n_saturated": np.zeros(shape, np.int16)})  # fmt: skip
    return out


def test_the_stamp_has_the_two_seasons_and_their_difference():
    sum_a, sum_b = sums()
    channels, bad = stamps.cube_from_sums(sum_a, sum_b)
    assert channels.shape == (3, 64, 64) and not bad.any()
    stamp = stamps.cut(channels, 31.75, 32.0)
    assert stamp.shape == (3, 32, 32) and stamp.dtype == np.float32
    difference = stamp[2]
    assert difference.min() < -3 and difference.max() > 3  # a dipole, in asinh(sigma) units
    assert np.argmin(difference) % 32 < np.argmax(difference) % 32  # negative west of positive
    assert abs(float(np.median(stamp[0]))) < 0.5  # background subtracted


def test_saturated_and_masked_pixels_are_zero():
    sum_a, sum_b = sums()
    sum_a["n_saturated"][32, 31] = 3
    sum_b["mask"][10, 10] = True
    channels, bad = stamps.cube_from_sums(sum_a, sum_b)
    assert bad[32, 31] and bad[10, 10]
    assert (channels[:, 32, 31] == 0).all() and (channels[:, 10, 10] == 0).all()


def test_a_stamp_at_the_edge_is_padded_with_zeros():
    channels = np.ones((3, 40, 40), np.float32)
    stamp = stamps.cut(channels, 2, 2)
    assert stamp[:, 0, 0].sum() == 0 and stamp[:, 31, 31].sum() == 3


@pytest.mark.skipif(not predict.available(), reason="no models/dipole_cnn.onnx")
def test_the_onnx_model_scores_a_moving_star_above_a_still_one():
    moving = stamps.cut(stamps.cube_from_sums(*sums(shift=(1.5, 0.5)))[0], 31.75, 32.25)
    still = stamps.cut(stamps.cube_from_sums(*sums(shift=(0.0, 0.0)))[0], 31.0, 32.0)
    p = predict.predict(np.stack([moving, still]))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
    assert p[0] > 0.5 > p[1]


def test_predict_checks_the_shape():
    with pytest.raises(ValueError):
        predict.predict(np.zeros((2, 3, 16, 16), np.float32))


def test_the_model_is_small():
    if not predict.available():
        pytest.skip("no models/dipole_cnn.onnx")
    assert predict.MODEL_PATH.stat().st_size < 1_000_000


def test_the_network_has_about_26k_parameters():
    torch = pytest.importorskip("torch")
    from training.model import DipoleCNN, parameters

    model = DipoleCNN()
    assert 20_000 < parameters(model) < 30_000
    assert model(torch.zeros(4, 3, 32, 32)).shape == (4,)


def test_an_injected_dipole_moves_the_star_between_the_sums():
    pytest.importorskip("torch")
    from training import dataset

    profile = dataset.point((40, 40), 20.0, 20.0, 0.75)
    assert profile.sum() == pytest.approx(1.0, abs=0.01)
    bad = dataset.saturate(np.zeros((40, 40), bool), 1000 * profile, level=50.0)
    assert bad[20, 20] and not bad[0, 0]  # the core is masked, the wings are not
    rng = np.random.default_rng(0)
    places = dataset.free_places(np.zeros((80, 80), bool), [(40.0, 40.0)], rng, 20)
    assert len(places) == 20
    assert (np.hypot(places[:, 0] - 40, places[:, 1] - 40) >= dataset.MOVER_PX).all()
    assert (places >= stamps.HALF).all() and (places < 80 - stamps.HALF).all()
