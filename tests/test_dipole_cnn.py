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
    assert channels.shape == (4, 64, 64) and not bad.any()
    stamp = stamps.cut(channels, 31.75, 32.0)
    assert stamp.shape == (4, 32, 32) and stamp.dtype == np.float32
    assert stamp[3].sum() == 0  # every pixel has data
    difference = stamp[2]
    assert difference.min() < -3 and difference.max() > 3  # a dipole, in asinh(sigma) units
    assert np.argmin(difference) % 32 < np.argmax(difference) % 32  # negative west of positive
    assert abs(float(np.median(stamp[0]))) < 0.5  # background subtracted


def test_saturated_and_masked_pixels_are_zero_and_flagged():
    """Version 2: a saturated core is 0 in the images and 1 in the no-data channel, so it no
    longer looks like empty sky."""
    sum_a, sum_b = sums()
    sum_a["n_saturated"][32, 31] = 3
    sum_b["mask"][10, 10] = True
    channels, bad = stamps.cube_from_sums(sum_a, sum_b)
    assert bad[32, 31] and bad[10, 10]
    for y, x in ((32, 31), (10, 10)):
        assert (channels[:3, y, x] == 0).all() and channels[3, y, x] == 1


def test_beyond_the_edge_is_no_data():
    channels = np.ones((4, 40, 40), np.float32)
    channels[3] = 0  # every pixel of the grid has data
    stamp = stamps.cut(channels, 2, 2)
    assert (stamp[:3, 0, 0] == 0).all() and stamp[3, 0, 0] == 1  # outside the grid
    assert (stamp[:3, 31, 31] == 1).all() and stamp[3, 31, 31] == 0  # inside


@pytest.mark.skipif(not predict.available(), reason="no models/dipole_cnn.onnx")
def test_the_onnx_model_scores_a_moving_star_above_a_still_one():
    moving = stamps.cut(stamps.cube_from_sums(*sums(shift=(1.5, 0.5)))[0], 31.75, 32.25)
    still = stamps.cut(stamps.cube_from_sums(*sums(shift=(0.0, 0.0)))[0], 31.0, 32.0)
    p = predict.predict(np.stack([moving, still]))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
    assert p[0] > 0.5 > p[1]


def test_a_model_of_another_version_is_not_used(tmp_path):
    """A 3-channel model (version 1) with 4-channel stamps: not available, never a crash."""
    torch = pytest.importorskip("torch")
    from training.model import DipoleCNN

    path = tmp_path / "old.onnx"
    torch.onnx.export(DipoleCNN(channels=3).eval(), torch.zeros(1, 3, 32, 32), str(path),
                      input_names=["stamps"], dynamo=False)  # fmt: skip
    assert not predict.available(path)
    assert not predict.available(tmp_path / "missing.onnx")


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
    assert model(torch.zeros(4, len(stamps.CHANNELS), 32, 32)).shape == (4,)


def test_an_ensemble_averages_its_members():
    torch = pytest.importorskip("torch")
    from training.model import DipoleCNN, Ensemble

    members = [DipoleCNN().eval() for _ in range(3)]
    x = torch.randn(2, len(stamps.CHANNELS), 32, 32)
    expected = torch.stack([m(x) for m in members]).mean(0)
    assert torch.allclose(Ensemble(members).eval()(x), expected)


def test_an_injected_dipole_moves_the_star_between_the_sums():
    pytest.importorskip("torch")
    from training import dataset

    profile = dataset.point((40, 40), 20.0, 20.0, 6)  # the fallback PSF: none measured here
    assert profile.sum() == pytest.approx(1.0, abs=0.01)
    wider = dataset.point((40, 40), 20.0, 20.0, 6, scale=1.1)
    assert wider.sum() == pytest.approx(1.0, abs=0.01) and wider.max() < profile.max()
    bad = dataset.saturate(np.zeros((40, 40), bool), 1000 * profile, level=50.0)
    assert bad[20, 20] and not bad[0, 0]  # the core is masked, the wings are not
    rng = np.random.default_rng(0)
    places = dataset.free_places(np.zeros((80, 80), bool), [(40.0, 40.0)], rng, 20)
    assert len(places) == 20
    assert (np.hypot(places[:, 0] - 40, places[:, 1] - 40) >= dataset.MOVER_PX).all()
    assert (places >= stamps.HALF).all() and (places < 80 - stamps.HALF).all()
