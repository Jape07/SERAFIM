"""Real-vs-artifact CNN inference with onnxruntime (task A10, AI & validation).

The model is trained with PyTorch in training/ and exported to
models/cnn_real_artifact.onnx (< 1 MB). If the CNN does not beat the classic rule
in task A7, p_real is shown on the website but does not change any status.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/cnn_real_artifact.onnx")
CUTOUT_SIZE = 32
CHANNELS = ("new", "reference", "difference")


def predict(cutouts: np.ndarray, model_path: Path = MODEL_PATH) -> np.ndarray:
    """Probability that each change is real.

    Args:
        cutouts: float32 array of shape (N, 3, 32, 32), channels in CHANNELS order.
        model_path: ONNX model to run.

    Returns:
        Array of shape (N,) with p_real in [0, 1].
    """
    raise NotImplementedError("Task A10")
