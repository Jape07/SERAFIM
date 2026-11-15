"""Dipole CNN inference with onnxruntime (task A9, stage 2).

The model is trained with PyTorch in training/ (outside GitHub, on CPU in minutes) and exported
to models/dipole_cnn.onnx (< 1 MB). The pipeline only needs onnxruntime. Stamps come from
serafim/ai/stamps.py, the same functions training used.

    from serafim.ai.predict import predict, available
    p = predict(stamps)   # stamps: float32 (N, 3, 32, 32) -> p_dipole in [0, 1]
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from serafim.ai.stamps import CHANNELS, STAMP

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "dipole_cnn.onnx"


def available(model_path: Path = MODEL_PATH) -> bool:
    return Path(model_path).is_file()


@cache
def _session(model_path: str):
    import onnxruntime

    return onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict(stamps: np.ndarray, model_path: Path = MODEL_PATH, batch: int = 512) -> np.ndarray:
    """Probability that each stamp has a motion dipole at its center.

    Args:
        stamps: float32 array of shape (N, 3, 32, 32), channels in CHANNELS order.
        model_path: ONNX model to run.

    Returns:
        Array of shape (N,) with p_dipole in [0, 1].
    """
    stamps = np.asarray(stamps, np.float32)
    if stamps.ndim != 4 or stamps.shape[1:] != (len(CHANNELS), STAMP, STAMP):
        raise ValueError(
            f"stamps must be (N, {len(CHANNELS)}, {STAMP}, {STAMP}), not {stamps.shape}"
        )
    session = _session(str(model_path))
    name = session.get_inputs()[0].name
    out = [
        session.run(None, {name: stamps[i : i + batch]})[0] for i in range(0, len(stamps), batch)
    ]
    logits = np.concatenate(out).reshape(-1) if out else np.zeros(0, np.float32)
    return 1.0 / (1.0 + np.exp(-logits))
