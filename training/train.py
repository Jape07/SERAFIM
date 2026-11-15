"""Train the motion-dipole CNN on CPU (task A9, stage 2).

    python -m training.train      # data/dataset/oof.parquet, models/dipole_cnn.onnx

1. Leave one target out: five trainings, each on the stamps of four targets, each predicting the
   stamps of the fifth and its real test stamps (training never sees the field it is judged on;
   neighbor stamps overlap, so a random split would cheat). The out-of-fold predictions go to
   data/dataset/oof.parquet for training/evaluate.py.
2. The final model, on the stamps of the five targets, exported to models/dipole_cnn.onnx and
   checked against onnxruntime.

Augmentation: the 8 rotations and flips of each stamp (a dipole has no preferred direction).
Fixed seed. Appends a row to training/experiments.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from training.dataset import OUT_DIR
from training.model import DipoleCNN, parameters

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "dipole_cnn.onnx"
EXPERIMENTS = ROOT / "training" / "experiments.csv"
CONFIG = {"lr": 2e-3, "base": 16, "dropout": 0.3, "batch": 128, "epochs": 15, "seed": 0,
          "weight_decay": 1e-4}  # fmt: skip


def load(folder: Path = OUT_DIR) -> tuple:
    found = np.load(folder / "dipoles.npz")["stamps"].astype(np.float32)
    labels = pd.read_parquet(folder / "labels.parquet")
    real = np.load(folder / "real.npz")["stamps"].astype(np.float32)
    return found, labels, real, pd.read_parquet(folder / "real.parquet")


def dihedral(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One random rotation by 90 degrees and maybe a flip, per batch."""
    k = int(torch.randint(4, (1,), generator=generator))
    batch = torch.rot90(batch, k, dims=(2, 3))
    if int(torch.randint(2, (1,), generator=generator)):
        batch = torch.flip(batch, dims=(3,))
    return batch


def fit(x: np.ndarray, y: np.ndarray, config: dict = CONFIG) -> DipoleCNN:
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = DipoleCNN(base=config["base"], dropout=config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])  # fmt: skip
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    loss_of = nn.BCEWithLogitsLoss()
    data = torch.from_numpy(x)
    target = torch.from_numpy(y.astype(np.float32))
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(data), generator=generator)
        total = 0.0
        for start in range(0, len(data), config["batch"]):
            pick = order[start : start + config["batch"]]
            batch = dihedral(data[pick], generator)
            optimizer.zero_grad()
            loss = loss_of(model(batch), target[pick])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(pick)
        schedule.step()
        log.debug("epoch %d loss %.4f", epoch + 1, total / len(data))
    model.eval()
    return model


def scores(model: DipoleCNN, x: np.ndarray) -> np.ndarray:
    """p_dipole, averaged over the 8 rotations and flips (test-time augmentation)."""
    with torch.no_grad():
        data = torch.from_numpy(np.asarray(x, np.float32))
        total = torch.zeros(len(data))
        for k in range(4):
            for flip in (False, True):
                view = torch.rot90(data, k, dims=(2, 3))
                view = torch.flip(view, dims=(3,)) if flip else view
                total += torch.sigmoid(model(view))
    return (total / 8).numpy()


def leave_one_target_out(found, labels, real, real_labels, config: dict = CONFIG) -> tuple:
    """Out-of-fold p_dipole for the synthetic and the real stamps."""
    oof = np.full(len(labels), np.nan)
    real_oof = np.full(len(real_labels), np.nan)
    for target_id in sorted(labels["target_id"].unique()):
        held = (labels["target_id"] == target_id).to_numpy()
        model = fit(found[~held], labels["label"].to_numpy()[~held], config)
        oof[held] = scores(model, found[held])
        mine = (real_labels["target_id"] == target_id).to_numpy()
        if mine.any():
            real_oof[mine] = scores(model, real[mine])
        log.info("%s held out: %d stamps scored", target_id, int(held.sum()))
    return oof, real_oof


def export(model: DipoleCNN, path: Path = MODEL_PATH) -> None:
    """ONNX with a dynamic batch, checked against onnxruntime."""
    import onnxruntime

    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(2, 3, 32, 32)
    torch.onnx.export(model, example, str(path), input_names=["stamps"], output_names=["logit"],
                      dynamic_axes={"stamps": {0: "n"}, "logit": {0: "n"}}, opset_version=17,
                      dynamo=False)  # fmt: skip
    check = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        expected = model(check).numpy()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"stamps": check.numpy()})[0].reshape(-1)
    if not np.allclose(expected, got, atol=1e-4):
        raise RuntimeError("the ONNX model does not give what PyTorch gives")
    log.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1024)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    keep = np.isfinite(p)
    return float(roc_auc_score(y[keep], p[keep]))


def record(config: dict, value: float, notes: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True, cwd=ROOT).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    with EXPERIMENTS.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow([
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), commit, "dipole_cnn", config["lr"],
            config["base"], config["dropout"], config["batch"], config["epochs"], config["seed"],
            round(value, 4), notes,
        ])  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.train")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--no-export", action="store_true", help="Only the out-of-fold run.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    config = {**CONFIG, "epochs": args.epochs}

    found, labels, real, real_labels = load()
    log.info("%d stamps (%d dipoles), %d real; model with %d parameters", len(labels),
             int(labels["label"].sum()), len(real_labels),
             parameters(DipoleCNN(config["base"], config["dropout"])))  # fmt: skip
    oof, real_oof = leave_one_target_out(found, labels, real, real_labels, config)
    labels.assign(p_dipole=oof).to_parquet(OUT_DIR / "oof.parquet", index=False)
    real_labels.assign(p_dipole=real_oof).to_parquet(OUT_DIR / "real_oof.parquet", index=False)
    value = auc(labels["label"].to_numpy(), oof)
    log.info("leave-one-target-out AUC: %.4f", value)
    record(config, value, "leave-one-target-out AUC on injected dipoles and hard negatives")
    if not args.no_export:
        export(fit(found, labels["label"].to_numpy(), config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
