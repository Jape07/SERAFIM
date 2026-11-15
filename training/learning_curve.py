"""Does the dipole CNN have enough data? A learning curve (task A9, stage 2).

    python -m training.learning_curve     # docs/dipole_cnn_learning.json, docs/img/

Trains on 1/8, 1/4, 1/2 and all of the training differences of four targets (whole differences
are kept together: their stamps share a background) and scores the fifth, for two held-out
targets. If the out-of-fold AUC still climbs at the full size, more data would help; if it is
flat, the network has learned what these stamps can teach. Also the gap between the AUC on the
stamps it trained on and on the stamps it never saw (a large gap is overfitting).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from training import train
from training.metrics import roc_auc

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs" / "dipole_cnn_learning.json"
IMAGE_PATH = ROOT / "docs" / "img" / "dipole_cnn_learning.png"
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
HELD_OUT = ("eps_indi_b", "rw_aur")  # a sparse field with a mover, a crowded one with a still star
SEED = 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    found, labels, _, _ = train.load()
    y = labels["label"].to_numpy()
    cell = (labels["target_id"] + "/" + labels["detector"].astype(str) + "/"
            + labels["season_a"].astype(str) + "/" + labels["season_b"].astype(str))  # fmt: skip
    rng = np.random.default_rng(SEED)
    rows = []
    for target_id in HELD_OUT:
        held = (labels["target_id"] == target_id).to_numpy()
        cells = cell[~held].unique()
        rng.shuffle(cells)
        for fraction in FRACTIONS:
            chosen = set(cells[: max(1, int(round(fraction * len(cells))))])
            use = (~held) & cell.isin(chosen).to_numpy()
            model = train.fit(found[use], y[use])
            seen = roc_auc(y[use], train.scores(model, found[use]))
            unseen = roc_auc(y[held], train.scores(model, found[held]))
            rows.append({"held_out": target_id, "fraction": fraction,
                         "differences": len(chosen), "stamps": int(use.sum()),
                         "auc_trained_on": round(seen, 4),
                         "auc_held_out": round(unseen, 4)})  # fmt: skip
            log.info("%s, %.0f%% of the data (%d stamps): AUC %.3f on what it saw, %.3f held out",
                     target_id, 100 * fraction, use.sum(), seen, unseen)  # fmt: skip
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    plot(rows, IMAGE_PATH)
    return 0


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for target_id, color in zip(HELD_OUT, ("#0B3D91", "#FC3D21"), strict=True):
        mine = [r for r in rows if r["held_out"] == target_id]
        stamps = [r["stamps"] for r in mine]
        ax.plot(stamps, [r["auc_held_out"] for r in mine], "o-", color=color,
                label=f"{target_id} left out")  # fmt: skip
        ax.plot(stamps, [r["auc_trained_on"] for r in mine], "o--", color=color, alpha=0.5,
                label=f"{target_id}: on its training stamps")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("training stamps")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("Learning curve of the dipole CNN", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
