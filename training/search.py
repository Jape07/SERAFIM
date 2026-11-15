"""Optional hyperparameter search of the dipole CNN with Optuna (task A9, stage 2).

    python -m training.search [--trials 12]

Learning rate (1e-4 to 1e-2), base filters (8, 16 or 32) and dropout (0 to 0.5), judged by the
leave-one-target-out AUC with fewer epochs (two held-out targets per trial, to stay in minutes).
The best configuration is appended to training/experiments.csv; copy it into train.CONFIG by
hand if it is clearly better. Never needed by the pipeline.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from training import train

log = logging.getLogger(__name__)

HELD_OUT = ("barnard", "wise_0855")  # one crowded field and one sparse one
EPOCHS = 6


def main(argv: list[str] | None = None) -> int:
    import optuna

    parser = argparse.ArgumentParser(prog="python -m training.search")
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    found, labels, _, _ = train.load()
    y = labels["label"].to_numpy()

    def objective(trial) -> float:
        config = {**train.CONFIG, "epochs": EPOCHS,
                  "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  "base": trial.suggest_categorical("base", [8, 16, 32]),
                  "dropout": trial.suggest_float("dropout", 0.0, 0.5)}  # fmt: skip
        values = []
        for target_id in HELD_OUT:
            held = (labels["target_id"] == target_id).to_numpy()
            model = train.fit(found[~held], y[~held], config)
            values.append(train.auc(y[held], train.scores(model, found[held])))
        return float(np.mean(values))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    best = {**train.CONFIG, "epochs": EPOCHS, **study.best_params}
    log.info("best AUC %.4f with %s", study.best_value, study.best_params)
    train.record(best, study.best_value, f"Optuna best of {args.trials} (held out {HELD_OUT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
