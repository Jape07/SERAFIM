"""Hyperparameter search with Optuna (task A6).

20 trials, 45-minute cap. Search space: learning rate (1e-4 to 1e-2), base
filters (8, 16 or 32), dropout (0 to 0.5) and batch size (64 or 128).
Uses the validation set only; the test set is reserved for task A7.

Output: best configuration as a new row in training/experiments.csv.
"""


def main() -> None:
    raise NotImplementedError("Task A6")


if __name__ == "__main__":
    main()
