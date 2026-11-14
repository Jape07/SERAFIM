"""Train the base CNN on CPU (task A5).

3 conv blocks (16, 32 and 64 filters), global pooling and 1 output with the
probability that the change is real; about 24k parameters. Augmentation with
90-degree rotations and flips, fixed seed.

Output: data/models/base.pt, validation AUC and confusion matrix, plus a new
row in training/experiments.csv.
"""


def main() -> None:
    raise NotImplementedError("Task A5")


if __name__ == "__main__":
    main()
