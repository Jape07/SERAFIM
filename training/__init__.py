"""Training of the motion-dipole CNN (task A9, stage 2, AI & validation).

Runs outside the pipeline and outside GitHub (CPU, minutes). Needs the `ai` extra:
pip install -e ".[dev,ai]", and silver sums on the machine (python -m serafim silver).

    python -m training.dataset    stamps: injected dipoles and hard negatives on real sums
    python -m training.train      leave-one-target-out CNN, then the final ONNX in models/
    python -m training.search     optional Optuna search of lr, filters and dropout
    python -m training.evaluate   CNN against the classic rule (docs/dipole_cnn.md)

Every training run appends a row to training/experiments.csv.
"""
