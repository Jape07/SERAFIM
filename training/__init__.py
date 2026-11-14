"""Training of the real-vs-artifact CNN (AI & validation).

Runs outside the pipeline. Needs the `ai` extra: pip install -e ".[dev,ai]".

    python -m training.dataset    task A4: cutouts and labels
    python -m training.train      task A5: base CNN
    python -m training.search     task A6: Optuna search (45 min cap)
    python -m training.evaluate   task A7: CNN vs classic rule on the test set

Every run appends a row to training/experiments.csv.
"""
