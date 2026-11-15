# Results of the motion-dipole CNN (task A9)

The results are generated, not written by hand:

- Stage 1, the classic rule on the 93 real differences: [docs/dipoles.md](../docs/dipoles.md).
- Stage 2, the CNN against the rule at the same false-alarm rate, out of fold (leave one target
  out): [docs/dipole_cnn.md](../docs/dipole_cnn.md).
- Every training run: [experiments.csv](experiments.csv).
- What the model is for and where it fails: [docs/model_card.md](../docs/model_card.md).

To reproduce (CPU, about 15 minutes; needs `pip install -e ".[dev,ai]"` and silver on disk):

```
python -m training.dataset
python -m training.train
python -m training.evaluate
python -m serafim.validation.dipoles
```
