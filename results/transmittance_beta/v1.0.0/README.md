# Run artifacts — transmittance beta CNN v1.0.0

These are the unmodified artifacts supplied for the 2026-10-04 training run:

- `best_model.pt` — best validation checkpoint (epoch 5).
- `metrics.json` — held-out test and validation summary.
- `training_history.csv` — epoch-level train/validation loss and accuracy.
- `random_test_20261004.csv` and `.json` — 30-image sanity check from the same synthetic data domain.

No retraining or metric recalculation occurred when these files were committed. See `src/transmittance_beta/README.md` for intended use and limitations.
