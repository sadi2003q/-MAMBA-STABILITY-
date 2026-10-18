# Reused runs for the learning-rate check (exp01c)

The learning rate 0.001 is the setting both networks were already trained with, on seed 0, in
exp01b (full Mamba, `certified_chained`) and exp01 (Long Short-Term Memory network, `lstm_chained`).
Training settings, data and code path are identical to the runs exp01c would repeat, so the two small
records below are used instead of spending about 31 minutes of Kaggle time on repeating them.

- `mamba_chained_lr_1e-3_seed0/` from exp01b `certified_chained_seed0` (Kaggle run, 18 October 2026)
- `lstm_chained_lr_1e-3_seed0/` from exp01 `lstm_chained_seed0` (Kaggle run, 18 October 2026)

Only `train_log.csv` (loss per epoch) and `metrics.json` are kept; `tools/pick_learning_rate.py` reads
exactly these two files. Nothing was edited.
