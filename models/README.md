# Trained models

`seed0/` holds the fifteen seed-0 models that Stage 1 trained and Stage 2 analyses
(`tools/diagnose_errors.py`, settings in `configs/analysis/stage2_diagnosis.yaml`).

- `exp01d_certificate_ablation_confirm/`: full Mamba (guarantee on and removed) and the Long Short-Term
  Memory network, each with teacher forcing, chained training and direct prediction. Learning rate 0.001.
- `exp01_certificate_ablation/`: simplified Mamba (guarantee on and removed), same three training methods.

Each folder: `checkpoint.pt` (weights and the full training configuration), `metrics.json` (Stage 1
evaluation) and `config.yaml`. Retrained on Kaggle (T4 graphics card) on 19 October 2026 after the
original session was lost; training is deterministic there, and the summaries reproduced exp01d and exp01
(50-step error with the guarantee removed: +2%, +4%, +0% for the full Mamba).

Only the final `checkpoint.pt` is kept; `checkpoint_last.pt` (training state for resuming) is not needed.
