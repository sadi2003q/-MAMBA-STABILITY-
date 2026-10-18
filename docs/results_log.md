# Results log

Dated. Each experiment: verdict in 2–3 lines and whether the mathematical prediction held.

---

## 9 October 2026 — a01 and a02 (mathematical checks, no training)

- a01: certified models with positive loop gain were unstable in 100% of random draws; the necessary condition det(-(D + C (I - A_bar)^(-1) B)) > 0 held for 100% of stable draws. Prediction held (algebra).
- a02: true Van der Pol one-step map stretches some direction on 100% of the cycle (the brief said "parts"); perfect-model 50-step amplification up to 5.2 at time step 0.1. Global contraction ruled out.

## 18 October 2026 — exp01 certificate ablation, minimal cell, seed 0 (direction only)

- Guarantee removed vs guarantee on, 50-step error: teacher forcing +6%, chained training −18%, direct +1% → **NOT worth confirming** (no arm beyond 20%). Free-sign decay stayed negative everywhere (0% positive). **Prediction held.**
- Started unstable blew up in all three regimes and its decay never turned negative (100% positive after 40 epochs). Prediction "diverges" held; the sub-prediction "non-finite loss from the first batch" held for chained and direct (epoch 1) but **not** for teacher forcing, which trained 40 epochs on a finite loss (one-step error 0.47) and only exploded in chained evaluation.
- Not predicted, worth following: (1) under teacher forcing the minimal cell is **too damped, not too explosive**: it amplifies a starting-state error 0.54× over 50 steps against the true system's 2.51×, and its free-running period is 15% too long (8.35 vs 7.27); chained training brings the period error to 5% and the 50-step error from 0.584 to 0.256 (−56%). (2) The parameter-matched Long Short-Term Memory network beats the minimal cell under teacher forcing (0.193 vs 0.584, −67%; one-step 0.016 vs 0.036) and ties it under chained training (0.267 vs 0.256). (3) Direct multi-step: the minimal cell is weak (1.09 vs 0.52 for the Long Short-Term Memory network); free-running beyond the 50-step training horizon explodes for both direct-trained minimal-cell models — expected, since the regime is never trained past 50 steps.

## 18 October 2026 — exp01b certificate ablation, full Mamba (3,282 parameters), seed 0 (direction only)

- Guarantee removed vs guarantee on, 50-step error (average over the horizon): teacher forcing +2%, chained training +4%, direct +0% → **NOT worth confirming** (no arm beyond 20%). Free-sign decay stayed negative in every run (0% positive). **Prediction held**, so exp01 replicates on the full architecture.
- Not predicted: the full Mamba is **clearly worse than the parameter-matched Long Short-Term Memory network in all three regimes** (50-step error 0.383 vs 0.193 teacher forcing, 0.465 vs 0.267 chained, 1.05 vs 0.52 direct; the network is 50%, 43% and 50% lower). Under chained training the full Mamba (0.465) is also **82% worse than the minimal cell** (0.256) although it is the larger design.
- Chained-training records for the full Mamba look unsettled: validation loss 0.049 against training loss 0.023 at epoch 40 (about 2×) and noisy from epoch to epoch (0.040–0.063 over the last six epochs), unlike the minimal cell (0.022 vs 0.016) and the Long Short-Term Memory network (0.023 vs 0.018). The first epoch also started at a training loss of 140. Cause unknown: training settings (learning rate, epochs) or the design. No claim until a pre-declared, equal-effort check for both models. Code review of `src/models/mamba_block.py` found the standard structure and no defect; the architecture checks passed.
- Timing drift again: free-running period 7% too long with teacher forcing and 24% too long with chained training (9.02 vs 7.27). Both direct-trained full-Mamba models blow up in free running (2 of 2), as in exp01.
- Note: the 50-step error in `summary.md` is the average over the 50 steps; the error exactly at step 50 is larger (for example 0.626 against 0.465 for the certified chained arm).
