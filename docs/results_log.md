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
