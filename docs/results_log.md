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

## 18 October 2026 — exp01c learning-rate check, chained training, seed 0 (direction only)

- Score = lowest validation loss over 40 epochs (rule fixed before the run). Full Mamba: 0.309 / 0.088 / 0.040 at learning rates 0.0001 / 0.0003 / 0.001. Long Short-Term Memory network: 0.264 / 0.085 / 0.022. No ties, so the rule chose **0.001 for both networks**. The best value is the highest one tested, so a learning rate above 0.001 was not tried.
- **The pre-declared prediction did not hold.** It said a lower learning rate would clearly lower the full Mamba's validation loss if the training settings were the cause. Lower was much worse (0.088 and 0.309 against 0.040), and the Long Short-Term Memory network behaved the same way. "Learning rate too high" is not supported in the direction tested; a design cause is not ruled out, and a higher rate remains untested.
- Validation loss divided by training loss at the best epoch: full Mamba 1.17 (0.0001), 1.69 (0.0003), 1.68 (0.001); Long Short-Term Memory network 0.98, 0.97, 1.23. The full Mamba generalises worse from training to unseen data at every rate that trained well. Cause unknown.
- For information only (test windows, not used in the choice): 50-step error 0.999 and 0.631 (full Mamba) and 0.885 and 0.471 (Long Short-Term Memory network) at 0.0001 and 0.0003; same ordering as the validation scores.
- The two runs at 0.001 were not repeated: records from exp01b and exp01 (same settings, seed 0) are kept in `docs/evidence/exp01c_reused_runs/`. exp01d below reproduced them to six digits, which confirms the reuse was valid.

## 18 October 2026 — exp01d closing run, full Mamba, seed 0 (direction only; eight seeds still to run)

- Same learning rate as exp01b (0.001), so every number reproduced exp01b and exp01 **exactly** (differences 0.0 to six digits). Training is deterministic on this graphics card with these settings.
- Guarantee removed vs guarantee on, 50-step error: teacher forcing +2%, chained +4%, direct +0% → **NOT worth confirming**. Free-sign decay stayed negative in every run (0% positive). **Prediction held** for every arm.
- Long Short-Term Memory network (same learning rate) vs the full Mamba: 50% lower with teacher forcing (0.193 vs 0.383), 43% lower chained (0.267 vs 0.465), 50% lower direct (0.521 vs 1.05).
- Where the full Mamba loses: free-running oscillation period error under chained training is 21% to 24% for the full Mamba (9.02 vs 7.27) and 0.3% for the Long Short-Term Memory network. Timing is the weakness, consistent with the drift picture; both direct-trained full Mamba models blow up in free running (2 of 2), the Long Short-Term Memory network does not.
- **Stage 1 closed as a direction.** The paper claim "removing the guarantee changes nothing" still needs seeds 0 to 7 (README step E, about 8.6 hours); the learning rates are now stored in `configs/models/*_final.yaml`, so step E runs in a fresh session without repeating the check. Evidence files: `docs/evidence/`.

## 19 October 2026 — Stage 2 diagnosis, seed 0 (direction only; no training)

- **Timing dominates for every model that feeds predictions back, not only Mamba:** timing share 0.71 full Mamba, 0.68 Long Short-Term Memory network, 0.62 simplified Mamba (chained training); cross-check 0.89, 0.84, 0.78. Q1 prediction **held**, but the effect is not Mamba-specific.
- **Feedback does not amplify after chained training:** chained error / added-up one-step mistakes = 0.71 (full Mamba), 0.72 (Long Short-Term Memory network), 0.50 (simplified). Teacher-forcing models are amplified (1.6, 1.8, 1.2) for every network. No model's timing error grows exponentially.
- **No slow clock on the test windows:** one-step mistakes have no consistent direction (consistency 0.11 for the full Mamba; implied clock 99.6%); at step 50 it is slightly ahead (−0.08 time units). Q3 prediction (slow clock) **did not hold**. The 24% slow loop appears only with zero input. *Limitation of the Q3 sign test:* it pooled all windows, so a bias consistent within a window but different between windows was invisible (checked separately, below).
- **Both correct shape errors** (shape push 1.8 full Mamba, 1.5 Long Short-Term Memory network; ≤ 3 = like the true system); Q4 prediction technically held but the gap is small. Chung et al.'s "cannot correct" does not carry over to chained prediction.
- **Guarantee on vs removed: about the same on every question** (repeats Stage 1).
- **Where the full Mamba actually loses (from details.json):** its one-step mistakes are smaller than the Long Short-Term Memory network's (0.096 vs 0.112) and feedback helps both equally, but its mistakes do not cancel when they add up: accumulation factor 6.8 against 3.3 (5.0 = unrelated mistakes, 29 = the same mistake every step, below 5.0 = cancelling). This alone accounts for the 1.74 times gap in 50-step error. Steady bias within a window is small (about 3% of the one-step error variance). Checked directly by Stage 2b.
