# Stage 3 plan — make Mamba's mistakes cancel, with the guarantee kept

Written 19 October 2026, before any Stage 3 code or result. Predictions and pass/fail rules below are fixed now.

## Goal in one sentence

Change the full Mamba's memory so that its next mistake can undo the last one (as the Long Short-Term
Memory network's does), keep the stability guarantee exactly as it is, and get the 50-step error at least
20% lower than today's 0.465.

## What Stages 1 and 2 give us (seed 0, direction only)

| Chained training | Full Mamba | Long Short-Term Memory network |
|---|---|---|
| 50-step error (smaller is better) | 0.465 | 0.267 |
| One-step mistake | 0.096 | 0.112 |
| Next mistake vs last one (correlation; negative = cancels) | +0.20 | −0.12 |
| Mistakes added over 50 steps ÷ one mistake (5.0 = unrelated) | 6.8 | 3.3 |
| Free-running loop time vs true (1.00 = right clock) | 1.24 | 1.00 |

- The guarantee changed nothing anywhere (Stage 1 and Stage 2).
- Feedback does not blow errors up after chained training (factor 0.71); the problem is that Mamba's
  mistakes line up instead of cancelling, and that is mostly seen as timing (71% of the error).
- `analysis/a03`: the guarantee cannot fix the clock or the growth of errors, so the fix must be in the
  model's design, not in a stronger version of the same guarantee.

## Mathematical signal

```
Mathematical signal: POSITIVE (for "the target is worth it"); INCONCLUSIVE (for "these designs reach it")
What was checked:    (1) accumulation over 50 steps for mistakes with a given step-to-step correlation;
                     (2) whether a clock built only from feedback gains is unusually sensitive to them
Result:              (1) correlation +0.20 -> factor 6.1; 0.00 -> 5.05; -0.12 -> 4.5. If the full Mamba's
                     mistakes stopped lining up (factor 6.8 -> 4.5 to 5.05) with its one-step mistake and
                     feedback unchanged, the 50-step error would fall from 0.465 to about 0.31-0.35
                     (26-33% lower), above the 20% bar.
                     (2) NEGATIVE for my own argument: a 1% change in the feedback gain changes the loop time
                     by about 1.2%, so the feedback-built clock is NOT unusually fragile. The case for a
                     rotating memory (arm B) rests on Mamba-3's results, not on this calculation.
Assumptions:         lag-1 correlation only (the measured factors 6.8 and 3.3 are more extreme than 6.1
                     and 4.5, so longer-gap correlations matter too); one-step mistake and feedback factor
                     unchanged by the new design
Implication:         worth one single-seed screening of three designs; no design is favoured by the maths
```

## Literature checked before designing (19 October 2026)

- **Mamba-3 (2026, arXiv 2603.15569):** memory update h_t = e^{Δ_t A_t} R_t h_{t-1} + Δ_t B_t x_t, where R_t
  is a data-dependent rotation (2 × 2 blocks). The decay e^{Δ A} stays in (0, 1) and rotations do not change
  size, so **our guarantee is kept exactly**. Without the rotation Mamba-3 fails parity tracking (2% against
  100%). Not tested on oscillating signals: that would be new.
- **Chung, Choi and Kim (2026):** memories whose update does not depend on their own contents cannot shrink
  certain errors; "state dependence does not guarantee correction, but it permits" it. Diagnostic paper, no
  drop-in design. Stage 2 found Chung's "cannot correct" does not carry over to shape errors in chained
  prediction, but Mamba's mistakes still fail to cancel, so a memory-dependent update is still a fair test.
- **Gated Delta Networks (2024) and Gated KalmaNet (2025):** memories that correct themselves by an error
  signal (delta rule; a Kalman-like least-squares update). Kept as a fallback (below).
- **Wake paper (2026, arXiv 2608.07189):** post-hoc phase realignment, no retraining. Kept as the cheap
  baseline every design must beat.

## The arms (all full Mamba, chained training, seed 0, learning rate 0.001 as fixed by exp01c)

| Arm | What changes | Guarantee kept? | Why |
|---|---|---|---|
| A. Baseline | nothing (existing model, no retraining: training is deterministic) | yes | reference: 0.465 |
| B. Rotating memory | Mamba-3 rotation: each memory pair turns by a learned, input-dependent angle every step | yes, exactly (decay in (0,1), rotation keeps size) | the oscillation can live inside the memory instead of being built only through feedback |
| C. Memory-dependent step size | the step size Δ also reads the memory's previous output: Δ_t = softplus(W x_t + U y_{t-1}) | yes, exactly (decay = e^{Δ A} with A < 0 and Δ > 0, whatever the memory holds) | Chung et al.: lets the memory react to its own contents, so a mistake can be undone |
| D. Both B and C | rotation + memory-dependent step size | yes, exactly | the two fixes may need each other |
| E. Parameter-matched control | baseline with extra width equal to the largest added parameter count | yes | rules out "it is just more parameters" (project rule) |
| F. Post-hoc realignment | arm A's predictions shifted in time by a drift rate estimated on the validation windows | — | cheap baseline from the wake paper |
| Reference | Long Short-Term Memory network (existing model) | no guarantee | the target to match: 0.267 |

Design rules (from the first attempt's pitfalls, CLAUDE.md Section 11):
- New gates start small but **not zero** (U and the rotation angles), so gradient reaches them; checked by
  `tools/check_model.py` (non-zero gradient, non-zero at the first step).
- Arm C has no parallel scan (the decay depends on the memory). Its step-by-step path is the only path; the
  check compares it with a slow reference loop instead.
- Arms B and D keep the parallel scan (the rotation is input-dependent only), checked as today (1e-4).

## Predictions and pass/fail rules (fixed now)

Primary: 50-step chained-prediction error on the test windows, against arm A (0.465).

- **Worth confirming (project rule):** at least 20% lower than arm A (0.372 or less), one-step error no more
  than 10% worse, no blow-up, **and** at least 10% lower than the parameter-matched control E.
- **Mechanism check (must agree, or the gain is not explained):** step-to-step correlation of one-step
  mistakes below arm A's +0.20 (prediction: 0.05 or less for C and D).
- **Recorded:** free-running loop time vs true (prediction: B and D closer to 1.00 than A's 1.24), timing
  share, guarantee status (must hold in every run, by construction), parameter count, training time.
- Predictions: C and D pass; B lowers the loop-time error but may not pass the 20% bar alone. Honest prior:
  INCONCLUSIVE; any arm may fail.

## Steps and who does what

1. **Claude:** maths for arms B and C at the first step (gradient reaches the new parts; decay stays in (0,1);
   the rotation does not change the guarantee), saved as `analysis/a04`; new model options in
   `src/models/mamba_block.py` (switched on by config, default off so old runs are unchanged);
   `configs/experiments/exp02_self_correcting_memory.yaml` with this plan's rules; post-hoc realignment
   (arm F) in the diagnosis tool; tests and model checks.
2. **Sadi:** `check_model.py` then one Kaggle run, seed 0, arms B–E. Estimated time: about 7 minutes per arm
   for B and E (as exp01d), up to 2–3 times longer for C and D (no parallel scan in teacher-forced parts):
   **about 45–75 minutes on a T4**.
3. **Sadi:** rerun the Stage 2 and 2b diagnosis on the new models (about 15 minutes on a processor): same
   tools, the new runs added to the configs.
4. **Claude:** summary, results log, decisions, one short page for the supervisor.
5. **Eight seeds** for any passing arm: after all single-seed experiments, as Sadi planned.

## If no arm passes (decided now, so it is not chosen after seeing results)

In this order, one single-seed screen each:
1. **Nonlinear memory read-out** (Chung's φ): y_t from tanh of the memory; guarantee kept.
2. **Error-correcting memory** (delta-rule update, Gated Delta Networks); guarantee checked by maths first.
3. **Training-side:** a penalty on mistakes that line up (step-to-step correlation of one-step mistakes),
   with its size fixed by the order-of-magnitude check before running.
If all fail, the paper is the negative result plus the mechanism (Stages 1–2, a01, a03), which is still
publishable, but the target stays a positive result while scope remains.

## After Stage 3 (not planned in detail yet)

- **Stage 4 — control:** model predictive control on Van der Pol with the best arm against the baseline Mamba
  and the Long Short-Term Memory network, under disturbances. This is where "the guarantee transfers to
  control" is finally tested, and where a guarantee on the whole prediction loop (a03) may be needed.
- **Stage 5 — second system** and the same change on the Long Short-Term Memory network.

## Sources

- [Mamba-3: Improved Sequence Modeling using State Space Principles (2026)](https://arxiv.org/pdf/2603.15569)
- [Chung, Choi, Kim — Rethinking State Tracking in Recurrent Models Through Error Control Dynamics (2026)](https://arxiv.org/html/2605.07755)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/pdf/2412.06464)
- [Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression](https://arxiv.org/pdf/2511.21016)
- [Autoregressive rollout error in latent-space reduced-order models of bluff-body wakes is accumulated phase drift (2026)](https://arxiv.org/pdf/2608.07189)
- [Learning orbitally stable dynamics via transverse contraction criteria (International Journal of Robotics Research, 2026)](https://metatoc.com/papers/141837-learning-orbitally-stable-dynamics-via-transverse-contraction-criteria-for-modeling-periodic-tasks)
