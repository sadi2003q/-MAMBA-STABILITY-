# Stage 2 plan — find out what actually makes the predictions go wrong

Written 19 October 2026, before any Stage 2 result exists. Predictions and pass/fail rules below are fixed now.

## Goal

Stage 1 showed that the built-in guarantee (the locked fading factor of the memory) does not protect the
predictions: removing it changes nothing, and the error is mostly drift. Stage 2 explains **why**, with
numbers and a mechanism, so that Stage 3 knows exactly what a new guarantee has to fix.

**No new training is needed.** Everything runs on the models already trained (seed 0), mostly on a normal
processor, in under about 30 minutes.

## What we already know (Stage 1, seed 0, direction only)

- The guarantee holds in every run; removing it changes the 50-step error by +2%, +4% and +0%.
- The full Mamba's free-running oscillation is about 24% too slow under chained training (one loop takes
  9.02 time units instead of 7.27); the Long Short-Term Memory network is within 0.3%.
- The real system keeps timing errors and wipes out shape errors: per loop, timing errors are multiplied by
  1.0 and shape errors by 0.00086 (`analysis/a02`).

## Words used below

- **Timing error:** the prediction is on the right path but ahead of or behind the real system.
- **Shape error:** the prediction is off the path.
- **Chained prediction:** each prediction is fed back in as the next input, as a controller does.
- **Jacobian:** how much a small change in one step's state changes the next step's state.

## Four questions

Each has a method, a prediction and a pass/fail rule, written before running.

### Q1. How much of the error is timing and how much is shape?

- **Method:** at every predicted step, split the error into the part along the direction the true system is
  moving (timing) and the part across it (shape). This works with the input switched on, unlike measuring
  against the free-running loop.
- **Prediction:** timing dominates for the full Mamba under chained training.
- **Rule:** "timing-dominated" if timing is at least 60% of the 50-step squared error (median over windows).

### Q2. Do errors just add up, or does feedback amplify them?

- **Method:** compare normal chained prediction with a version in which each step's correction is computed
  from the true history (nothing is fed back), so one-step mistakes only add up.
- **Rule:** "amplified by feedback" if the chained error is more than 1.5 times the no-feedback error.
  Close to 1 means the problem is small mistakes piling up; much larger means the prediction-memory loop
  amplifies them (the mechanism of `analysis/a01`).

### Q3. Slow clock or amplifier?

- **Method:** how the timing error grows with step number. A straight line means a constant speed mistake
  (a slow clock); growth that speeds up means instability. Also: do the one-step errors consistently point
  "behind" rather than at random?
- **Rule:** "slow clock" if the mean timing component of the one-step errors is at least half of its
  root-mean-square value (a consistent sign) and a straight line fits the timing error at least as well as
  an exponential.
- Also recorded, following Pervez and Locatello (2026): the largest error magnification over the horizon
  and how far each step's Jacobian is from "normal" (non-normality).

### Q4. Why does the Long Short-Term Memory network keep time when Mamba does not?

- **Background:** Chung, Choi and Kim (2026) prove that recurrences whose memory update does not depend on
  the memory itself cannot correct drift; Mamba is in that class, the Long Short-Term Memory network is not.
  Their result is for symbolic tasks, and in chained prediction Mamba's own outputs feed back into its input.
- **Method:** push each trained model a little along the timing direction and along the shape direction, at
  points on the loop, and follow the push for 50 steps. Compare with the real system (1.0 and 0.00086 per
  loop).
- **Payoff:** tests whether their result carries over to continuous, chained prediction — the planned
  "extend Chung et al. to continuous dynamics" item and a possible theory contribution.
- **Rule (added 19 October, before any result):** shape-push ratio = the model's part across the path
  after 50 steps divided by the true system's, median over 12 points on the loop. 3 or less: "corrects
  shape errors like the true system"; 10 or more: "does not correct"; in between: "partly corrects".
  **Prediction:** the full Mamba's ratio is larger than the Long Short-Term Memory network's (both chained
  training). Chung et al. measure recovery the same way (push the state, follow the ratio of sizes), but on
  the memory and on symbolic tasks; they do not test the Long Short-Term Memory network.

## Theory piece (mathematical check, done by Claude)

- **Claim:** any pattern of closed-loop error growth and oscillation speed can be produced by a model that
  satisfies the guarantee.
- **Sketch (one-memory case):** the closed loop is the 2 × 2 matrix [[1 + d, c], [b, a]] with trace
  1 + d + a and determinant (1 + d) a − b c. For any guaranteed fading factor a between 0 and 1, choose
  d = λ1 + λ2 − 1 − a and b c = (1 + d) a − λ1 λ2 to obtain any two target eigenvalues λ1, λ2 (real or a
  complex pair). So the guarantee places no constraint on timing or growth.
- **Deliverables:** `analysis/a03_certificate_places_no_constraint.py` and a write-up in `docs/math_notes.md`,
  extended to the general (many-memory) case.

## Mathematical signal

```
Mathematical signal: INCONCLUSIVE (for Q1 on the real input-driven data)
What was checked:    slow-clock estimate from the measured period (9.02 vs 7.27)
Result:              after 50 steps a 24% slow clock would be about 13% of a loop behind,
                     a step-50 error of about 1.7 on the free-running loop; the measured
                     step-50 error is 0.63, so the input partly pulls the model back into step
Assumptions:         free-running period applies to input-driven windows (it clearly does not fully)
Implication:         timing drift is the likely main cause, but its size must be measured,
                     not assumed; Q1 settles it cheaply
```

## What gets built and who runs it

1. **Claude:** `analysis/a03` (theory) and `tools/diagnose_errors.py` (answers Q1–Q4 on saved models), with
   settings, predictions and rules in `configs/analysis/stage2_diagnosis.yaml`. Recent literature is searched
   again before the code is written.
2. **Claude:** the seed-0 trained models are tiny (full Mamba 42 KB, Long Short-Term Memory network 18 KB,
   simplified Mamba 19 KB), so they are stored in the repository. The analysis then runs anywhere, even on
   the Mac, without a Kaggle session.
3. **Sadi:** one command, under about 30 minutes on a normal processor, covering the full Mamba, the Long
   Short-Term Memory network and the simplified Mamba in all three training methods.
4. **Claude:** plain-language summary, results log entry, and a Part 6 in the supervisor document.
5. **In parallel, when graphics-card time allows:** the eight-seed Stage 1 repeat (README step E, about
   8.6 hours); Stage 2 is then rerun on those models to confirm.

**Status (19 October 2026, evening).** Stage 2 and the Stage 2b follow-up are run and logged (`docs/results_log.md`); plain-language update for the supervisor written as a separate document. Earlier the same day — steps 1 and 2 done: `analysis/a03` (run; result in `docs/math_notes.md`),
`tools/diagnose_errors.py` with its calculations in `src/evaluation/error_diagnosis.py`, settings and rules
in `configs/analysis/stage2_diagnosis.yaml`, and the fifteen seed-0 models (retrained on Kaggle the same
day; identical to Stage 1) in `models/seed0/`. Next: Sadi runs README "Stage 2 diagnosis".

## How Stage 2 feeds Stage 3

- **If timing dominates and grows in a straight line (most likely):** the new guarantee must keep the
  model's clock right — leave timing errors alone, pull shape errors back, and use a memory update that can
  correct itself. The phase autoencoder of Yawata and colleagues (learned timing and shape coordinates for
  oscillators) is a strong building block.
- **If feedback amplification dominates:** the guarantee must cover the whole prediction-plus-memory loop,
  which is what `analysis/a01` points at.

## Novelty warning

A 2026 paper reports that rollout error in learned models of the wake behind a bluff body is "accumulated
phase drift", and a 2026 sea-temperature result also drifted in phase. So "the error is timing drift" alone
is not new. What is new here:

1. the proof that the guarantee cannot constrain timing;
2. the mechanism (memory update that does or does not depend on the memory) in the continuous, chained
   setting;
3. the link to control.

The paper must cite those works and position itself on these three points.

**Wake paper, content checked 19 October 2026** (arXiv 2608.07189): a convolutional autoencoder with a
Long Short-Term Memory network on cylinder wakes; 97–98% of the error variance is phase, 2% amplitude; the
phase drifts in a straight line ("one number per latent coordinate: the drift rate"); the fix is a
post-hoc phase realignment that needs no retraining (41–71% lower error on average). No Mamba, no stability
certificate, no control. Two consequences: (1) their Long Short-Term Memory network drifts, ours keeps time
(0.3% period error), so the drift is not a fixed property of a network family; (2) a post-hoc phase
realignment is a strong, cheap baseline that any Stage 3 method must be compared with.

## Sources

- [Autoregressive rollout error in latent-space reduced-order models of bluff-body wakes is accumulated phase drift (2026)](https://arxiv.org/pdf/2608.07189)
- [Chung, Choi, Kim — Rethinking State Tracking in Recurrent Models Through Error Control Dynamics (2026)](https://arxiv.org/pdf/2605.07755)
- [Pervez, Locatello — Controlling Transient Amplification Improves Long-horizon Rollouts (2026)](https://www.alphaxiv.org/abs/2605.08856.md)
- [Eigenanalysis framework for autoregressive neural emulators of multi-scale chaotic dynamics (2026)](https://papers.cool/arxiv/2608.16084)
- [Biswas — A comparative study of accuracy and rollout stability of temporal surrogate models (2026)](https://arxiv.org/pdf/2605.24868)
- [Phase autoencoder for limit-cycle oscillators](https://arxiv.org/pdf/2403.06992)
- [Artifacts of Numerical Integration in Learning Dynamical Systems](https://arxiv.org/pdf/2507.14491)
