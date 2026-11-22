# Space-engineering pilots A and B (22 November 2026)

Two quick single-seed pilots that ask whether your current system — the memory-removed chained predictor
(exp04 `chain_mamba_no_memory`: 2 gated feed-forward layers, anchor, increment scale, about 3,300 parameters)
trained with your recipe (chained prediction, curriculum from 5 steps to the full horizon, one-step loss
weight 0.5), plus the running correction (offset-free control, mpc01c) — solves an open problem stated in a
2026 space paper. **One seed shows direction only.** Each pilot ends with GO (worth confirming with 8 seeds)
or STOP.

Nothing in the model, the recipe or the controller was changed. New: two simulators, data generators,
controller costs, and the comparison controllers from the papers.

## Pilot A — satellite pointing without the steady pointing error

- **Paper:** Cena, Martini, Chiaberge (2026), arXiv 2602.15954, "Hybrid Model Predictive Control with
  Physics-Informed Neural Network for Satellite Attitude Control".
- **Open problem they report:** model predictive control with the learned model alone ends with a steady
  pointing error (median 0.2901 degrees against 0.0023 for their hybrid). Their fix switches to a linear
  model below 1 degree, which "leads to considerably higher torques than necessary"; their networks were
  trained without reaction-wheel friction.
- **What is tested:** 5 controllers (true equations, linear model only, learned model only, their hybrid
  switch, learned + running correction) × 2 conditions of the real satellite (no friction / wheel friction
  not in the training data), 32 satellites starting 20-60 degrees off target, 80 seconds.
- **Mathematical signal (analysis/p01):** POSITIVE. Unmodelled friction of 3.5% of the push limit leaves
  0.191 degrees of steady error with an exact linear controller; the running correction removes it (0.000).
- **GO if:** the problem is real (learned alone at least 5× the true equations' steady error), and the
  running correction is at most 1.1× the hybrid's steady error without friction and at most 0.5× with
  friction, settles no slower than 1.1× the hybrid, and loses no satellite.
- **Differences from the paper (to disclose):** own simulator of the same physics instead of Basilisk;
  dt 0.5 s instead of 0.1 s; Adam solver; 40 training epochs instead of 80.

## Pilot B — staying on a planned orbit near an irregular asteroid

- **Paper:** Cesarini and Silvestrini (Politecnico di Milano), "Transformer-based Neural Predictive Control
  for Small-Body Proximity Operations" (venue and date not printed on the copy read; it cites 2026 data).
- **Open problem they report:** a large Transformer surrogate (embedding 512, 8 heads, hidden 2048) trained
  on a low-fidelity Eros model, shown only "in the vicinity of a reference orbit"; "future studies should
  extend the analysis to more complex reference trajectories and a wider range of operational regimes".
  Planning took 3.58 s per step on an embedded computer.
- **What is tested:** your predictor trained on the low-fidelity model over a wide shell (25-70 km),
  5 controllers (true physics, low-fidelity physics = the paper's standard controller, low-fidelity physics
  + correction, learned only, learned + correction) on 4 different orbits near the real asteroid
  (equatorial 40 km, polar 45 km, inclined 32 km, retrograde 55 km), 8 spacecraft each, 12 hours.
- **Mathematical signal (analysis/p02):** POSITIVE that the problem is real and correctable. The
  low-fidelity model is wrong by 1.9-2.9e-7 km/s² (mostly the tilted spin axis), a 2.2-3.3 km drift over an
  80-minute plan, but only 12-17% of the thrusters' largest acceleration. INCONCLUSIVE on the learned model's
  own fitting error.
- **GO if:** the problem is real (low-fidelity physics at least 1.2× the true physics' error), and learned +
  correction loses no spacecraft, is at most 1.2× the low-fidelity physics controller's error on every orbit,
  and uses at most 1.2× its fuel.
- **Differences from the paper (to disclose):** mass-point gravity of an Eros-like two-lobe shape instead of
  the polyhedral Eros model (so planning times are not comparable with the paper's); no comparison with
  their Transformer itself (not public); Adam solver instead of IPOPT; 40 training epochs.

## How to run

```bash
# both pilots, one command (Kaggle cell)
!bash /kaggle/working/-MAMBA-STABILITY-/scripts/run_space_pilots.sh

# or one at a time
python /kaggle/working/-MAMBA-STABILITY-/tools/run_pilot_attitude.py \
    --config /kaggle/working/-MAMBA-STABILITY-/configs/pilots/pilot_a_attitude_offset.yaml \
    --seed 0 \
    --device cpu \
    --quick no

python /kaggle/working/-MAMBA-STABILITY-/tools/run_pilot_asteroid.py \
    --config /kaggle/working/-MAMBA-STABILITY-/configs/pilots/pilot_b_asteroid_proximity.yaml \
    --seed 0 \
    --device auto \
    --quick no
```

`--quick yes` is a code check only (tiny data, 2 epochs, a few steps) — never interpret its numbers.
Results: `outputs/pilot_a_attitude_offset/` and `outputs/pilot_b_asteroid_proximity/` (summary.md,
results.csv, metrics.json, predictor_check.json, log.txt, models/).
