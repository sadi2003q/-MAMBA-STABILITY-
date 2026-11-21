# Drone payload test - results (seed 0, 20 flights, this machine's 2-core processor)

| Controller | Crashes | Error before payload (cm) | Error after payload (cm) | Largest after (cm) | Planning per step (ms) |
|---|---|---|---|---|---|
| 1 Cascaded PID (used today) | 2/20 | 1.80 | 4.00 | 11.11 | 0.0 |
| 2 Physics planner | 0/20 | 3.81 | 10.94 | 12.65 | 6.3 |
| 3 Physics planner + correction | 0/20 | 2.66 | 3.40 | 6.73 | 6.5 |
| 4 Plain one-step network + correction | 0/20 | 2.01 | 2.40 | 5.40 | 5.0 |
| 5 Same model, one-step trained + correction | 0/20 | 1.96 | 2.69 | 5.23 | 80.2 |
| 6 Chained predictor, no correction | 0/20 | 0.93 | 12.64 | 14.33 | 9.7 |
| 7 Chained predictor + correction (YOUR METHOD) | 0/20 | 2.26 | 3.72 | 7.57 | 10.4 |
| 8 Oracle (knows the payload) - reference | 0/20 | 0.87 | 0.84 | 2.30 | 30.9 |
| 3b Physics planner + slow correction | 0/20 | 1.91 | 3.63 | 6.73 | 6.2 |
| 4b Plain one-step network + slow correction | 0/20 | 1.49 | 2.19 | 6.12 | 5.0 |
| 5b Same model, one-step trained + slow correction | 0/20 | 1.42 | 2.89 | 6.59 | 80.3 |
| 7b Chained predictor + slow correction (YOUR METHOD, step 1b) | 0/20 | 0.84 | 3.20 | 7.06 | 9.4 |
| 5c Same model, one-step trained, no correction (extra) | 0/20 | 1.66 | 11.02 | 12.48 | 79.0 |
| 4c Plain one-step network, no correction (extra) | 0/20 | 2.02 | 11.13 | 12.71 | 4.5 |

Errors are medians over flights of the root-mean-square position error. Crashed flights are left out of these columns and counted as 1 m in the pass-rule test.

## Pass rule (written before the results, drone/plan.md)
- H1: best baseline = 4 Plain one-step network + correction (2.40 cm). Your method 3.72 cm -> -55.2% lower, Wilcoxon p = 1.0000, zero crashes = True, planning 10.4 ms. **NOT PASSED**
- H2_recipe: -38.5% lower than 5 Same model, one-step trained + correction (p = 1.0000). **NOT PASSED**
- H3_correction: 70.5% lower than 6 Chained predictor, no correction (p = 0.0000). **PASS**
- Step 1b (added after the first results): best baseline = 4b Plain one-step network + slow correction (2.19 cm). Your method with the slow correction 3.20 cm -> -46.0% lower, p = 1.0000, zero crashes = True, planning 9.4 ms. **NOT PASSED**

## Prediction check (held-out normal-mass flights, median position error in cm)
| Model | after 1 step | 10 steps | 25 steps | 50 steps | Parameters |
|---|---|---|---|---|---|
| physics (textbook equations) | 0.46 | 0.93 | 3.65 | 11.27 | - |
| plain | 0.46 | 1.04 | 2.46 | 5.62 | 6284 |
| one_step | 0.46 | 1.04 | 2.45 | 5.00 | 8040 |
| chained | 0.45 | 0.57 | 1.13 | 2.71 | 8040 |

One training seed only: this shows direction, not proof.

## Notes (read before using these numbers)
- NORMAL FLIGHT (before the payload, measured for every controller but with no pass rule):
  chained predictor + slow correction 0.84 cm, the same as the oracle that knows the true physics (0.87 cm).
  It was lower than every other non-oracle controller on all 20 flights: 41% lower than the best other learned
  model (one-step trained, slow correction, 1.42 cm, Wilcoxon p < 0.000001) and 53% lower than the cascaded
  PID (1.80 cm, 18 of 18 flights without a crash). This is a finding from a reported measurement, not a
  pre-declared headline: it needs its own pre-declared test with more seeds.
- PAYLOAD (the headline): not passed. The chained predictor adapted WORST of the three learned models.
- Planning time of the one-step-trained model (80 ms) is a processor artifact: its trained weights contain
  extremely small "denormal" numbers that slow the processor down. With torch.set_flush_denormal(True)
  both it and the chained predictor take 6.8 ms for one 256-sample, 26-step prediction.
- Training time: chained 29.3 min, one-step 17.8 min (both slowed by other jobs running at the same time),
  plain 1.6 min. Not a fair timing comparison.
- Steps 1b and the "no correction" extras were added after the first results (disclosed in plan.md).
- Simulator is our own (Crazyflie-size, drag, motor lag 0.03 s, 0.02 s command delay, sensor noise);
  not yet checked against gym-pybullet-drones or a real drone. One training seed.
