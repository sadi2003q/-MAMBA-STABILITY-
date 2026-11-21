# Drone test — plan written BEFORE any result (20 November 2026)

## The question
Can a small learned drone model that predicts many steps ahead accurately (the memory-free chained
predictor trained with the chained-prediction recipe), plus the offset-free disturbance correction
(the same idea as mpc01c), keep a drone on its path after a heavier payload is attached in the middle of
a flight, with no retraining and no change of the model's weights?

## The drone simulator (drone/quadrotor_sim.py)
- Crazyflie 2.x size quadrotor: mass 0.027 kg, inertia 1.4e-5 / 1.4e-5 / 2.17e-5 kg m^2.
- 12 measured numbers (the state): position (3), velocity (3), roll / pitch / yaw angles (3), turning rates (3).
- 4 commands: total thrust (0 to 0.6 newton) and three turning-rate commands (limit 6 rad/s each).
  A fast inner turning-rate loop (500 times per second) follows the turning-rate commands, as on real drones.
- The planner decides 50 times per second (every 0.02 s).
- The real drone has air drag (0.35 / 0.35 / 0.10 per second, in the drone's own frame). The textbook
  equations used by the physics planner do NOT have drag (this is the usual mismatch in drone papers).
- Small sensor noise: position 2 mm, velocity 1 cm/s, angles 0.003 rad, turning rates 0.02 rad/s.
- Payload: at a random moment between 3 s and 5 s the mass AND the inertia become 1.35 times larger
  (a payload of 35% of the drone's weight, the same size as Altawaitan and Atanasov, 2026).

## Training data
Flights with the normal mass only (the models never see the payload). A cascaded
proportional-integral-derivative controller flies random smooth paths with random extra command noise.
Seed 0.

## Learned models (all trained on the same data, seed 0)
- A. Chained predictor (the user's method): Chain-Mamba core from src/models/mamba_chain.py with the
  memory removed (mixer false), d_model 24, ffn_hidden 48, anchor (linear skip) + increment scaling.
  Recipe: chained training, horizon curriculum 5 -> 50 over 25 epochs, one-step loss weight 0.5,
  80 epochs, cosine learning rate 2e-3, batch 128.
- B. The SAME model trained one step at a time (teacher forcing), same epochs and learning rate.
  (Isolates the recipe.)
- C. A plain one-step network: 2 hidden layers of 64 units, predicts the change of state, trained one step
  at a time (the common choice in learned-model drone control papers).

## Controllers compared on the same 20 test flights
1. Cascaded proportional-integral-derivative controller (what small drones use today; has an integral term).
2. Physics planner: textbook equations, normal mass, no drag, no correction.
3. Physics planner + offset-free correction.
4. Plain one-step network (C) + offset-free correction.
5. Same model trained one step at a time (B) + offset-free correction.
6. Chained predictor (A), no correction.
7. Chained predictor (A) + offset-free correction  <- the user's method.
8. Oracle: planner with the TRUE simulator equations, which also knows the payload (reference only,
   not a baseline to beat).

Planner for 2-8 (the same for all): sampling-based model predictive path integral planner, looking 25 steps
(0.5 s) ahead, 256 sampled command sequences, one update per step with a warm start. Planner settings are
tuned once on the physics planner on a circle path (not a test path) and then frozen for every model.
Offset-free correction: disturbance = 0.5 x old disturbance + 0.5 x (measured state - state predicted one
step earlier), added to every predicted step (the mpc01c rule).

## Test flights
20 figure-eight paths at 1 m height: size 0.6-1.0 m, lap time 4-6 s, 10 s long, payload at 3-5 s
(random, start_seed 2026). Same 20 flights for every controller.

## Measurements
- MAIN: root-mean-square position error from the payload moment to the end of the flight (metres).
- Also: error before the payload, largest error after the payload, crashes (error above 1 m, height below
  0.05 m, or tilt above 80 degrees), median planning time per step on this 2-core processor.
- Prediction check: chained position/velocity error of models A, B, C and the textbook equations at
  25 and 50 steps on held-out normal-mass flights.

## Pass rule (fixed now)
H1 (headline) PASS if ALL hold for controller 7:
  (a) zero crashes in 20 flights;
  (b) median after-payload error at least 20% lower than the BEST of controllers 1-6;
  (c) paired Wilcoxon signed-rank test against that best controller, one-sided p < 0.05;
  (d) median planning time below 20 ms (fits 50 decisions per second).
H2 (does the recipe matter?): controller 7 at least 20% lower median than controller 5.
H3 (does the correction matter?): controller 7 at least 20% lower median than controller 6.
One training seed = direction only; 3+ seeds are needed before any paper claim.

## Follow-up step 1b — written AFTER the first results, BEFORE running it (21 November 2026)
Must be disclosed as added after the first results (like mpc01b / mpc01c).

What the first results showed (seed 0): H1 not passed. After the payload, the physics planner + correction
(3.40 cm) was slightly better than the chained predictor + correction (3.72 cm). But BEFORE the payload the
chained predictor WITHOUT correction was the most accurate controller (0.93 cm, against 1.80 cm for the
cascaded PID and 2.26 cm for the chained predictor WITH correction): the correction filter 0.5 lets sensor
noise into every prediction.

Reasoning (not tuned on results): a payload is a constant change, so a slow filter is enough to learn it,
and a slow filter averages away sensor noise. With filter a, noise spread shrinks by sqrt(a / (2 - a)):
a = 0.1 -> 0.23 x, while the correction still settles in about 1 / a = 10 steps (0.2 s).

Change: disturbance filter 0.1 instead of 0.5, for EVERY corrected controller (physics, plain network,
one-step model, chained predictor). Nothing else changes. One value only, chosen before running.
New controllers: physics_slow, plain_slow, one_step_slow, chained_slow.

Pass rule 1b (same form as H1): chained_slow PASSES if zero crashes, median after-payload error at least
20% lower than the best of {pid, physics, physics_slow, plain_slow, one_step_slow, chained}, one-sided
paired Wilcoxon p < 0.05, planning below 20 ms.
Also reported (no pass rule): error before the payload for every controller.

## Extra measurement (added after the first results, no pass rule)
To compare normal flight fairly (error BEFORE the payload), the two one-step-trained models are also flown
WITHOUT correction: one_step_none and plain_none. Reported only; not part of any pass rule.
