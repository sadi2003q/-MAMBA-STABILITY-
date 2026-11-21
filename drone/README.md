# Drone payload test (chained predictor inside a planner)

Question, controllers, pass rules and every change made after the first results: `drone/plan.md`.
Results of seed 0 (run on 20-21 November 2026, 2-core processor): `drone/outputs/seed0/summary.md`
and `drone/outputs/seed0/flights_figure.png`.

Files, in the order they are used:
1. `quadrotor_sim.py`    drone simulator (Crazyflie size, air drag, motor lag, command delay, sensor noise)
                          and the cascaded proportional-integral-derivative controller
2. `collect_data.py`     training flights, normal mass only (about 1 minute; data is not stored in git)
3. `drone_models.py`     the three learned models
4. `train_models.py`     training (chained recipe / one step at a time)
5. `prediction_check.py` error after 1, 10, 25, 50 chained steps on held-out flights
6. `planner.py`          sampling planner + offset-free correction
7. `run_flights.py`      the 20 payload test flights for every controller
8. `analyse_results.py`  pass-rule check, summary.md, figure

Commands (processor only; use --threads 1 for the flights):
```
python drone/collect_data.py
python drone/train_models.py --model chained --seed 0
python drone/train_models.py --model one_step --seed 0
python drone/train_models.py --model plain --seed 0
python drone/prediction_check.py --seed 0
python drone/run_flights.py \
    --mode test \
    --seed 0 \
    --threads 1 \
    --controllers all
python drone/analyse_results.py --seed 0
```
