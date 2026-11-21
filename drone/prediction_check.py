"""
Prediction check: how far off is each model after 1, 10, 25 and 50 chained steps?

Uses the 30 held-out normal-mass flights (never used for training or for picking the best epoch).
From 200 random starting points, each model is given the TRUE commands and predicts on its own
(chained prediction). Errors are in physical units: position in centimetres, velocity in cm/s.

Run:  python drone/prediction_check.py --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone import drone_models as dm
from drone.planner import LearnedPredictor, PhysicsPredictor
from drone.train_models import load_flights, load_model, split_flights

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_STEPS = [1, 10, 25, 50]
N_STARTS = 200


def main(seed):
    torch.set_num_threads(2)
    measured, commands = load_flights()
    _, _, test_flights = split_flights(measured.shape[0])
    numbers = torch.load(os.path.join(HERE, "models", "normalisation.pt"), weights_only=False)

    # Step 1: random starting points on the held-out flights
    rng = np.random.default_rng(123)
    starts = []
    for _ in range(N_STARTS):
        flight = int(rng.choice(test_flights))
        step = int(rng.integers(0, commands.shape[1] - 50))
        starts.append((flight, step))

    # Step 2: the predictors
    predictors = {"physics (textbook equations)": PhysicsPredictor(oracle=False)}
    for name in ["plain", "one_step", "chained"]:
        model, saved = load_model(name, seed)
        predictors[name] = LearnedPredictor(model, numbers)

    # Step 3: predict 50 steps from every start, measure the error
    results = {}
    for name, predictor in predictors.items():
        position_errors = {k: [] for k in REPORT_STEPS}
        velocity_errors = {k: [] for k in REPORT_STEPS}
        for flight, step in starts:
            x_now = measured[flight, step]
            applied = commands[flight, step:step + 50, 0:4][None]
            previous = commands[flight, step:step + 50, 4:8][None]
            predicted = predictor.predict(x_now, applied, previous, None)[0]       # [50, 12]
            truth = measured[flight, step + 1:step + 51]
            for k in REPORT_STEPS:
                position_errors[k].append(np.linalg.norm(predicted[k - 1, 0:3] - truth[k - 1, 0:3]))
                velocity_errors[k].append(np.linalg.norm(predicted[k - 1, 3:6] - truth[k - 1, 3:6]))
        results[name] = {}
        for k in REPORT_STEPS:
            results[name][f"position_cm_step{k}"] = float(np.median(position_errors[k]) * 100)
            results[name][f"velocity_cm_s_step{k}"] = float(np.median(velocity_errors[k]) * 100)

    # Step 4: print and save
    print("Median position error (cm) after 1 / 10 / 25 / 50 chained steps (0.02 s each):")
    for name in results:
        values = [f"{results[name][f'position_cm_step{k}']:.2f}" for k in REPORT_STEPS]
        print(f"  {name:32s} " + " / ".join(values))
    for name in ["plain", "one_step", "chained"]:
        _, saved = load_model(name, seed)
        results[name]["parameters"] = saved["parameters"]
        results[name]["training_minutes"] = saved["minutes"]
    out_dir = os.path.join(HERE, "outputs", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prediction_check.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
