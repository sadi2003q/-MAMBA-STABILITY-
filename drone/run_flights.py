"""
Fly the 20 test flights with every controller, then check the pass rule written in drone/plan.md.

Each test flight: a figure-eight path at 1 m height, 10 s long. At a random moment between 3 s and 5 s a
payload is attached: mass and inertia become 1.35 times larger. No controller is told about it
(except the oracle, which is only a reference).

Run:
    python drone/run_flights.py --mode test --seed 0
    python drone/run_flights.py --mode tune          (planner settings, circle path, physics planner only)

Output in drone/outputs/: results.csv, summary.md, metrics.json, flights_figure.png, log.txt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone import quadrotor_sim as sim
from drone.planner import LearnedPredictor, PhysicsPredictor, SamplingPlanner
from drone.train_models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
FLIGHT_SECONDS = 10.0
N_TEST_FLIGHTS = 20
TEST_START_SEED = 2026
PAYLOAD_FACTOR = 1.35

CONTROLLER_NAMES = [
    "pid",
    "physics",
    "physics_corrected",
    "plain_corrected",
    "one_step_corrected",
    "chained",
    "chained_corrected",
    "oracle",
    "physics_slow",
    "plain_slow",
    "one_step_slow",
    "chained_slow",
    "one_step_none",
    "plain_none",
]
NICE_NAMES = {
    "pid": "1 Cascaded PID (used today)",
    "physics": "2 Physics planner",
    "physics_corrected": "3 Physics planner + correction",
    "plain_corrected": "4 Plain one-step network + correction",
    "one_step_corrected": "5 Same model, one-step trained + correction",
    "chained": "6 Chained predictor, no correction",
    "chained_corrected": "7 Chained predictor + correction (YOUR METHOD)",
    "oracle": "8 Oracle (knows the payload) - reference",
    "physics_slow": "3b Physics planner + slow correction",
    "plain_slow": "4b Plain one-step network + slow correction",
    "one_step_slow": "5b Same model, one-step trained + slow correction",
    "chained_slow": "7b Chained predictor + slow correction (YOUR METHOD, step 1b)",
    "one_step_none": "5c Same model, one-step trained, no correction (extra)",
    "plain_none": "4c Plain one-step network, no correction (extra)",
}

# Planner settings: chosen with --mode tune on the circle path, then frozen (see drone/plan.md)
PLANNER_SETTINGS = {"temperature": 0.03, "noise_scale": 0.5}     # from --mode tune (20 Nov 2026), frozen


# ======================================================================================================
# Paths and flights
# ======================================================================================================
def circle_path(size, lap_time):
    def path(t):
        w = 2.0 * np.pi / lap_time
        position = np.array([size * np.cos(w * t), size * np.sin(w * t), 1.0])
        velocity = np.array([-size * w * np.sin(w * t), size * w * np.cos(w * t), 0.0])
        acceleration = np.array([-size * w * w * np.cos(w * t), -size * w * w * np.sin(w * t), 0.0])
        return position, velocity, acceleration
    return path


def eight_path(size, lap_time):
    def path(t):
        return sim.figure_eight(t, size, lap_time)
    return path


def make_test_flights():
    rng = np.random.default_rng(TEST_START_SEED)
    flights = []
    for i in range(N_TEST_FLIGHTS):
        flight = {
            "index": i,
            "size": float(rng.uniform(0.6, 1.0)),
            "lap_time": float(rng.uniform(4.0, 6.0)),
            "payload_time": float(rng.uniform(3.0, 5.0)),
        }
        flight["path"] = eight_path(flight["size"], flight["lap_time"])
        flights.append(flight)
    return flights


# ======================================================================================================
# Controllers
# ======================================================================================================
def make_controller(name, seed, settings):
    """Returns (kind, object): kind is 'pid' or 'planner'."""
    if name == "pid":
        return "pid", sim.CascadedPID(1)

    # every other controller is the same planner with a different predictor
    planner_options = {
        "temperature": settings["temperature"],
        "noise": tuple(np.array([0.04, 1.5, 1.5, 0.5]) * settings["noise_scale"]),
        "seed": seed,
    }
    # step 1b (added after the first results): the slow correction filter 0.1 instead of 0.5
    if name.endswith("_slow"):
        planner_options["disturbance_filter"] = 0.1

    if name == "physics":
        predictor = PhysicsPredictor(oracle=False)
        return "planner", SamplingPlanner(predictor, offset_free=False, **planner_options)
    if name in ["physics_corrected", "physics_slow"]:
        predictor = PhysicsPredictor(oracle=False)
        return "planner", SamplingPlanner(predictor, offset_free=True, **planner_options)
    if name == "oracle":
        predictor = PhysicsPredictor(oracle=True)
        return "planner", SamplingPlanner(predictor, offset_free=False, **planner_options)

    # learned models
    numbers = torch.load(os.path.join(HERE, "models", "normalisation.pt"), weights_only=False)
    if name in ["plain_corrected", "plain_slow"]:
        model, _ = load_model("plain", seed)
        offset_free = True
    elif name in ["one_step_corrected", "one_step_slow"]:
        model, _ = load_model("one_step", seed)
        offset_free = True
    elif name == "chained":
        model, _ = load_model("chained", seed)
        offset_free = False
    elif name == "one_step_none":
        model, _ = load_model("one_step", seed)
        offset_free = False
    elif name == "plain_none":
        model, _ = load_model("plain", seed)
        offset_free = False
    elif name in ["chained_corrected", "chained_slow"]:
        model, _ = load_model("chained", seed)
        offset_free = True
    else:
        raise ValueError(name)
    return "planner", SamplingPlanner(LearnedPredictor(model, numbers), offset_free=offset_free,
                                      **planner_options)


# ======================================================================================================
# One flight
# ======================================================================================================
def fly(name, path, payload_time, noise_seed, seed, settings, payload=True):
    kind, controller = make_controller(name, seed, settings)
    rng = np.random.default_rng(noise_seed)          # the same sensor noise for every controller
    n_steps = int(round(FLIGHT_SECONDS / sim.CONTROL_STEP))
    horizon = 25

    # Step 1: start on the path, normal mass, hovering motors
    state = np.zeros((1, 12))
    p0, v0, _ = path(0.0)
    state[0, 0:3] = p0
    state[0, 3:6] = v0
    motor_thrust = np.array([sim.HOVER_THRUST])
    mass = np.array([sim.NORMAL_MASS])
    inertia = sim.NORMAL_INERTIA[None].copy()
    hover = np.array([sim.HOVER_THRUST, 0.0, 0.0, 0.0])
    applied_now = hover.copy()
    applied_before = hover.copy()

    errors = []
    crashed = False
    payload_step = int(round(payload_time / sim.CONTROL_STEP))

    for k in range(n_steps):
        t = k * sim.CONTROL_STEP

        # Step 2: the payload is attached (the controllers are NOT told)
        if payload and k == payload_step:
            mass = mass * PAYLOAD_FACTOR
            inertia = inertia * PAYLOAD_FACTOR

        # Step 3: measure (with sensor noise)
        measurement = state[0] + rng.normal(0.0, 1.0, size=12) * sim.SENSOR_NOISE

        # Step 4: decide the command for the NEXT step
        if kind == "pid":
            ref_p, ref_v, ref_a = path(t + sim.CONTROL_STEP)
            decided = controller.command(measurement[None], ref_p[None], ref_v[None], ref_a[None])[0]
        else:
            if name == "oracle":
                controller.predictor.mass = float(mass[0])
                controller.predictor.inertia = inertia[0].copy()
                controller.predictor.motor_thrust = float(motor_thrust[0])
            ref_positions = np.zeros((horizon + 1, 3))
            ref_velocities = np.zeros((horizon + 1, 3))
            for j in range(horizon + 1):
                p, v, _ = path(t + (j + 1) * sim.CONTROL_STEP)
                ref_positions[j] = p
                ref_velocities[j] = v
            decided = controller.decide(measurement, applied_now, applied_before,
                                        (ref_positions, ref_velocities))

        # Step 5: the drone moves one step with the command decided one step ago
        state, motor_thrust = sim.control_step(state, motor_thrust, applied_now[None], mass, inertia)
        applied_before = applied_now
        applied_now = sim.clip_commands(decided[None])[0]

        # Step 6: tracking error (true position against the path)
        ref_p, _, _ = path(t + sim.CONTROL_STEP)
        error = float(np.linalg.norm(state[0, 0:3] - ref_p))
        errors.append(error)

        # Step 7: crash check
        tilt = float(np.abs(state[0, 6:8]).max())
        if (not np.isfinite(error)) or error > 1.0 or state[0, 2] < 0.05 or tilt > np.radians(80):
            crashed = True
            break

    errors = np.array(errors)
    result = {"crashed": crashed, "errors": errors}
    warm_up = 25            # skip the first 0.5 s (every controller starts from hover)
    if not crashed:
        before = errors[warm_up:payload_step]
        after = errors[payload_step:]
        result["rmse_before"] = float(np.sqrt(np.mean(before ** 2)))
        result["rmse_after"] = float(np.sqrt(np.mean(after ** 2)))
        result["largest_after"] = float(after.max())
    if kind == "planner":
        result["planning_ms"] = float(np.median(controller.planning_times) * 1000.0)
    else:
        result["planning_ms"] = 0.0
    return result


# ======================================================================================================
# Tuning the planner (physics planner, circle path only)
# ======================================================================================================
def tune():
    scores = []
    for temperature in [0.03, 0.1, 0.3]:
        for noise_scale in [0.25, 0.5, 1.0]:
            settings = {"temperature": temperature, "noise_scale": noise_scale}
            path = circle_path(0.8, 5.0)
            result = fly("physics", path, 4.0, noise_seed=1, seed=0, settings=settings)
            score = result.get("rmse_after", 99.0) + result.get("rmse_before", 99.0)
            scores.append((score, temperature, noise_scale, result.get("rmse_before"), result.get("rmse_after"),
                           result["planning_ms"]))
            print(f"temperature {temperature} noise {noise_scale}: before {result.get('rmse_before')}"
                  f" after {result.get('rmse_after')} planning {result['planning_ms']:.1f} ms")
    scores.sort()
    print("best:", scores[0])


# ======================================================================================================
# The test
# ======================================================================================================
def test(seed, names, threads):
    torch.set_num_threads(threads)
    flights = make_test_flights()
    out_dir = os.path.join(HERE, "outputs", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for name in names:
        results = []
        for flight in tqdm(flights, desc=name):
            r = fly(name, flight["path"], flight["payload_time"], noise_seed=100 + flight["index"],
                    seed=seed, settings=PLANNER_SETTINGS)
            r["flight"] = flight["index"]
            results.append(r)
        all_results[name] = results
        np.save(os.path.join(out_dir, f"errors_{name}.npy"),
                np.array([r["errors"] for r in results], dtype=object), allow_pickle=True)
        crashes = sum(r["crashed"] for r in results)
        after = [r["rmse_after"] for r in results if not r["crashed"]]
        median_after = float(np.median(after)) if after else float("nan")
        print(f"{NICE_NAMES[name]}: crashes {crashes}, median after-payload error {median_after * 100:.2f} cm,"
              f" planning {np.median([r['planning_ms'] for r in results]):.1f} ms")

    # save one table per controller (so controllers can be flown in separate runs)
    for name in names:
        with open(os.path.join(out_dir, f"results_{name}.csv"), "w") as f:
            f.write("controller,flight,crashed,rmse_before_m,rmse_after_m,largest_after_m,planning_ms\n")
            for r in all_results[name]:
                row = [name, r["flight"], int(r["crashed"]), r.get("rmse_before", float("nan")),
                       r.get("rmse_after", float("nan")), r.get("largest_after", float("nan")),
                       r["planning_ms"]]
                f.write(",".join(str(v) for v in row) + "\n")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "tune"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controllers", default="all", help="comma-separated names, or all")
    parser.add_argument("--threads", type=int, default=2, help="processor threads for the learned models")
    args = parser.parse_args()
    if args.mode == "tune":
        tune()
    else:
        if args.controllers == "all":
            chosen = CONTROLLER_NAMES
        else:
            chosen = args.controllers.split(",")
        test(args.seed, chosen, args.threads)
