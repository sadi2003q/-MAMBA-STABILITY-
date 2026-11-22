"""
Space pilot A: satellite pointing without the steady pointing error (single seed: direction only).

What it does, in order:
    1. makes the training data (small satellite with reaction wheels, NO friction, as Cena et al.)
    2. trains your memory-removed chained predictor with your recipe (or loads it if trained before)
    3. checks the predictor: pointing error of its chained prediction after 1 and 10 steps, next to the linear model
    4. runs 5 controllers x 2 conditions (no friction / wheel friction), 32 satellites at once, 80 seconds each
    5. applies the decision rule from the config and writes summary.md  ->  GO or STOP

python tools/run_pilot_attitude.py \
    --config configs/pilots/pilot_a_attitude_offset.yaml \
    --seed 0 \
    --device cpu \
    --quick no

Screen: a few lines, one progress bar at a time, one line per finished run, the verdict.
Everything else: outputs/pilot_a_attitude_offset/log.txt
Files: results.csv (one row per controller and condition; used to resume), metrics.json, summary.md, models/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.pilots.attitude import make_attitude_cost, make_test_starts, make_trajectories, run_closed_loop  # noqa: E402
from src.pilots.mpc import PilotMPC  # noqa: E402
from src.pilots.predictors import HybridPredictor, LearnedPredictor, PhysicsPredictor  # noqa: E402
from src.pilots.report import (append_row, chained_prediction_errors, change_words, fmt, median,  # noqa: E402
                               ratio, read_rows)
from src.pilots.training import build_windows, train_predictor  # noqa: E402
from src.systems.cubesat_wheels import CubeSatWithWheels  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402

RESULT_COLUMNS = ["seed", "version", "condition", "lost_control_count", "steady_angle_median", "final_angle_median",
                  "settle_time_median", "effort_median", "holding_push_median", "planning_ms_mean", "minutes"]

LABELS = {
    "true_equations_mpc": "True equations (knows friction)",
    "linear_mpc": "Linear model only",
    "learned_mpc": "Learned model only",
    "hybrid_switch": "Hybrid switch (Cena et al.)",
    "learned_offset_free": "Learned + running correction",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-root", default=None, help="where outputs/ lives (default: the repository's)")
    add_yes_no(parser, "--quick", False, "code check only: tiny data, 2 epochs, 4 satellites, 12 steps")
    return parser.parse_args()


def make_predictor(kind: str, system: CubeSatWithWheels, condition: dict, learned_model, norm, cfg, device):
    """Build the predictor a controller plans with."""
    # Step 1: true equations (with the real friction of this condition)
    if kind == "true_equations":
        viscous = float(condition["viscous"])
        coulomb = float(condition["coulomb"])

        def true_step(state, push):
            return system.step(state, push, viscous, coulomb)

        return PhysicsPredictor(true_step)

    # Step 2: Cena et al.'s linear model
    if kind == "linear":
        return PhysicsPredictor(system.linear_step)

    # Step 3: your learned predictor
    learned = LearnedPredictor(learned_model, norm, device)
    if kind == "learned":
        return learned

    # Step 4: the hybrid switch (learned above switch_angle, linear below)
    if kind == "hybrid":
        linear = PhysicsPredictor(system.linear_step)
        return HybridPredictor(learned, linear, CubeSatWithWheels.pointing_error_degrees,
                               float(cfg["controller"]["switch_angle"]))
    raise ValueError(f"Unknown predictor kind '{kind}'")


def pointing_difference_degrees(predicted, true):
    """Small-angle pointing difference between two attitudes: 4 |sigma_pred - sigma_true|, in degrees. [W]"""
    difference = predicted[:, 0:3] - true[:, 0:3]
    size = torch.sqrt((difference * difference).sum(dim=1))
    return 4.0 * size * 180.0 / math.pi


def row_from_metrics(seed, version, condition, metrics, planning_seconds, minutes):
    return {
        "seed": seed,
        "version": version,
        "condition": condition,
        "lost_control_count": sum(1 for value in metrics["lost_control"] if value),
        "steady_angle_median": median(metrics["steady_angle"]),
        "final_angle_median": median(metrics["final_angle"]),
        "settle_time_median": median(metrics["settle_time"]),
        "effort_median": median(metrics["effort"]),
        "holding_push_median": median(metrics["holding_push"]),
        "planning_ms_mean": 1000.0 * sum(planning_seconds) / max(1, len(planning_seconds)),
        "minutes": minutes,
    }


def main():
    args = parse_arguments()

    # ------------------------------------------------------------------ Step 0: settings and folders
    cfg = load_yaml_with_inherit(args.config)
    if args.quick:
        cfg = deep_merge(cfg, cfg.get("quick", {}))
    name = cfg["name"] + ("_quick" if args.quick else "")
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs"
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(output_dir / "log.txt")
    log.capture_warnings()
    device = pick_device(args.device)
    seed = int(args.seed)

    log.say(f"SPACE PILOT A — {name}   (seed {seed}; one seed shows direction only)")
    if args.quick:
        log.say("QUICK MODE: code check only — do not interpret any number below")
    log.say(f"Device: {describe(device)}")
    log.detail(f"Resolved configuration: {cfg}")

    # ------------------------------------------------------------------ Step 1: data (no friction)
    system = CubeSatWithWheels.from_config(cfg)
    data_cfg = cfg["data"]
    data_seed = int(data_cfg["data_seed"])
    train = make_trajectories(system, data_cfg, int(data_cfg["train_trajectories"]), data_seed)
    val = make_trajectories(system, data_cfg, int(data_cfg["val_trajectories"]), data_seed + 1)
    test = make_trajectories(system, data_cfg, int(data_cfg["test_trajectories"]), data_seed + 2)
    data = build_windows(cfg, train, val, test)
    largest_wheel = float(train[0][:, :, 6:9].abs().max())
    log.say(f"Training windows: {data['train_x'].shape[0]:,}   (largest wheel speed in the data {largest_wheel:.0f} rad/s)")

    # ------------------------------------------------------------------ Step 2: your predictor
    model_path = output_dir / "models" / f"learned_seed{seed}.pt"
    model = train_predictor(cfg, data, seed, device, model_path, log, f"train memory-removed predictor seed{seed}")

    # ------------------------------------------------------------------ Step 3: predictor check
    log.say("")
    log.say("Predictor check (pointing error of the chained prediction on test trajectories, degrees, smaller is better):")
    learned_check = LearnedPredictor(model, data["norm"], device)
    linear_check = PhysicsPredictor(system.linear_step)
    horizon = int(cfg["controller"]["horizon"])
    check = {}
    for label, predictor in [("learned", learned_check), ("linear", linear_check)]:
        errors = chained_prediction_errors(predictor, test[0], test[1], horizon, 10, pointing_difference_degrees, device)
        check[label] = errors
        log.say(f"  {label:8s} after 1 step {errors[0]:.4f} | after {horizon} steps {errors[-1]:.4f}")

    # ------------------------------------------------------------------ Step 4: closed-loop runs
    results_path = output_dir / "results.csv"
    done = set()
    for row in read_rows(results_path):
        done.add((int(row["seed"]), row["version"], row["condition"]))

    metrics_path = output_dir / "metrics.json"
    all_metrics = {}
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text())

    starts = make_test_starts(cfg["test"], device)
    steps = int(cfg["test"]["steps"])
    context_len = int(data_cfg["context_len"])
    cost_function = make_attitude_cost(cfg["controller"])
    plan = []
    for condition in cfg["conditions"]:
        for version in cfg["versions"]:
            plan.append((condition, version))

    log.say("")
    log.say(f"Closed-loop runs: {len(plan)} (each: {starts.shape[0]} satellites, {steps} steps of {system.dt} s)")
    for index, (condition, version) in enumerate(plan):
        key = (seed, version["name"], condition["name"])
        if key in done:
            log.say(f"  run {index + 1}/{len(plan)} {version['name']} | {condition['name']}: already done, skipped")
            continue

        predictor = make_predictor(version["predictor"], system, condition, model, data["norm"], cfg, device)
        controller = PilotMPC(predictor, cost_function, horizon, 3, system.u_max, cfg["solver"],
                              bool(version["offset_free"]), float(cfg["controller"]["correction_filter"]), device)
        bar = progress_bar(steps, f"run {index + 1}/{len(plan)} {version['name']} | {condition['name']}", "step")
        started = time.time()
        metrics = run_closed_loop(controller, system, starts, cfg["test"], condition, context_len, device, bar)
        bar.close()
        minutes = (time.time() - started) / 60.0

        row = row_from_metrics(seed, version["name"], condition["name"], metrics, controller.planning_seconds, minutes)
        append_row(results_path, row, RESULT_COLUMNS)
        all_metrics[f"{version['name']}|{condition['name']}|seed{seed}"] = metrics
        save_json(all_metrics, metrics_path)
        log.say(f"  run {index + 1}/{len(plan)} {version['name']:20s} | {condition['name']:14s} "
                f"steady error {row['steady_angle_median']:.4f} deg, settle {fmt(row['settle_time_median'], 1)} s, "
                f"lost {row['lost_control_count']}, {minutes:.1f} min")

    # ------------------------------------------------------------------ Step 5: decision and summary
    rows = {}
    for row in read_rows(results_path):
        if int(row["seed"]) == seed:
            rows[(row["version"], row["condition"])] = row
    write_summary(cfg, rows, check, seed, args.quick, output_dir, log)
    save_json({"predictor_check_degrees": check}, output_dir / "predictor_check.json")
    log.close()


def number(rows, version, condition, column) -> float:
    return float(rows[(version, condition)][column])


def write_summary(cfg, rows, check, seed, quick, output_dir, log):
    rule = cfg["decision_rule"]
    proposal = rule["proposal"]
    competitor = rule["competitor"]
    conditions = [condition["name"] for condition in cfg["conditions"]]
    versions = [version["name"] for version in cfg["versions"]]
    lines = []
    lines.append(f"RESULT — {cfg['name']}   (1 seed: direction only)")
    if quick:
        lines.append("QUICK MODE: code check only — do not interpret")
    lines.append("")
    lines.append("Question: can the running correction replace Cena et al.'s switch to a linear model, and does it")
    lines.append("still work when the real wheels have friction the training data never had?")
    lines.append("")

    # Step 1: one small table per condition
    for condition in conditions:
        lines.append(f"Condition: {condition.replace('_', ' ')}   (smaller is better in every column)")
        lines.append("                                     Steady pointing   Settling   Push effort   Lost")
        lines.append("                                     error (degrees)   time (s)")
        for version in versions:
            if (version, condition) not in rows:
                continue
            steady = number(rows, version, condition, "steady_angle_median")
            settle = number(rows, version, condition, "settle_time_median")
            effort = number(rows, version, condition, "effort_median")
            lost = int(float(rows[(version, condition)]["lost_control_count"]))
            lines.append(f"  {LABELS.get(version, version):35s} {steady:12.4f}     {fmt(settle, 1):>8s}   "
                         f"{effort:9.2f}     {lost}")
        lines.append("")

    # Step 2: the decision rule
    missing = []
    for condition in conditions:
        for version in versions:
            if (version, condition) not in rows:
                missing.append(f"{version}|{condition}")
    if missing:
        lines.append(f"Decision: NOT READY (runs missing: {', '.join(missing)})")
    else:
        real = rule["problem_real"]
        problem_real = False
        for condition in conditions:
            learned = number(rows, "learned_mpc", condition, "steady_angle_median")
            truth = number(rows, "true_equations_mpc", condition, "steady_angle_median")
            if learned >= float(real["steady_angle_times"]) * truth and learned >= float(real["steady_angle_at_least_degrees"]):
                problem_real = True

        go = rule["go"]
        p_free = number(rows, proposal, "no_friction", "steady_angle_median")
        c_free = number(rows, competitor, "no_friction", "steady_angle_median")
        p_fric = number(rows, proposal, "wheel_friction", "steady_angle_median")
        c_fric = number(rows, competitor, "wheel_friction", "steady_angle_median")
        lost = 0
        settle_ok = True
        for condition in conditions:
            lost = lost + int(float(rows[(proposal, condition)]["lost_control_count"]))
            p_settle = number(rows, proposal, condition, "settle_time_median")
            c_settle = number(rows, competitor, condition, "settle_time_median")
            if ratio(p_settle, c_settle) > float(go["settle_time_at_most_times"]):
                settle_ok = False
        free_ok = ratio(p_free, c_free) <= float(go["no_friction_steady_angle_at_most_times"])
        fric_ok = ratio(p_fric, c_fric) <= float(go["friction_steady_angle_at_most_times"])
        lost_ok = lost <= int(go["lost_control_allowed"])

        lines.append("What this means:")
        for condition in conditions:
            learned = number(rows, "learned_mpc", condition, "steady_angle_median")
            truth = number(rows, "true_equations_mpc", condition, "steady_angle_median")
            lines.append(f"  • {condition.replace('_', ' ')}: learned model alone ends {fmt(learned, 4)} degrees off "
                         f"(true equations {fmt(truth, 4)}).")
        lines.append(f"  • Problem real (learned alone at least {real['steady_angle_times']}x the true equations): "
                     f"{'yes' if problem_real else 'no'}")
        lines.append(f"  • Running correction vs hybrid switch, no friction: {change_words(p_free, c_free)}")
        lines.append(f"  • Running correction vs hybrid switch, wheel friction: {change_words(p_fric, c_fric)}")
        for condition in conditions:
            p_effort = number(rows, proposal, condition, "effort_median")
            c_effort = number(rows, competitor, condition, "effort_median")
            lines.append(f"  • Push effort vs hybrid switch, {condition.replace('_', ' ')}: {change_words(p_effort, c_effort)}")
        lines.append("")
        lines.append("Math predicted: under friction the linear model, the learned model alone and the hybrid keep a")
        lines.append("steady error (about 0.1-0.3 degrees); the running correction removes it.")
        hybrid_kept = c_fric >= 3.0 * max(number(rows, "true_equations_mpc", "wheel_friction", "steady_angle_median"), 1e-6)
        held = hybrid_kept and fric_ok
        lines.append(f"Result {'agreed with' if held else 'did NOT agree with'} the prediction.")
        lines.append("")
        checks = [
            ("problem is real", problem_real),
            (f"no friction: at most {go['no_friction_steady_angle_at_most_times']}x the hybrid's error", free_ok),
            (f"wheel friction: at most {go['friction_steady_angle_at_most_times']}x the hybrid's error", fric_ok),
            (f"settling no slower than {go['settle_time_at_most_times']}x the hybrid", settle_ok),
            ("no satellite lost", lost_ok),
        ]
        for text, ok in checks:
            lines.append(f"  [{'pass' if ok else 'FAIL'}] {text}")
        verdict = all(ok for _, ok in checks)
        lines.append("")
        lines.append(f"Decision: {'GO — worth confirming with more seeds' if verdict else 'STOP — not worth continuing as designed'}")

    lines.append("")
    lines.append(f"Predictor check (pointing error after {len(check['learned'])} chained steps, degrees): "
                 f"learned {check['learned'][-1]:.4f}, linear {check['linear'][-1]:.4f}")
    text = "\n".join(lines)
    (output_dir / "summary.md").write_text(text + "\n")
    log.say("")
    for line in lines:
        log.say(line)


if __name__ == "__main__":
    main()
