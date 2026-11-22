"""
Space pilot B: staying on a planned orbit near an irregular, spinning asteroid (single seed: direction only).

What it does, in order:
    1. builds the two asteroid models: high fidelity (the real asteroid) and low fidelity (the paper's errors)
    2. makes training data from the LOW-fidelity model over a wide shell of orbits (as the paper, but wider)
    3. trains your memory-removed chained predictor with your recipe (or loads it if trained before)
    4. checks the predictors against the REAL asteroid: position error of an 80-minute chained prediction
    5. runs 5 controllers on 4 different orbits (8 spacecraft each, 12 hours) near the real asteroid
    6. applies the decision rule from the config and writes summary.md  ->  GO or STOP

python tools/run_pilot_asteroid.py \
    --config configs/pilots/pilot_b_asteroid_proximity.yaml \
    --seed 0 \
    --device auto \
    --quick no

Screen: a few lines, one progress bar at a time, one line per finished run, the verdict.
Everything else: outputs/pilot_b_asteroid_proximity/log.txt
Files: results.csv (one row per controller and orbit; used to resume), metrics.json, summary.md, models/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.pilots.asteroid import (make_asteroid_cost, make_orbit_starts, make_trajectories,  # noqa: E402
                                 run_closed_loop, spin_rate_from_config)
from src.pilots.mpc import PilotMPC  # noqa: E402
from src.pilots.predictors import LearnedPredictor, PhysicsPredictor  # noqa: E402
from src.pilots.report import (append_row, chained_prediction_errors, change_words, fmt, median,  # noqa: E402
                               ratio, read_rows)
from src.pilots.training import build_windows, train_predictor  # noqa: E402
from src.systems.small_body import SmallBody  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402

RESULT_COLUMNS = ["seed", "version", "orbit", "lost_count", "position_error_median_m", "largest_error_median_m",
                  "fuel_median_m_per_s", "planning_ms_mean", "minutes"]

LABELS = {
    "true_physics_mpc": "True physics (real asteroid)",
    "nominal_physics_mpc": "Low-fidelity physics (paper's)",
    "nominal_physics_offset_free": "Low-fidelity physics + correction",
    "learned_mpc": "Learned model only",
    "learned_offset_free": "Learned + running correction",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-root", default=None, help="where outputs/ lives (default: the repository's)")
    add_yes_no(parser, "--quick", False, "code check only: tiny data, 2 epochs, 4 spacecraft, 6 steps")
    return parser.parse_args()


def position_difference_m(predicted, true):
    """Distance between predicted and true position, metres. [W]"""
    difference = predicted[:, 0:3] - true[:, 0:3]
    return 1000.0 * torch.sqrt((difference * difference).sum(dim=1))


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

    log.say(f"SPACE PILOT B — {name}   (seed {seed}; one seed shows direction only)")
    if args.quick:
        log.say("QUICK MODE: code check only — do not interpret any number below")
    log.say(f"Device: {describe(device)}")
    log.detail(f"Resolved configuration: {cfg}")

    # ------------------------------------------------------------------ Step 1: the two asteroid models
    real_body = SmallBody.from_config(cfg, "high")
    low_body = SmallBody.from_config(cfg, "low")
    gm = float(cfg["system"]["gm"])
    true_spin = spin_rate_from_config(cfg)
    log.say(f"Asteroid models: real {real_body.mass_points.shape[0]} mass points, "
            f"low fidelity {low_body.mass_points.shape[0]} (+10% density, +1% spin, spin axis tilted 3 degrees)")

    # ------------------------------------------------------------------ Step 2: data (low fidelity, as the paper)
    data_cfg = cfg["data"]
    data_seed = int(data_cfg["data_seed"])
    bar = progress_bar(3, "making trajectories", "set")
    train = make_trajectories(low_body, data_cfg, int(data_cfg["train_trajectories"]), data_seed, gm, true_spin)
    bar.update(1)
    val = make_trajectories(low_body, data_cfg, int(data_cfg["val_trajectories"]), data_seed + 1, gm, true_spin)
    bar.update(1)
    test = make_trajectories(low_body, data_cfg, int(data_cfg["test_trajectories"]), data_seed + 2, gm, true_spin)
    bar.update(1)
    bar.close()
    # trajectories near the REAL asteroid, only for checking predictions (never for training)
    real_test = make_trajectories(real_body, data_cfg, int(data_cfg["test_trajectories"]), data_seed + 3, gm, true_spin)
    data = build_windows(cfg, train, val, test)
    log.say(f"Training windows: {data['train_x'].shape[0]:,} from {train[0].shape[0]:,} low-fidelity trajectories")

    # ------------------------------------------------------------------ Step 3: your predictor
    model_path = output_dir / "models" / f"learned_seed{seed}.pt"
    model = train_predictor(cfg, data, seed, device, model_path, log, f"train memory-removed predictor seed{seed}")

    # ------------------------------------------------------------------ Step 4: predictor check vs the REAL asteroid
    horizon = int(cfg["controller"]["horizon"])
    log.say("")
    log.say(f"Predictor check (position error of a chained prediction vs the REAL asteroid, metres, smaller is better):")
    predictors_to_check = [
        ("learned", LearnedPredictor(model, data["norm"], device)),
        ("low-fidelity physics", PhysicsPredictor(low_body.step)),
    ]
    check = {}
    for label, predictor in predictors_to_check:
        errors = chained_prediction_errors(predictor, real_test[0], real_test[1], horizon, 8, position_difference_m,
                                           device)
        check[label] = errors
        log.say(f"  {label:22s} after 1 step {errors[0]:8.1f} | after {horizon} steps {errors[-1]:8.1f}")
    learned_self = chained_prediction_errors(LearnedPredictor(model, data["norm"], device), test[0], test[1], horizon, 8,
                                             position_difference_m, device)
    check["learned_vs_low_fidelity"] = learned_self
    log.say(f"  learned vs the low-fidelity model it was trained on: after {horizon} steps {learned_self[-1]:.1f} m "
            f"(its own fitting error)")

    # ------------------------------------------------------------------ Step 5: closed-loop runs
    results_path = output_dir / "results.csv"
    done = set()
    for row in read_rows(results_path):
        done.add((int(row["seed"]), row["version"]))

    metrics_path = output_dir / "metrics.json"
    all_metrics = {}
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text())

    test_cfg = cfg["test"]
    starts, references, orbit_names = make_orbit_starts(test_cfg, gm, true_spin, real_body.dt, horizon, device)
    steps = int(test_cfg["steps"])
    context_len = int(data_cfg["context_len"])
    cost_function = make_asteroid_cost(cfg["controller"])
    orbit_list = [orbit["name"] for orbit in test_cfg["orbits"]]

    log.say("")
    log.say(f"Closed-loop runs: {len(cfg['versions'])} controllers (each: {starts.shape[0]} spacecraft on "
            f"{len(orbit_list)} orbits, {steps} steps of {real_body.dt / 60:.0f} minutes)")
    for index, version in enumerate(cfg["versions"]):
        if (seed, version["name"]) in done:
            log.say(f"  run {index + 1}/{len(cfg['versions'])} {version['name']}: already done, skipped")
            continue

        # Step 5a: the predictor this controller plans with
        kind = version["predictor"]
        if kind == "true_physics":
            predictor = PhysicsPredictor(real_body.step)
        elif kind == "low_fidelity_physics":
            predictor = PhysicsPredictor(low_body.step)
        else:
            predictor = LearnedPredictor(model, data["norm"], device)
        controller = PilotMPC(predictor, cost_function, horizon, 3, real_body.u_max, cfg["solver"],
                              bool(version["offset_free"]), float(cfg["controller"]["correction_filter"]), device)

        # Step 5b: run every orbit at once
        bar = progress_bar(steps, f"run {index + 1}/{len(cfg['versions'])} {version['name']}", "step")
        started = time.time()
        metrics = run_closed_loop(controller, real_body, starts, references, test_cfg, horizon, context_len, device, bar)
        bar.close()
        minutes = (time.time() - started) / 60.0
        planning_ms = 1000.0 * sum(controller.planning_seconds) / max(1, len(controller.planning_seconds))

        # Step 5c: one row per orbit
        for orbit in orbit_list:
            chosen = [i for i in range(len(orbit_names)) if orbit_names[i] == orbit]
            row = {
                "seed": seed,
                "version": version["name"],
                "orbit": orbit,
                "lost_count": sum(1 for i in chosen if metrics["lost"][i]),
                "position_error_median_m": median([metrics["mean_position_error_m"][i] for i in chosen]),
                "largest_error_median_m": median([metrics["largest_position_error_m"][i] for i in chosen]),
                "fuel_median_m_per_s": median([metrics["fuel_m_per_s"][i] for i in chosen]),
                "planning_ms_mean": planning_ms,
                "minutes": minutes,
            }
            append_row(results_path, row, RESULT_COLUMNS)
        metrics["orbit"] = orbit_names
        all_metrics[f"{version['name']}|seed{seed}"] = metrics
        save_json(all_metrics, metrics_path)
        overall = median(metrics["mean_position_error_m"])
        lost = sum(1 for value in metrics["lost"] if value)
        log.say(f"  run {index + 1}/{len(cfg['versions'])} {version['name']:28s} position error {overall:8.1f} m "
                f"(median over all), lost {lost}, planning {planning_ms:.0f} ms per step, {minutes:.1f} min")

    # ------------------------------------------------------------------ Step 6: decision and summary
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    rows = {}
    for row in read_rows(results_path):
        if int(row["seed"]) == seed:
            rows[(row["version"], row["orbit"])] = row
    write_summary(cfg, rows, all_metrics, check, seed, args.quick, orbit_list, output_dir, log)
    save_json({"predictor_check_metres": check}, output_dir / "predictor_check.json")
    log.close()


def pooled(all_metrics, version, seed, key):
    """Median over ALL spacecraft (every orbit) of one metric for one controller."""
    metrics = all_metrics.get(f"{version}|seed{seed}")
    if metrics is None:
        return None
    return median(metrics[key])


def write_summary(cfg, rows, all_metrics, check, seed, quick, orbit_list, output_dir, log):
    rule = cfg["decision_rule"]
    proposal = rule["proposal"]
    competitor = rule["competitor"]
    strong_baseline = rule["strong_baseline"]
    versions = [version["name"] for version in cfg["versions"]]
    lines = []
    lines.append(f"RESULT — {cfg['name']}   (1 seed: direction only)")
    if quick:
        lines.append("QUICK MODE: code check only — do not interpret")
    lines.append("")
    lines.append("Question: can a small learned predictor trained on a rough asteroid model, plus the running")
    lines.append("correction, keep spacecraft on four different orbits near the real asteroid as well as planning")
    lines.append("with the rough physics model itself?")
    lines.append("")

    # Step 1: table — median position error per orbit, then fuel and planning time
    header = "                                     " + "".join(f"{orbit[:16]:>18s}" for orbit in orbit_list)
    lines.append("Position error from the planned orbit (metres, median per orbit; smaller is better)")
    lines.append(header)
    for version in versions:
        values = []
        for orbit in orbit_list:
            if (version, orbit) in rows:
                values.append(f"{float(rows[(version, orbit)]['position_error_median_m']):18.1f}")
            else:
                values.append(f"{'-':>18s}")
        lines.append(f"  {LABELS.get(version, version):35s}" + "".join(values))
    lines.append("")
    lines.append("                                     Fuel (m/s, median)   Lost   Planning time per step (ms)")
    for version in versions:
        fuel = pooled(all_metrics, version, seed, "fuel_m_per_s")
        if fuel is None:
            continue
        lost = sum(1 for value in all_metrics[f"{version}|seed{seed}"]["lost"] if value)
        planning = float(rows[(version, orbit_list[0])]["planning_ms_mean"])
        lines.append(f"  {LABELS.get(version, version):35s} {fuel:12.2f}        {lost:4d}   {planning:10.0f}")
    lines.append("")

    # Step 2: the decision rule
    missing = [version for version in versions if pooled(all_metrics, version, seed, "fuel_m_per_s") is None]
    if missing:
        lines.append(f"Decision: NOT READY (runs missing: {', '.join(missing)})")
    else:
        truth = pooled(all_metrics, "true_physics_mpc", seed, "mean_position_error_m")
        nominal = pooled(all_metrics, competitor, seed, "mean_position_error_m")
        problem_real = ratio(nominal, truth) >= float(rule["problem_real"]["position_error_times"])

        go = rule["go"]
        every_orbit_ok = True
        orbit_words = []
        for orbit in orbit_list:
            p = float(rows[(proposal, orbit)]["position_error_median_m"])
            c = float(rows[(competitor, orbit)]["position_error_median_m"])
            orbit_words.append(f"{orbit}: {change_words(p, c)}")
            if ratio(p, c) > float(go["position_error_at_most_times_every_orbit"]):
                every_orbit_ok = False
        p_fuel = pooled(all_metrics, proposal, seed, "fuel_m_per_s")
        c_fuel = pooled(all_metrics, competitor, seed, "fuel_m_per_s")
        fuel_ok = ratio(p_fuel, c_fuel) <= float(go["fuel_at_most_times"])
        lost = sum(1 for value in all_metrics[f"{proposal}|seed{seed}"]["lost"] if value)
        lost_ok = lost <= int(go["lost_allowed"])
        p_all = pooled(all_metrics, proposal, seed, "mean_position_error_m")
        s_all = pooled(all_metrics, strong_baseline, seed, "mean_position_error_m")
        strong = ratio(p_all, s_all) <= float(rule["strong"]["position_error_at_most_times_strong_baseline"])

        lines.append("What this means:")
        lines.append(f"  • Rough physics vs true physics: {change_words(nominal, truth)} -> problem real: "
                     f"{'yes' if problem_real else 'no'}")
        lines.append(f"  • Learned + correction vs rough physics (the paper's controller), per orbit:")
        for words in orbit_words:
            lines.append(f"      {words}")
        lines.append(f"  • Fuel vs rough physics: {change_words(p_fuel, c_fuel)}")
        lines.append(f"  • Learned + correction vs rough physics + correction (strong baseline): "
                     f"{change_words(p_all, s_all)}")
        lines.append("")
        lines.append("Math predicted: rough physics clearly worse than true physics; the correction helps both")
        lines.append("planners; the learned planner close to rough physics + correction.")
        nominal_free = pooled(all_metrics, "nominal_physics_offset_free", seed, "mean_position_error_m")
        held = problem_real and nominal_free < nominal
        lines.append(f"Result {'agreed with' if held else 'did NOT agree with'} the prediction (first two parts).")
        lines.append("")
        checks = [
            ("problem is real", problem_real),
            (f"every orbit: at most {go['position_error_at_most_times_every_orbit']}x the rough-physics controller's error",
             every_orbit_ok),
            (f"fuel at most {go['fuel_at_most_times']}x", fuel_ok),
            ("no spacecraft lost", lost_ok),
        ]
        for text, ok in checks:
            lines.append(f"  [{'pass' if ok else 'FAIL'}] {text}")
        verdict = all(ok for _, ok in checks)
        lines.append(f"  (extra, not part of the decision) within 10% of the strong baseline: {'yes' if strong else 'no'}")
        lines.append("")
        lines.append(f"Decision: {'GO — worth confirming with more seeds' if verdict else 'STOP — not worth continuing as designed'}")

    lines.append("")
    lines.append(f"Predictor check vs the real asteroid after {len(check['learned'])} chained steps (metres): "
                 f"learned {fmt(check['learned'][-1], 1)}, low-fidelity physics {fmt(check['low-fidelity physics'][-1], 1)}; "
                 f"learned vs its own training model {fmt(check['learned_vs_low_fidelity'][-1], 1)}")
    lines.append("Planning times here are NOT comparable with the paper's 3.58 s (different gravity model and computer).")
    text = "\n".join(lines)
    (output_dir / "summary.md").write_text(text + "\n")
    log.say("")
    for line in lines:
        log.say(line)


if __name__ == "__main__":
    main()
