"""
Space paper, pilot 1: quantum hardware defects in a learned spacecraft controller.

What it does, in order:
    1. checks the quantum simulator's readout-error formula against a slow exact calculation
    2. makes the spacecraft training data (inertia different in every trajectory, shrinking over time)
    3. trains three Chain-Mamba predictors with your recipe: classical, quantum, quantum practising with noise
       (or loads them if they were trained before)
    4. checks the predictors (1-step and 10-step chained error, perfect hardware and combined defects)
    5. runs every controller version under every hardware condition (32 spacecraft at once, 30 seconds each)
    6. applies the pass rules from the config and writes summary.md

python tools/run_space.py \
    --config configs/space/space01_quantum_hardware.yaml \
    --seed 0 \
    --device auto \
    --quick no

Screen: a few header lines, one progress bar at a time, one line per finished run, the verdict.
Everything else: outputs/space01_quantum_hardware/log.txt
Files: results.csv (one row per version and condition; used to resume), metrics.json (per-start numbers),
       summary.md, models/*.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.space.closed_loop_space import make_test_starts, run_spacecraft_loop  # noqa: E402
from src.space.data_space import build_space_data  # noqa: E402
from src.space.mpc_space import LearnedSpacecraftPredictor, SpacecraftMPC, TrueSpacecraftPredictor  # noqa: E402
from src.space.quantum import check_readout_formula  # noqa: E402
from src.space.training_space import chained_error, train_space_model  # noqa: E402
from src.systems.spacecraft import Spacecraft  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402

RESULT_COLUMNS = ["seed", "version", "condition", "at_rest_percent", "lost_control_count", "tracking_cost_mean",
                  "tracking_cost_median", "final_angle_median", "effort_mean", "minutes"]

VERSION_LABELS = {
    "true_equations_mpc": "True equations (knows the inertia)",
    "classical_mpc": "Classical Chain-Mamba",
    "classical_mpc_offset_free": "Classical Chain-Mamba + running correction",
    "quantum_no_fix": "Quantum, no fix",
    "quantum_noise_aware_only": "Quantum, practising with noise only",
    "quantum_offset_free_only": "Quantum, running correction only",
    "quantum_both_fixes": "Quantum, both fixes",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-root", default=None, help="where outputs/ lives (default: the repository's)")
    add_yes_no(parser, "--quick", False, "code check only: tiny data, 2 epochs, 4 spacecraft, 8 steps")
    return parser.parse_args()


def read_finished_rows(results_path: Path) -> list:
    if not results_path.exists():
        return []
    with open(results_path) as handle:
        return list(csv.DictReader(handle))


def append_row(results_path: Path, row: dict) -> None:
    is_new = not results_path.exists()
    with open(results_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def summarise_run(metrics: dict) -> dict:
    """Turn per-start lists into the numbers of one results.csv row."""
    number = len(metrics["tracking_cost"])
    costs = sorted(metrics["tracking_cost"])
    angles = sorted(metrics["final_angle"])
    return {
        "at_rest_percent": 100.0 * sum(1 for value in metrics["at_rest"] if value) / number,
        "lost_control_count": sum(1 for value in metrics["lost_control"] if value),
        "tracking_cost_mean": sum(costs) / number,
        "tracking_cost_median": costs[number // 2],
        "final_angle_median": angles[number // 2],
        "effort_mean": sum(metrics["effort"]) / number,
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

    log.say(f"SPACE PILOT — {name}   (seed {seed}; one seed shows direction only)")
    if args.quick:
        log.say("QUICK MODE: code check only — do not interpret any number below")
    log.say(f"Device: {describe(device)}")
    log.detail(f"Resolved configuration: {cfg}")

    # ------------------------------------------------------------------ Step 1: simulator check
    difference = check_readout_formula()
    log.say(f"Readout-error formula vs exact calculation: largest difference {difference:.2e}")
    if difference > 1e-5:
        log.say("STOP: the readout-error formula is wrong. Nothing else was run.")
        log.close()
        return

    # ------------------------------------------------------------------ Step 2: data
    system = Spacecraft.from_config(cfg)
    data = build_space_data(cfg, system)
    log.say(f"Training windows: {data['train_x'].shape[0]:,}   (largest spin in data {data['largest_spin']:.2f} rad/s, "
            f"largest pointing error {data['largest_angle']:.0f} degrees)")

    # ------------------------------------------------------------------ Step 3: predictors
    needed = set()
    for version in cfg["versions"]:
        if version["predictor"] != "true_equations":
            needed.add(version["predictor"])
    models = {}
    for model_version in ["classical", "quantum", "quantum_noise_aware"]:
        if model_version not in needed:
            continue
        save_path = output_dir / "models" / f"{model_version}_seed{seed}.pt"
        models[model_version] = train_space_model(cfg, data, model_version, seed, device, save_path, log,
                                                  f"train {model_version} seed{seed}")

    # ------------------------------------------------------------------ Step 4: predictor check
    log.say("")
    log.say("Predictor check (chained error on test windows, normalised units, smaller is better):")
    combined = {"shots": 100, "gate_error": 0.05, "readout_a": 0.02, "readout_b": 0.20}
    check = {}
    for model_version, model in models.items():
        one_step = chained_error(model, data, cfg, device, 1)
        ten_steps = chained_error(model, data, cfg, device, 10)
        line = f"  {model_version:22s} 1 step {one_step:.4f} | 10 steps {ten_steps:.4f}"
        check[model_version] = {"one_step": one_step, "ten_steps": ten_steps}
        if model_version != "classical":
            ten_steps_defects = chained_error(model, data, cfg, device, 10, **combined)
            check[model_version]["ten_steps_combined_defects"] = ten_steps_defects
            line = line + f" | 10 steps with combined defects {ten_steps_defects:.4f}"
        log.say(line)

    # ------------------------------------------------------------------ Step 5: closed-loop runs
    results_path = output_dir / "results.csv"
    finished = read_finished_rows(results_path)
    done = set()
    for row in finished:
        done.add((int(row["seed"]), row["version"], row["condition"]))

    plan = []
    for version in cfg["versions"]:
        if version["predictor"] in ("quantum", "quantum_noise_aware"):
            conditions = cfg["conditions"]
        else:
            conditions = ["perfect"]
        for condition in conditions:
            plan.append((version, condition))

    starts = make_test_starts(cfg["test"], device)
    context_len = int(cfg["data"]["context_len"])
    steps = int(cfg["test"]["steps"])
    metrics_path = output_dir / "metrics.json"
    all_metrics = {}
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text())

    log.say("")
    log.say(f"Closed-loop runs: {len(plan)} (each: {starts.shape[0]} spacecraft, {steps} steps)")
    run_started = time.time()
    runs_this_session = 0
    for run_number, (version, condition) in enumerate(plan, start=1):
        key = (seed, version["name"], condition)
        if key in done:
            continue

        # 5a: the predictor and the controller for this version
        if version["predictor"] == "true_equations":
            predictor = TrueSpacecraftPredictor(system)
            quantum_model = None
        else:
            model = models[version["predictor"]]
            predictor = LearnedSpacecraftPredictor(model, data["norm"], device)
            quantum_model = model if version["predictor"] != "classical" else None
        controller = SpacecraftMPC(predictor, cfg["controller"], cfg["solver"], system.u_max,
                                   bool(version["offset_free"]), device)

        # 5b: run it (the random measurement scatter starts from the same seed in every run)
        torch.manual_seed(1000 + seed)
        start_time = time.time()
        bar = progress_bar(steps, f"run {run_number}/{len(plan)} {version['name']} {condition}", "step")
        metrics = run_spacecraft_loop(controller, system, starts, cfg, condition, quantum_model, context_len,
                                      device, bar)
        bar.close()
        minutes = (time.time() - start_time) / 60.0

        # 5c: save
        row = {"seed": seed, "version": version["name"], "condition": condition, "minutes": round(minutes, 2)}
        row.update(summarise_run(metrics))
        append_row(results_path, row)
        all_metrics[f"seed{seed}|{version['name']}|{condition}"] = metrics
        save_json(all_metrics, metrics_path)
        runs_this_session = runs_this_session + 1
        log.say(f"  {version['name']:28s} {condition:24s} at rest {row['at_rest_percent']:5.1f}% | "
                f"lost {row['lost_control_count']:2d} | cost {row['tracking_cost_mean']:9.1f} | {minutes:.1f} min")
        if runs_this_session == 3:
            remaining = len(plan) - run_number
            per_run = (time.time() - run_started) / 3.0 / 60.0
            log.say(f"  (estimate: about {remaining * per_run:.0f} more minutes for the remaining {remaining} runs)")

    # ------------------------------------------------------------------ Step 6: verdict
    rows = [row for row in read_finished_rows(results_path) if int(row["seed"]) == seed]
    table = {}
    for row in rows:
        table[(row["version"], row["condition"])] = {
            "at_rest": float(row["at_rest_percent"]),
            "lost": int(row["lost_control_count"]),
            "cost": float(row["tracking_cost_mean"]),
            "angle": float(row["final_angle_median"]),
        }
    write_verdict(cfg, table, check, output_dir, log, args.quick)
    log.close()


def percent_change(new_value: float, old_value: float) -> float:
    return (new_value / old_value - 1.0) * 100.0 if old_value != 0 else float("nan")


def word_for(change: float) -> str:
    if abs(change) <= 10:
        return "about the same"
    if abs(change) <= 20:
        return "slightly better" if change < 0 else "slightly worse"
    return "clearly better (on this seed)" if change < 0 else "clearly worse (on this seed)"


def write_verdict(cfg: dict, table: dict, check: dict, output_dir: Path, log, quick: bool) -> None:
    rule = cfg["decision_rule"]
    reference_name = rule["reference_version"]
    lines = []
    lines.append(f"RESULT — {cfg['name']}   (1 seed: direction only)" + ("   QUICK MODE: code check only" if quick else ""))
    lines.append("")
    lines.append("Question: " + " ".join(str(cfg["question"]).split()))
    lines.append("")

    reference = table.get((reference_name, "perfect"))
    if reference is None:
        lines.append("The reference run is missing, so no verdict can be given yet.")
        for line in lines:
            log.say(line)
        (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
        return

    # Small table: perfect hardware for every version, then the quantum versions in the combined condition
    lines.append(f"{'':48s} {'At rest':>8s} {'Lost':>5s} {'Tracking cost':>14s} {'vs reference':>13s}")
    for version in cfg["versions"]:
        for condition in ["perfect", "combined_realistic"]:
            entry = table.get((version["name"], condition))
            if entry is None:
                continue
            label = VERSION_LABELS.get(version["name"], version["name"]) + ("" if condition == "perfect" else ", combined defects")
            change = percent_change(entry["cost"], reference["cost"])
            lines.append(f"  {label:46s} {entry['at_rest']:7.0f}% {entry['lost']:5d} {entry['cost']:14.1f} {change:+12.1f}%")
    lines.append("  (tracking cost: sum of squared pointing error in degrees over the run; smaller is better)")
    lines.append("")

    # Question 1: is the problem real?
    baseline = table.get(("quantum_no_fix", "perfect"))
    hurt = []
    lines.append("1. Is the problem real? (quantum, no fix, against itself on perfect hardware)")
    if baseline is not None:
        for condition in cfg["defect_conditions"]:
            entry = table.get(("quantum_no_fix", condition))
            if entry is None:
                continue
            cost_rise = percent_change(entry["cost"], baseline["cost"])
            rest_drop = baseline["at_rest"] - entry["at_rest"]
            is_hurt = (entry["lost"] > 0) or (rest_drop >= rule["problem_real"]["at_rest_drop_points"]) \
                or (cost_rise > rule["problem_real"]["cost_rise_percent"])
            if is_hurt:
                hurt.append(condition)
            lines.append(f"   {condition:24s} cost {cost_rise:+7.1f}%  at rest {entry['at_rest']:4.0f}%  "
                         f"lost {entry['lost']:2d}  -> {'HURTS' if is_hurt else 'ok'}")
    answer_1 = "REAL" if hurt else "NOT SHOWN"
    lines.append(f"   => Problem: {answer_1}")
    lines.append("")

    # Question 2: do both fixes work?
    failed = []
    lines.append("2. Do both fixes work? (quantum with both fixes, against the reference on perfect hardware)")
    for condition in cfg["conditions"]:
        entry = table.get(("quantum_both_fixes", condition))
        if entry is None:
            continue
        cost_gap = percent_change(entry["cost"], reference["cost"])
        rest_gap = reference["at_rest"] - entry["at_rest"]
        passes = (entry["lost"] <= rule["fix_works"]["lost_control_allowed"]) \
            and (rest_gap <= rule["fix_works"]["at_rest_below_reference_points"]) \
            and (cost_gap <= rule["fix_works"]["cost_above_reference_percent"])
        if not passes:
            failed.append(condition)
        lines.append(f"   {condition:24s} cost {cost_gap:+7.1f}%  at rest {entry['at_rest']:4.0f}%  "
                     f"lost {entry['lost']:2d}  -> {'PASS' if passes else 'FAIL'}")
    if not failed:
        answer_2 = "WORKS"
    elif len(failed) <= 2:
        answer_2 = "PARTLY"
    else:
        answer_2 = "NO"
    lines.append(f"   => Fixes: {answer_2}")
    lines.append("")

    # Question 3: which fix matters?
    lines.append("3. Which fix matters? (tracking cost against the reference, %; lost-control count in brackets)")
    header = "   " + f"{'condition':24s}" + "".join(f"{name[8:]:>22s}" for name in
                                                     ["quantum_no_fix", "quantum_noise_aware_only",
                                                      "quantum_offset_free_only", "quantum_both_fixes"])
    lines.append(header)
    for condition in cfg["conditions"]:
        cells = []
        for name in ["quantum_no_fix", "quantum_noise_aware_only", "quantum_offset_free_only", "quantum_both_fixes"]:
            entry = table.get((name, condition))
            if entry is None:
                cells.append(f"{'—':>22s}")
            else:
                text = f"{percent_change(entry['cost'], reference['cost']):+.1f}% ({entry['lost']})"
                cells.append(f"{text:>22s}")
        lines.append("   " + f"{condition:24s}" + "".join(cells))
    lines.append("")

    # Question 4: is pure measurement noise negligible?
    lines.append("4. Is pure measurement noise negligible? (quantum, no fix, against itself on perfect hardware)")
    if baseline is not None:
        for condition in ["measurement_100_shots", "measurement_10_shots"]:
            entry = table.get(("quantum_no_fix", condition))
            if entry is None:
                continue
            change = percent_change(entry["cost"], baseline["cost"])
            negligible = abs(change) <= rule["measurement_noise_negligible_percent"] and entry["lost"] == 0
            lines.append(f"   {condition:24s} cost {change:+6.1f}%  lost {entry['lost']}  -> "
                         f"{'NEGLIGIBLE' if negligible else 'NOT negligible'}")
    lines.append("")

    # What this means
    lines.append("What this means:")
    true_entry = table.get(("true_equations_mpc", "perfect"))
    classical_entry = table.get(("classical_mpc", "perfect"))
    if true_entry is not None and classical_entry is not None:
        change = percent_change(classical_entry["cost"], true_entry["cost"])
        lines.append(f"  • Classical Chain-Mamba vs the true equations while the mass changes: {change:+.1f}% "
                     f"tracking cost -> {word_for(change)} (smaller is better).")
    if baseline is not None and classical_entry is not None:
        change = percent_change(baseline["cost"], classical_entry["cost"])
        lines.append(f"  • Quantum vs classical predictor, both on perfect hardware, no fixes: {change:+.1f}% "
                     f"-> {word_for(change)}.")
    no_fix_bias = table.get(("quantum_no_fix", "readout_bias_severe"))
    corrected_bias = table.get(("quantum_offset_free_only", "readout_bias_severe"))
    if no_fix_bias is not None and corrected_bias is not None:
        lines.append(f"  • Severe readout bias, final pointing error (median): no fix {no_fix_bias['angle']:.2f} degrees, "
                     f"with the running correction {corrected_bias['angle']:.2f} degrees.")
    lines.append("")
    lines.append("Math predicted: " + " ".join(str(cfg["math_signal"]["prediction"]).split()))
    lines.append("(Compare by hand: did the correction remove the offsets, and did it fail to stop any loss of control?)")
    lines.append("")
    lines.append(f"Decision: problem {answer_1}; fixes {answer_2}. One seed only: direction, not proof.")

    for line in lines:
        log.say(line)
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    save_json({"predictor_check": check, "problem": answer_1, "fixes": answer_2,
               "conditions_hurt": hurt, "conditions_failed": failed}, output_dir / "verdict.json")


if __name__ == "__main__":
    main()
