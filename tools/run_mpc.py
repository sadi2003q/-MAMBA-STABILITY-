"""
Model predictive control, step 1: bring the Van der Pol oscillator to rest at the origin.

Three controllers, the same 100 starting points, the same real oscillator:
    no_control          push always 0
    perfect_model_mpc   model predictive control planning with the TRUE equations (best possible)
    learned_model_mpc   model predictive control planning with the memory-removed chained predictor

python tools/run_mpc.py \
    --config configs/control/mpc01_stabilise.yaml

Options:
    --checkpoint PATH   the trained predictor (default: the path in the config)
    --device cpu        cpu | cuda | auto  (default cpu: the model is tiny, the processor is fastest)
    --quick yes         code check only (5 starts, 20 steps); default no
    --output-root PATH  where outputs/ lives (default: the repository's outputs/)

Screen: a few header lines, the predictor check, one progress bar per controller, the result table.
Everything else goes to outputs/mpc01_stabilise/log.txt.
Files written: results.csv (one row per controller and start), metrics.json, summary.md, figures/*.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw figures to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.control.closed_loop import run_closed_loop, run_metrics  # noqa: E402
from src.control.mpc import MPCController  # noqa: E402
from src.control.predictors import LearnedModelPredictor, TrueModelPredictor  # noqa: E402
from src.data.datasets import build_data  # noqa: E402
from src.models.registry import build_model, count_parameters  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

# Labels used on screen, in the summary and in the figures (no abbreviations)
CONTROLLER_LABELS = {
    "no_control": "No control (push = 0)",
    "perfect_model_mpc": "Predictive control, true equations",
    "learned_model_mpc": "Predictive control, learned model",
}


# ==========================================================================================================
# Command line
# ==========================================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Model predictive control of the Van der Pol oscillator.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="trained predictor (default: from the config)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto (default cpu)")
    add_yes_no(p, "--quick", False, "code check only: few starts, few steps")
    p.add_argument("--output-root", default=None, help="where outputs/ lives (default: repository outputs/)")
    return p.parse_args()


# ==========================================================================================================
# Loading the trained predictor
# ==========================================================================================================
def load_trained_model(checkpoint_path: Path, device: torch.device):
    """Rebuild the model from the checkpoint's own settings, then load the trained weights."""

    # Step 1: read the file (weights + the full training configuration)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]

    # Step 2: build the same model as in training, then put the trained weights in
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, model_cfg


# ==========================================================================================================
# Check the learned predictor BEFORE using it for control
# ==========================================================================================================
def predictor_check(learned, perfect, test_split, horizon: int, windows: int, device) -> dict:
    """
    Take pieces of the TEST trajectory: a true starting state and the true next `horizon` pushes.
    Ask both predictors where the oscillator goes, and compare with where it really went.

    Why: a mistake in the unit conversion (normalised <-> physical) would make the learned controller fail
    for a reason that has nothing to do with the model. This check catches it before control starts.
    Errors are in physical units (position and velocity together).
    """
    # Step 1: evenly spaced starting points along the test trajectory
    number_of_pushes_available = len(test_split.u)
    last_start = number_of_pushes_available - horizon - 1
    start_indices = np.linspace(0, last_start, windows).astype(int)

    # Step 2: build the inputs and the true answers, one piece at a time
    x_now_list = []
    u_plan_list = []
    truth_list = []
    for i in start_indices:
        x_now_list.append(test_split.x[i])                              # [2]
        u_plan_list.append(test_split.u[i:i + horizon, 0])              # [N]
        truth_list.append(test_split.x[i + 1:i + horizon + 1])          # [N, 2]

    x_now = torch.tensor(np.array(x_now_list), dtype=torch.float32, device=device)
    u_plan = torch.tensor(np.array(u_plan_list), dtype=torch.float32, device=device)
    truth = np.array(truth_list)

    # Step 3: predictions (no gradient needed here)
    with torch.no_grad():
        learned_prediction = learned.predict(x_now, u_plan).cpu().numpy()
        perfect_prediction = perfect.predict(x_now, u_plan).cpu().numpy()

    # Step 4: root-mean-square error at the first step and at the last planned step
    def rms_error(prediction, step_index):
        difference = prediction[:, step_index, :] - truth[:, step_index, :]
        return float(np.sqrt(np.mean(difference * difference)))

    last = horizon - 1
    result = {}
    result["learned_step1"] = rms_error(learned_prediction, 0)
    result["learned_last_step"] = rms_error(learned_prediction, last)
    result["perfect_last_step"] = rms_error(perfect_prediction, last)
    result["typical_state_size"] = float(np.sqrt(np.mean(truth * truth)))
    result["windows"] = int(len(start_indices))
    return result


# ==========================================================================================================
# Summary helpers
# ==========================================================================================================
def median_of(values) -> float:
    """Median that ignores NaN values (returns NaN if there is nothing left)."""
    clean = []
    for v in values:
        if not math.isnan(v):
            clean.append(v)
    if len(clean) == 0:
        return float("nan")
    return float(np.median(clean))


def describe_difference(percent: float) -> str:
    """Word label for an error difference in percent (positive = worse, because smaller error is better)."""
    if math.isnan(percent):
        return "cannot compare"
    size = abs(percent)
    if size <= 10:
        return "about the same"
    if percent > 0:
        direction = "worse"
    else:
        direction = "better"
    if size <= 20:
        return f"slightly {direction}"
    return f"clearly {direction} (on this seed)"


def summarise_controller(rows: list) -> dict:
    """Medians and counts over all starts of one controller."""
    position_errors = []
    settle_times = []
    push_energies = []
    at_rest_count = 0
    blew_up_count = 0
    for row in rows:
        position_errors.append(row["position_error"])
        settle_times.append(row["settle_time"])
        push_energies.append(row["push_energy"])
        if row["at_rest"]:
            at_rest_count = at_rest_count + 1
        if row["blew_up"]:
            blew_up_count = blew_up_count + 1

    summary = {}
    summary["starts"] = len(rows)
    summary["at_rest"] = at_rest_count
    summary["blew_up"] = blew_up_count
    summary["median_position_error"] = median_of(position_errors)
    summary["median_settle_time"] = median_of(settle_times)
    summary["median_push_energy"] = median_of(push_energies)
    return summary


def format_number(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}g}"


# ==========================================================================================================
# Figures
# ==========================================================================================================
def draw_figures(runs: dict, dt: float, target, figure_dir: Path, how_many: int = 20) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = list(runs.keys())

    # Figure 1: paths in the position-velocity plane, one panel per controller
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharex=True, sharey=True)
    for panel, name in zip(axes, names):
        states = runs[name]["states"]
        number_to_draw = min(how_many, states.shape[0])
        for s in range(number_to_draw):
            panel.plot(states[s, :, 0], states[s, :, 1], linewidth=0.8)
            panel.plot(states[s, 0, 0], states[s, 0, 1], "o", markersize=3, color="black")
        panel.plot(target[0], target[1], "x", markersize=10, color="red", label="target")
        panel.set_title(CONTROLLER_LABELS[name])
        panel.set_xlabel("position")
        panel.grid(alpha=0.3)
    axes[0].set_ylabel("velocity")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Paths from the first {how_many} starting points (black dot = start)")
    fig.tight_layout()
    fig.savefig(figure_dir / "paths_position_velocity.png", dpi=130)
    plt.close(fig)

    # Figure 2: first starting point over time: position (top) and push (bottom)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name in names:
        states = runs[name]["states"]
        pushes = runs[name]["pushes"]
        time_of_states = np.arange(states.shape[1]) * dt
        time_of_pushes = np.arange(pushes.shape[1]) * dt
        top.plot(time_of_states, states[0, :, 0], label=CONTROLLER_LABELS[name])
        bottom.step(time_of_pushes, pushes[0, :], where="post", label=CONTROLLER_LABELS[name])
    top.axhline(target[0], color="gray", linestyle="--", linewidth=0.8)
    top.set_ylabel("position")
    top.legend()
    top.grid(alpha=0.3)
    bottom.set_ylabel("push")
    bottom.set_xlabel("time (seconds)")
    bottom.grid(alpha=0.3)
    fig.suptitle("First starting point over time")
    fig.tight_layout()
    fig.savefig(figure_dir / "first_start_over_time.png", dpi=130)
    plt.close(fig)


# ==========================================================================================================
# Main program
# ==========================================================================================================
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: settings
    # ------------------------------------------------------------------
    cfg = load_yaml_with_inherit(args.config)
    if args.quick:
        cfg = deep_merge(cfg, cfg.get("quick", {}))

    controller_cfg = cfg["controller"]
    solver_cfg = cfg["solver"]
    test_cfg = cfg["test"]
    horizon = int(controller_cfg["horizon"])
    target = controller_cfg["target"]

    # ------------------------------------------------------------------
    # Step 2: output folder and log file
    # ------------------------------------------------------------------
    if args.output_root is None:
        root = REPO_ROOT / "outputs"
    else:
        root = Path(args.output_root)
    folder_name = cfg["name"]
    if args.quick:
        folder_name = folder_name + "_quick"
    out = root / folder_name
    out.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(out / "log.txt", fresh=True)

    set_seed(0)
    device = pick_device(args.device)

    log.say(f"Model predictive control — {cfg['name']}")
    if args.quick:
        log.say("  QUICK MODE: code check only — do not interpret the numbers")
    log.say(f"  Device: {describe(device)}")

    # ------------------------------------------------------------------
    # Step 3: the trained predictor
    # ------------------------------------------------------------------
    if args.checkpoint is None:
        checkpoint_path = Path(cfg["checkpoint"])
    else:
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        log.say(f"  Trained predictor not found: {checkpoint_path}")
        log.say("  Pass a trained predictor with --checkpoint PATH (see configs/control/mpc01_stabilise.yaml).")
        log.close()
        sys.exit(1)

    model, model_cfg = load_trained_model(checkpoint_path, device)
    number_of_parameters = count_parameters(model)
    log.say(f"  Predictor: {checkpoint_path}  ({number_of_parameters:,} parameters)")

    # ------------------------------------------------------------------
    # Step 4: the true oscillator and the normalisation numbers
    # (rebuilt from the same settings and data seed as in training, so they are identical)
    # ------------------------------------------------------------------
    data = build_data(model_cfg)
    system = data["system"]
    norm = data["norm"]
    dt = system.dt
    log.detail(f"Normalisation numbers: {norm.to_dict()}")
    log.detail(f"Settings: controller {controller_cfg}, solver {solver_cfg}, test {test_cfg}")

    # ------------------------------------------------------------------
    # Step 5: the two predictors and the predictor check
    # ------------------------------------------------------------------
    perfect_predictor = TrueModelPredictor(system.mu, system.dt, include_minus_x1=(system.k == 1.0))
    learned_predictor = LearnedModelPredictor(model, norm, device)
    if learned_predictor.has_memory:
        log.say("  Note: this model has memory; every plan starts with an empty memory (no warm-up).")

    check = predictor_check(learned_predictor, perfect_predictor, data["splits"]["test"], horizon,
                            int(cfg["predictor_check"]["windows"]), device)
    log.say(f"  Predictor check on test data ({check['windows']} pieces, physical units): "
            f"error after 1 step {check['learned_step1']:.4f}, after {horizon} steps "
            f"{check['learned_last_step']:.4f}  (typical state size {check['typical_state_size']:.2f})")
    log.detail(f"Predictor check, true equations after {horizon} steps (should be about 0): "
               f"{check['perfect_last_step']:.2e}")

    # ------------------------------------------------------------------
    # Step 6: the starting points (the same for every controller)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(test_cfg["start_seed"]))
    number_of_starts = int(test_cfg["starts"])
    low = np.array(test_cfg["start_low"], dtype=float)
    high = np.array(test_cfg["start_high"], dtype=float)
    starts = rng.uniform(low, high, size=(number_of_starts, 2))
    steps = int(test_cfg["steps"])
    settle_radius = float(test_cfg["settle_radius"])

    # ------------------------------------------------------------------
    # Step 7: the three controllers
    # ------------------------------------------------------------------
    controllers = {}
    controllers["no_control"] = None
    controllers["perfect_model_mpc"] = MPCController(perfect_predictor, controller_cfg, solver_cfg, device)
    controllers["learned_model_mpc"] = MPCController(learned_predictor, controller_cfg, solver_cfg, device)

    # ------------------------------------------------------------------
    # Step 8: run every controller on all starting points at once
    # ------------------------------------------------------------------
    runs = {}
    rows = []
    for name, controller in controllers.items():
        bar = progress_bar(steps, CONTROLLER_LABELS[name], "step")
        run = run_closed_loop(controller, system, starts, steps, device, bar=bar)
        bar.close()
        runs[name] = run

        # metrics for each start separately
        for s in range(number_of_starts):
            metrics = run_metrics(run["states"][s], run["pushes"][s], run["blew_up"][s], target, dt,
                                  settle_radius)
            row = {"controller": name, "start": s,
                   "start_position": float(starts[s, 0]), "start_velocity": float(starts[s, 1])}
            row.update(metrics)
            rows.append(row)

        log.detail(f"{name}: finished, total planning time {run['plan_seconds'].sum():.1f} s "
                   f"for {number_of_starts} oscillators together")

    # ------------------------------------------------------------------
    # Step 9: planning time for ONE oscillator (what a real controller would need)
    # ------------------------------------------------------------------
    planning_time = {}
    timing_steps = int(test_cfg["timing_steps"])
    for name, controller in controllers.items():
        if controller is None:
            planning_time[name] = float("nan")
            continue
        one_start = starts[:1]
        timing_run = run_closed_loop(controller, system, one_start, timing_steps, device)
        planning_time[name] = float(np.median(timing_run["plan_seconds"]))
        log.detail(f"{name}: planning time per control step for one oscillator, "
                   f"median {planning_time[name] * 1000:.1f} ms over {timing_steps} steps")

    # ------------------------------------------------------------------
    # Step 10: results per controller
    # ------------------------------------------------------------------
    per_controller = {}
    for name in controllers:
        controller_rows = []
        for row in rows:
            if row["controller"] == name:
                controller_rows.append(row)
        summary = summarise_controller(controller_rows)
        summary["median_planning_seconds"] = planning_time[name]
        per_controller[name] = summary

    # learned against perfect: percent difference of the median position error (positive = worse)
    perfect_error = per_controller["perfect_model_mpc"]["median_position_error"]
    learned_error = per_controller["learned_model_mpc"]["median_position_error"]
    if perfect_error > 0:
        percent_vs_perfect = (learned_error - perfect_error) / perfect_error * 100.0
    else:
        percent_vs_perfect = float("nan")

    # ------------------------------------------------------------------
    # Step 11: the decision rule (fixed in the config before any result)
    # ------------------------------------------------------------------
    rule = cfg["decision_rule"]
    learned = per_controller["learned_model_mpc"]
    tolerance = float(rule["position_error_tolerance_percent"])
    time_limit = float(rule["planning_time_below_seconds"])

    passed_rest = learned["at_rest"] == learned["starts"]
    passed_error = (not math.isnan(percent_vs_perfect)) and percent_vs_perfect <= tolerance
    passed_time = learned["median_planning_seconds"] < time_limit
    worth_continuing = passed_rest and passed_error and passed_time

    # ------------------------------------------------------------------
    # Step 12: save the files
    # ------------------------------------------------------------------
    # 12a: one row per controller and start
    with open(out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 12b: everything else in one file
    all_metrics = {
        "name": cfg["name"],
        "quick": bool(args.quick),
        "checkpoint": str(checkpoint_path),
        "parameters": count_parameters(model),
        "predictor_check": check,
        "per_controller": per_controller,
        "learned_vs_perfect_percent": percent_vs_perfect,
        "decision": {"all_at_rest": passed_rest, "position_error_within_tolerance": passed_error,
                     "planning_time_below_limit": passed_time, "worth_continuing": worth_continuing},
    }
    save_json(all_metrics, out / "metrics.json")

    # 12c: figures
    draw_figures(runs, dt, target, out / "figures")

    # ------------------------------------------------------------------
    # Step 13: the plain-language summary (screen + summary.md)
    # ------------------------------------------------------------------
    lines = []
    if args.quick:
        lines.append(f"RESULT — {cfg['name']}   (QUICK MODE: code check only — do not interpret)")
    else:
        lines.append(f"RESULT — {cfg['name']}   (1 trained predictor, seed 0: direction only)")
    lines.append("")
    lines.append(f"Question: {cfg['question']}")
    lines.append("")
    lines.append(f"  {'':38s} {'At rest':>9s} {'Position error':>15s} {'Settle time':>12s} "
                 f"{'Planning time':>14s}")
    lines.append(f"  {'':38s} {'':>9s} {'(smaller=better)':>15s} {'(seconds)':>12s} {'(1 oscillator)':>14s}")
    for name, summary in per_controller.items():
        at_rest_text = f"{summary['at_rest']} of {summary['starts']}"
        planning_text = "—"
        if not math.isnan(summary["median_planning_seconds"]):
            planning_text = f"{summary['median_planning_seconds'] * 1000:.0f} ms"
        lines.append(f"  {CONTROLLER_LABELS[name]:38s} {at_rest_text:>9s} "
                     f"{format_number(summary['median_position_error']):>15s} "
                     f"{format_number(summary['median_settle_time']):>12s} {planning_text:>14s}")
    lines.append("  Numbers are medians over the starting points.")
    for name, summary in per_controller.items():
        if summary["blew_up"] > 0:
            lines.append(f"  {CONTROLLER_LABELS[name]}: {summary['blew_up']} of {summary['starts']} runs blew up.")
    lines.append("")
    lines.append("What this means:")
    lines.append(f"  • Learned model against true equations: position error {percent_vs_perfect:+.0f}%  "
                 f"→ {describe_difference(percent_vs_perfect)}.")
    lines.append(f"  • The learned predictor's error after {horizon} planned steps is "
                 f"{check['learned_last_step']:.4f} (typical state size {check['typical_state_size']:.2f}).")
    lines.append(f"  • Planning must finish within the sampling time ({dt} s): learned controller needs "
                 f"{learned['median_planning_seconds'] * 1000:.0f} ms per step.")
    lines.append("")
    lines.append(f"Math predicted: {cfg['math_signal']['prediction'].strip()}")
    lines.append("")
    if worth_continuing:
        lines.append("Decision: WORTH CONTINUING (all at rest, error within "
                     f"{tolerance:.0f}% of the true-equations controller, planning faster than {time_limit} s).")
    else:
        reasons = []
        if not passed_rest:
            reasons.append("not every start came to rest")
        if not passed_error:
            reasons.append(f"position error more than {tolerance:.0f}% above the true-equations controller")
        if not passed_time:
            reasons.append(f"planning slower than {time_limit} s")
        lines.append("Decision: NOT YET (" + "; ".join(reasons) + ").")

    summary_text = "\n".join(lines)
    with open(out / "summary.md", "w") as fh:
        fh.write(summary_text + "\n")

    log.say("")
    log.say(summary_text)
    log.say("")
    log.say(f"Files: {out}")
    log.close()


if __name__ == "__main__":
    main()
