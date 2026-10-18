"""Train every arm of one experiment from one config, then write the plain-language summary.

python train.py \
    --config configs/experiments/exp01_certificate_ablation.yaml \
    --seed 0

Options (on/off options take yes or no):
    --quick yes        code check only (tiny data, 2 epochs); default no
    --resume no        start over, ignoring earlier results and checkpoints; default yes
    --ask-first no     do not ask before a long run; default yes (never asked when there is no keyboard)
    --estimate no      skip the timing estimate; default yes
    --seeds 0 1 2 3 4 5 6 7   Stage 2, only after Stage 1 says "worth confirming"

What appears on screen (kept short on purpose):
    - three header lines: experiment, device, how many runs and roughly how long;
    - one progress bar for the run in progress;
    - one line per finished run with its two key numbers;
    - the short result table at the end.
Everything else (settings, parameter table, time estimate per arm, every epoch's losses,
warnings) is written to outputs/<experiment>/log.txt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import time  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.models.registry import REFERENCE_PARAMETERS, build_model, count_parameters  # noqa: E402
from src.training.trainer import result_row, time_batches, train_run  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import load_experiment, output_dir, resolve_arm, save_yaml  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import append_result, completed_runs, save_json  # noqa: E402
from src.utils.summary import build_summary, read_results, short_summary  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train all arms of one experiment.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+")
    add_yes_no(p, "--quick", False, "code check only: tiny data, 2 epochs")
    p.add_argument("--device", default=None, help="auto | cpu | mps | cuda (default: from config)")
    p.add_argument("--arms", nargs="+", help="run only these arms")
    add_yes_no(p, "--resume", True, "continue finished and interrupted runs; no = start over")
    add_yes_no(p, "--ask-first", True, "ask before starting a run longer than the configured limit")
    p.add_argument("--output-root", default=None, help="where outputs/ lives (default: repository outputs/)")
    add_yes_no(p, "--estimate", True, "time a few batches per arm and estimate the run time")
    return p.parse_args()


def choose_arms(exp: dict, wanted_names) -> list:
    """All arms of the experiment, or only the ones named with --arms."""
    # Step 1: no --arms given -> use every arm
    if not wanted_names:
        return list(exp["arms"])

    # Step 2: keep only the named arms
    chosen = []
    for arm in exp["arms"]:
        if arm["name"] in wanted_names:
            chosen.append(arm)

    # Step 3: stop with a clear message if a name does not exist
    known_names = set()
    for arm in chosen:
        known_names.add(arm["name"])
    missing = set(wanted_names) - known_names
    if missing:
        sys.exit(f"Unknown arm(s): {', '.join(sorted(missing))}")
    return chosen


def write_settings_to_log(log, exp, first_cfg, data, seeds, device, out, quick):
    """Everything about the set-up goes to the log file, not the screen."""
    n_train = data["splits"]["train"].xw.shape[0]
    n_val = data["splits"]["val"].xw.shape[0]
    n_test = data["splits"]["test"].xw.shape[0]
    log.detail(f"Experiment : {exp['name']}  (quick run: {quick})")
    log.detail(f"Hypothesis : {exp['hypothesis'].strip()}")
    log.detail(f"Device     : {describe(device)}")
    log.detail(f"System     : {first_cfg['system']['name']}, mu = {first_cfg['system']['mu']}, "
               f"time step = {first_cfg['system']['dt']}")
    log.detail(f"Data       : {n_train} training windows, {n_val} validation, {n_test} test; "
               f"context {data['context_len']} steps, horizon {data['horizon']} steps")
    log.detail(f"Training   : {first_cfg['training']['epochs']} epochs, batch size "
               f"{first_cfg['training']['batch_size']}, learning rate {first_cfg['training']['lr']}, seeds {seeds}")
    log.detail(f"Outputs    : {out}")


def estimate_minutes(arms, cfgs, data, device, batches_per_epoch, log, do_estimate):
    """Rough minutes per run for every arm. Details go to the log file; the screen sees one bar."""
    estimates = {}
    log.detail("")
    log.detail(f"{'arm':26s}{'regime':17s}{'model':13s}{'parameters':>11s}{'reference':>11s}{'est. minutes':>14s}")

    # One progress bar while the arms are timed (a few training batches each)
    bar = progress_bar(total=len(arms), description="estimating run time", unit="arm")
    for arm in arms:
        name = arm["name"]
        cfg = cfgs[name]

        # Step 1: parameter count and the published reference (for the log)
        n_parameters = count_parameters(build_model(cfg))
        reference = REFERENCE_PARAMETERS.get(cfg["model"]["name"])
        if reference:
            reference_text = f"{reference:,}"
        else:
            reference_text = "—"

        # Step 2: time a few batches and scale up to a whole run
        if do_estimate:
            seconds_per_batch = time_batches(cfg, data, device)
            # 1.4 adds time for validation and the final evaluation
            minutes = seconds_per_batch * batches_per_epoch * cfg["training"]["epochs"] * 1.4 / 60.0
            minutes_text = f"{minutes:.1f}"
        else:
            minutes = None
            minutes_text = "—"
        estimates[name] = minutes

        # Step 3: one row in the log file
        log.detail(f"{name:26s}{cfg['training']['regime']:17s}{cfg['model']['name']:13s}"
                   f"{n_parameters:>11,d}{reference_text:>11s}{minutes_text:>14s}")
        bar.update(1)
    bar.close()
    return estimates


def finished_run_line(position: str, arm_name: str, seed: int, row: dict, expected: str) -> str:
    """One short line per finished run, e.g. '  ✓ run 1/12  certified_teacher seed0   next-step 0.0041 ...'."""
    if row["diverged"]:
        if expected == "diverges":
            note = "BLEW UP (expected)"
        else:
            note = "BLEW UP (not expected!)"
        return f"  ✗ {position}  {arm_name} seed{seed}   {note}"

    one_step = row["one_step_rmse"]
    rollout = row["rollout_rmse"]
    minutes = row["train_minutes"]
    return (f"  ✓ {position}  {arm_name} seed{seed}   "
            f"next-step {one_step:.3g}   50-step {rollout:.3g}   ({minutes:.1f} min)")


def main():
    args = parse_args()

    # Step 1: read the experiment and decide which arms and seeds to run
    exp = load_experiment(args.config)
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed]
    arms = choose_arms(exp, args.arms)
    cfgs = {}
    for arm in arms:
        cfgs[arm["name"]] = resolve_arm(exp, arm, quick=args.quick)

    # Step 2: output folder and the log file
    out = output_dir(exp, args.quick, args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "results.csv"
    if not args.resume and results_csv.exists():
        # keep the old results under a new name instead of deleting them
        stamp = time.strftime("%Y%m%d_%H%M%S")
        results_csv.rename(out / f"results_replaced_{stamp}.csv")
    log = ConsoleLog(out / "log.txt")
    log.capture_warnings()

    # Step 3: device and data
    device = pick_device(args.device or exp.get("device", "auto"))
    first_cfg = cfgs[arms[0]["name"]]
    data = build_data(first_cfg)
    n_train = data["splits"]["train"].xw.shape[0]
    batch_size = first_cfg["training"]["batch_size"]
    batches_per_epoch = -(-n_train // batch_size)    # rounds up

    # Step 4: save the settings (files) and write them into the log
    exp_clean = {}
    for key, value in exp.items():
        if not key.startswith("_"):
            exp_clean[key] = value
    save_yaml(exp_clean, out / "experiment_config.yaml")
    window_counts = {}
    for split_name, split in data["splits"].items():
        window_counts[split_name] = int(split.xw.shape[0])
    save_json({"normaliser": data["norm"].to_dict(), "windows": window_counts,
               "context_len": data["context_len"], "horizon": data["horizon"]}, out / "data_info.json")
    write_settings_to_log(log, exp, first_cfg, data, seeds, device, out, args.quick)

    # Step 5: which runs are still to do (resume skips finished ones)
    done = completed_runs(results_csv)
    pending = []
    for seed in seeds:
        for arm in arms:
            if (arm["name"], seed) not in done:
                pending.append((arm, seed))

    # Step 6: estimate the run time (details in the log, one bar on screen)
    estimates = estimate_minutes(arms, cfgs, data, device, batches_per_epoch, log, args.estimate)
    total_minutes = 0.0
    for arm, seed in pending:
        if estimates[arm["name"]] is not None:
            total_minutes = total_minutes + estimates[arm["name"]]

    # Step 7: the short header on screen
    if args.quick:
        log.say("QUICK RUN — code check only, do not interpret the numbers")
    log.say(f"{exp['name']}   seeds {seeds}")
    if args.estimate:
        time_text = f"about {total_minutes:.0f} minutes"
    else:
        time_text = "time not estimated"
    log.say(f"  device: {describe(device)} | {len(pending)} runs to do, {len(done)} already done | {time_text}")
    log.say(f"  full details: {log.path}")
    log.say("")

    # Step 8: ask before a long run (only when someone is at a keyboard)
    limit = exp.get("run", {}).get("long_run_confirm_minutes", 30)
    if total_minutes > limit and args.ask_first and sys.stdin.isatty():
        answer = input(f"This is longer than {limit} minutes. Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            log.close()
            sys.exit("Stopped before training.")

    # Step 9: train every pending run, one progress bar each, one result line each
    for number, (arm, seed) in enumerate(pending, start=1):
        position = f"run {number}/{len(pending)}"
        bar_label = f"{position} {arm['name']} seed{seed}"
        run_dir = out / "runs" / f"{arm['name']}_seed{seed}"

        metrics = train_run(cfgs[arm["name"]], data, seed, run_dir, device,
                            fresh=not args.resume, position_label=bar_label, log=log)

        row = result_row(cfgs[arm["name"]], seed, metrics)
        append_result(results_csv, row)
        log.say(finished_run_line(position, arm["name"], seed, row, arm.get("expected", "")))

    # Step 10: full summary to files, short summary on screen
    rows = read_results(results_csv)
    full_text, machine = build_summary(exp, rows, quick=args.quick)
    (out / "summary.md").write_text(full_text)
    save_json(machine, out / "summary.json")
    log.detail(full_text)

    log.say("")
    log.say(short_summary(machine, full_summary_path=out / "summary.md"))
    log.close()


if __name__ == "__main__":
    main()
