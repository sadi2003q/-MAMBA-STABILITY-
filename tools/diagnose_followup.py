"""Stage 2b — follow-up of the Stage 2 diagnosis (configs/analysis/stage2b_followup.yaml).

Three checks on the saved seed-0 models (no training):
  C1  Do the one-step mistakes cancel each other from step to step?
  C2  Is the part that does not cancel the timing part or the shape part?
  C3  Is the model slow on every step when the input is zero, or does the slowness come from feedback?

python tools/diagnose_followup.py \
    --config configs/analysis/stage2b_followup.yaml \
    --quick no \
    --device cpu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import yaml  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.evaluation import error_diagnosis as diag  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from tools.diagnose_errors import data_key, describe_run, load_run, load_settings  # noqa: E402

CSV_COLUMNS = [
    "run", "family", "guarantee", "training",
    "one_step_rms", "accumulation_total", "accumulation_timing", "accumulation_shape",
    "correlation_gap1", "correlation_gap1_timing", "correlation_gap1_shape",
    "zero_input_clock_speed", "zero_input_sign_consistency", "zero_input_verdict", "error",
]


def check_one(model, data, settings, rules, device):
    """C1, C2 and C3 for one model. Returns (row, details)."""
    C = data["context_len"]
    H = data["horizon"]
    norm = data["norm"]
    system = data["system"]
    split = diag.keep_first_windows(data["splits"]["test"], int(settings["max_windows"]))
    row = {}
    details = {}

    # Step 1: one-step mistakes (true history) on the test windows, physical units
    outputs = diag.predictions_on_windows(model, "chained", split, norm, C, H, device,
                                          int(settings["batch_size"]))
    true_targets = outputs["true"][:, C + 1:C + H + 1]
    mistakes = outputs["one_step"] - true_targets                        # [W, H, 2]
    velocity = diag.true_velocity(split, C, H, system.dt)
    direction, _ = diag.unit_vectors(velocity)
    timing, shape = diag.signed_parts(mistakes, direction)
    row["one_step_rms"] = diag.root_mean_square(mistakes)

    # Step 2: C1 — step-to-step correlation (whole mistake, timing part, shape part)
    max_gap = int(settings["max_gap"])
    total_corr = diag.step_to_step_correlation(mistakes, max_gap)
    timing_corr = diag.step_to_step_correlation(timing, max_gap)
    shape_corr = diag.step_to_step_correlation(shape, max_gap)
    row["correlation_gap1"] = total_corr[0]
    row["correlation_gap1_timing"] = timing_corr[0]
    row["correlation_gap1_shape"] = shape_corr[0]
    details["correlation"] = {"total": total_corr, "timing": timing_corr, "shape": shape_corr}

    # Step 3: C2 — accumulation factor, whole mistake and each part
    row["accumulation_total"] = diag.accumulation_factor(mistakes)
    row["accumulation_timing"] = diag.accumulation_factor(timing)
    row["accumulation_shape"] = diag.accumulation_factor(shape)

    # Step 4: C3 — one-step clock on the true zero-input loop
    trajectory, indices, steps_per_loop = diag.true_limit_cycle_points(
        system, settings["loop_start"], int(settings["loop_settle_steps"]), 1, C)
    clock = diag.zero_input_one_step_clock(model, norm, system, trajectory, indices[0], steps_per_loop,
                                           int(settings["zero_input_windows"]), C, H, device,
                                           float(settings["min_speed_fraction"]))
    row["zero_input_clock_speed"] = clock["implied_clock_speed"]
    row["zero_input_sign_consistency"] = clock["sign_consistency"]
    slow = rules["c3_zero_input_clock"]["slow_threshold"]
    on_time = rules["c3_zero_input_clock"]["on_time_threshold"]
    if clock["implied_clock_speed"] <= slow:
        row["zero_input_verdict"] = "slow on every step"
    elif clock["implied_clock_speed"] >= on_time:
        row["zero_input_verdict"] = "keeps time; slowness from feedback"
    else:
        row["zero_input_verdict"] = "partly both"
    details["zero_input_clock"] = clock
    return row, details


def find_row(rows, family, training):
    for r in rows:
        if r.get("family") == family and r.get("training") == training and r.get("guarantee") != "guarantee removed":
            return r
    return None


def number(value, digits=2):
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Stage 2b: do one-step mistakes cancel; where the slow clock comes from.")
    parser.add_argument("--config", required=True)
    add_yes_no(parser, "--quick", False, "a few windows only; checks that the code runs, never interpret")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    # Step 1: settings, output folder, log file
    config, settings = load_settings(args.config, args.quick)
    rules = config["checks"]
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs"
    out = output_root / (config["name"] + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(out / "log.txt", fresh=True)
    log.capture_warnings()
    device = pick_device(args.device)
    set_seed(0)
    runs = config["runs"]
    log.say(f"Stage 2b follow-up: {len(runs)} trained models, no training   full details: {log.path}")
    if args.quick:
        log.say("QUICK MODE — code check only, do not interpret these numbers")

    # Step 2: every model
    rows = []
    all_details = {}
    data_cache = {}
    coverage = None
    bar = progress_bar(total=len(runs), description="checking", unit="model")
    for position, run in enumerate(runs, start=1):
        bar.set_description(f"model {position}/{len(runs)} {Path(run).name}")
        run_folder = REPO_ROOT / config["models_root"] / run
        row = {"run": run}
        try:
            model, cfg = load_run(run_folder, device)
            family, guarantee, training, _ = describe_run(cfg)
            row["family"] = family
            row["guarantee"] = guarantee
            row["training"] = training
            key = data_key(cfg)
            if key not in data_cache:
                data_cache[key] = build_data(copy.deepcopy(cfg))
            data = data_cache[key]
            if coverage is None:
                coverage = diag.input_coverage(data["splits"]["train"], float(settings["small_input_size"]),
                                               data["horizon"])
            result, details = check_one(model, data, settings, rules, device)
            row.update(result)
            all_details[run] = details
            log.detail(json.dumps(row, default=str))
        except Exception as exc:
            row["error"] = repr(exc)
            log.say(f"  {run}: failed ({exc!r})")
        rows.append(row)
        bar.update(1)
    bar.close()

    # Step 3: save
    with open(out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in CSV_COLUMNS})
    save_json({"settings": settings, "rows": rows, "details": all_details, "input_coverage": coverage},
              out / "details.json")

    # Step 4: short summary
    unrelated, repeated = diag.reference_accumulation(next(iter(data_cache.values()))["horizon"]) if data_cache else (5.05, 29.3)
    text = []
    title = "RESULT — stage2b_followup   (1 seed: direction only)"
    if args.quick:
        title = title + "   QUICK MODE: code check only, do not interpret"
    text.append(title)
    text.append("")
    text.append("Question: do the one-step mistakes cancel each other, and where does the slow clock come from?")
    text.append(f"  Adding up: about {unrelated:.1f} = unrelated mistakes, about {repeated:.0f} = same mistake every step,")
    text.append(f"  below {unrelated:.1f} = mistakes cancel. Smaller is better.")
    text.append("  Step-to-step: negative = the next mistake tends to undo the last one (good).")
    text.append("  Clock at zero input: 1.00 = keeps time, 0.81 would explain the slow free-running loop fully.")
    text.append("")
    text.append(f"  {'Model':<46} {'Training':<17} {'Adding up':>9} {'timing':>7} {'shape':>6} {'Step-to-step':>12} {'Clock at 0':>10}")
    for r in rows:
        if "family" not in r:
            continue
        label = f"{r['family']} ({r['guarantee']})"
        text.append(f"  {label:<46} {r['training']:<17} {number(r.get('accumulation_total'), 1):>9} "
                    f"{number(r.get('accumulation_timing'), 1):>7} {number(r.get('accumulation_shape'), 1):>6} "
                    f"{number(r.get('correlation_gap1')):>12} {number(r.get('zero_input_clock_speed')):>10}")
    text.append("")
    if coverage is not None:
        text.append(f"  Training input near zero (below {settings['small_input_size']} in size, peak {coverage['input_peak']:.0f}):"
                    f" {100 * coverage['share_of_steps_small']:.1f}% of steps; longest stretch"
                    f" {coverage['longest_small_stretch_steps']} steps (the free-running test needs {coverage['stretch_needed_steps']}+).")
        text.append("")

    # Step 5: predictions written before running
    mamba = find_row(rows, "Full Mamba", "chained training")
    lstm = find_row(rows, "Long Short-Term Memory network", "chained training")
    text.append("Math predicted (written before running):")
    if mamba and lstm and "correlation_gap1" in mamba and "correlation_gap1" in lstm:
        margin = rules["c1_cancelling"]["margin"]
        held = mamba["correlation_gap1"] >= lstm["correlation_gap1"] + margin
        text.append(f"  C1: full Mamba's step-to-step {mamba['correlation_gap1']:.2f} vs Long Short-Term Memory network"
                    f" {lstm['correlation_gap1']:.2f} (needs at least {margin:.2f} higher) → {'HELD' if held else 'DID NOT HOLD'}")
        held = (mamba.get("accumulation_timing") or 0) > (mamba.get("accumulation_shape") or 0)
        text.append(f"  C2: full Mamba's timing part adds up {number(mamba.get('accumulation_timing'), 1)} vs shape part"
                    f" {number(mamba.get('accumulation_shape'), 1)} → {'HELD' if held else 'DID NOT HOLD'}")
        text.append(f"  C3: no prediction. Full Mamba's clock at zero input {number(mamba.get('zero_input_clock_speed'))}"
                    f" → {mamba.get('zero_input_verdict')}")
    else:
        text.append("  Could not be checked (a model failed; see the log file).")
    summary = "\n".join(text)
    (out / "summary.md").write_text(summary + "\n")
    log.say("")
    log.say(summary)
    log.close()


if __name__ == "__main__":
    main()
