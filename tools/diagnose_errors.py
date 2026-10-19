"""Stage 2 — find out what actually makes the predictions go wrong (docs/stage2_plan.md).

Loads the trained seed-0 models from models/seed0/ (no training) and answers four questions:
  Q1  How much of the error is timing and how much is shape?
  Q2  Do errors just add up, or does feedback amplify them?
  Q3  Slow clock or amplifier?
  Q4  Does the model correct a push like the true system does?
Questions, predictions and pass/fail rules: configs/analysis/stage2_diagnosis.yaml (fixed before running).

Screen: one progress bar, then a short table and the verdicts.
Files in outputs/stage2_diagnosis/:
  log.txt        everything in detail
  results.csv    one row per model
  details.json   every number, including the curves
  summary.md     plain-language summary

python tools/diagnose_errors.py \
    --config configs/analysis/stage2_diagnosis.yaml \
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.evaluation import error_diagnosis as diag  # noqa: E402
from src.evaluation.diagnostics import limit_cycle  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.device import pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

FAMILY_NAMES = {"mamba_full": "Full Mamba",
                "ssm_minimal": "Simplified Mamba",
                "lstm": "Long Short-Term Memory network"}
GUARANTEE_NAMES = {"certified": "guarantee on", "free_sign": "guarantee removed"}
TRAINING_NAMES = {"teacher_forcing": "teacher forcing", "chained": "chained training", "direct": "direct"}

CSV_COLUMNS = [
    "run", "family", "guarantee", "training", "evaluated_as", "blew_up", "chained_or_direct_rmse",
    "q1_timing_share_median", "q1_timing_share_pooled", "q1_timing_share_step50",
    "q1_alignment_timing_share_median", "q1_alignment_fraction_at_edge", "q1_alignment_mean_lag_step50",
    "q1_verdict",
    "q2_ratio_to_added_up", "q2_ratio_to_true_feedback", "q2_step1_check", "q2_verdict",
    "q3_misfit_line", "q3_misfit_exponential", "q3_exponential_rate", "q3_late_over_early_growth",
    "q3_sign_consistency", "q3_sign_consistency_time_units", "q3_mean_one_step_lag",
    "q3_implied_clock_speed", "q3_verdict",
    "q4_timing_push_ratio", "q4_shape_push_ratio", "q4_model_shape_survival", "q4_true_shape_survival",
    "q4_points_used", "q4_verdict",
    "loop_time_model", "loop_time_true", "loop_time_ratio", "free_run_status",
    "magnification_model", "magnification_true", "magnification_ratio",
    "step_spectral_radius_median", "step_henrici_median", "step_pervez_defect_median", "step_state_size",
    "error",
]


# =================================================================================================
# Settings and models
# =================================================================================================
def load_settings(config_path, quick):
    """Read the YAML file; with --quick yes, apply the quick block on top of the settings."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    settings = dict(config["settings"])
    if quick:
        for key, value in config.get("quick", {}).items():
            settings[key] = value
    return config, settings


def load_run(run_folder, device):
    """Load one trained model and the configuration it was trained with."""
    checkpoint = torch.load(run_folder / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


def describe_run(cfg):
    """Plain-language labels for one run."""
    model_name = cfg["model"]["name"]
    family = FAMILY_NAMES.get(model_name, model_name)
    if model_name == "lstm":
        guarantee = "no guarantee"
    else:
        decay_mode = cfg["model"].get("decay_mode", "certified")
        guarantee = GUARANTEE_NAMES.get(decay_mode, decay_mode)
    regime = cfg["training"]["regime"]
    training = TRAINING_NAMES.get(regime, regime)
    if regime == "direct":
        evaluated_as = "direct"
    else:
        evaluated_as = "chained"
    return family, guarantee, training, evaluated_as


def data_key(cfg):
    """The settings that decide the test data. Runs with the same key share one data set."""
    key = {"system": cfg["system"],
           "data": {k: v for k, v in cfg["data"].items() if not k.startswith("max_train") and not k.startswith("max_val")},
           "max_eval_windows": cfg.get("evaluation", {}).get("max_eval_windows")}
    return json.dumps(key, sort_keys=True, default=str)


# =================================================================================================
# One model
# =================================================================================================
def diagnose_one(model, cfg, data, settings, rules, device, log):
    """All four questions for one model. Returns (csv row, details dictionary)."""
    C = data["context_len"]
    H = data["horizon"]
    norm = data["norm"]
    system = data["system"]
    dt = system.dt
    _, _, _, evaluated_as = describe_run(cfg)
    split = diag.keep_first_windows(data["splits"]["test"], int(settings["max_windows"]))

    row = {}
    details = {}

    # ---------------------------------------------------------------- predictions on every window
    outputs = diag.predictions_on_windows(model, evaluated_as, split, norm, C, H, device,
                                          int(settings["batch_size"]))
    blew_up = int(outputs["hit_clamp"].sum())
    row["blew_up"] = f"{blew_up} of {len(outputs['hit_clamp'])} windows"
    true_targets = outputs["true"][:, C + 1:C + H + 1]                  # x_{C+1} ... x_{C+H}
    true_before = outputs["true"][:, C:C + H]                           # x_C ... x_{C+H-1}
    error = outputs["predicted"] - true_targets
    one_step_error = outputs["one_step"] - true_targets
    row["chained_or_direct_rmse"] = diag.root_mean_square(error)

    # ---------------------------------------------------------------- free-running loop (the clock)
    try:
        cycle = limit_cycle(model, evaluated_as, norm, system, C, int(settings["limit_cycle_steps"]),
                            settings["loop_start"], device)
        row["loop_time_model"] = cycle.get("period_model")
        row["loop_time_true"] = cycle.get("period_true")
        row["free_run_status"] = cycle.get("status")
        if cycle.get("period_model") and cycle.get("period_true"):
            row["loop_time_ratio"] = cycle["period_model"] / cycle["period_true"]
        details["free_running_loop"] = cycle
    except Exception as exc:
        row["free_run_status"] = f"error: {exc!r}"

    if blew_up > 0:
        # A run that hit the clamp is reported as a count, never analysed as an error number
        log.detail(f"  {blew_up} windows blew up: questions 1-4 skipped for this model")
        row["q1_verdict"] = "blew up"
        row["q2_verdict"] = "blew up"
        row["q3_verdict"] = "blew up"
        row["q4_verdict"] = "blew up"
        return row, details

    velocity = diag.true_velocity(split, C, H, dt)
    direction, _ = diag.unit_vectors(velocity)

    # ---------------------------------------------------------------- Q1
    q1 = diag.q1_timing_and_shape(error, direction)
    q1_check = diag.q1_alignment_cross_check(split, outputs["predicted"], C, H, dt,
                                             int(settings["alignment_search_steps"]))
    row["q1_timing_share_median"] = q1["median_timing_share"]
    row["q1_timing_share_pooled"] = q1["pooled_timing_share"]
    row["q1_timing_share_step50"] = q1["step50_timing_share"]
    row["q1_alignment_timing_share_median"] = q1_check["median_timing_share"]
    row["q1_alignment_fraction_at_edge"] = q1_check["fraction_at_search_edge"]
    row["q1_alignment_mean_lag_step50"] = q1_check["mean_lag_at_step50"]
    if q1["median_timing_share"] >= rules["q1_timing_or_shape"]["threshold"]:
        row["q1_verdict"] = "timing-dominated"
    else:
        row["q1_verdict"] = "not timing-dominated"
    details["q1"] = {"timing_curve": q1["timing_curve"].tolist(),
                     "shape_curve": q1["shape_curve"].tolist(),
                     "alignment_mean_lag_curve": q1_check["mean_lag_curve"].tolist()}

    # ---------------------------------------------------------------- Q2 (chained models only)
    if evaluated_as == "chained":
        q2 = diag.q2_feedback(error, one_step_error, true_before, system)
        row["q2_ratio_to_added_up"] = q2["ratio_to_added_up"]
        row["q2_ratio_to_true_feedback"] = q2["ratio_to_true_feedback"]
        row["q2_step1_check"] = q2["step1_chained_minus_one_step_max"]
        if q2["ratio_to_added_up"] > rules["q2_feedback_or_pile_up"]["threshold"]:
            row["q2_verdict"] = "amplified by feedback"
        else:
            row["q2_verdict"] = "errors mostly add up"
        details["q2"] = q2
    else:
        row["q2_verdict"] = "not applicable (direct)"

    # ---------------------------------------------------------------- Q3
    growth = diag.q3_growth_shape(q1["timing_curve"])
    row["q3_misfit_line"] = growth["misfit_line"]
    row["q3_misfit_exponential"] = growth["misfit_exponential"]
    row["q3_exponential_rate"] = growth["exponential_rate_per_step"]
    row["q3_late_over_early_growth"] = growth["late_over_early_growth"]
    line_fits_better = growth["misfit_line"] <= growth["misfit_exponential"]
    details["q3_growth"] = growth
    if evaluated_as == "chained":
        sign = diag.q3_one_step_sign(one_step_error, velocity, dt, float(settings["min_speed_fraction"]))
        row["q3_sign_consistency"] = sign["consistency"]
        row["q3_sign_consistency_time_units"] = sign["consistency_time_units"]
        row["q3_mean_one_step_lag"] = sign["mean_one_step_lag"]
        row["q3_implied_clock_speed"] = sign["implied_clock_speed"]
        details["q3_sign"] = sign
        consistent = sign["consistency"] >= rules["q3_slow_clock_or_amplifier"]["consistency_threshold"]
        if consistent and line_fits_better:
            row["q3_verdict"] = "slow clock"
        elif (not line_fits_better) and growth["exponential_rate_per_step"] > 0:
            row["q3_verdict"] = "amplifier"
        else:
            row["q3_verdict"] = "mixed"
    else:
        if line_fits_better:
            row["q3_verdict"] = "curve only: straight line"
        else:
            row["q3_verdict"] = "curve only: exponential"

    # ---------------------------------------------------------------- Q4
    trajectory, indices, steps_per_loop = diag.true_limit_cycle_points(
        system, settings["loop_start"], int(settings["loop_settle_steps"]), int(settings["loop_points"]), C)
    q4 = diag.q4_push_response(model, evaluated_as, norm, system, trajectory, indices, C, H,
                               float(settings["push_size"]), device)
    row["q4_points_used"] = q4["points_used"]
    if q4["points_used"] > 0:
        row["q4_timing_push_ratio"] = q4["timing_push_ratio"]
        row["q4_shape_push_ratio"] = q4["shape_push_ratio"]
        row["q4_model_shape_survival"] = q4["model_shape_survival"]
        row["q4_true_shape_survival"] = q4["true_shape_survival"]
        good = rules["q4_error_correction"]["good_threshold"]
        bad = rules["q4_error_correction"]["bad_threshold"]
        if q4["shape_push_ratio"] <= good:
            row["q4_verdict"] = "corrects shape errors"
        elif q4["shape_push_ratio"] >= bad:
            row["q4_verdict"] = "does not correct shape errors"
        else:
            row["q4_verdict"] = "partly corrects"
    else:
        row["q4_verdict"] = "no usable points"
    q4["steps_per_true_loop"] = steps_per_loop
    details["q4"] = q4

    # ---------------------------------------------------------------- recorded only
    try:
        magnification = diag.largest_magnification(model, evaluated_as, split, norm, system, C, H,
                                                   int(settings["magnification_windows"]), device)
        row["magnification_model"] = magnification.get("model_median")
        row["magnification_true"] = magnification.get("true_median")
        row["magnification_ratio"] = magnification.get("ratio_median")
        details["magnification"] = magnification
    except Exception as exc:     # a recorded-only number must never lose the main answers
        details["magnification"] = {"error": repr(exc)}
        log.detail(f"  magnification failed: {exc!r}")
    if evaluated_as == "chained":
        try:
            shape = diag.step_jacobian_shape(model, split, C, H, int(settings["step_jacobian_windows"]),
                                             [int(s) for s in settings["step_jacobian_steps"]], device)
            row["step_spectral_radius_median"] = shape.get("spectral_radius_median")
            row["step_henrici_median"] = shape.get("henrici_median")
            row["step_pervez_defect_median"] = shape.get("pervez_defect_median")
            row["step_state_size"] = shape.get("state_size")
            details["step_jacobian"] = shape
        except Exception as exc:
            details["step_jacobian"] = {"error": repr(exc)}
            log.detail(f"  step Jacobian failed: {exc!r}")

    return row, details


# =================================================================================================
# Output: table, predictions, summary
# =================================================================================================
def number(value, digits=2):
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def short_table(rows):
    """The few columns that answer the four questions, one line per model."""
    lines = []
    header = (f"  {'Model':<36} {'Training':<17} {'Timing':>7} {'Feedback':>9} "
              f"{'Clock':<14} {'Shape push':>11} {'Loop time':>10}")
    lines.append(header)
    lines.append(f"  {'':<36} {'':<17} {'share':>7} {'ratio':>9} {'(Q3)':<14} {'vs true':>11} {'vs true':>10}")
    for r in rows:
        label = f"{r['family']} ({r['guarantee']})"
        clock = r.get("q3_verdict", "—")
        if clock.startswith("curve only"):
            clock = "curve only"
        lines.append(f"  {label:<36} {r['training']:<17} {number(r.get('q1_timing_share_median')):>7} "
                     f"{number(r.get('q2_ratio_to_added_up')):>9} {clock:<14} "
                     f"{number(r.get('q4_shape_push_ratio'), 1):>11} {number(r.get('loop_time_ratio')):>10}")
    return lines


def find_row(rows, family, guarantee, training):
    for r in rows:
        if r["family"] == family and r["guarantee"] == guarantee and r["training"] == training:
            return r
    return None


def check_predictions(rows, rules):
    """Compare the results with the predictions written in the config before running."""
    lines = []
    mamba = find_row(rows, "Full Mamba", "guarantee on", "chained training")
    lstm = find_row(rows, "Long Short-Term Memory network", "no guarantee", "chained training")
    agreed = {}

    if mamba is None or "q1_timing_share_median" not in mamba:
        lines.append("  The full Mamba (guarantee on, chained training) could not be analysed; predictions not checked.")
        return lines, agreed

    # Q1
    held = mamba["q1_verdict"] == "timing-dominated"
    agreed["q1"] = held
    lines.append(f"  Q1 predicted timing-dominated for the full Mamba (chained training): timing share "
                 f"{mamba['q1_timing_share_median']:.2f} (rule: at least {rules['q1_timing_or_shape']['threshold']:.2f})"
                 f" → prediction {'HELD' if held else 'DID NOT HOLD'}.")

    # Q2 has no prediction
    lines.append("  Q2 had no prediction (rule only).")

    # Q3
    held = mamba.get("q3_verdict") == "slow clock"
    agreed["q3"] = held
    lines.append(f"  Q3 predicted a slow clock for the full Mamba (chained training): verdict "
                 f"'{mamba.get('q3_verdict')}' → prediction {'HELD' if held else 'DID NOT HOLD'}.")

    # Q4
    if lstm is not None and mamba.get("q4_shape_push_ratio") is not None and lstm.get("q4_shape_push_ratio") is not None:
        held = mamba["q4_shape_push_ratio"] > lstm["q4_shape_push_ratio"]
        agreed["q4"] = held
        lines.append(f"  Q4 predicted the full Mamba corrects shape errors less than the Long Short-Term Memory network"
                     f" (chained training): shape-push ratio {mamba['q4_shape_push_ratio']:.1f} against "
                     f"{lstm['q4_shape_push_ratio']:.1f} → prediction {'HELD' if held else 'DID NOT HOLD'}.")
    else:
        lines.append("  Q4 prediction could not be checked (a model had no usable points).")
    return lines, agreed


def stage3_direction(rows):
    """How Stage 2 feeds Stage 3 (docs/stage2_plan.md), read from the full Mamba, chained training."""
    mamba = find_row(rows, "Full Mamba", "guarantee on", "chained training")
    if mamba is None or "q1_verdict" not in mamba:
        return "  Not decided: the full Mamba (chained training) could not be analysed."
    clock_path = mamba.get("q1_verdict") == "timing-dominated" and mamba.get("q3_verdict") == "slow clock"
    loop_path = mamba.get("q2_verdict") == "amplified by feedback"
    if clock_path and loop_path:
        return ("  Both: the Stage 3 guarantee must keep the clock right AND cover the whole "
                "prediction-plus-memory loop.")
    if clock_path:
        return ("  Clock: the Stage 3 guarantee must keep the model's clock right (leave timing errors alone, "
                "pull shape errors back, memory update that can correct itself).")
    if loop_path:
        return "  Loop: the Stage 3 guarantee must cover the whole prediction-plus-memory loop."
    return ("  Neither pattern is clean: look at the details (log.txt, details.json) before designing Stage 3.")


def meaning_bullets(rows):
    """Two to four plain-language sentences about the full Mamba against the Long Short-Term Memory network."""
    bullets = []
    mamba = find_row(rows, "Full Mamba", "guarantee on", "chained training")
    lstm = find_row(rows, "Long Short-Term Memory network", "no guarantee", "chained training")
    if mamba is None or "q1_timing_share_median" not in mamba:
        return bullets
    share = mamba["q1_timing_share_median"]
    bullets.append(f"  • Full Mamba, chained training: {100 * share:.0f}% of the error is timing "
                   f"(the prediction is on the path but ahead or behind), {100 * (1 - share):.0f}% is shape.")
    if mamba.get("q2_ratio_to_added_up") is not None:
        bullets.append(f"  • Its chained error is {mamba['q2_ratio_to_added_up']:.2f} times what the same one-step "
                       f"mistakes give when they just add up (above 1.5 would mean feedback amplifies them).")
    if mamba.get("q3_implied_clock_speed") is not None:
        speed = mamba["q3_implied_clock_speed"]
        bullets.append(f"  • Its one-step mistakes imply a clock running at {100 * speed:.0f}% of the true speed "
                       f"(sign consistency {mamba['q3_sign_consistency']:.2f}; 1 = always the same direction).")
    if lstm is not None and lstm.get("q4_shape_push_ratio") is not None and mamba.get("q4_shape_push_ratio") is not None:
        bullets.append(f"  • After 50 steps, {mamba['q4_shape_push_ratio']:.1f} times as much of a push off the path is "
                       f"left as in the true system for the full Mamba, {lstm['q4_shape_push_ratio']:.1f} times for the "
                       f"Long Short-Term Memory network (smaller is better; 1 = like the true system).")
    return bullets


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            clean = {}
            for k in CSV_COLUMNS:
                v = r.get(k)
                if v is None:
                    clean[k] = ""
                elif isinstance(v, float):
                    clean[k] = f"{v:.6g}"
                else:
                    clean[k] = v
            writer.writerow(clean)


# =================================================================================================
# Main
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 2: what makes the predictions go wrong.")
    parser.add_argument("--config", required=True)
    add_yes_no(parser, "--quick", False, "a few windows only; checks that the code runs, never interpret")
    parser.add_argument("--device", default="cpu", help="cpu (default; fastest for these small models), cuda or mps")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    # Step 1: settings, output folder, log file
    config, settings = load_settings(args.config, args.quick)
    rules = config["questions"]
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs"
    folder_name = config["name"] + ("_quick" if args.quick else "")
    out = output_root / folder_name
    out.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(out / "log.txt", fresh=True)
    log.capture_warnings()
    device = pick_device(args.device)
    set_seed(0)

    models_root = REPO_ROOT / config["models_root"]
    runs = config["runs"]
    log.say(f"Stage 2 diagnosis: {len(runs)} trained models, no training   full details: {log.path}")
    if args.quick:
        log.say("QUICK MODE — code check only, do not interpret these numbers")
    log.detail("Settings: " + json.dumps(settings))

    # Step 2: analyse every model
    rows = []
    all_details = {}
    data_cache = {}
    bar = progress_bar(total=len(runs), description="diagnosing", unit="model")
    for position, run in enumerate(runs, start=1):
        run_folder = models_root / run
        bar.set_description(f"model {position}/{len(runs)} {Path(run).name}")
        row = {"run": run}
        if not (run_folder / "checkpoint.pt").exists():
            row["error"] = f"missing {run_folder / 'checkpoint.pt'}"
            log.say(f"  missing model: {run_folder / 'checkpoint.pt'}")
            rows.append(row)
            bar.update(1)
            continue
        try:
            model, cfg = load_run(run_folder, device)
            family, guarantee, training, evaluated_as = describe_run(cfg)
            row["family"] = family
            row["guarantee"] = guarantee
            row["training"] = training
            row["evaluated_as"] = evaluated_as

            # the test data are the same for every run trained on the same settings: build them once
            key = data_key(cfg)
            if key not in data_cache:
                data_cache[key] = build_data(copy.deepcopy(cfg))
            data = data_cache[key]

            log.detail(f"\n--- {run}: {family}, {guarantee}, {training} (evaluated {evaluated_as})")
            result, details = diagnose_one(model, cfg, data, settings, rules, device, log)
            row.update(result)
            all_details[run] = details
            log.detail(json.dumps({k: v for k, v in row.items()}, default=str))
        except Exception as exc:
            row["error"] = repr(exc)
            log.say(f"  {run}: failed ({exc!r}); see the log file")
        rows.append(row)
        bar.update(1)
    bar.close()
    if len(data_cache) > 1:
        log.say(f"  note: the runs used {len(data_cache)} different data settings (see the log file)")

    # Step 3: save the numbers
    write_csv(out / "results.csv", rows)
    save_json({"settings": settings, "rows": rows, "details": all_details}, out / "details.json")

    # Step 4: the short result on screen and the summary file
    analysed = [r for r in rows if "family" in r]
    prediction_lines, _ = check_predictions(analysed, rules)
    text = []
    title = "RESULT — stage2_diagnosis   (1 seed: direction only)"
    if args.quick:
        title = title + "   QUICK MODE: code check only, do not interpret"
    text.append(title)
    text.append("")
    text.append("Question: what makes the chained predictions go wrong?")
    text.append("  Timing share (Q1): part of the error that is timing; at least 0.60 = timing-dominated.")
    text.append("  Feedback ratio (Q2): chained error / one-step mistakes just added up; above 1.5 = amplified by feedback.")
    text.append("  Shape push vs true (Q4): how much of a push off the path is left after 50 steps, divided by the")
    text.append("  true system's; 3 or less = corrects like the true system, 10 or more = does not correct. Smaller is better.")
    text.append("  Loop time vs true: free-running loop time with zero input divided by the true one (1.00 = right clock).")
    text.append("")
    text.extend(short_table(analysed))
    text.append("")
    bullets = meaning_bullets(analysed)
    if bullets:
        text.append("What this means:")
        text.extend(bullets)
        text.append("")
    text.append("Math predicted (written before running):")
    text.extend(prediction_lines)
    text.append("")
    text.append("Stage 3 direction:")
    text.append(stage3_direction(analysed))
    failed = [r for r in rows if r.get("error")]
    if failed:
        text.append("")
        text.append(f"  {len(failed)} model(s) could not be analysed (see the log file).")
    summary = "\n".join(text)
    (out / "summary.md").write_text(summary + "\n")
    log.say("")
    log.say(summary)
    log.say(f"\nFull summary: {out / 'summary.md'}   every number: {out / 'details.json'}")
    log.close()


if __name__ == "__main__":
    main()
