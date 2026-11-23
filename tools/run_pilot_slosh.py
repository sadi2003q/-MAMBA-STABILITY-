"""
Space pilot C: turning a spacecraft during an engine burn while its fuel sloshes (single seed: direction only).

What it does, in order:
    1. makes flight-like training data at many fill levels (only angle and turn rate are recorded)
    2. trains your Chain-Mamba twice with your recipe: memory kept, and memory removed (or loads them)
    3. checks the predictors: angle error of chained prediction after 1, 10 and 25 steps, next to the frozen-fuel model
    4. runs 6 controllers on 16 spacecraft (4 unseen fill levels x 2 turns x 2 starting slosh levels), 60 s each
    5. applies the decision rule from the config and writes summary.md  ->  GO, STOP or INVALID

python tools/run_pilot_slosh.py \
    --config configs/pilots/pilot_c_slosh.yaml \
    --seed 0 \
    --device cpu \
    --quick no
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.pilots.mpc import PilotMPC  # noqa: E402
from src.pilots.predictors import LearnedPredictor, PhysicsPredictor  # noqa: E402
from src.pilots.report import append_row, read_rows  # noqa: E402
from src.pilots.slosh import (AttitudeLaw, build_system, chained_errors_with_warmup, make_cost,  # noqa: E402
                              make_test_cases, make_trajectories, run_closed_loop)
from src.pilots.training import build_windows, train_predictor  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.io import save_json  # noqa: E402

RESULT_COLUMNS = ["seed", "version", "settled", "cases", "settle_median_s", "settle_mean_settled_s",
                  "final_error_median_deg", "final_slosh_median_deg", "effort_median", "lost", "planning_ms",
                  "minutes"]

# (version, what it plans with, is it told the fuel state?)
CONTROLLERS = [
    ("attitude_law", "pd", False),
    ("frozen_fuel_mpc", "rigid", False),
    ("chain_mamba_no_memory_mpc", "no_memory", False),
    ("chain_mamba_memory_mpc", "memory", False),
    ("pendulum_60_mpc_told_slosh", "pendulum_nominal", True),
    ("true_model_mpc_told_slosh", "true", True),
]

LABELS = {
    "attitude_law": "Standard attitude law (0.1 Hz)",
    "frozen_fuel_mpc": "Predictive control, frozen-fuel model",
    "chain_mamba_no_memory_mpc": "Predictive control, Chain-Mamba memory removed",
    "chain_mamba_memory_mpc": "Predictive control, Chain-Mamba with memory (yours)",
    "pendulum_60_mpc_told_slosh": "Pendulum model tuned at 60% fill, told the slosh*",
    "true_model_mpc_told_slosh": "True model, told the slosh (best possible)*",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Space pilot C: spacecraft with sloshing fuel")
    parser.add_argument("--config", default="configs/pilots/pilot_c_slosh.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    add_yes_no(parser, "--quick", False, "tiny run to check the code (never interpret)")
    return parser.parse_args()


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def make_controller(kind, system, cfg, targets, fills, models, norm, device):
    nominal = float(cfg["system"]["nominal_fill"])
    if kind == "pd":
        return AttitudeLaw(system, cfg, targets)

    if kind == "rigid":
        predictor = PhysicsPredictor(lambda s, u: system.rigid_step(s, u, nominal))
    elif kind == "true":
        predictor = PhysicsPredictor(lambda s, u: system.step(s, u, fills[: s.shape[0]], substeps=1))
    elif kind == "pendulum_nominal":
        predictor = PhysicsPredictor(lambda s, u: system.step(s, u, torch.full((s.shape[0],), nominal), substeps=1))
    else:
        predictor = LearnedPredictor(models[kind], norm, device)

    c = cfg["controller"]
    return PilotMPC(predictor, make_cost(cfg, targets), int(c["horizon"]), 1, system.u_max, cfg["solver"],
                    offset_free=False, correction_filter=0.0, device=device)


def main():
    args = parse_arguments()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = load_yaml_with_inherit(REPO_ROOT / args.config)
    device = torch.device(args.device)
    if args.quick:
        cfg["data"].update({"train_trajectories": 40, "val_trajectories": 8, "test_trajectories": 8})
        cfg["training"]["epochs"] = 2
        cfg["test"]["steps"] = 20
        cfg["test"]["fills"] = [0.5]
    out = REPO_ROOT / "outputs" / (cfg["name"] + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(out / "log.txt")
    system = build_system(cfg)
    seed = args.seed

    log.say("=" * 78)
    log.say(f" SPACE PILOT C — sloshing fuel   (seed {seed}: direction only{'; CODE CHECK ONLY' if args.quick else ''})")
    log.say("=" * 78)

    # ------------------------------------------------------------------ 1. data
    data_path = out / "data.pt"
    if data_path.exists():
        saved = torch.load(data_path)
        train, val, test = saved["train"], saved["val"], saved["test"]
        log.say(" data: loaded")
    else:
        started = time.perf_counter()
        d = cfg["data"]
        base = int(d["data_seed"])
        train = make_trajectories(system, cfg, int(d["train_trajectories"]), base)
        val = make_trajectories(system, cfg, int(d["val_trajectories"]), base + 1)
        test = make_trajectories(system, cfg, int(d["test_trajectories"]), base + 2)
        torch.save({"train": train, "val": val, "test": test}, data_path)
        log.say(f" data: {train[0].shape[0]} / {val[0].shape[0]} / {test[0].shape[0]} trajectories of "
                f"{train[1].shape[1]} steps ({time.perf_counter() - started:.0f} s)")
    data = build_windows(cfg, train[:2], val[:2], test[:2])

    # ------------------------------------------------------------------ 2. train both predictors
    models = {}
    for kind in ["memory", "no_memory"]:
        arm_cfg = deep_merge(copy.deepcopy(cfg), cfg["models"][kind])
        models[kind] = train_predictor(arm_cfg, data, seed, device, out / "models" / f"{kind}_seed{seed}.pt", log,
                                       f"train {kind}")

    # ------------------------------------------------------------------ 3. prediction check
    context_len = int(cfg["data"]["context_len"])
    horizon = int(cfg["data"]["horizon"])
    nominal = float(cfg["system"]["nominal_fill"])
    check = {}
    rigid = PhysicsPredictor(lambda s, u: system.rigid_step(s, u, nominal))
    for name, predictor in [("frozen_fuel", rigid),
                            ("no_memory", LearnedPredictor(models["no_memory"], data["norm"], device)),
                            ("memory", LearnedPredictor(models["memory"], data["norm"], device))]:
        errors = chained_errors_with_warmup(predictor, test[0], test[1], context_len, horizon, every=10)
        check[name] = {"1": errors[0], "10": errors[9], str(horizon): errors[-1]}
    save_json(check, out / "predictor_check.json")
    log.say(" chained prediction, angle error (degrees) on unseen trajectories:")
    log.say(f"   {'model':<22}{'1 step':>10}{'10 steps':>10}{str(horizon) + ' steps':>10}")
    for name, values in check.items():
        log.say(f"   {name:<22}{values['1']:>10.4f}{values['10']:>10.4f}{values[str(horizon)]:>10.4f}")

    # ------------------------------------------------------------------ 4. closed loop
    fills, targets, starts = make_test_cases(cfg)
    results_path = out / "results.csv"
    done = {(int(r["seed"]), r["version"]) for r in read_rows(results_path)}
    all_metrics = {}
    for index, (version, kind, full_state) in enumerate(CONTROLLERS):
        if (seed, version) in done:
            continue
        started = time.perf_counter()
        controller = make_controller(kind, system, cfg, targets, fills, models, data["norm"], device)
        bar = progress_bar(int(cfg["test"]["steps"]), f"run {index + 1}/{len(CONTROLLERS)} {version}", "step")
        metrics = run_closed_loop(controller, system, fills, targets, starts, cfg, full_state, context_len,
                                  noise_seed=1000 + seed, bar=bar)
        bar.close()
        minutes = (time.perf_counter() - started) / 60.0
        settled = [t for t in metrics["settle_time"] if math.isfinite(t)]
        row = {
            "seed": seed, "version": version, "settled": len(settled), "cases": len(metrics["settle_time"]),
            "settle_median_s": median(metrics["settle_time"]),
            "settle_mean_settled_s": sum(settled) / len(settled) if settled else float("inf"),
            "final_error_median_deg": median(metrics["final_error_degrees"]),
            "final_slosh_median_deg": median(metrics["final_slosh_degrees"]),
            "effort_median": median(metrics["effort"]), "lost": sum(metrics["lost"]),
            "planning_ms": metrics["planning_ms"], "minutes": minutes,
        }
        append_row(results_path, row, RESULT_COLUMNS)
        all_metrics[version] = metrics
        save_json(metrics, out / f"metrics_{version}_seed{seed}.json")
        log.say(f"  {version:<30} settled {row['settled']:>2}/{row['cases']}  median {row['settle_median_s']:.1f} s  "
                f"final error {row['final_error_median_deg']:.3f} deg  ({minutes:.1f} min)")

    write_summary(cfg, read_rows(results_path), check, seed, args.quick, out, log, models)
    log.close()


def write_summary(cfg, rows, check, seed, quick, out, log, models):
    rows = {r["version"]: r for r in rows if int(r["seed"]) == seed}
    if len(rows) < len(CONTROLLERS):
        log.say(" (not all controllers finished; summary skipped)")
        return

    def value(version, column):
        return float(rows[version][column])

    true_settle = value("true_model_mpc_told_slosh", "settle_median_s")
    frozen = value("frozen_fuel_mpc", "settle_median_s")
    law = value("attitude_law", "settle_median_s")
    ours = value("chain_mamba_memory_mpc", "settle_median_s")
    cases = int(rows["true_model_mpc_told_slosh"]["cases"])
    valid = int(rows["true_model_mpc_told_slosh"]["settled"]) >= 0.75 * cases
    problem_real = frozen >= 1.5 * true_settle
    best_standard = min(frozen, law)
    settled_ok = int(rows["chain_mamba_memory_mpc"]["settled"]) >= max(int(rows["frozen_fuel_mpc"]["settled"]),
                                                                      int(rows["attitude_law"]["settled"]))
    faster = ours <= 0.7 * best_standard
    if not valid:
        verdict = "INVALID (true-model controller did not settle often enough: solver floor)"
    elif problem_real and faster and settled_ok:
        verdict = "GO (worth confirming with 8 seeds)"
    else:
        verdict = "STOP"

    def time_text(t):
        return "never" if not math.isfinite(t) else f"{t:.1f} s"

    lines = []
    lines.append(f"RESULT — pilot_c_slosh   (1 seed: direction only{'; CODE CHECK ONLY — do not interpret' if quick else ''})")
    lines.append("")
    lines.append("Question: can predictive control with your Chain-Mamba (memory kept, trained only on measured angle")
    lines.append("and turn rate) settle a turning spacecraft with sloshing fuel faster than the standard controllers?")
    lines.append("")
    sizes = {kind: sum(p.numel() for p in models[kind].parameters()) for kind in models}
    lines.append(f"Parameters: with memory {sizes['memory']:,}; memory removed {sizes['no_memory']:,}")
    lines.append("")
    lines.append(f"  {'Controller':<55}{'Settled':>9}{'Median settling':>17}{'Pointing error':>16}{'Slosh left':>12}")
    for version, _, _ in CONTROLLERS:
        r = rows[version]
        lines.append(f"  {LABELS[version]:<55}{r['settled'] + '/' + r['cases']:>9}"
                     f"{time_text(float(r['settle_median_s'])):>17}{float(r['final_error_median_deg']):>13.3f} deg"
                     f"{float(r['final_slosh_median_deg']):>9.2f} deg")
    lines.append("  * told the true fuel state, which a real spacecraft cannot measure (reference only)")
    lines.append("")
    lines.append("  Settling = pointing stays within 0.1 degrees until the end (60 s run). Smaller is better.")
    lines.append("  Pointing error and slosh left = last 15 s.")
    lines.append("")
    lines.append("Chained prediction, angle error after 1 / 10 / 25 steps (degrees, unseen trajectories):")
    for name, values in check.items():
        lines.append(f"  {name:<14} {values['1']:.4f} / {values['10']:.4f} / {values['25']:.4f}")
    lines.append("")
    lines.append("Decision rule (fixed before the run):")
    lines.append(f"  valid (true model settles >= 75% of cases):           {'yes' if valid else 'NO'}")
    lines.append(f"  problem real (frozen-fuel >= 1.5 x true model):       {'yes' if problem_real else 'no'}"
                 f"  ({time_text(frozen)} vs {time_text(true_settle)})")
    lines.append(f"  yours <= 0.7 x best standard controller:              {'yes' if faster else 'no'}"
                 f"  ({time_text(ours)} vs {time_text(best_standard)})")
    lines.append(f"  yours settles at least as often as both:              {'yes' if settled_ok else 'no'}")
    lines.append("")
    lines.append(f"Decision: {verdict}")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    log.say("")
    log.say(text)


if __name__ == "__main__":
    main()
