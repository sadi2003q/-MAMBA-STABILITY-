"""
Space pilot C — PREDICTION test: a spacecraft with sloshing fuel (single seed: direction only).

Question: during an engine burn the fuel sloshes, but only the spacecraft's angle and turn rate are measured.
Does a chained predictor WITH memory work out the hidden fuel motion and predict the next 5 seconds better than
the same model with the memory removed, and at least as well as two standard memory networks?

Four models, all with your anchor and recipe, all about 3,100-3,300 parameters:
    chain_mamba_memory       Chain-Mamba with memory (yours)
    chain_mamba_no_memory    Chain-Mamba with memory removed (exp04)
    lstm                     long short-term memory network
    gru                      gated recurrent unit

Decision rule (fixed before the run), on the angle error after 25 chained steps averaged over the 4 test fill
levels (never used in training):
    GO   if Chain-Mamba with memory is at least 30% lower than with memory removed
         AND not more than 10% higher than the better of the long short-term memory network and gated recurrent unit
    STOP otherwise

python tools/run_pilot_c_prediction.py \
    --config configs/pilots/pilot_c_slosh_prediction.yaml \
    --seed 0 \
    --simulator pendulum \
    --data-style normal_operations \
    --device auto \
    --quick no

Files: outputs/pilot_c_slosh_prediction/ (summary_seed0.md, results_seed0.json, error_growth_seed0.png, log.txt, models/)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.pilots.predictors import LearnedPredictor, PhysicsPredictor  # noqa: E402
from src.pilots.slosh import build_system  # noqa: E402
from src.pilots.slosh_prediction import chained_errors, draw_training_fills, make_mission_trajectories  # noqa: E402
from src.pilots.training import build_windows, train_predictor  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog  # noqa: E402

# The arms are read from the config (in the order written there); the rotation config adds one more.

LABELS = {
    "frozen_fuel_model": "Frozen-fuel physics model (reference)",
    "chain_mamba_rotating": "Chain-Mamba, rotating memory (new)",
    "chain_mamba_memory": "Chain-Mamba with memory (yours)",
    "chain_mamba_no_memory": "Chain-Mamba memory removed",
    "lstm": "Long short-term memory network",
    "gru": "Gated recurrent unit",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Space pilot C: prediction with sloshing fuel")
    parser.add_argument("--config", default="configs/pilots/pilot_c_slosh_prediction.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulator", default="pendulum", choices=["pendulum", "fluid"])
    parser.add_argument("--data-style", default="normal_operations", choices=["normal_operations", "rich"])
    parser.add_argument("--device", default="auto")
    add_yes_no(parser, "--quick", False, "tiny run to check the code (never interpret)")
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(choice)


def change_words(new: float, base: float) -> str:
    """Percentage change with the project's word labels."""
    change = (new - base) / base * 100.0
    if abs(change) <= 10:
        word = "about the same"
    elif abs(change) <= 20:
        if change < 0:
            word = "slightly lower"
        else:
            word = "slightly higher"
    else:
        if change < 0:
            word = "clearly lower"
        else:
            word = "clearly higher"
    return f"{change:+.0f}% ({word})"


def main():
    args = parse_arguments()

    # ------------------------------------------------------------------ Step 1: settings
    cfg = load_yaml_with_inherit(REPO_ROOT / args.config)
    if args.quick:
        cfg["data"]["train_trajectories"] = 40
        cfg["data"]["val_trajectories"] = 8
        cfg["data"]["test_per_fill"] = 2
        cfg["training"]["epochs"] = 2
    device = pick_device(args.device)
    seed = args.seed
    ARM_NAMES = list(cfg["arms"].keys())
    rule_type = cfg.get("decision_rule", {}).get("type", "memory_vs_standard")
    out = REPO_ROOT / "outputs" / (cfg["name"] + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = ConsoleLog(out / "log.txt")

    log.say("+" + "-" * 78 + "+")
    title = f"SPACE PILOT C — {cfg['name']}   (seed {seed}: direction only)"
    if args.quick:
        title = title + "   CODE CHECK ONLY"
    log.say("| " + title.ljust(77) + "|")
    log.say("+" + "-" * 78 + "+")
    log.say(f"  simulator {args.simulator}  |  data {args.data_style}  |  device {device}")

    # ------------------------------------------------------------------ Step 2: simulator
    if args.simulator == "pendulum":
        system = build_system(cfg)
    else:
        raise NotImplementedError("The fluid simulator is not connected yet: run tools/inspect_fluid_simulator.py "
                                  "and send Claude its output.")

    # ------------------------------------------------------------------ Step 3: data
    d = cfg["data"]
    test_fills = d["test_fills"]
    generator = torch.Generator().manual_seed(int(d["data_seed"]))
    train_fills = draw_training_fills(int(d["train_trajectories"]), float(d["fill_low"]), float(d["fill_high"]),
                                      test_fills, float(d["test_gap"]), generator)
    val_fills = draw_training_fills(int(d["val_trajectories"]), float(d["fill_low"]), float(d["fill_high"]),
                                    test_fills, float(d["test_gap"]), generator)

    started = time.perf_counter()
    train = make_mission_trajectories(system, cfg, train_fills, int(d["data_seed"]), args.data_style)
    val = make_mission_trajectories(system, cfg, val_fills, int(d["data_seed"]) + 1, args.data_style)
    test_by_fill = {}
    for index in range(len(test_fills)):
        fill = test_fills[index]
        fills = torch.full((int(d["test_per_fill"]),), float(fill))
        test_by_fill[fill] = make_mission_trajectories(system, cfg, fills, int(d["data_seed"]) + 10 + index,
                                                       args.data_style)
    data = build_windows(cfg, train, val, test_by_fill[test_fills[0]])
    log.say(f"  data: {len(train_fills)} training / {len(val_fills)} checking trajectories, "
            f"{len(test_fills)} unseen fill levels x {d['test_per_fill']} "
            f"({time.perf_counter() - started:.0f} s, {data['train_x'].shape[0]:,} training windows)")

    # ------------------------------------------------------------------ Step 4: train the four models
    models = {}
    sizes = {}
    minutes = {}
    for name in ARM_NAMES:
        # 4a: shared settings with this arm's model put in (replaced, not merged)
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg["model"] = copy.deepcopy(cfg["arms"][name]["model"])

        # 4b: train, or load if trained before
        started = time.perf_counter()
        save_path = out / "models" / f"{name}_seed{seed}.pt"
        models[name] = train_predictor(arm_cfg, data, seed, device, save_path, log, f"train {name}")
        minutes[name] = (time.perf_counter() - started) / 60.0

        # 4c: size (all weights)
        total = 0
        for parameter in models[name].parameters():
            total = total + parameter.numel()
        sizes[name] = total

    # ------------------------------------------------------------------ Step 5: prediction check
    context_len = int(d["context_len"])
    horizon = int(d["horizon"])
    nominal_fill = float(cfg["system"]["nominal_fill"])

    def frozen_fuel_step(measured, push):
        return system.rigid_step(measured, push, nominal_fill)

    predictors = {"frozen_fuel_model": PhysicsPredictor(frozen_fuel_step)}
    for name in ARM_NAMES:
        predictors[name] = LearnedPredictor(models[name], data["norm"], device)

    results = {}
    for name in predictors:
        results[name] = {"angle_by_fill": {}, "rate_by_fill": {}}
        for fill in test_fills:
            measured, pushes = test_by_fill[fill]
            angle, rate = chained_errors(predictors[name], measured, pushes, context_len, horizon, 10, device)
            results[name]["angle_by_fill"][str(fill)] = angle
            results[name]["rate_by_fill"][str(fill)] = rate

    def average_over_fills(name, kind, step):
        total = 0.0
        for fill in test_fills:
            total = total + results[name][kind][str(fill)][step - 1]
        return total / len(test_fills)

    with open(out / f"results_seed{seed}.json", "w") as handle:
        json.dump({"results": results, "sizes": sizes, "minutes": minutes, "simulator": args.simulator,
                   "data_style": args.data_style, "quick": args.quick}, handle, indent=1)

    # ------------------------------------------------------------------ Step 6: decision
    plain = average_over_fills("chain_mamba_memory", "angle_by_fill", horizon)
    no_memory = average_over_fills("chain_mamba_no_memory", "angle_by_fill", horizon)
    lstm_error = average_over_fills("lstm", "angle_by_fill", horizon)
    gru_error = average_over_fills("gru", "angle_by_fill", horizon)
    best_standard = min(lstm_error, gru_error)
    rule_lines = []
    if rule_type == "rotation_vs_standard":
        # rule written in configs/pilots/pilot_c_slosh_rotation.yaml before the run
        rotating = average_over_fills("chain_mamba_rotating", "angle_by_fill", horizon)
        beats_lstm = rotating <= 0.9 * lstm_error
        beats_gru = rotating <= gru_error
        passed = beats_lstm and beats_gru
        meaning = [
            "   - rotating vs long short-term memory network:  " + change_words(rotating, lstm_error),
            "   - rotating vs gated recurrent unit:            " + change_words(rotating, gru_error),
            "   - rotating vs plain Chain-Mamba with memory:   " + change_words(rotating, plain),
        ]
        rule_lines.append(f"   - at least 10% below long short-term memory:  {'yes' if beats_lstm else 'no'}"
                          f"   ({rotating:.4f} vs {lstm_error:.4f})")
        rule_lines.append(f"   - not above gated recurrent unit:             {'yes' if beats_gru else 'no'}"
                          f"   ({rotating:.4f} vs {gru_error:.4f})")
    else:
        # rule written in configs/pilots/pilot_c_slosh_prediction.yaml before the run
        memory_helps = plain <= 0.7 * no_memory
        as_good_as_standard = plain <= 1.1 * best_standard
        passed = memory_helps and as_good_as_standard
        meaning = [
            "   - with memory vs memory removed:               " + change_words(plain, no_memory),
            "   - with memory vs best standard memory network: " + change_words(plain, best_standard),
        ]
        rule_lines.append(f"   - at least 30% below memory removed:        {'yes' if memory_helps else 'no'}"
                          f"   ({plain:.4f} vs {no_memory:.4f})")
        rule_lines.append(f"   - at most 10% above best standard network:  {'yes' if as_good_as_standard else 'no'}"
                          f"   ({plain:.4f} vs {best_standard:.4f})")
    if passed:
        verdict = "GO (worth confirming with 8 seeds)"
    else:
        verdict = "STOP"

    # ------------------------------------------------------------------ Step 7: summary
    lines = []
    lines.append("")
    lines.append("  Angle error of chained prediction, degrees (average of the 4 unseen fill levels). Smaller is better.")
    lines.append(f"  {'Model':<40}{'Size':>8}{'1 step':>10}{'10 steps':>10}{'25 steps':>10}")
    lines.append("  " + "-" * 78)
    for name in ["frozen_fuel_model"] + ARM_NAMES:
        if name in sizes:
            size_text = f"{sizes[name]:,}"
        else:
            size_text = "-"
        lines.append(f"  {LABELS[name]:<40}{size_text:>8}"
                     f"{average_over_fills(name, 'angle_by_fill', 1):>10.4f}"
                     f"{average_over_fills(name, 'angle_by_fill', 10):>10.4f}"
                     f"{average_over_fills(name, 'angle_by_fill', horizon):>10.4f}")
    lines.append("")
    lines.append("  25-step angle error by fill level (degrees):")
    header = f"  {'Model':<40}"
    for fill in test_fills:
        header = header + f"{str(int(round(fill * 100))) + '% full':>10}"
    lines.append(header)
    lines.append("  " + "-" * 78)
    for name in ["frozen_fuel_model"] + ARM_NAMES:
        row = f"  {LABELS[name]:<40}"
        for fill in test_fills:
            row = row + f"{results[name]['angle_by_fill'][str(fill)][horizon - 1]:>10.4f}"
        lines.append(row)
    lines.append("")
    lines.append("  What this means:")
    for line in meaning:
        lines.append(line)
    lines.append("")
    lines.append("  Decision rule (fixed before the run), 25-step angle error:")
    for line in rule_lines:
        lines.append(line)
    lines.append("")
    lines.append("  Decision: " + verdict)
    lines.append("+" + "-" * 78 + "+")
    text = "\n".join(lines)
    log.say(text)
    (out / f"summary_seed{seed}.md").write_text(text + "\n")

    # ------------------------------------------------------------------ Step 8: figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    steps = list(range(1, horizon + 1))
    for name in ["frozen_fuel_model"] + ARM_NAMES:
        angle_curve = []
        rate_curve = []
        for step in steps:
            angle_curve.append(average_over_fills(name, "angle_by_fill", step))
            rate_curve.append(average_over_fills(name, "rate_by_fill", step))
        width = 2.6 if name in ("chain_mamba_memory", "chain_mamba_rotating") else 1.4
        axes[0].plot(steps, angle_curve, label=LABELS[name], linewidth=width)
        axes[1].plot(steps, rate_curve, label=LABELS[name], linewidth=width)
    axes[0].set_title("Angle error (degrees)")
    axes[1].set_title("Turn-rate error (degrees per second)")
    for axis in axes:
        axis.set_xlabel("chained prediction step (0.2 seconds each)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Sloshing fuel, unseen fill levels (seed {seed}, direction only)")
    figure.tight_layout()
    figure.savefig(out / f"error_growth_seed{seed}.png", dpi=150)
    log.say(f"  files: {out}")
    log.close()


if __name__ == "__main__":
    main()
