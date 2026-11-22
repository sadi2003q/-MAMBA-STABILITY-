"""
Space paper, pilot 1 — circuit-usage test (no training; uses the predictors saved by tools/run_space.py).

Question: does the trained quantum predictor actually USE its quantum circuit?

How: replace every circuit's outputs with their AVERAGE value (measured on training windows), so the circuit no
longer reacts to its input, and measure how much the 10-step chained-prediction error grows.

Rule (fixed before running, the same as the language-model test):
    error grows by less than 0.5%   -> circuit MOSTLY IGNORED  (hardware defects in it cannot matter much)
    0.5% to 5%                      -> circuit USED A LITTLE
    more than 5%                    -> circuit CLEARLY USED

python tools/check_space_circuit.py \
    --config configs/space/space01_quantum_hardware.yaml \
    --seed 0 \
    --device cpu

Files: outputs/space01_quantum_hardware/circuit_usage.md and circuit_usage.json, details in circuit_check_log.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from src.space.data_space import build_space_data  # noqa: E402
from src.space.quantum import get_circuits, set_hardware  # noqa: E402
from src.space.training_space import build_space_model, chained_error  # noqa: E402
from src.systems.spacecraft import Spacecraft  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, deep_merge, load_yaml_with_inherit  # noqa: E402
from src.utils.console import ConsoleLog  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402

IGNORED_BELOW_PERCENT = 0.5
CLEARLY_USED_ABOVE_PERCENT = 5.0


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    parser.add_argument("--output-root", default=None)
    add_yes_no(parser, "--quick", False, "use the models of the quick code-check run")
    return parser.parse_args()


@torch.no_grad()
def average_circuit_outputs(model, data: dict, cfg: dict, device, windows: int = 512) -> dict:
    """Run the model on training windows and record the average output of every circuit."""
    # Step 1: one recording hook per circuit
    sums = {}
    counts = {}
    hooks = []
    for circuit in get_circuits(model):
        sums[circuit] = None
        counts[circuit] = 0

        def record(module, inputs, output):
            batch_sum = output.detach().sum(dim=0)
            if sums[module] is None:
                sums[module] = batch_sum
            else:
                sums[module] = sums[module] + batch_sum
            counts[module] = counts[module] + output.shape[0]

        hooks.append(circuit.register_forward_hook(record))

    # Step 2: chained prediction on training windows (perfect hardware), exactly as during control
    set_hardware(model)
    context_len = int(cfg["data"]["context_len"])
    x = data["train_x"][:windows].to(device)
    u = data["train_u"][:windows].to(device)
    model.chained(x[:, :context_len + 1], u, 10)

    # Step 3: remove the hooks, return the averages
    for hook in hooks:
        hook.remove()
    averages = {}
    for circuit in sums:
        averages[circuit] = sums[circuit] / counts[circuit]
    return averages


def error_with_constant_circuits(model, data: dict, cfg: dict, device, averages: dict, steps: int) -> float:
    """Chained error when every circuit always outputs its average value (ignores its input)."""
    hooks = []
    for circuit, average in averages.items():

        def constant(module, inputs, output, value=average):
            return value.unsqueeze(0).expand_as(output)

        hooks.append(circuit.register_forward_hook(constant))
    error = chained_error(model, data, cfg, device, steps)
    for hook in hooks:
        hook.remove()
    return error


def main():
    args = parse_arguments()

    # Step 1: settings, folders, log
    cfg = load_yaml_with_inherit(args.config)
    if args.quick:
        cfg = deep_merge(cfg, cfg.get("quick", {}))
    name = cfg["name"] + ("_quick" if args.quick else "")
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs"
    output_dir = output_root / name
    log = ConsoleLog(output_dir / "circuit_check_log.txt")
    log.capture_warnings()
    device = pick_device(args.device)
    seed = int(args.seed)
    log.say(f"CIRCUIT-USAGE TEST — {name}   (seed {seed}; no training)")
    log.say(f"Device: {describe(device)}")

    # Step 2: the same data as the main run (identical, from the same data seed)
    system = Spacecraft.from_config(cfg)
    data = build_space_data(cfg, system)

    # Step 3: load the saved predictors (never train here)
    models = {}
    for model_version in ["classical", "quantum", "quantum_noise_aware"]:
        path = output_dir / "models" / f"{model_version}_seed{seed}.pt"
        if not path.exists():
            log.say(f"  {model_version}: no saved model at {path} — run tools/run_space.py first (skipped)")
            continue
        model = build_space_model(cfg, model_version, data["increment_std"]).to(device)
        saved = torch.load(path, map_location=device)
        model.load_state_dict(saved["model"])
        model.eval()
        models[model_version] = model

    if "quantum" not in models and "quantum_noise_aware" not in models:
        log.say("STOP: no quantum predictor found.")
        log.close()
        return

    # Step 4: the test
    results = {}
    lines = []
    lines.append(f"CIRCUIT-USAGE TEST — {cfg['name']}   (seed {seed}: direction only)")
    lines.append("")
    lines.append("Question: does the trained quantum predictor actually use its quantum circuit?")
    lines.append("")
    lines.append(f"{'':34s} {'10-step error':>14s} {'circuit = average':>18s} {'change':>9s}  verdict")
    if "classical" in models:
        classical_error = chained_error(models["classical"], data, cfg, device, 10)
        results["classical"] = {"ten_steps": classical_error}
        lines.append(f"  {'Classical Chain-Mamba (reference)':32s} {classical_error:14.4f} {'—':>18s} {'—':>9s}")

    for model_version in ["quantum", "quantum_noise_aware"]:
        if model_version not in models:
            continue
        model = models[model_version]
        normal = chained_error(model, data, cfg, device, 10)
        averages = average_circuit_outputs(model, data, cfg, device)
        constant = error_with_constant_circuits(model, data, cfg, device, averages, 10)
        change = (constant / normal - 1.0) * 100.0
        if change < IGNORED_BELOW_PERCENT:
            verdict = "MOSTLY IGNORED"
        elif change <= CLEARLY_USED_ABOVE_PERCENT:
            verdict = "USED A LITTLE"
        else:
            verdict = "CLEARLY USED"
        results[model_version] = {"ten_steps": normal, "ten_steps_circuit_average": constant,
                                  "change_percent": change, "verdict": verdict}
        label = "Quantum" if model_version == "quantum" else "Quantum, practising with noise"
        lines.append(f"  {label:32s} {normal:14.4f} {constant:18.4f} {change:+8.2f}%  {verdict}")

    lines.append("  (chained-prediction error after 10 steps, normalised units; smaller is better)")
    lines.append("")
    lines.append("What this means:")
    lines.append(f"  • Below +{IGNORED_BELOW_PERCENT}%: the circuit hardly matters, so its hardware defects cannot matter")
    lines.append("    either -> a 'defects do not hurt' result would be a null result, not robustness.")
    lines.append(f"  • Above +{CLEARLY_USED_ABOVE_PERCENT}%: the circuit carries real information -> if defects still do not hurt")
    lines.append("    control, that is genuine robustness.")

    for line in lines:
        log.say(line)
    (output_dir / "circuit_usage.md").write_text("\n".join(lines) + "\n")
    save_json(results, output_dir / "circuit_usage.json")
    log.close()


if __name__ == "__main__":
    main()
