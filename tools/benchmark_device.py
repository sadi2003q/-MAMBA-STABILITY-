"""Time training batches on every available device (processor, Apple graphics, NVIDIA graphics card),
so the faster one can be used. The models loop over time step by step, so small models are often
as fast on the processor.

On screen: one progress bar, one line per training regime, and the overall winner.
The full timing table goes to outputs/<experiment>/benchmark_log.txt.

python tools/benchmark_device.py --config configs/experiments/exp01_certificate_ablation.yaml
python tools/benchmark_device.py --config ... --arms certified_chained lstm_chained
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.training.trainer import time_batches  # noqa: E402
from src.utils.config import load_experiment, output_dir, resolve_arm  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402

DEVICE_NAMES = {"cpu": "processor", "mps": "Apple graphics", "cuda": "graphics card"}


def available_devices() -> list[str]:
    """The processor always; Apple graphics and the NVIDIA graphics card when present."""
    devices = ["cpu"]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def main():
    p = argparse.ArgumentParser(description="Time one run on each available device.")
    p.add_argument("--config", required=True)
    p.add_argument("--arms", nargs="+", help="default: one arm per training regime")
    p.add_argument("--batches", type=int, default=5)
    args = p.parse_args()

    # Step 1: pick the arms to time (default: the first arm of each training regime)
    exp = load_experiment(args.config)
    arms = []
    if args.arms:
        for arm in exp["arms"]:
            if arm["name"] in args.arms:
                arms.append(arm)
    else:
        seen_regimes = set()
        for arm in exp["arms"]:
            if arm["training_regime"] not in seen_regimes:
                seen_regimes.add(arm["training_regime"])
                arms.append(arm)

    # Step 2: log file and data
    log = ConsoleLog(output_dir(exp, quick=False) / "benchmark_log.txt", fresh=True)
    log.capture_warnings()
    cfgs = {}
    for arm in arms:
        cfgs[arm["name"]] = resolve_arm(exp, arm)
    data = build_data(next(iter(cfgs.values())))
    n_train = data["splits"]["train"].xw.shape[0]
    devices = available_devices()

    # Step 3: time every (arm, device) pair
    log.detail(f"{'arm':26s}{'device':8s}{'seconds per batch':>20s}{'minutes per epoch':>20s}{'minutes per run':>18s}")
    minutes = {}            # minutes[arm name][device] = minutes per run
    total_per_device = {}   # summed over the timed arms
    for dev in devices:
        total_per_device[dev] = 0.0
    bar = progress_bar(total=len(arms) * len(devices), description="timing", unit="test")
    for name, cfg in cfgs.items():
        minutes[name] = {}
        batches_per_epoch = -(-n_train // cfg["training"]["batch_size"])    # rounds up
        for dev in devices:
            seconds = time_batches(cfg, data, torch.device(dev), n_batches=args.batches)
            per_epoch = seconds * batches_per_epoch / 60.0
            per_run = per_epoch * cfg["training"]["epochs"] * 1.4     # 1.4: validation + evaluation
            minutes[name][dev] = per_run
            total_per_device[dev] = total_per_device[dev] + per_run
            log.detail(f"{name:26s}{dev:8s}{seconds:>20.3f}{per_epoch:>20.2f}{per_run:>18.1f}")
            bar.update(1)
    bar.close()

    # Step 4: one short line per arm, e.g. "certified_teacher   processor 4.1 min | graphics card 6.3 min → processor"
    log.say("Minutes per training run:")
    for name, by_device in minutes.items():
        parts = []
        for dev, value in by_device.items():
            parts.append(f"{DEVICE_NAMES[dev]} {value:.1f}")
        fastest = min(by_device, key=by_device.get)
        log.say(f"  {name:26s}" + " | ".join(parts) + f"   → {DEVICE_NAMES[fastest]}")

    # Step 5: the overall winner
    best = min(total_per_device, key=total_per_device.get)
    log.say(f"Fastest overall: {DEVICE_NAMES[best]}. Use it with:  --device {best}")
    log.close()


if __name__ == "__main__":
    main()
