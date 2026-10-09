"""Time training batches on every available device (processor, Apple graphics, NVIDIA graphics card),
so the faster one can be written into configs/base.yaml. The models loop over time step by step,
so small models are often as fast on the processor.

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
from src.utils.config import load_experiment, resolve_arm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Time one epoch on each available device.")
    p.add_argument("--config", required=True)
    p.add_argument("--arms", nargs="+", help="default: one arm per training regime")
    p.add_argument("--batches", type=int, default=5)
    args = p.parse_args()
    exp = load_experiment(args.config)
    if args.arms:
        arms = [a for a in exp["arms"] if a["name"] in args.arms]
    else:
        seen, arms = set(), []
        for a in exp["arms"]:
            if a["training_regime"] not in seen:
                seen.add(a["training_regime"])
                arms.append(a)
    cfgs = {a["name"]: resolve_arm(exp, a) for a in arms}
    data = build_data(next(iter(cfgs.values())))
    n_train = data["splits"]["train"].xw.shape[0]

    devices = ["cpu"]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    print(f"{'arm':26s}{'device':8s}{'seconds per batch':>20s}{'minutes per epoch':>20s}{'minutes per run':>18s}")
    best = {}
    for name, cfg in cfgs.items():
        bpe = -(-n_train // cfg["training"]["batch_size"])
        for dev in devices:
            sec = time_batches(cfg, data, torch.device(dev), n_batches=args.batches)
            per_epoch = sec * bpe / 60.0
            per_run = per_epoch * cfg["training"]["epochs"] * 1.4
            print(f"{name:26s}{dev:8s}{sec:>20.3f}{per_epoch:>20.2f}{per_run:>18.1f}")
            best.setdefault(name, []).append((per_run, dev))
    print()
    for name, opts in best.items():
        per_run, dev = min(opts)
        print(f"Fastest for {name}: {dev} (about {per_run:.1f} minutes per run)")
    print("Write the overall fastest into 'device:' in configs/base.yaml.")


if __name__ == "__main__":
    main()
