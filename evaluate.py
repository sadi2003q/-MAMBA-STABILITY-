"""Re-evaluate saved checkpoints of an experiment (for example after adding a diagnostic).
Rewrites each run's metrics.json and results.csv; never retrains.

python evaluate.py --config configs/experiments/exp01_certificate_ablation.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.models.registry import build_model, count_parameters  # noqa: E402
from src.training.trainer import evaluate_model, result_row  # noqa: E402
from src.utils.config import load_experiment, output_dir, resolve_arm  # noqa: E402
from src.utils.device import pick_device  # noqa: E402
from src.utils.io import append_result, load_json, save_json  # noqa: E402
from src.utils.summary import build_summary, read_results  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Re-evaluate saved checkpoints.")
    p.add_argument("--config", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    exp = load_experiment(args.config)
    out = output_dir(exp, args.quick, args.output_root)
    device = pick_device(args.device or exp.get("device", "auto"))
    cfgs = {a["name"]: resolve_arm(exp, a, quick=args.quick) for a in exp["arms"]}
    data = build_data(next(iter(cfgs.values())))
    run_dirs = sorted((out / "runs").glob("*_seed*"))
    if not run_dirs:
        sys.exit(f"No runs found in {out / 'runs'}.")
    new_csv = out / "results.csv.new"
    if new_csv.exists():
        new_csv.unlink()
    for run_dir in tqdm(run_dirs, desc="re-evaluating"):
        name, seed = run_dir.name.rsplit("_seed", 1)
        if name not in cfgs or not (run_dir / "checkpoint.pt").exists():
            tqdm.write(f"  skipping {run_dir.name} (unknown arm or no final checkpoint)")
            continue
        cfg = cfgs[name]
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)["model"])
        old = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
        metrics = evaluate_model(model, cfg, data, device)
        metrics["training"] = old.get("training", {"diverged_during_training": False, "nonfinite_steps": 0,
                                                   "epochs_run": 0, "train_minutes": 0.0})
        metrics["parameters"] = count_parameters(model)
        metrics["diverged"] = bool(metrics["training"].get("diverged_during_training") or metrics["primary"]["diverged"])
        save_json(metrics, run_dir / "metrics.json")
        append_result(new_csv, result_row(cfg, int(seed), metrics))
    new_csv.replace(out / "results.csv")
    exp_clean = {k: v for k, v in exp.items() if not k.startswith("_")}
    text, machine = build_summary(exp_clean, read_results(out / "results.csv"), quick=args.quick)
    (out / "summary.md").write_text(text)
    save_json(machine, out / "summary.json")
    print(text)


if __name__ == "__main__":
    main()
