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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import time  # noqa: E402

from tqdm.auto import tqdm  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.models.registry import REFERENCE_PARAMETERS, build_model, count_parameters  # noqa: E402
from src.training.trainer import result_row, time_batches, train_run  # noqa: E402
from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import load_experiment, output_dir, resolve_arm, save_yaml  # noqa: E402
from src.utils.device import describe, pick_device  # noqa: E402
from src.utils.io import append_result, completed_runs, save_json  # noqa: E402
from src.utils.summary import build_summary, read_results  # noqa: E402


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
    add_yes_no(p, "--estimate", True, "time a few batches per arm and print the run-time estimate")
    return p.parse_args()


def main():
    args = parse_args()
    exp = load_experiment(args.config)
    seeds = args.seeds or [args.seed]
    arms = [a for a in exp["arms"] if not args.arms or a["name"] in args.arms]
    if args.arms:
        missing = set(args.arms) - {a["name"] for a in arms}
        if missing:
            sys.exit(f"Unknown arm(s): {', '.join(sorted(missing))}")
    cfgs = {a["name"]: resolve_arm(exp, a, quick=args.quick) for a in arms}
    out = output_dir(exp, args.quick, args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "results.csv"
    if not args.resume and results_csv.exists():
        results_csv.rename(out / f"results_replaced_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    device = pick_device(args.device or exp.get("device", "auto"))
    first = cfgs[arms[0]["name"]]
    data = build_data(first)
    n_train = data["splits"]["train"].xw.shape[0]
    batches_per_epoch = -(-n_train // first["training"]["batch_size"])

    save_yaml({k: v for k, v in exp.items() if not k.startswith("_")}, out / "experiment_config.yaml")
    save_json({"normaliser": data["norm"].to_dict(), "windows": {k: int(s.xw.shape[0]) for k, s in data["splits"].items()},
               "context_len": data["context_len"], "horizon": data["horizon"]}, out / "data_info.json")

    print("=" * 88)
    if args.quick:
        print("QUICK RUN — code check only, do not interpret the numbers")
    print(f"Experiment : {exp['name']}")
    print(f"Hypothesis : {exp['hypothesis'].strip()}")
    print(f"Device     : {describe(device)}")
    print(f"System     : {first['system']['name']}, mu = {first['system']['mu']}, time step = {first['system']['dt']}")
    print(f"Data       : {n_train} training windows ({batches_per_epoch} batches of {first['training']['batch_size']}), "
          f"{data['splits']['val'].xw.shape[0]} validation, {data['splits']['test'].xw.shape[0]} test; "
          f"context {data['context_len']} steps, horizon {data['horizon']} steps")
    print(f"Training   : {first['training']['epochs']} epochs, learning rate {first['training']['lr']}, seeds {seeds}")
    print(f"Outputs    : {out}")
    print("-" * 88)

    done = completed_runs(results_csv)
    pending = [(a, s) for s in seeds for a in arms if (a["name"], s) not in done]
    est_minutes = {}
    print(f"{'arm':26s}{'regime':17s}{'model':13s}{'parameters':>11s}{'reference':>11s}{'est. minutes':>14s}")
    for a in arms:
        cfg = cfgs[a["name"]]
        n_par = count_parameters(build_model(cfg))
        ref = REFERENCE_PARAMETERS.get(cfg["model"]["name"])
        ref_s = f"{ref:,}" if ref else "—"
        if not args.estimate:
            est = None
        else:
            sec = time_batches(cfg, data, device)
            est = sec * batches_per_epoch * cfg["training"]["epochs"] * 1.4 / 60.0   # 1.4: validation + evaluation
        est_minutes[a["name"]] = est
        est_s = "—" if est is None else f"{est:.1f}"
        print(f"{a['name']:26s}{cfg['training']['regime']:17s}{cfg['model']['name']:13s}{n_par:>11,d}{ref_s:>11s}{est_s:>14s}")
    total = sum(est_minutes[a["name"]] or 0 for a, _ in pending)
    print("-" * 88)
    print(f"Runs: {len(pending)} pending, {len(done)} already finished (resuming). "
          f"Estimated time for pending runs: {total:.0f} minutes" + (" (rough)" if total else ""))
    print("=" * 88)
    limit = exp.get("run", {}).get("long_run_confirm_minutes", 30)
    if total > limit and args.ask_first and sys.stdin.isatty():
        if input(f"This is longer than {limit} minutes. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Stopped before training.")

    outer = tqdm(pending, desc="runs", unit="run")
    for i, (arm, seed) in enumerate(outer, 1):
        label = f"run {i}/{len(pending)}: {arm['name']} seed{seed}"
        outer.set_description(label)
        run_dir = out / "runs" / f"{arm['name']}_seed{seed}"
        metrics = train_run(cfgs[arm["name"]], data, seed, run_dir, device, fresh=not args.resume, position_label=label)
        row = result_row(cfgs[arm["name"]], seed, metrics)
        append_result(results_csv, row)
        status = "BLEW UP" if row["diverged"] else (f"next-step {row['one_step_rmse']:.3g}, "
                                                   f"50-step {row['rollout_rmse']:.3g}")
        tqdm.write(f"  finished {arm['name']} seed{seed}: {status} ({row['train_minutes']:.1f} min training)")
    outer.close()

    rows = read_results(results_csv)
    text, machine = build_summary(exp, rows, quick=args.quick)
    (out / "summary.md").write_text(text)
    save_json(machine, out / "summary.json")
    print()
    print(text)
    print(f"Summary written to {out / 'summary.md'}")


if __name__ == "__main__":
    main()
