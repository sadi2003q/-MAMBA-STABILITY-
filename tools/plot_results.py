"""Figures for an experiment folder: error against prediction step, and 50-step error per arm.

python tools/plot_results.py --experiment outputs/exp01_certificate_ablation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.utils.io import load_json  # noqa: E402
from src.utils.summary import GROUP_TITLES, read_results  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    args = p.parse_args()
    folder = Path(args.experiment)
    exp = yaml.safe_load((folder / "experiment_config.yaml").read_text())
    rows = read_results(folder / "results.csv")
    figs = folder / "figures"
    figs.mkdir(exist_ok=True)
    groups = {}
    for a in exp["arms"]:
        groups.setdefault(a.get("group", "all"), []).append(a["name"])

    # 1. error against prediction step, seed 0 (or the lowest seed)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.2), squeeze=False)
    for ax, (group, names) in zip(axes[0], groups.items()):
        for name in names:
            runs = sorted((folder / "runs").glob(f"{name}_seed*"))
            if not runs or not (runs[0] / "metrics.json").exists():
                continue
            m = load_json(runs[0] / "metrics.json")
            if m.get("diverged"):
                continue
            curve = m["primary"]["per_step_rmse"]
            ax.plot(np.arange(1, len(curve) + 1), curve, label=name)
        ax.set_yscale("log")
        ax.set_title(GROUP_TITLES.get(group, group), fontsize=9)
        ax.set_xlabel("prediction step")
        ax.set_ylabel("root-mean-square error (smaller is better)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "error_against_step.png", dpi=130)
    plt.close(fig)

    # 2. 50-step error per arm, all seeds (diverged runs left out and counted in the label)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(exp["arms"])), 4.5))
    data, labels = [], []
    for a in exp["arms"]:
        rs = [r for r in rows if r["arm"] == a["name"]]
        ok = [r["rollout_rmse"] for r in rs if not r["diverged"] and r["rollout_rmse"] is not None]
        div = sum(r["diverged"] for r in rs)
        data.append(ok if ok else [np.nan])
        labels.append(a["name"] + (f"\n({div} blew up)" if div else ""))
    ax.boxplot(data, tick_labels=labels)
    ax.set_yscale("log")
    ax.set_ylabel("50-step root-mean-square error (smaller is better)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "rollout_error_per_arm.png", dpi=130)
    plt.close(fig)
    print(f"Figures written to {figs}")


if __name__ == "__main__":
    main()
