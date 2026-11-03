"""Side-by-side comparison of trained runs on the same test windows, with figures. No training.

python tools/compare_runs.py \
    --runs "Full Mamba (exp01d)=models/seed0/exp01d_certificate_ablation_confirm/certified_chained_seed0" \
           "Long Short-Term Memory network (exp01d)=models/seed0/exp01d_certificate_ablation_confirm/lstm_chained_seed0" \
           "Chain-Mamba=outputs/exp03_chain_mamba/runs/chain_mamba_recipe_seed0" \
    --out outputs/exp03_chain_mamba/comparison

Writes comparison.csv, comparison.md and three figures:
  error_vs_step.png    chained-prediction error at every step of the 50-step horizon (log scale)
  example_window.png   one test window: true states and each model's 50-step chained prediction
  free_running.png     zero input for 1,000 steps from (2, 0): the limit cycle of each model against the truth
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.evaluation.diagnostics import limit_cycle  # noqa: E402
from src.evaluation.metrics import rollout_errors, teacher_forced_one_step  # noqa: E402
from src.models.registry import build_model, count_parameters  # noqa: E402
from src.training.trainer import attach_data_statistics  # noqa: E402
from src.utils.config import load_yaml_with_inherit  # noqa: E402

# Reference categorical palette, fixed order (blue, orange, aqua, yellow, magenta); truth in dark ink
COLOURS = ["#eb6834", "#1baf7a", "#2a78d6", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED = "#0b0b0b", "#52514e"


def load(path: Path, data):
    cfg = load_yaml_with_inherit(path / "config.yaml")
    model = build_model(cfg)
    attach_data_statistics(model, data)
    ck = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description="Compare trained runs on the same test windows.")
    p.add_argument("--runs", nargs="+", required=True, help='"label=run folder" pairs')
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=300, help="test window shown in example_window.png")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.runs]
    first_cfg = load_yaml_with_inherit(runs[0][1] / "config.yaml")
    data = build_data(first_cfg)
    C, H = data["context_len"], data["horizon"]
    test, norm, system = data["splits"]["test"], data["norm"], data["system"]
    has_cycle = getattr(system, "n_x", 2) == 2          # the free-running limit-cycle check is for Van der Pol
    names = first_cfg["system"].get("state_names", [f"x{i + 1}" for i in range(system.n_x)])

    rows, curves, examples, cycles = [], {}, {}, {}
    w = args.window
    xw, uw = test.xw[w:w + 1], test.uw[w:w + 1]
    for label, path in runs:
        cfg, model = load(path, data)
        r = rollout_errors(model, "chained", test, norm, C, H, device)
        lc = (limit_cycle(model, "chained", norm, system, C, 1000, [2.0, 0.0], device) if has_cycle
              else {"status": "not applicable"})
        with torch.no_grad():
            pred, _ = model.chained(xw[:, :C + 1], uw, H)
        examples[label] = norm.x_inverse(pred[0].numpy())
        curves[label] = np.array(r["per_step_rmse"])
        cycles[label] = lc
        rows.append({"model": label, "parameters": count_parameters(model),
                     "one_step_rmse": r["one_step_rmse"], "error_at_10": r["error_at_10"],
                     "error_at_50": r["error_at_50"], "rollout_rmse_50": r["rollout_rmse"],
                     "growth_1_to_50": r["growth_1_to_end"], "diverged": r["diverged"],
                     "teacher_forced_one_step_rmse": teacher_forced_one_step(model, test, norm, device),
                     "limit_cycle_amplitude_error": lc.get("amplitude_error"),
                     "limit_cycle_period_error": lc.get("period_error"),
                     "limit_cycle_status": lc.get("status")})

    with open(out / "comparison.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    base = rows[0]["rollout_rmse_50"]
    lines = ["Chained prediction on the test windows (smaller is better; root-mean-square error, physical units)",
             "",
             f"{'model':44s}{'params':>8s}{'1-step':>9s}{'10-step':>9s}{'50-step':>9s}{'avg 1-50':>10s}"
             f"{'vs first':>10s}{'cycle amp':>11s}{'cycle period':>14s}"]
    for r in rows:
        pct = 100.0 * (r["rollout_rmse_50"] - base) / base
        amp = r["limit_cycle_amplitude_error"]
        per = r["limit_cycle_period_error"]
        lines.append(f"{r['model'][:43]:44s}{r['parameters']:>8d}{r['one_step_rmse']:>9.4f}{r['error_at_10']:>9.4f}"
                     f"{r['error_at_50']:>9.4f}{r['rollout_rmse_50']:>10.4f}{pct:>+9.0f}%"
                     f"{(f'{100 * amp:.1f}%' if amp is not None else '-'):>11s}"
                     f"{(f'{100 * per:.1f}%' if per is not None else '-'):>14s}")
    lines += ["", "cycle amp / cycle period: free-running limit cycle (zero input) against the true one, relative error."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    colour = {label: COLOURS[i % len(COLOURS)] for i, (label, _) in enumerate(runs)}
    steps = np.arange(1, H + 1)

    # 1. error at every step
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for label, _ in runs:
        ax.plot(steps, curves[label], color=colour[label], linewidth=2, label=label)
    # end labels, 3 significant figures, pushed apart (log scale) so they never overlap
    ends = sorted(((curves[label][-1], label) for label, _ in runs))
    placed = []
    for value, label in ends:
        y = np.log10(value)
        if placed and y - placed[-1] < 0.07:
            y = placed[-1] + 0.07
        placed.append(y)
        ax.annotate(f"{value:.3g}", (H, value), xytext=(H + 1, 10 ** y), textcoords="data",
                    va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel(f"prediction step (time step {system.dt:g})", fontsize=9, color=INK)
    ax.set_ylabel("root-mean-square error (log scale)", fontsize=9, color=INK)
    ax.set_title("Chained prediction error over the horizon, test windows", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.set_xlim(1, H + 6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "error_vs_step.png")
    plt.close(fig)

    # 2. one example window
    true = norm.x_inverse(xw[0, C + 1:C + 1 + H].numpy())
    ctx = norm.x_inverse(xw[0, :C + 1].numpy())
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 2.5 * len(names)), dpi=150, sharex=True)
    for k, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(np.arange(-C, 1), ctx[:, k], color=MUTED, linewidth=1.5, label="true context (given)")
        ax.plot(steps, true[:, k], color=INK, linewidth=2, linestyle="--", label="true")
        for label, _ in runs:
            ax.plot(steps, examples[label][:, k], color=colour[label], linewidth=2, label=label)
        ax.set_ylabel(name, fontsize=9, color=INK)
        style(ax)
    axes[0].set_title(f"Test window {w}: 50-step chained prediction from the true state at step 0",
                      fontsize=10, color=INK, loc="left")
    axes[-1].set_xlabel("step", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out / "example_window.png")
    plt.close(fig)

    if not has_cycle:
        print(f"\nFigures and tables in {out}")
        return

    # 3. free running, zero input
    x = np.empty((1001, 2))
    x[0] = [2.0, 0.0]
    for k in range(1000):
        x[k + 1] = system.step(x[k], 0.0)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 3.2), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, path) in zip(axes, runs):
        _, model = load(path, data)
        traj = free_run(model, data, 1000)
        ax.plot(x[500:, 0], x[500:, 1], color=INK, linewidth=1.5, linestyle="--", label="true")
        if traj is not None:
            ax.plot(traj[500:, 0], traj[500:, 1], color=colour[label], linewidth=1.6, label="model")
        lc = cycles[label]
        note = lc.get("status", "")
        if lc.get("amplitude_error") is not None:
            note = f"amplitude {100 * lc['amplitude_error']:.1f}%, period {100 * (lc.get('period_error') or 0):.1f}%"
        ax.set_title(f"{label}\n{note}", fontsize=8, color=INK)
        ax.set_xlabel("x1", fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("x2", fontsize=8, color=INK)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3.5, 3.5)
    fig.suptitle("Free running with zero input, steps 500-1000 (dashed = true limit cycle)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "free_running.png")
    plt.close(fig)
    print(f"\nFigures and tables in {out}")


@torch.no_grad()
def free_run(model, data, steps: int):
    """Same protocol as src/evaluation/diagnostics.limit_cycle: true context, then zero input, chained."""
    norm, system, C = data["norm"], data["system"], data["context_len"]
    x = np.empty((C + 1, 2))
    x[0] = [2.0, 0.0]
    for k in range(C):
        x[k + 1] = system.step(x[k], 0.0)
    x_ctx = torch.tensor(norm.x(x), dtype=torch.float32)[None]
    u = torch.tensor(norm.u(np.zeros((C + steps, 1))), dtype=torch.float32)[None]
    pred, hit = model.chained(x_ctx, u, steps)
    traj = np.concatenate([x, norm.x_inverse(pred[0].numpy())], 0)[:steps + 1]
    return None if bool(hit.any()) else traj


if __name__ == "__main__":
    main()
