"""
Read the flight results, check the pass rule from drone/plan.md, write summary.md and a figure.

Run:  python drone/analyse_results.py --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.run_flights import CONTROLLER_NAMES, NICE_NAMES, make_test_flights

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = ["pid", "physics", "physics_corrected", "plain_corrected", "one_step_corrected", "chained"]
METHOD = "chained_corrected"


def read_results(out_dir):
    table = {}
    for name in CONTROLLER_NAMES:
        path = os.path.join(out_dir, f"results_{name}.csv")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                rows.append({
                    "flight": int(parts[1]),
                    "crashed": parts[2] == "1",
                    "rmse_before": float(parts[3]),
                    "rmse_after": float(parts[4]),
                    "largest_after": float(parts[5]),
                    "planning_ms": float(parts[6]),
                })
        table[name] = rows
    return table


def after_errors(rows):
    """After-payload errors per flight; a crashed flight counts as 1 m (the crash limit) for the paired test."""
    values = []
    for r in rows:
        if r["crashed"]:
            values.append(1.0)
        else:
            values.append(r["rmse_after"])
    return np.array(values)


def main(seed):
    out_dir = os.path.join(HERE, "outputs", f"seed{seed}")
    table = read_results(out_dir)

    # Step 1: a summary line per controller
    summary = {}
    for name, rows in table.items():
        ok = [r for r in rows if not r["crashed"]]
        summary[name] = {
            "crashes": len(rows) - len(ok),
            "median_before_cm": float(np.median([r["rmse_before"] for r in ok]) * 100) if ok else None,
            "median_after_cm": float(np.median([r["rmse_after"] for r in ok]) * 100) if ok else None,
            "median_largest_after_cm": float(np.median([r["largest_after"] for r in ok]) * 100) if ok else None,
            "median_after_all_flights_cm": float(np.median(after_errors(rows)) * 100),
            "planning_ms": float(np.median([r["planning_ms"] for r in rows])),
        }

    # Step 2: the pass rule (H1, H2, H3)
    decision = {}
    if METHOD in table:
        method_after = after_errors(table[METHOD])
        method_median = float(np.median(method_after))

        # H1: against the best baseline (lowest median, crashed flights counted as 1 m)
        available = [b for b in BASELINES if b in table]
        best = min(available, key=lambda b: np.median(after_errors(table[b])))
        best_after = after_errors(table[best])
        best_median = float(np.median(best_after))
        improvement = 1.0 - method_median / best_median
        test = wilcoxon(method_after, best_after, alternative="less")
        h1 = {
            "best_baseline": best,
            "method_median_cm": method_median * 100,
            "best_baseline_median_cm": best_median * 100,
            "improvement_percent": improvement * 100,
            "wilcoxon_p": float(test.pvalue),
            "zero_crashes": summary[METHOD]["crashes"] == 0,
            "planning_ms": summary[METHOD]["planning_ms"],
        }
        h1["pass"] = bool(h1["zero_crashes"] and improvement >= 0.20 and test.pvalue < 0.05
                          and h1["planning_ms"] < 20.0)
        decision["H1"] = h1

        # H2 and H3: at least 20% lower median than one controller
        for key, other in [("H2_recipe", "one_step_corrected"), ("H3_correction", "chained")]:
            if other in table:
                other_median = float(np.median(after_errors(table[other])))
                gain = 1.0 - method_median / other_median
                p = float(wilcoxon(method_after, after_errors(table[other]), alternative="less").pvalue)
                decision[key] = {"against": other, "improvement_percent": gain * 100, "wilcoxon_p": p,
                                 "pass": bool(gain >= 0.20)}

    # Step 2b: follow-up step 1b (slow correction filter), only if it was run
    BASELINES_1B = ["pid", "physics", "physics_slow", "plain_slow", "one_step_slow", "chained"]
    if "chained_slow" in table:
        method_after = after_errors(table["chained_slow"])
        available = [b for b in BASELINES_1B if b in table]
        best = min(available, key=lambda b: np.median(after_errors(table[b])))
        best_after = after_errors(table[best])
        improvement = 1.0 - np.median(method_after) / np.median(best_after)
        p = float(wilcoxon(method_after, best_after, alternative="less").pvalue)
        step_1b = {"best_baseline": best, "method_median_cm": float(np.median(method_after) * 100),
                   "best_baseline_median_cm": float(np.median(best_after) * 100),
                   "improvement_percent": float(improvement * 100), "wilcoxon_p": p,
                   "zero_crashes": summary["chained_slow"]["crashes"] == 0,
                   "planning_ms": summary["chained_slow"]["planning_ms"],
                   "baselines_flown": available}
        step_1b["pass"] = bool(step_1b["zero_crashes"] and improvement >= 0.20 and p < 0.05
                               and step_1b["planning_ms"] < 20.0)
        decision["step_1b"] = step_1b

    # Step 3: prediction check (if it was run)
    prediction_path = os.path.join(out_dir, "prediction_check.json")
    prediction = None
    if os.path.exists(prediction_path):
        with open(prediction_path) as f:
            prediction = json.load(f)

    # Step 4: write summary.md
    lines = ["# Drone payload test - results (seed 0, 20 flights, this machine's 2-core processor)", ""]
    lines.append("| Controller | Crashes | Error before payload (cm) | Error after payload (cm) |"
                 " Largest after (cm) | Planning per step (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for name in CONTROLLER_NAMES:
        if name not in summary:
            continue
        s = summary[name]

        def show(v):
            return "-" if v is None else f"{v:.2f}"
        lines.append(f"| {NICE_NAMES[name]} | {s['crashes']}/20 | {show(s['median_before_cm'])} |"
                     f" {show(s['median_after_cm'])} | {show(s['median_largest_after_cm'])} |"
                     f" {s['planning_ms']:.1f} |")
    lines.append("")
    lines.append("Errors are medians over flights of the root-mean-square position error. Crashed flights are"
                 " left out of these columns and counted as 1 m in the pass-rule test.")
    lines.append("")
    if decision:
        h1 = decision["H1"]
        lines.append("## Pass rule (written before the results, drone/plan.md)")
        lines.append(f"- H1: best baseline = {NICE_NAMES[h1['best_baseline']]} ({h1['best_baseline_median_cm']:.2f} cm)."
                     f" Your method {h1['method_median_cm']:.2f} cm -> {h1['improvement_percent']:.1f}% lower,"
                     f" Wilcoxon p = {h1['wilcoxon_p']:.4f}, zero crashes = {h1['zero_crashes']},"
                     f" planning {h1['planning_ms']:.1f} ms. **{'PASS' if h1['pass'] else 'NOT PASSED'}**")
        for key in ["H2_recipe", "H3_correction"]:
            if key in decision:
                d = decision[key]
                lines.append(f"- {key}: {d['improvement_percent']:.1f}% lower than {NICE_NAMES[d['against']]}"
                             f" (p = {d['wilcoxon_p']:.4f}). **{'PASS' if d['pass'] else 'NOT PASSED'}**")
        if "step_1b" in decision:
            d = decision["step_1b"]
            lines.append(f"- Step 1b (added after the first results): best baseline = {NICE_NAMES[d['best_baseline']]}"
                         f" ({d['best_baseline_median_cm']:.2f} cm). Your method with the slow correction"
                         f" {d['method_median_cm']:.2f} cm -> {d['improvement_percent']:.1f}% lower,"
                         f" p = {d['wilcoxon_p']:.4f}, zero crashes = {d['zero_crashes']},"
                         f" planning {d['planning_ms']:.1f} ms. **{'PASS' if d['pass'] else 'NOT PASSED'}**")
        lines.append("")
    if prediction:
        lines.append("## Prediction check (held-out normal-mass flights, median position error in cm)")
        lines.append("| Model | after 1 step | 10 steps | 25 steps | 50 steps | Parameters |")
        lines.append("|---|---|---|---|---|---|")
        for name, values in prediction.items():
            parameters = values.get("parameters", "-")
            lines.append(f"| {name} | {values['position_cm_step1']:.2f} | {values['position_cm_step10']:.2f} |"
                         f" {values['position_cm_step25']:.2f} | {values['position_cm_step50']:.2f} | {parameters} |")
    lines.append("")
    lines.append("One training seed only: this shows direction, not proof.")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "decision": decision}, f, indent=2)
    print("\n".join(lines))

    # Step 5: a figure: error over time for one flight, and after-payload errors for all flights
    make_figure(out_dir, table)


def make_figure(out_dir, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flights = make_test_flights()
    show = ["pid", "physics_corrected", "one_step_corrected", "chained", "chained_corrected", "oracle"]
    colours = {"pid": "#888888", "physics_corrected": "#e08a2c", "one_step_corrected": "#7a5cc7",
               "chained": "#9ec5e8", "chained_corrected": "#1f5fa8", "oracle": "#2e9e5b"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: error over time in flight 0
    ax = axes[0]
    for name in show:
        path = os.path.join(out_dir, f"errors_{name}.npy")
        if not os.path.exists(path):
            continue
        errors = np.load(path, allow_pickle=True)[0]
        t = np.arange(len(errors)) * 0.02
        ax.plot(t, np.array(errors) * 100, label=NICE_NAMES[name][2:], color=colours[name], linewidth=1.5)
    ax.axvline(flights[0]["payload_time"], color="black", linestyle="--", linewidth=1)
    ax.text(flights[0]["payload_time"] + 0.1, ax.get_ylim()[1] * 0.9, "payload attached", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (cm)")
    ax.set_title("Test flight 1: distance from the figure-eight path")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)

    # right: after-payload error of every flight, one column per controller
    ax = axes[1]
    names = [n for n in CONTROLLER_NAMES if n in table]
    for i, name in enumerate(names):
        values = [r["rmse_after"] * 100 for r in table[name] if not r["crashed"]]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=14, color=colours.get(name, "#444444"))
        if values:
            ax.hlines(np.median(values), i - 0.3, i + 0.3, color="black", linewidth=2)
        crashes = sum(r["crashed"] for r in table[name])
        if crashes:
            ax.text(i, 0.5, f"{crashes} crash", ha="center", fontsize=8, color="red")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NICE_NAMES[n].split(" ")[0] for n in names])
    ax.set_xlabel("controller number (see table)")
    ax.set_ylabel("error after payload (cm)")
    ax.set_title("All 20 flights (black line = median)")
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "flights_figure.png"), dpi=130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
