"""Plain-language result summary (CLAUDE.md Section 10)."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.utils.stats import bootstrap_median_ci, bootstrap_ratio_ci, compare

GROUP_TITLES = {
    "teacher_forcing": "Trained with the true previous state (teacher forcing); tested on chained prediction",
    "chained": "Trained on its own chained predictions; tested on chained prediction",
    "direct": "Direct multi-step prediction (whole horizon at once, nothing fed back)",
}
DECAY_LABELS = {"certified": "Guarantee on", "free_sign": "Guarantee removed", "unstable_init": "Started unstable"}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_results(results_csv: Path) -> list[dict]:
    with open(results_csv) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["diverged"] = str(r["diverged"]).lower() == "true"
        for k in ("one_step_rmse", "rollout_rmse", "error_at_10", "error_at_50", "growth_1_to_50"):
            r[k] = _float(r.get(k))
    return rows


def arm_label(arm: dict, row: dict) -> str:
    if arm.get("label"):
        return arm["label"]
    if row["model"] == "lstm":
        return "Long Short-Term Memory network"
    base = DECAY_LABELS.get(row.get("decay_mode", ""), arm["name"])
    if row["model"] == "mamba_full":
        base += " (full Mamba)"
    return base


def word(pct: float) -> str:
    a = abs(pct)
    if a <= 10:
        return "about the same"
    side = "worse" if pct > 0 else "better"
    return f"slightly {side}" if a <= 20 else f"clearly {side} (on this seed)"


def _fmt(v):
    return "—" if v is None or not np.isfinite(v) else f"{v:.3g}"


def build_summary(exp_cfg: dict, rows: list[dict], quick: bool = False) -> tuple[str, dict]:
    arms = OrderedDict((a["name"], a) for a in exp_cfg["arms"])
    rule = exp_cfg["decision_rule"]
    metric = rule.get("metric", "rollout_rmse")
    thr = float(rule.get("threshold_percent", 20))
    seeds = sorted({r["seed"] for r in rows})
    n_seeds = len(seeds)
    stage2 = n_seeds >= 8

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def agg(name):
        rs = by_arm.get(name, [])
        ok = [r for r in rs if not r["diverged"]]
        med = lambda k: float(np.median([r[k] for r in ok if r[k] is not None])) if ok else None  # noqa: E731
        return {"runs": len(rs), "diverged": sum(r["diverged"] for r in rs), "one_step": med("one_step_rmse"),
                "rollout": med(metric), "growth": med("growth_1_to_50"),
                "values": [r[metric] for r in ok], "by_seed": {r["seed"]: r[metric] for r in ok},
                "row": rs[0] if rs else None}

    seed_note = "1 seed: direction only" if n_seeds == 1 else (
        f"{n_seeds} seeds" + ("" if stage2 else ": direction only (fewer than 8 seeds)"))
    lines = []
    if quick:
        lines.append("CODE CHECK ONLY — DO NOT INTERPRET (--quick run: tiny data, 2 epochs)")
        lines.append("")
    lines.append(f"RESULT — {exp_cfg['name']}   ({seed_note})")
    lines.append("")
    lines.append(f"Question: {exp_cfg.get('question', exp_cfg['hypothesis'].strip())}")
    lines.append("Smaller error is better. Errors are root-mean-square, in the system's own units.")

    groups = OrderedDict()
    for name, arm in arms.items():
        groups.setdefault(arm.get("group", "all"), []).append(name)

    decision_hits, agreements, stage2_lines = [], [], []
    machine = {"seeds": seeds, "groups": {}}
    for group, names in groups.items():
        base_name = next((n for n in names if arms[n].get("role") == "baseline"), names[0])
        base = agg(base_name)
        lines.append("")
        lines.append(GROUP_TITLES.get(group, group))
        lines.append(f"  {'':34s}{'Next-step error':>17s}{'50-step error':>16s}{'Blew up?':>12s}")
        bullets = []
        ordered = [base_name] + [n for n in names if n != base_name]
        machine["groups"][group] = {}
        for name in ordered:
            a = agg(name)
            if a["row"] is None:
                lines.append(f"  {arms[name]['name']:34s}{'not run yet':>17s}")
                continue
            label = arm_label(arms[name], a["row"])
            role = arms[name].get("role", "")
            if name == base_name:
                label += " (baseline)"
            elif role == "reference":
                label += " (reference)"
            blew = ("yes" if a["diverged"] else "no") if a["runs"] == 1 else f"{a['diverged']} of {a['runs']}"
            all_div = a["diverged"] == a["runs"]
            lines.append(f"  {label[:34]:34s}{_fmt(None if all_div else a['one_step']):>17s}"
                         f"{_fmt(None if all_div else a['rollout']):>16s}{blew:>12s}")
            entry = {"label": label, "role": role, "diverged": a["diverged"], "runs": a["runs"],
                     "one_step": a["one_step"], "rollout": a["rollout"]}
            machine["groups"][group][name] = entry
            if name == base_name:
                continue
            if all_div:
                bullets.append(f"{label} blew up in {a['diverged']} of {a['runs']} run(s); no error is reported for it.")
                pct = None
            elif base["rollout"] and a["rollout"] is not None:
                pct = 100.0 * (a["rollout"] - base["rollout"]) / base["rollout"]
                one = 100.0 * (a["one_step"] - base["one_step"]) / base["one_step"] if base["one_step"] else None
                extra = f" (next-step error {one:+.0f}%)" if one is not None else ""
                bullets.append(f"{label}: 50-step error {pct:+.0f}% against the baseline → {word(pct)}{extra}.")
                if a["diverged"]:
                    bullets.append(f"  ({a['diverged']} of {a['runs']} runs of it blew up and are left out.)")
                if role == "ablation" and abs(pct) > thr:
                    decision_hits.append(f"{label} in '{group}' ({pct:+.0f}%)")
                if role == "proposed":
                    ok = (pct <= -thr and (one is None or one <= float(rule.get("one_step_tolerance_percent", 10)))
                          and a["diverged"] == 0)
                    if ok:
                        decision_hits.append(f"{label} in '{group}' ({pct:+.0f}%)")
                entry["percent_vs_baseline"] = pct
            else:
                pct = None
            exp = arms[name].get("expected")
            if exp:
                if exp == "same":
                    agree = (not all_div) and pct is not None and abs(pct) <= thr
                elif exp == "diverges":
                    agree = all_div
                elif exp == "better":
                    agree = pct is not None and pct < -thr
                elif exp == "worse":
                    agree = pct is not None and pct > thr
                else:
                    agree = None
                agreements.append((label, group, exp, agree))
            if stage2 and not all_div and base["values"] and a["values"]:
                c = compare(a["values"], base["values"])
                ratio, lo, hi = bootstrap_ratio_ci(a["values"], base["values"])
                common = set(a["by_seed"]) & set(base["by_seed"])
                better = sum(a["by_seed"][s] < base["by_seed"][s] for s in common)
                verdict = "the difference is reliable" if c["significant"] else "the difference is not reliable"
                margin = float(rule.get("equivalence_margin_percent", 10)) / 100.0
                equiv = lo >= 1 - margin and hi <= 1 + margin
                stage2_lines.append(
                    f"{label} ({group}): lower error than the baseline in {better} of {len(common)} seeds; {verdict}. "
                    f"Error ratio {ratio:.2f} (95% interval {lo:.2f} to {hi:.2f})"
                    + (f"; within ±{margin*100:.0f}% → equivalent." if equiv else "."))
                entry["stage2"] = {**c, "ratio": ratio, "ratio_ci": [lo, hi], "equivalent": equiv,
                                   "better_in": better, "paired_seeds": len(common)}
        if base["growth"] and base["row"] is not None and base["diverged"] < base["runs"]:
            bullets.append(f"Baseline errors grow about {base['growth']:.0f}× between 1 step and 50 steps.")
        if base["row"] is not None and base["rollout"] is not None:
            lo_, hi_ = (None, None)
            if stage2 and base["values"]:
                _, lo_, hi_ = bootstrap_median_ci(base["values"])
                bullets.append(f"Baseline 50-step error {base['rollout']:.3g} (95% interval {lo_:.3g} to {hi_:.3g}).")
        if bullets:
            lines.append("  What this means:")
            lines += [f"    • {b}" for b in bullets]

    if stage2_lines:
        lines.append("")
        lines.append("Across seeds:")
        lines += [f"  • {s}" for s in stage2_lines]

    lines.append("")
    ms = exp_cfg.get("math_signal", {})
    lines.append(f"Math predicted: {str(ms.get('prediction', '(not stated)')).strip()}")
    if agreements:
        bad = [a for a in agreements if a[3] is False]
        unknown = [a for a in agreements if a[3] is None]
        if not bad and not unknown:
            lines.append("Result agreed with the prediction for every arm.")
        else:
            for label, group, exp, agree in bad:
                lines.append(f"Result DISAGREED for {label} in '{group}' (predicted: {exp}).")
            for label, group, exp, agree in unknown:
                lines.append(f"Could not check the prediction for {label} in '{group}'.")
    lines.append("")
    if rule.get("type") == "ablation":
        if decision_hits:
            decision = "WORTH CONFIRMING (" + "; ".join(decision_hits) + f" differ by more than {thr:.0f}%)"
        else:
            decision = f"NOT worth confirming (no arm that trained differed by more than {thr:.0f}%)"
        decision += ". Blown-up arms are reported as counts, not errors."
    else:
        decision = ("WORTH CONFIRMING (" + "; ".join(decision_hits) + ")") if decision_hits else \
            "NOT worth confirming (no proposed arm met the declared rule)"
    if stage2:
        decision = "Stage 2 complete — see 'Across seeds' above. Stage 1 rule: " + decision
    lines.append(f"Decision: {decision}")
    machine["decision"] = decision
    machine["agreements"] = [{"arm": a[0], "group": a[1], "expected": a[2], "agreed": a[3]} for a in agreements]
    return "\n".join(lines) + "\n", machine
