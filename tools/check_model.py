"""Architecture checks before any training (CLAUDE.md Section 8). Fails loudly (exit code 1).

For every arm of the experiment:
  1. parameter count, next to Cevaal et al.'s reference (Mamba 3,418; Long Short-Term Memory 3,052);
  2. step-by-step and parallel forward passes agree within 1e-4 (and the parallel scan matches a plain loop);
  3. every parameter receives a non-zero, finite gradient from the arm's own training loss on real data;
  4. certificate status at initialisation (all decay values negative?);
  5. components listed under an arm's `new_components` are not exactly zero at the first time step.

On screen: one progress bar, the parameter count of each model type, and the verdict.
Every number of every check goes to outputs/<experiment>/check_log.txt.

python tools/check_model.py --config configs/experiments/exp01_certificate_ablation.yaml
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
from src.losses.regimes import regime_loss  # noqa: E402
from src.models.registry import REFERENCE_PARAMETERS, build_model, count_parameters  # noqa: E402
from src.utils.config import deep_merge, load_experiment, output_dir, resolve_arm  # noqa: E402
from src.utils.console import ConsoleLog, progress_bar  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

TOL = 1e-4


def rel_diff(a, b) -> float:
    """Largest difference between two outputs, relative to the size of b."""
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        same_pattern = torch.equal(torch.isfinite(a), torch.isfinite(b))
        if not same_pattern:
            return float("inf")
        m = torch.isfinite(a)
        a, b = a[m], b[m]
        if a.numel() == 0:
            return 0.0
    return float((a - b).abs().max() / max(1.0, float(b.abs().max())))


def check_arm(arm, cfg, data, log) -> tuple[list[str], dict]:
    """Run the five checks for one arm.

    Returns
      failures : list of short failure messages (empty = all passed)
      info     : facts shown on screen (parameter count, expected non-finite loss)
    """
    failures = []
    name = arm["name"]
    expected_diverge = arm.get("expected") == "diverges"
    info = {"model": cfg["model"]["name"], "expected_nonfinite": False}

    set_seed(0)
    model = build_model(cfg).double()     # double precision on the processor for the equivalence check
    core = model.core

    # ---- 1. parameter count ----
    n_par = count_parameters(model)
    info["parameters"] = n_par
    log.detail("")
    log.detail(f"[{name}]  model {cfg['model']['name']}, regime {cfg['training']['regime']}, "
               f"decay {cfg['model'].get('decay_mode', '—')}")
    log.detail(f"  1. parameters: {n_par:,}")

    # ---- 2. the three ways of running the model must give the same output ----
    if cfg["model"].get("decay_mode") == "unstable_init":
        T = 8      # short sequence: an unstable model overflows on long ones
    else:
        T = data["context_len"] + data["horizon"]
    g = torch.Generator().manual_seed(1)
    inp = torch.randn(4, T, model.in_dim(model.n_x, model.n_u), generator=g, dtype=torch.float64)
    with torch.no_grad():
        # (a) parallel scan, (b) plain loop, (c) one step at a time
        y_par, _ = core.sequence(inp, parallel=True)
        y_seq, _ = core.sequence(inp, parallel=False)
        state = None
        ys = []
        for t in range(T):
            y, state = core.step(inp[:, t], state)
            ys.append(y)
        y_step = torch.stack(ys, dim=1)
        # (d) run the first half, then continue from the carried memory
        k = T // 2
        _, st = core.sequence(inp[:, :k], parallel=True)
        y_cont, _ = core.sequence(inp[:, k:], st, parallel=True)
    d1 = rel_diff(y_par, y_seq)
    d2 = rel_diff(y_par, y_step)
    d3 = rel_diff(y_par[:, k:], y_cont)
    same_output = d1 <= TOL and d2 <= TOL and d3 <= TOL
    if same_output:
        verdict = "pass"
    else:
        verdict = "FAIL"
    log.detail(f"  2. parallel vs loop {d1:.2e} | parallel vs step-by-step {d2:.2e} | "
               f"split-and-continue {d3:.2e}  (T = {T})  {verdict}")
    if not same_output:
        failures.append(f"{name}: the parallel and step-by-step outputs disagree (more than {TOL})")

    # ---- 3. every parameter must receive gradient ----
    set_seed(0)
    model32 = build_model(cfg)
    model32.train()
    split = data["splits"]["train"]
    x = split.xw[:16]
    u = split.uw[:16]
    loss, _ = regime_loss(model32, x, u, cfg["training"]["regime"], data["context_len"], data["horizon"])
    if not torch.isfinite(loss):
        if expected_diverge:
            info["expected_nonfinite"] = True
            log.detail(f"  3. loss at initialisation is not finite ({float(loss)}) — expected for an unstable start")
        else:
            log.detail(f"  3. loss at initialisation is not finite ({float(loss)})  FAIL")
            failures.append(f"{name}: loss is not finite at the start")
    else:
        loss.backward()
        norms = {}
        for param_name, param in model32.named_parameters():
            if param.grad is None:
                norms[param_name] = 0.0
            else:
                norms[param_name] = float(param.grad.norm())
        dead = []
        bad = []
        for param_name, value in norms.items():
            if value == 0.0:
                dead.append(param_name)
            if not torch.isfinite(torch.tensor(value)):
                bad.append(param_name)
        smallest = sorted(norms.items(), key=lambda kv: kv[1])[:3]
        smallest_text = ", ".join(f"{n} {v:.2e}" for n, v in smallest)
        log.detail(f"  3. loss {float(loss):.3e}; gradient norms min {min(norms.values()):.2e}, "
                   f"max {max(norms.values()):.2e}; smallest: {smallest_text}")
        if not expected_diverge:
            if dead:
                failures.append(f"{name}: zero gradient for {', '.join(dead)}")
            if bad:
                failures.append(f"{name}: non-finite gradient for {', '.join(bad)}")

    # ---- 4. is the stability guarantee in place at the start? ----
    mods = model32.decay_modules()
    if mods:
        largest_decay = max(float(m.A().max()) for m in mods)
        holds = largest_decay < 0
        log.detail(f"  4. largest decay value {largest_decay:+.3g} → certificate "
                   f"{'holds' if holds else 'does NOT hold'} at start")
        mode = cfg["model"].get("decay_mode")
        if mode in ("certified", "free_sign") and not holds:
            failures.append(f"{name}: decay values should start negative")
        if mode == "unstable_init" and holds:
            failures.append(f"{name}: an unstable start should have positive decay values")
    else:
        log.detail("  4. no decay parameter (no certificate in this model)")

    # ---- 5. new components must not be exactly zero at the first step ----
    new = arm.get("new_components") or []
    if not new:
        log.detail("  5. no new components declared — not applicable")
    else:
        acts = {}
        modules = dict(model32.named_modules())
        hooks = []
        for comp in new:
            hook = modules[comp].register_forward_hook(
                lambda m, i, o, n=comp: acts.setdefault(n, o if torch.is_tensor(o) else o[0]))
            hooks.append(hook)
        with torch.no_grad():
            model32.teacher_forced(x[:, :1], u[:, :1])
        for hook in hooks:
            hook.remove()
        for comp in new:
            zero = comp in acts and float(acts[comp].abs().max()) == 0.0
            if zero:
                log.detail(f"  5. {comp}: exactly ZERO at the first step  FAIL")
                failures.append(f"{name}: new component {comp} is zero at the first step")
            else:
                log.detail(f"  5. {comp}: non-zero at the first step  pass")

    if failures:
        log.detail("  → FAILED")
    else:
        log.detail("  → all checks passed")
    return failures, info


def parameter_line(model_name: str, n_parameters: int) -> str:
    """e.g. '  lstm          3,152 parameters  (Cevaal et al.: 3,052, +3.3%)'."""
    reference = REFERENCE_PARAMETERS.get(model_name)
    if reference:
        percent = 100.0 * (n_parameters - reference) / reference
        reference_text = f"(Cevaal et al.: {reference:,}, {percent:+.1f}%)"
    else:
        reference_text = "(no published reference)"
    return f"  {model_name:13s}{n_parameters:>7,d} parameters  {reference_text}"


def main():
    p = argparse.ArgumentParser(description="Architecture checks before training.")
    p.add_argument("--config", required=True)
    p.add_argument("--arms", nargs="+")
    args = p.parse_args()

    # Step 1: read the experiment and pick the arms
    exp = load_experiment(args.config)
    arms = []
    for arm in exp["arms"]:
        if not args.arms or arm["name"] in args.arms:
            arms.append(arm)

    # Step 2: log file (overwritten every time, so it always holds the latest check)
    out = output_dir(exp, quick=False)
    log = ConsoleLog(out / "check_log.txt", fresh=True)
    log.capture_warnings()
    log.say(f"Architecture checks — {exp['name']} ({len(arms)} arms)")

    # Step 3: a small amount of data is enough for the checks
    first = resolve_arm(exp, arms[0])
    small = deep_merge(first, {"data": {"max_train_windows": 64, "max_val_windows": 16},
                               "evaluation": {"max_eval_windows": 16}})
    data = build_data(small)

    # Step 4: check every arm (one progress bar)
    failures = []
    parameters_by_model = {}
    expected_nonfinite_arms = []
    bar = progress_bar(total=len(arms), description="checking", unit="arm")
    for arm in arms:
        arm_failures, info = check_arm(arm, resolve_arm(exp, arm), data, log)
        failures = failures + arm_failures
        parameters_by_model[info["model"]] = info["parameters"]
        if info["expected_nonfinite"]:
            expected_nonfinite_arms.append(arm["name"])
        bar.update(1)
    bar.close()

    # Step 5: short report on screen
    for model_name, n_parameters in parameters_by_model.items():
        log.say(parameter_line(model_name, n_parameters))
    if expected_nonfinite_arms:
        log.say(f"  note: {len(expected_nonfinite_arms)} unstable-start arm(s) have a non-finite loss "
                f"at the start — expected, not a failure")

    if failures:
        log.say("CHECKS FAILED — do not train until these are fixed:")
        for message in failures:
            log.say(f"  ✗ {message}")
        log.say(f"Full details: {log.path}")
        log.close()
        sys.exit(1)

    log.say(f"All checks passed for {len(arms)} arm(s).   Full details: {log.path}")
    log.close()


if __name__ == "__main__":
    main()
