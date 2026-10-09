"""Architecture checks before any training (CLAUDE.md Section 8). Fails loudly (exit code 1).

For every arm of the experiment:
  1. parameter count, next to Cevaal et al.'s reference (Mamba 3,418; Long Short-Term Memory 3,052);
  2. step-by-step and parallel forward passes agree within 1e-4 (and the parallel scan matches a plain loop);
  3. every parameter receives a non-zero, finite gradient from the arm's own training loss on real data;
  4. certificate status at initialisation (all decay values negative?);
  5. components listed under an arm's `new_components` are not exactly zero at the first time step.

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
from src.utils.config import deep_merge, load_experiment, resolve_arm  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

TOL = 1e-4


def rel_diff(a, b) -> float:
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        same_pattern = torch.equal(torch.isfinite(a), torch.isfinite(b))
        if not same_pattern:
            return float("inf")
        m = torch.isfinite(a)
        a, b = a[m], b[m]
        if a.numel() == 0:
            return 0.0
    return float((a - b).abs().max() / max(1.0, float(b.abs().max())))


def check_arm(arm, cfg, data) -> list[str]:
    failures = []
    name = arm["name"]
    expected_diverge = arm.get("expected") == "diverges"
    set_seed(0)
    model = build_model(cfg).double()     # double precision on the processor for the equivalence check
    core = model.core
    n_par = count_parameters(model)
    ref = REFERENCE_PARAMETERS.get(cfg["model"]["name"])
    ref_s = f"reference {ref:,} ({100 * (n_par - ref) / ref:+.1f}%)" if ref else "no published reference"
    print(f"\n[{name}]  model {cfg['model']['name']}, regime {cfg['training']['regime']}, "
          f"decay {cfg['model'].get('decay_mode', '—')}")
    print(f"  1. parameters: {n_par:,}   {ref_s}")

    # 2. equivalence
    T = 8 if cfg["model"].get("decay_mode") == "unstable_init" else data["context_len"] + data["horizon"]
    g = torch.Generator().manual_seed(1)
    inp = torch.randn(4, T, model.in_dim(model.n_x, model.n_u), generator=g, dtype=torch.float64)
    with torch.no_grad():
        y_par, _ = core.sequence(inp, parallel=True)
        y_seq, _ = core.sequence(inp, parallel=False)
        state, ys = None, []
        for t in range(T):
            y, state = core.step(inp[:, t], state)
            ys.append(y)
        y_step = torch.stack(ys, dim=1)
        # continuing from a carried state must match one long pass (tests the warm-up hand-over)
        k = T // 2
        _, st = core.sequence(inp[:, :k], parallel=True)
        y_cont, _ = core.sequence(inp[:, k:], st, parallel=True)
    d1, d2, d3 = rel_diff(y_par, y_seq), rel_diff(y_par, y_step), rel_diff(y_par[:, k:], y_cont)
    ok = d1 <= TOL and d2 <= TOL and d3 <= TOL
    print(f"  2. parallel vs loop {d1:.2e} | parallel vs step-by-step {d2:.2e} | split-and-continue {d3:.2e}"
          f"  (T = {T})  {'pass' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name}: forward passes disagree (> {TOL})")

    # 3. gradient reach, float32, real windows, the arm's own loss
    set_seed(0)
    model32 = build_model(cfg)
    model32.train()
    split = data["splits"]["train"]
    x, u = split.xw[:16], split.uw[:16]
    loss, _ = regime_loss(model32, x, u, cfg["training"]["regime"], data["context_len"], data["horizon"])
    if not torch.isfinite(loss):
        msg = f"  3. loss at initialisation is not finite ({float(loss)})"
        if expected_diverge:
            print(msg + " — expected for an unstable start; recorded, not a failure")
        else:
            print(msg + "  FAIL")
            failures.append(f"{name}: non-finite loss at initialisation")
    else:
        loss.backward()
        norms = {n: (float(p.grad.norm()) if p.grad is not None else 0.0) for n, p in model32.named_parameters()}
        dead = [n for n, v in norms.items() if v == 0.0]
        bad = [n for n, v in norms.items() if not torch.isfinite(torch.tensor(v))]
        smallest = sorted(norms.items(), key=lambda kv: kv[1])[:3]
        healthy = not dead and not bad
        verdict = "pass" if healthy else ("expected for an unstable start; recorded, not a failure"
                                          if expected_diverge else "FAIL")
        print(f"  3. loss {float(loss):.3e}; gradient norms: min {min(norms.values()):.2e}, "
              f"max {max(norms.values()):.2e}; smallest: " + ", ".join(f"{n} {v:.2e}" for n, v in smallest)
              + f"  {verdict}")
        if not expected_diverge:
            if dead:
                failures.append(f"{name}: zero gradient for {', '.join(dead)}")
            if bad:
                failures.append(f"{name}: non-finite gradient for {', '.join(bad)}")

    # 4. certificate status
    mods = model32.decay_modules()
    if mods:
        mx = max(float(m.A().max()) for m in mods)
        holds = mx < 0
        mode = cfg["model"].get("decay_mode")
        print(f"  4. largest decay value {mx:+.3g} → certificate {'holds' if holds else 'does NOT hold'} at start "
              f"(decay factor exp(step * decay) is {'below' if holds else 'not below'} 1)")
        if mode in ("certified", "free_sign") and not holds:
            failures.append(f"{name}: decay should start negative")
        if mode == "unstable_init" and holds:
            failures.append(f"{name}: unstable start should have positive decay values")
    else:
        print("  4. no decay parameter (no certificate in this model)")

    # 5. new components
    new = arm.get("new_components") or []
    if not new:
        print("  5. no new components declared — not applicable")
    else:
        acts = {}
        hooks = [dict(model32.named_modules())[n].register_forward_hook(
            lambda m, i, o, n=n: acts.setdefault(n, o if torch.is_tensor(o) else o[0])) for n in new]
        with torch.no_grad():
            model32.teacher_forced(x[:, :1], u[:, :1])
        for h in hooks:
            h.remove()
        for n in new:
            zero = n in acts and float(acts[n].abs().max()) == 0.0
            print(f"  5. {n}: {'exactly ZERO at the first step  FAIL' if zero else 'non-zero at the first step  pass'}")
            if zero:
                failures.append(f"{name}: new component {n} is zero at the first step")
    return failures


def main():
    p = argparse.ArgumentParser(description="Architecture checks before training.")
    p.add_argument("--config", required=True)
    p.add_argument("--arms", nargs="+")
    args = p.parse_args()
    exp = load_experiment(args.config)
    arms = [a for a in exp["arms"] if not args.arms or a["name"] in args.arms]
    # small data are enough for the checks
    first = resolve_arm(exp, arms[0])
    data = build_data(deep_merge(first, {"data": {"max_train_windows": 64, "max_val_windows": 16},
                                         "evaluation": {"max_eval_windows": 16}}))
    failures = []
    for arm in arms:
        failures += check_arm(arm, resolve_arm(exp, arm), data)
    print("\n" + "=" * 80)
    if failures:
        print("CHECKS FAILED — do not train until these are fixed:")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    print(f"All checks passed for {len(arms)} arm(s).")


if __name__ == "__main__":
    main()
