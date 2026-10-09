"""Diagnostics that go to metrics.json (not to the plain-language summary):

- certificate status: are all decay parameters still negative after training?
- error amplification: how much the model magnifies a small error in the starting state over the
  horizon, d x_hat(C+H) / d x(C), against the true system's own amplification along the same stretch
  (product of one-step Jacobians). Largest singular value of each, in physical units.
- limit cycle: free-running prediction with zero input; amplitude and period against the true system.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd.functional import jacobian


def certificate_status(model) -> dict:
    mods = model.decay_modules()
    if not mods:
        return {"applies": False, "certificate_holds_at_end": None, "fraction_decay_positive": None}
    stats = [m.stats() for m in mods]
    frac = float(np.mean([s["fraction_positive"] for s in stats]))
    return {"applies": True, "certificate_holds_at_end": bool(all(s["max"] < 0 for s in stats)),
            "fraction_decay_positive": frac, "max_decay": float(max(s["max"] for s in stats)),
            "per_module": stats}


def error_amplification(model, regime: str, split, norm, system, context_len: int, horizon: int,
                        n_windows: int, device) -> dict:
    """Largest singular value of d x_hat(C+H) / d x(C) for the model and for the true system."""
    was_training = model.training
    model.train()   # recurrent-network backward on graphics cards needs training mode; no dropout is used
    std = torch.as_tensor(norm.x_std, dtype=torch.float32, device=device)
    model_sv, true_sv, ratios = [], [], []
    n = min(n_windows, split.xw.shape[0])
    idx = np.linspace(0, split.xw.shape[0] - 1, n).astype(int)
    try:
        for i in idx:
            x = split.xw[i:i + 1].to(device)
            u = split.uw[i:i + 1].to(device)
            x_ctx = x[:, :context_len + 1]
            if regime == "direct":
                def f(start):
                    preds, _ = model.direct(x_ctx, u, horizon, start=start.unsqueeze(0))
                    return preds[0, -1]
            else:
                with torch.no_grad():
                    state = model.warmup(x_ctx, u)

                def f(start):
                    preds, _ = model.chained(x_ctx, u, horizon, state=state, start=start.unsqueeze(0))
                    return preds[0, -1]
            J = jacobian(f, x_ctx[0, context_len].detach().clone())            # normalised units
            J_phys = (std[:, None] * J / std[None, :]).detach().cpu().double().numpy()
            if not np.all(np.isfinite(J_phys)):
                continue
            s_model = float(np.linalg.svd(J_phys, compute_uv=False)[0])
            x_true = norm.x_inverse(x[0, context_len:context_len + horizon].detach().cpu().numpy())
            Phi = np.eye(2)
            for J_t in system.jacobian(x_true):
                Phi = J_t @ Phi
            s_true = float(np.linalg.svd(Phi, compute_uv=False)[0])
            model_sv.append(s_model)
            true_sv.append(s_true)
            ratios.append(s_model / s_true)
    finally:
        model.train(was_training)
    if not model_sv:
        return {"windows": 0}
    return {"windows": len(model_sv),
            "model_amplification_median": float(np.median(model_sv)),
            "true_amplification_median": float(np.median(true_sv)),
            "amplification_ratio_median": float(np.median(ratios)),
            "amplification_ratio_max": float(np.max(ratios)),
            "model_amplification": model_sv, "true_amplification": true_sv}


def _period(signal: np.ndarray, dt: float):
    s = signal - signal.mean()
    up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(up) < 2:
        return None
    t = up + (-s[up]) / (s[up + 1] - s[up])         # linear interpolation of the crossing
    return float(np.mean(np.diff(t)) * dt)


@torch.no_grad()
def limit_cycle(model, regime: str, norm, system, context_len: int, steps: int, start, device) -> dict:
    model.eval()
    u_phys = np.zeros((context_len + steps, 1))
    x_true = system.simulate(np.asarray(start, dtype=float), u_phys)
    x_ctx = torch.tensor(norm.x(x_true[:context_len + 1])[None], dtype=torch.float32, device=device)
    u = torch.tensor(norm.u(u_phys)[None], dtype=torch.float32, device=device)
    if regime == "direct":
        preds, hit = model.direct(x_ctx, u, steps)
    else:
        preds, hit = model.chained(x_ctx, u, steps)
    pred = norm.x_inverse(preds[0].cpu().numpy())
    true = x_true[context_len + 1:context_len + 1 + steps]
    half = steps // 2
    amp_m, amp_t = float(np.abs(pred[half:, 0]).max()), float(np.abs(true[half:, 0]).max())
    per_m, per_t = _period(pred[half:, 0], system.dt), _period(true[half:, 0], system.dt)
    if bool(hit.any()) or not np.all(np.isfinite(pred)):
        status = "diverged"
    elif amp_m < 0.5 * amp_t or per_m is None:
        status = "collapsed"
    else:
        status = "ok"
    return {"status": status, "amplitude_model": amp_m, "amplitude_true": amp_t,
            "amplitude_error": abs(amp_m - amp_t) / amp_t,
            "period_model": per_m, "period_true": per_t,
            "period_error": (abs(per_m - per_t) / per_t) if (per_m and per_t) else None}
