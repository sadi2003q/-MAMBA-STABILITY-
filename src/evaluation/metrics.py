"""Prediction-error metrics, in physical units (root-mean-square error over both state components)."""

from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def rollout_errors(model, regime: str, split, norm, context_len: int, horizon: int, device,
                   report_steps=(1, 10, 50), batch_size: int = 512) -> dict:
    """Chained or direct prediction over every window of a split."""
    model.eval()
    sq_sum = torch.zeros(horizon, dtype=torch.float64)
    count = 0
    hits = 0
    nonfinite = False
    for i in range(0, split.xw.shape[0], batch_size):
        x = split.xw[i:i + batch_size].to(device)
        u = split.uw[i:i + batch_size].to(device)
        preds, target, hit = model.rollout(regime, x, u, context_len, horizon)
        err = (norm.x_inverse(preds) - norm.x_inverse(target)).double().cpu()
        if not torch.isfinite(err).all():
            nonfinite = True
            err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
        sq_sum += (err ** 2).sum(dim=(0, 2))
        count += err.shape[0] * err.shape[2]
        hits += int(hit.sum())
    per_step = torch.sqrt(sq_sum / count).numpy()
    out = {
        "regime": regime,
        "per_step_rmse": per_step.tolist(),
        "one_step_rmse": float(per_step[0]),
        "rollout_rmse": float(math.sqrt(float(sq_sum.sum()) / (count * horizon))),
        "growth_1_to_end": float(per_step[-1] / per_step[0]) if per_step[0] > 0 else None,
        "windows_hit_clamp": hits,
        "windows": int(split.xw.shape[0]),
        "diverged": bool(hits > 0 or nonfinite),
    }
    for k in report_steps:
        if 1 <= k <= horizon:
            out[f"error_at_{k}"] = float(per_step[k - 1])
    return out


@torch.no_grad()
def teacher_forced_one_step(model, split, norm, device, batch_size: int = 512) -> float:
    """One-step error with true states fed at every position (all positions of every window)."""
    model.eval()
    sq, n = 0.0, 0
    for i in range(0, split.xw.shape[0], batch_size):
        x = split.xw[i:i + batch_size].to(device)
        u = split.uw[i:i + batch_size].to(device)
        pred = model.teacher_forced(x[:, :-1], u)
        err = (norm.x_inverse(pred) - norm.x_inverse(x[:, 1:])).double()
        err = torch.nan_to_num(err, nan=1e6, posinf=1e6, neginf=1e6)
        sq += float((err ** 2).sum())
        n += err.numel()
    return float(np.sqrt(sq / n))
