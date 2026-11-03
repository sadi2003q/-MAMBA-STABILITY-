"""Training losses for the three regimes. All in normalised units, mean squared error."""

from __future__ import annotations

import torch

from src.models.dynamics import DynamicsModel


def regime_loss(model: DynamicsModel, x, u, regime: str, context_len: int, horizon: int,
                train_horizon: int | None = None, noise_std: float = 0.0, teacher_weight: float = 0.0):
    """x [B, C+H+1, n_x], u [B, C+H, n_u]. Returns (loss, number of rows that hit the clamp).

    Optional, chained regime only (all off by default, so earlier experiments are unchanged):
      train_horizon   roll out only this many steps (curriculum: grows during training); default = horizon
      noise_std       Gaussian noise added to fed-back states (normalised units)
      teacher_weight  adds teacher_weight * (one-step teacher-forced loss over the whole window)
    """
    if regime == "teacher_forcing":
        pred = model.teacher_forced(x[:, :-1], u)
        pred, bad = model._safe(pred)
        return torch.mean((pred - x[:, 1:]) ** 2), int(bad.any(dim=1).sum())
    if regime == "direct":
        preds, target, hit = model.rollout("direct", x, u, context_len, horizon)
        return torch.mean((preds - target) ** 2), int(hit.sum())
    H = horizon if train_horizon is None else max(1, min(int(train_horizon), horizon))
    x_ctx = x[:, :context_len + 1]
    target = x[:, context_len + 1:context_len + 1 + H]
    preds, hit = model.chained(x_ctx, u, H, noise_std=noise_std)
    loss = torch.mean((preds - target) ** 2)
    if teacher_weight > 0:
        pred_tf, _ = model._safe(model.teacher_forced(x[:, :-1], u))
        loss = loss + teacher_weight * torch.mean((pred_tf - x[:, 1:]) ** 2)
    return loss, int(hit.sum())
