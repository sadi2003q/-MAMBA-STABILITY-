"""Training losses for the three regimes. All in normalised units, mean squared error."""

from __future__ import annotations

import torch

from src.models.dynamics import DynamicsModel


def regime_loss(model: DynamicsModel, x, u, regime: str, context_len: int, horizon: int):
    """x [B, C+H+1, n_x], u [B, C+H, n_u]. Returns (loss, number of rows that hit the clamp)."""
    if regime == "teacher_forcing":
        pred = model.teacher_forced(x[:, :-1], u)
        pred, bad = model._safe(pred)
        return torch.mean((pred - x[:, 1:]) ** 2), int(bad.any(dim=1).sum())
    eval_regime = "direct" if regime == "direct" else "chained"
    preds, target, hit = model.rollout(eval_regime, x, u, context_len, horizon)
    return torch.mean((preds - target) ** 2), int(hit.sum())
