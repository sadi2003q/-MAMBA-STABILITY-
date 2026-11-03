"""Dynamics model = recurrent core + residual readout:  x_{t+1} = x_t + W_out y_t  (normalised units).

Input at every step is [x_t, u_t, mask_t]. mask_t = 0 when the state slot holds a state (true or
predicted), 1 when it is empty (future steps of direct multi-step prediction).

Three ways to use it, for a window with true states x_0 ... x_{C+H} and inputs u_0 ... u_{C+H-1}:
  teacher_forced : feed every true state, predict the next one (one step ahead everywhere)
  chained        : feed true x_0 ... x_C, then feed its own predictions for H steps
  direct         : feed true x_0 ... x_C, then only future inputs (state slot empty); the
                   predictions are accumulated from the readout and never fed back
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    """Optional read-out changes (both off by default, so earlier checkpoints load unchanged):

    linear_skip:            increment = head(y_t) + W_skip [x_t, u_t, mask_t]. A memory-free linear path from
                            the current (true or predicted) state and input to the increment: the "anchor".
                            For Van der Pol it can represent the linear part of the Euler step exactly, so the
                            network only has to learn the x1^2 x2 correction.
    increment_normalisation: the increment is multiplied by the per-component standard deviation of the true
                            one-step increments (normalised units, training data), so the network's raw output
                            has unit size (state-derivative normalisation, Beintema et al., 2023).
    """

    def __init__(self, core: nn.Module, n_x: int, n_u: int, clamp: float = 50.0,
                 linear_skip: bool = False, increment_normalisation: bool = False):
        super().__init__()
        self.core = core
        self.n_x, self.n_u = n_x, n_u
        self.head = nn.Linear(core.out_dim, n_x)
        self.clamp = float(clamp)
        self.linear_skip = bool(linear_skip)
        self.increment_normalisation = bool(increment_normalisation)
        if self.linear_skip:
            self.skip = nn.Linear(self.in_dim(n_x, n_u), n_x, bias=False)
            nn.init.zeros_(self.skip.weight)
        if self.increment_normalisation:
            self.register_buffer("increment_scale", torch.ones(n_x))

    @torch.no_grad()
    def set_increment_scale(self, scale) -> None:
        if self.increment_normalisation:
            self.increment_scale.copy_(torch.as_tensor(scale, dtype=self.increment_scale.dtype))

    def _increment(self, y, inp):
        """Increment x_{t+1} - x_t (normalised units) from core features y and the step input inp."""
        d = self.head(y)
        if self.linear_skip:
            d = d + self.skip(inp)
        if self.increment_normalisation:
            d = d * self.increment_scale
        return d

    @staticmethod
    def in_dim(n_x: int, n_u: int) -> int:
        return n_x + n_u + 1

    def _inp(self, x, u, mask: float):
        return torch.cat([x, u, torch.full_like(x[..., :1], mask)], dim=-1)

    def _safe(self, x):
        """Clamp predictions so a diverging model still yields a finite loss; report which rows hit it."""
        bad = (~torch.isfinite(x)).any(-1) | (x.abs() > self.clamp).any(-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=self.clamp, neginf=-self.clamp).clamp(-self.clamp, self.clamp)
        return x, bad

    # ------------------------------------------------------------------ regimes
    def teacher_forced(self, x, u):
        """x [B, T, n_x] true states x_0..x_{T-1}; u [B, T, n_u] -> predictions of x_1..x_T."""
        inp = self._inp(x, u, 0.0)
        y, _ = self.core.sequence(inp)
        return x + self._increment(y, inp)

    def warmup(self, x_ctx, u):
        """Run the core over true x_0..x_{C-1}; return its state (None when C = 0)."""
        C = x_ctx.shape[1] - 1
        if C == 0:
            return None
        _, state = self.core.sequence(self._inp(x_ctx[:, :C], u[:, :C], 0.0))
        return state

    def chained(self, x_ctx, u, horizon: int, state=None, start=None, noise_std: float = 0.0):
        """x_ctx [B, C+1, n_x] = true x_0..x_C; u [B, C+H, n_u]. Returns (predictions [B, H, n_x], hit [B]).

        `state` / `start` allow the diagnostics to supply a precomputed warm-up state and a
        (differentiable) starting state x_C. `noise_std` > 0 (training only) adds Gaussian noise to the
        state that is fed back, never to the prediction that is scored.
        """
        C = x_ctx.shape[1] - 1
        if state is None:
            state = self.warmup(x_ctx, u)
        xh = x_ctx[:, C] if start is None else start
        hit = torch.zeros(xh.shape[0], dtype=torch.bool, device=xh.device)
        preds = []
        for k in range(horizon):
            fed = xh + noise_std * torch.randn_like(xh) if noise_std > 0 else xh
            inp = self._inp(fed, u[:, C + k], 0.0)
            y, state = self.core.step(inp, state)
            xh, bad = self._safe(xh + self._increment(y, inp))
            hit = hit | bad
            preds.append(xh)
        return torch.stack(preds, dim=1), hit

    def direct(self, x_ctx, u, horizon: int, start=None):
        """Direct multi-step prediction: the whole horizon in one forward pass, nothing fed back."""
        B, C1, _ = x_ctx.shape
        C = C1 - 1
        if start is not None:
            x_ctx = torch.cat([x_ctx[:, :C], start.unsqueeze(1)], dim=1)
        ctx = self._inp(x_ctx, u[:, :C + 1], 0.0)
        future_u = u[:, C + 1:C + horizon]
        fut = self._inp(future_u.new_zeros(B, future_u.shape[1], self.n_x), future_u, 1.0)
        inp = torch.cat([ctx, fut], dim=1)
        y, _ = self.core.sequence(inp)
        increments = self._increment(y[:, C:C + horizon], inp[:, C:C + horizon])
        preds, bad = self._safe(x_ctx[:, C:C + 1] + torch.cumsum(increments, dim=1))
        return preds, bad.any(dim=1)

    def rollout(self, regime: str, x_window, u_window, context_len: int, horizon: int):
        """Prediction for the evaluation regime ('chained' or 'direct'). Returns (preds, targets, hit)."""
        x_ctx = x_window[:, :context_len + 1]
        target = x_window[:, context_len + 1:context_len + 1 + horizon]
        if regime == "direct":
            preds, hit = self.direct(x_ctx, u_window, horizon)
        else:
            preds, hit = self.chained(x_ctx, u_window, horizon)
        return preds, target, hit

    def decay_modules(self):
        return self.core.decay_modules()
