"""Pieces shared by the state-space models: the decay parameter (where the certificate lives),
step-size initialisation, the parallel scan, and root-mean-square normalisation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DECAY_MODES = ("certified", "free_sign", "unstable_init")


class DecayParam(nn.Module):
    """Continuous-time decay A, shape [channels, d_state].

    certified:      A = -exp(A_log)  < 0 always  -> A_bar = exp(Delta * A) in (0, 1): the latent certificate
    free_sign:      A = A_raw, starts at the same negative values, may become positive during training
    unstable_init:  A = A_raw, starts at the mirrored positive values (memory grows at the start)
    Initial magnitudes 1, 2, ..., d_state per channel (the usual Mamba / S4D-real start).
    """

    def __init__(self, channels: int, d_state: int, mode: str = "certified", unstable_init_scale: float = 1.0):
        super().__init__()
        if mode not in DECAY_MODES:
            raise ValueError(f"decay_mode must be one of {DECAY_MODES}, got '{mode}'.")
        self.mode = mode
        magnitude = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)
        if mode == "certified":
            self.A_log = nn.Parameter(torch.log(magnitude))
        elif mode == "free_sign":
            self.A_raw = nn.Parameter(-magnitude)
        else:
            self.A_raw = nn.Parameter(unstable_init_scale * magnitude)
        for p in self.parameters():
            p._no_weight_decay = True

    def A(self) -> torch.Tensor:
        return -torch.exp(self.A_log) if self.mode == "certified" else self.A_raw

    @torch.no_grad()
    def stats(self) -> dict:
        A = self.A()
        return {"fraction_positive": float((A > 0).float().mean()), "max": float(A.max()),
                "min": float(A.min())}


def init_dt_bias(linear: nn.Linear, dt_min: float, dt_max: float) -> None:
    """Bias so that softplus(bias) starts log-uniform in [dt_min, dt_max] (Mamba initialisation)."""
    with torch.no_grad():
        dt = torch.exp(torch.rand(linear.out_features) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        linear.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse of softplus


def parallel_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """All states of h_t = a_t * h_{t-1} + b_t along dim 1 (time), by recursive doubling.

    a, b: [batch, T, ...]; h0: [batch, ...] or None (zero). Returns h: [batch, T, ...].
    Exact (no division by cumulative products), so it is safe for decay factors near 0.
    """
    if h0 is not None:
        b = torch.cat([b[:, :1] + a[:, :1] * h0.unsqueeze(1), b[:, 1:]], dim=1)
    T = a.shape[1]
    offset = 1
    while offset < T:
        a_prev = torch.cat([torch.ones_like(a[:, :offset]), a[:, :-offset]], dim=1)
        b_prev = torch.cat([torch.zeros_like(b[:, :offset]), b[:, :-offset]], dim=1)
        b = a * b_prev + b
        a = a * a_prev
        offset *= 2
    return b


def sequential_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """Same as parallel_scan, one step at a time (reference implementation)."""
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
