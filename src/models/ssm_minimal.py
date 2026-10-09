"""Minimal selective state-space cell (one layer; no convolution, no gate branch).

Core interface shared by all cores:
    out_dim                         features passed to the dynamics model's readout
    sequence(inp, state=None)       inp [B, T, in_dim] -> (y [B, T, out_dim], final state)
    step(inp, state=None)           inp [B, in_dim]    -> (y [B, out_dim], new state)
    decay_modules()                 the DecayParam modules (for certificate statistics)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import DecayParam, init_dt_bias, parallel_scan, sequential_scan


class MinimalSelectiveSSM(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 32, d_state: int = 16, dt_min: float = 0.001,
                 dt_max: float = 0.1, decay_mode: str = "certified", unstable_init_scale: float = 1.0,
                 use_parallel_scan: bool = True):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.use_parallel_scan = use_parallel_scan
        self.embed = nn.Linear(in_dim, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        init_dt_bias(self.dt_proj, dt_min, dt_max)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.decay = DecayParam(d_model, d_state, decay_mode, unstable_init_scale)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_dim = d_model

    def _gates(self, inp):
        s = F.silu(self.embed(inp))
        dt = F.softplus(self.dt_proj(s))
        return s, dt, self.B_proj(s), self.C_proj(s)

    def _ab(self, s, dt, B):
        A = self.decay.A()
        a = torch.exp(dt.unsqueeze(-1) * A)                 # [..., D, N]
        b = (dt * s).unsqueeze(-1) * B.unsqueeze(-2)        # [..., D, N]
        return a, b

    def sequence(self, inp, state=None, parallel: bool | None = None):
        parallel = self.use_parallel_scan if parallel is None else parallel
        s, dt, B, C = self._gates(inp)
        a, b = self._ab(s, dt, B)
        H = (parallel_scan if parallel else sequential_scan)(a, b, state)
        y = (H * C.unsqueeze(-2)).sum(-1) + self.D * s
        return y, H[:, -1]

    def step(self, inp, state=None):
        s, dt, B, C = self._gates(inp)
        a, b = self._ab(s, dt, B)
        h = b if state is None else a * state + b
        y = (h * C.unsqueeze(-2)).sum(-1) + self.D * s
        return y, h

    def decay_modules(self):
        return [self.decay]

    @torch.no_grad()
    def decay_factors(self, inp):
        s, dt, B, _ = self._gates(inp)
        a, _ = self._ab(s, dt, B)
        return a
