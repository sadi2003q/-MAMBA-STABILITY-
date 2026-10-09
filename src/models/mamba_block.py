"""Full Mamba block stack (pure PyTorch, no custom kernels), with a step-by-step mode for chained
prediction. Layer state = (convolution buffer [B, E, d_conv - 1], hidden state [B, E, d_state])."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import DecayParam, RMSNorm, init_dt_bias, parallel_scan, sequential_scan


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dt_rank: int,
                 dt_min: float, dt_max: float, decay_mode: str, unstable_init_scale: float, bias: bool):
        super().__init__()
        E = expand * d_model
        self.E, self.d_state, self.d_conv, self.dt_rank = E, d_state, d_conv, dt_rank
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * E, bias=bias)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=0)   # causal: we left-pad with the buffer
        self.x_proj = nn.Linear(E, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, E)
        init_dt_bias(self.dt_proj, dt_min, dt_max)
        self.decay = DecayParam(E, d_state, decay_mode, unstable_init_scale)
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d_model, bias=bias)

    def sequence(self, x, state=None, parallel: bool = True):
        Bsz, T, _ = x.shape
        residual = x
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)                 # [B, T, E] each
        if state is None:
            buf = x.new_zeros(Bsz, self.E, self.d_conv - 1)
            h0 = None
        else:
            buf, h0 = state
        padded = torch.cat([buf, xi.transpose(1, 2)], dim=2)                # [B, E, T + d_conv - 1]
        xc = F.silu(self.conv(padded)).transpose(1, 2)                      # [B, T, E]
        new_buf = padded[:, :, padded.shape[2] - (self.d_conv - 1):]
        dt_in, Bm, Cm = torch.split(self.x_proj(xc), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))                                # [B, T, E]
        a = torch.exp(dt.unsqueeze(-1) * self.decay.A())                    # [B, T, E, N]
        b = (dt * xc).unsqueeze(-1) * Bm.unsqueeze(-2)
        H = (parallel_scan if parallel else sequential_scan)(a, b, h0)
        y = (H * Cm.unsqueeze(-2)).sum(-1) + self.D * xc
        y = y * F.silu(z)
        return residual + self.out_proj(y), (new_buf, H[:, -1])


class MambaCore(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 8, d_state: int = 8, d_conv: int = 10, n_layers: int = 6,
                 expand: int = 1, dt_rank: int = 1, dt_min: float = 0.001, dt_max: float = 0.1,
                 bias: bool = True, decay_mode: str = "certified", unstable_init_scale: float = 1.0,
                 use_parallel_scan: bool = True):
        super().__init__()
        self.use_parallel_scan = use_parallel_scan
        self.embed = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, decay_mode,
                       unstable_init_scale, bias) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.out_dim = d_model

    def sequence(self, inp, state=None, parallel: bool | None = None):
        parallel = self.use_parallel_scan if parallel is None else parallel
        h = self.embed(inp)
        state = state if state is not None else [None] * len(self.layers)
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st = layer.sequence(h, st, parallel)
            new_state.append(st)
        return self.norm_f(h), new_state

    def step(self, inp, state=None):
        y, state = self.sequence(inp.unsqueeze(1), state, parallel=True)   # T = 1: exact single step
        return y[:, 0], state

    def decay_modules(self):
        return [layer.decay for layer in self.layers]
