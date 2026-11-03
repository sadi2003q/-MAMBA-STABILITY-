"""Chain-Mamba: a Mamba block stack modified for chained (one-step, fed-back) prediction.

Why each change exists (diagnosis from exp01d seed 0 and the literature, see docs/chain_mamba_plan.md):

1. Memory with a decay MARGIN (the certificate, made quantitative).
   Standard Mamba: A = -exp(A_log) < 0 and Delta > 0, so the decay factor A_bar = exp(Delta * A) lies in (0, 1).
   That is the latent certificate, but it allows A_bar -> 1. In the trained exp01d models the largest decay
   factor is 0.996 to 0.999 in every layer (step size Delta never left its initial range 0.001 to 0.1), so the
   memory's steady-state gain 1 / (1 - A_bar) is 250 to 1,200. Under chained prediction a mistake that enters
   the memory is fed back with that gain. Here
       Delta_t = delta_floor + softplus(.)      >= delta_floor
       A       = -(a_min + exp(A_log))          <= -a_min
   so every decay factor is at most rho_max = exp(-delta_floor * a_min) < 1 for every input and every set of
   weights. With the defaults (0.05, 1.0): rho_max = 0.951 and the memory gain is at most 20.5.

2. A gated feed-forward layer (SwiGLU) after the selective state-space mixer in every layer, as in Mamba-3's
   block. Plain Mamba has no product between the current state and the input inside a time step (Mamba-X,
   arXiv 2604.17221); Van der Pol's one-step map needs x1^2 * x2.

3. A short causal convolution (kernel 4 by default, not 10), so at most three earlier, possibly wrong,
   predictions are mixed directly into the current step.

4. No root-mean-square normalisation in front of the read-out: that normalisation removes the size of the
   feature vector, and the size of the next increment changes by more than a factor of 10 along the orbit.

The read-out anchor (a linear path from the current state and input to the increment) and the increment
normalisation live in src/models/dynamics.py (options under `readout:` in the config), because they are not
specific to this core.

Core interface (same as the other cores):
    out_dim; sequence(inp, state=None) -> (y, state); step(inp, state=None) -> (y, state); decay_modules()
Layer state = (convolution buffer [B, E, d_conv - 1], hidden state [B, E, d_state]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import RMSNorm, parallel_scan, sequential_scan


class MarginDecay(nn.Module):
    """A = -(a_min + exp(A_log)): strictly below -a_min. Same `A()` / `stats()` interface as DecayParam."""

    def __init__(self, channels: int, d_state: int, a_min: float = 1.0):
        super().__init__()
        self.mode = "certified_margin"
        self.a_min = float(a_min)
        magnitude = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)
        self.A_log = nn.Parameter(torch.log(magnitude))
        self.A_log._no_weight_decay = True

    def A(self) -> torch.Tensor:
        return -(self.a_min + torch.exp(self.A_log))

    @torch.no_grad()
    def stats(self) -> dict:
        A = self.A()
        return {"fraction_positive": float((A > 0).float().mean()), "max": float(A.max()),
                "min": float(A.min()), "a_min": self.a_min}


class ChainMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dt_rank: int, dt_min: float,
                 dt_max: float, delta_floor: float, a_min: float, ffn_hidden: int, bias: bool,
                 mixer: bool = True):
        super().__init__()
        E = expand * d_model
        self.mixer = bool(mixer)
        self.E, self.d_state, self.d_conv, self.dt_rank = E, d_state, d_conv, dt_rank
        self.delta_floor = float(delta_floor)
        if self.mixer:
            # selective state-space mixer
            self.norm = RMSNorm(d_model)
            self.in_proj = nn.Linear(d_model, 2 * E, bias=bias)
            self.conv_weight = nn.Parameter(torch.empty(E, d_conv))
            self.conv_bias = nn.Parameter(torch.zeros(E))
            nn.init.uniform_(self.conv_weight, -1.0 / math.sqrt(d_conv), 1.0 / math.sqrt(d_conv))
            self.x_proj = nn.Linear(E, dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(dt_rank, E)
            self._init_dt(dt_min, dt_max)
            self.decay = MarginDecay(E, d_state, a_min)
            self.D = nn.Parameter(torch.ones(E))
            self.out_proj = nn.Linear(E, d_model, bias=bias)
        # gated feed-forward (SwiGLU): per-step products between features
        self.ffn = ffn_hidden > 0
        if self.ffn:
            self.norm2 = RMSNorm(d_model)
            self.ffn_in = nn.Linear(d_model, 2 * ffn_hidden, bias=False)
            self.ffn_out = nn.Linear(ffn_hidden, d_model, bias=False)

    @torch.no_grad()
    def _init_dt(self, dt_min: float, dt_max: float) -> None:
        """Total step size delta_floor + softplus(bias) starts log-uniform in [dt_min, dt_max]."""
        lo, hi = max(dt_min - self.delta_floor, 1e-4), max(dt_max - self.delta_floor, 2e-4)
        extra = torch.exp(torch.rand(self.E) * (math.log(hi) - math.log(lo)) + math.log(lo))
        self.dt_proj.bias.copy_(extra + torch.log(-torch.expm1(-extra)))   # inverse softplus

    # --------------------------------------------------------------------- pieces
    def _conv(self, xi, buf):
        """Causal depthwise convolution. xi [B, T, E]; buf [B, E, K-1] (earlier inputs) -> [B, T, E], new buf."""
        padded = torch.cat([buf, xi.transpose(1, 2)], dim=2)                  # [B, E, T + K - 1]
        out = F.conv1d(padded, self.conv_weight.unsqueeze(1), self.conv_bias, groups=self.E)
        return out.transpose(1, 2), padded[:, :, padded.shape[2] - (self.d_conv - 1):]

    def _selective(self, xc):
        dt_in, Bm, Cm = torch.split(self.x_proj(xc), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.delta_floor + F.softplus(self.dt_proj(dt_in))              # >= delta_floor
        a = torch.exp(dt.unsqueeze(-1) * self.decay.A())                     # <= exp(-delta_floor * a_min)
        b = (dt * xc).unsqueeze(-1) * Bm.unsqueeze(-2)
        return a, b, Cm

    def _ffn(self, h):
        if not self.ffn:
            return h
        g, v = self.ffn_in(self.norm2(h)).chunk(2, dim=-1)
        return h + self.ffn_out(F.silu(g) * v)

    # --------------------------------------------------------------------- forward
    def sequence(self, x, state=None, parallel: bool = True):
        if not self.mixer:                       # control: no memory, feed-forward layer only
            return self._ffn(x), None
        Bsz = x.shape[0]
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        if state is None:
            buf, h0 = x.new_zeros(Bsz, self.E, self.d_conv - 1), None
        else:
            buf, h0 = state
        xc, new_buf = self._conv(xi, buf)
        xc = F.silu(xc)
        a, b, Cm = self._selective(xc)
        H = (parallel_scan if parallel else sequential_scan)(a, b, h0)
        y = ((H * Cm.unsqueeze(-2)).sum(-1) + self.D * xc) * F.silu(z)
        h = x + self.out_proj(y)
        return self._ffn(h), (new_buf, H[:, -1])

    def step(self, x, state=None):
        """One time step without building length-1 sequences. x [B, d_model]."""
        if not self.mixer:
            return self._ffn(x), None
        Bsz = x.shape[0]
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        if state is None:
            buf = x.new_zeros(Bsz, self.E, self.d_conv - 1)
            h_prev = None
        else:
            buf, h_prev = state
        window = torch.cat([buf, xi.unsqueeze(-1)], dim=2)                   # [B, E, K]
        xc = F.silu((window * self.conv_weight).sum(-1) + self.conv_bias)
        a, b, Cm = self._selective(xc)
        h_new = b if h_prev is None else a * h_prev + b
        y = ((h_new * Cm.unsqueeze(-2)).sum(-1) + self.D * xc) * F.silu(z)
        h = x + self.out_proj(y)
        return self._ffn(h), (window[:, :, 1:], h_new)

    @torch.no_grad()
    def rho_max(self) -> float:
        """Guaranteed upper bound on every decay factor of this layer."""
        return math.exp(-self.delta_floor * self.decay.a_min)


class ChainMambaCore(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 16, d_state: int = 8, d_conv: int = 4, n_layers: int = 2,
                 expand: int = 1, dt_rank: int = 1, dt_min: float = 0.06, dt_max: float = 0.5,
                 delta_floor: float = 0.05, a_min: float = 1.0, ffn_hidden: int = 24, bias: bool = True,
                 final_norm: bool = False, use_parallel_scan: bool = True, mixer: bool = True):
        super().__init__()
        self.use_parallel_scan = use_parallel_scan
        self.embed = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            ChainMambaLayer(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, delta_floor, a_min,
                            ffn_hidden, bias, mixer) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model) if final_norm else nn.Identity()
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
        h = self.embed(inp)
        state = state if state is not None else [None] * len(self.layers)
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st = layer.step(h, st)
            new_state.append(st)
        return self.norm_f(h), new_state

    def decay_modules(self):
        return [layer.decay for layer in self.layers if layer.mixer]

    def rho_max(self) -> float:
        return max((layer.rho_max() for layer in self.layers if layer.mixer), default=0.0)
