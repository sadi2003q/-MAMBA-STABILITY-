"""
Chain-Mamba: a stack of Mamba-style layers, changed so that it works well for CHAINED prediction
(the model is fed its own predictions, one step after another).

BEGINNER-FRIENDLY VERSION. Every calculation, every name and the order of every random draw are the same
as in the compact version, so a trained model, a saved checkpoint and a seed-0 run give the same numbers.

----------------------------------------------------------------------------------------------------------
Sizes used in this project (configs/models/chain_mamba.yaml)
----------------------------------------------------------------------------------------------------------
    d_model = 12     features that flow between the layers
    E       = 12     channels inside the memory part (E = expand x d_model, expand = 1)
    d_state = 8      memory slots per channel  -> 12 x 8 = 96 memory numbers per layer
    d_conv  = 4      the short convolution looks at the current step and the 3 steps before it
    layers  = 2

    Shapes in the comments: B = batch size, T = number of time steps.

----------------------------------------------------------------------------------------------------------
What one layer does (mixer = True)
----------------------------------------------------------------------------------------------------------
    Part 1, the MEMORY part (selective state-space "mixer"):
        1. normalise the input, split it into a memory branch xi and a gate branch z
        2. short convolution: mix xi with the last 3 steps of xi, then SiLU          -> xc
        3. gates computed FROM xc at every step ("selective"):
               step size  dt  (never below delta_floor = 0.05)
               B          which memory slots receive new information
               C          which memory slots are read out
        4. decay factor  a = exp(dt x A),  with A <= -a_min = -1  ->  a <= exp(-0.05) = 0.951
           (this is the stability CERTIFICATE, with a margin: the memory always forgets a little)
        5. memory update:  h_new = a x h_old + b,   where b = dt x xc x B
        6. read out:  y = sum over slots (h_new x C) + D x xc,   then  y = y x SiLU(z)
        7. project back to d_model and add to the layer's input (residual connection)

    Part 2, the gated FEED-FORWARD part (SwiGLU), looks only at the current step:
        h + ffn_out( SiLU(g) x v ),  with g, v from ffn_in(normalise(h))
        The product g x v lets the model build terms like x1^2 x2 (Van der Pol needs them).

    The layer's memory STATE is carried from one time step to the next:
        state = (convolution buffer [B, E, 3],  memory h [B, E, 8])

----------------------------------------------------------------------------------------------------------
Why each design choice exists (diagnosis from exp01d seed 0, see docs/chain_mamba_plan.md)
----------------------------------------------------------------------------------------------------------
1. Decay MARGIN. Standard Mamba only guarantees 0 < a < 1, so a can creep up to 0.999. In the trained
   exp01d models the largest decay factor was 0.996 to 0.999, so the memory's steady-state gain 1 / (1 - a)
   was 250 to 1,200: a mistake that entered the memory was fed back that strongly. Here
       dt = delta_floor + softplus(.)      >= delta_floor
       A  = -(a_min + exp(A_log))          <= -a_min
   so every decay factor is at most rho_max = exp(-delta_floor x a_min) = 0.951 for every input and every
   set of weights, and the memory gain is at most 1 / (1 - 0.951) = 20.5.
2. Gated feed-forward layer after the memory part in every layer (as in Mamba-3's block): plain Mamba has
   no product between features inside one time step (Mamba-X, arXiv 2604.17221).
3. Short convolution (kernel 4, not 10): at most three earlier, possibly wrong, predictions are mixed
   directly into the current step.
4. No root-mean-square normalisation in front of the read-out: it would remove the size of the feature
   vector, and the size of the next increment changes by more than a factor of 10 along the orbit.

The read-out anchor and the increment normalisation live in src/models/dynamics.py (under `readout:` in the
config), because they are not specific to this core.

Core interface (the same for every core in this project):
    out_dim;  sequence(inp, state=None) -> (y, state);  step(inp, state=None) -> (y, state);  decay_modules()
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import RMSNorm, parallel_scan, sequential_scan


# ==========================================================================================================
# The decay rates A, with a margin (the stability certificate)
# ==========================================================================================================
class MarginDecay(nn.Module):
    """
    Decay rates  A = -(a_min + exp(A_log)),  so A is always below -a_min (never closer to zero).
    Same `A()` / `stats()` interface as DecayParam in the other models.
    """

    def __init__(self, channels: int, d_state: int, a_min: float = 1.0):
        super().__init__()

        # Step 1: a label used in logs and diagnostics
        self.mode = "certified_margin"

        # Step 2: the smallest allowed size of a decay rate (1.0 in this project)
        self.a_min = float(a_min)

        # Step 3: starting values: slot k of every channel starts with size k  (1, 2, ..., 8)
        one_row = torch.arange(1, d_state + 1, dtype=torch.float32)    # [d_state]       e.g. 1..8
        magnitude = one_row.repeat(channels, 1)                        # [channels, d_state]

        # Step 4: the trainable numbers are the logarithms (so exp(A_log) is always positive)
        self.A_log = nn.Parameter(torch.log(magnitude))

        # Step 5: no weight decay on these (weight decay would pull them towards zero)
        self.A_log._no_weight_decay = True

    def A(self) -> torch.Tensor:
        # size of each decay rate: a_min plus a positive number -> always bigger than a_min
        magnitude = self.a_min + torch.exp(self.A_log)

        # the decay rate itself is negative
        return -magnitude

    @torch.no_grad()
    def stats(self) -> dict:
        A = self.A()
        fraction_positive = float((A > 0).float().mean())    # always 0 here (certificate holds)
        largest = float(A.max())
        smallest = float(A.min())
        return {"fraction_positive": fraction_positive, "max": largest, "min": smallest, "a_min": self.a_min}


# ==========================================================================================================
# One Chain-Mamba layer:  memory part (optional)  ->  gated feed-forward part
# ==========================================================================================================
class ChainMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dt_rank: int, dt_min: float,
                 dt_max: float, delta_floor: float, a_min: float, ffn_hidden: int, bias: bool,
                 mixer: bool = True):
        super().__init__()

        # Step 1: width of the expanded channels inside the memory part
        E = expand * d_model

        # Step 2: remember whether this layer has the memory part (the "mixer")
        #   mixer = True  -> normal Chain-Mamba layer: memory + gated feed-forward layer
        #   mixer = False -> memory removed: only the gated feed-forward layer is kept
        # bool(...) turns values such as 1, 0 or false from the settings file into True / False
        self.mixer = bool(mixer)

        # Step 3: store the sizes we need later
        self.E = E
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.delta_floor = float(delta_floor)

        # Step 4: the selective state-space memory (the "mixer"), only if it is switched on
        # Keep these lines in EXACTLY this order: each one draws random starting numbers,
        # and a different order would give different starting weights for seed 0.
        if self.mixer:
            # 4a: normalise the input of the memory part
            self.norm = RMSNorm(d_model)

            # 4b: split the input into two branches (memory branch and gate branch)
            self.in_proj = nn.Linear(d_model, 2 * E, bias=bias)

            # 4c: short causal convolution over the last few time steps (kernel d_conv = 4)
            self.conv_weight = nn.Parameter(torch.empty(E, d_conv))
            self.conv_bias = nn.Parameter(torch.zeros(E))
            limit = 1.0 / math.sqrt(d_conv)
            nn.init.uniform_(self.conv_weight, -limit, limit)

            # 4d: the input-dependent gates: step size, B (what enters memory), C (what is read out)
            self.x_proj = nn.Linear(E, dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(dt_rank, E)
            self._init_dt(dt_min, dt_max)

            # 4e: the decay rates with a margin (the stability certificate lives here)
            self.decay = MarginDecay(E, d_state, a_min)

            # 4f: direct path around the memory, and the output projection back to d_model
            self.D = nn.Parameter(torch.ones(E))
            self.out_proj = nn.Linear(E, d_model, bias=bias)

        # Step 5: the gated feed-forward layer (SwiGLU): products between features inside one step
        # ffn_hidden = 0 would remove it
        self.ffn = ffn_hidden > 0
        if self.ffn:
            self.norm2 = RMSNorm(d_model)
            self.ffn_in = nn.Linear(d_model, 2 * ffn_hidden, bias=False)    # makes g and v together
            self.ffn_out = nn.Linear(ffn_hidden, d_model, bias=False)

    # ------------------------------------------------------------------------------------------------------
    # Starting values of the step size
    # ------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _init_dt(self, dt_min: float, dt_max: float) -> None:
        """
        The total step size  delta_floor + softplus(bias)  should START spread evenly on a log scale
        between dt_min (0.06) and dt_max (0.5). This sets dt_proj.bias so that happens.
        """

        # Step 1: the part ABOVE delta_floor must lie between lo and hi (small safety minimums)
        lo = max(dt_min - self.delta_floor, 1e-4)
        hi = max(dt_max - self.delta_floor, 2e-4)

        # Step 2: pick one random number per channel, evenly spread on a log scale between lo and hi
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        random_0_to_1 = torch.rand(self.E)                    # one random draw per channel
        log_value = random_0_to_1 * (log_hi - log_lo) + log_lo
        extra = torch.exp(log_value)                          # wanted value of softplus(bias)

        # Step 3: inverse softplus: find the bias whose softplus equals `extra`
        #   softplus(b) = log(1 + exp(b))   ->   b = extra + log(1 - exp(-extra))
        one_minus_exp = -torch.expm1(-extra)                  # 1 - exp(-extra), computed accurately
        bias_value = extra + torch.log(one_minus_exp)
        self.dt_proj.bias.copy_(bias_value)

    # ------------------------------------------------------------------------------------------------------
    # Pieces of the memory part
    # ------------------------------------------------------------------------------------------------------
    def _conv(self, xi, buf):
        """
        Causal depthwise convolution over a whole sequence ("causal" = only the past and the present).
            xi  [B, T, E]          memory-branch input for T steps
            buf [B, E, d_conv - 1] the last 3 inputs from BEFORE this sequence (zeros at the start)
        Returns the convolution output [B, T, E] and the new buffer (the last 3 inputs) for next time.
        """

        # Step 1: put the time axis last, and glue the 3 earlier inputs in front
        xi_time_last = xi.transpose(1, 2)                                 # [B, E, T]
        padded = torch.cat([buf, xi_time_last], dim=2)                    # [B, E, T + 3]

        # Step 2: every channel has its own 4 weights (groups = E means "one filter per channel")
        filters = self.conv_weight.unsqueeze(1)                           # [E, 1, 4]
        out = F.conv1d(padded, filters, self.conv_bias, groups=self.E)    # [B, E, T]

        # Step 3: back to [B, T, E]
        out = out.transpose(1, 2)

        # Step 4: keep the last 3 inputs for the next call
        total_length = padded.shape[2]
        keep_from = total_length - (self.d_conv - 1)
        new_buf = padded[:, :, keep_from:]                                # [B, E, 3]

        return out, new_buf

    def _selective(self, xc):
        """
        The 'selective' gates, recomputed from the current input xc at every step.
            xc [..., E]
        Returns
            a  [..., E, d_state]   decay factors: how much of each memory slot is KEPT (at most 0.951)
            b  [..., E, d_state]   what is WRITTEN into each memory slot
            Cm [..., d_state]      how much of each memory slot is READ OUT
        """

        # Step 1: one linear map gives three pieces: dt_in (1 number), B (8 numbers), C (8 numbers)
        all_gates = self.x_proj(xc)
        sizes = [self.dt_rank, self.d_state, self.d_state]
        dt_in, Bm, Cm = torch.split(all_gates, sizes, dim=-1)

        # Step 2: step size per channel, never below delta_floor
        softplus_part = F.softplus(self.dt_proj(dt_in))                    # always > 0
        dt = self.delta_floor + softplus_part                              # [..., E], >= delta_floor

        # Step 3: decay factor  a = exp(dt x A).  dt >= 0.05 and A <= -1  ->  a <= exp(-0.05) = 0.951
        A = self.decay.A()                                                 # [E, d_state]
        a = torch.exp(dt.unsqueeze(-1) * A)                                # [..., E, d_state]

        # Step 4: what is written into memory:  (dt x xc) for each channel, spread over slots by B
        dt_times_input = dt * xc                                           # [..., E]
        b = dt_times_input.unsqueeze(-1) * Bm.unsqueeze(-2)                # [..., E, d_state]

        return a, b, Cm

    def _ffn(self, h):
        """Gated feed-forward part (SwiGLU). Looks only at the current step."""

        # Step 0: no feed-forward layer -> pass the input through unchanged
        if not self.ffn:
            return h

        # Step 1: normalise, then make two branches g and v
        both = self.ffn_in(self.norm2(h))
        g, v = both.chunk(2, dim=-1)

        # Step 2: the gate: SiLU(g) multiplied by v (this is where feature products come from)
        gated = F.silu(g) * v

        # Step 3: project back and add to the input (residual connection)
        return h + self.ffn_out(gated)

    # ------------------------------------------------------------------------------------------------------
    # Running the layer
    # ------------------------------------------------------------------------------------------------------
    def sequence(self, x, state=None, parallel: bool = True):
        """
        Run the layer over a whole sequence at once (used for warm-up and teacher forcing).
            x     [B, T, d_model]
            state (convolution buffer, memory) from before this sequence, or None to start from zeros
        Returns (output [B, T, d_model], new state).
        """
        if not self.mixer:
            output = self._ffn(x)
            return output, None

        batch_size = x.shape[0]

        # Step 1: normalise and split into the memory branch xi and the gate branch z
        both = self.in_proj(self.norm(x))
        xi, z = both.chunk(2, dim=-1)                                      # each [B, T, E]

        # Step 2: where do we start? (empty convolution buffer and empty memory, or the given state)
        if state is None:
            buf = x.new_zeros(batch_size, self.E, self.d_conv - 1)
            h0 = None
        else:
            buf, h0 = state

        # Step 3: short convolution, then SiLU
        xc, new_buf = self._conv(xi, buf)
        xc = F.silu(xc)

        # Step 4: the gates for every time step
        a, b, Cm = self._selective(xc)

        # Step 5: memory update for all T steps:  H_t = a_t x H_(t-1) + b_t
        # (parallel_scan and sequential_scan give the same answer; parallel is faster)
        if parallel:
            H = parallel_scan(a, b, h0)                                    # [B, T, E, d_state]
        else:
            H = sequential_scan(a, b, h0)

        # Step 6: read out the memory with C, add the direct path D x xc, then apply the gate
        read_out = (H * Cm.unsqueeze(-2)).sum(-1)                          # [B, T, E]
        direct = self.D * xc
        y = (read_out + direct) * F.silu(z)

        # Step 7: back to d_model, added to the layer's input (residual connection)
        h = x + self.out_proj(y)

        # Step 8: gated feed-forward part; the new state is (last 3 inputs, memory after the last step)
        output = self._ffn(h)
        last_memory = H[:, -1]
        return output, (new_buf, last_memory)

    def step(self, x, state=None):
        """
        Run the layer for ONE time step (used in chained prediction).
            x     [B, d_model]
            state (convolution buffer, memory) from the previous step, or None at the very start
        Returns (output [B, d_model], new state).  Same maths as sequence(), for one step.
        """
        if not self.mixer:
            output = self._ffn(x)
            return output, None

        # Step 1: normalise and split into the memory branch xi and the gate branch z
        both = self.in_proj(self.norm(x))
        xi, z = both.chunk(2, dim=-1)                                      # each [B, E]

        # Step 2: where do we start?
        if state is None:
            buf = x.new_zeros(batch_size, self.E, self.d_conv - 1)
            h_prev = None
        else:
            buf, h_prev = state

        # Step 3: short convolution for this one step: the last 3 inputs + the current one
        window = torch.cat([buf, xi.unsqueeze(-1)], dim=2)                 # [B, E, 4]
        weighted_sum = (window * self.conv_weight).sum(-1)                 # [B, E]
        xc = F.silu(weighted_sum + self.conv_bias)

        # Step 4: the gates for this step
        a, b, Cm = self._selective(xc)

        # Step 5: memory update:  keep a part of the old memory, add the new information
        if h_prev is None:
            h_new = b                                                      # empty memory at the start
        else:
            h_new = a * h_prev + b

        # Step 6: read out the memory with C, add the direct path D x xc, then apply the gate
        read_out = (h_new * Cm.unsqueeze(-2)).sum(-1)                      # [B, E]
        direct = self.D * xc
        y = (read_out + direct) * F.silu(z)

        # Step 7: back to d_model, added to the layer's input (residual connection)
        h = x + self.out_proj(y)

        # Step 8: gated feed-forward part; the new state drops the oldest input from the window
        output = self._ffn(h)
        new_buf = window[:, :, 1:]                                         # [B, E, 3]
        return output, (new_buf, h_new)

    @torch.no_grad()
    def rho_max(self) -> float:
        """Guaranteed upper bound on every decay factor of this layer:  exp(-delta_floor x a_min)."""
        return math.exp(-self.delta_floor * self.decay.a_min)


# ==========================================================================================================
# The core: embedding  ->  n_layers Chain-Mamba layers  ->  (optional) final normalisation
# ==========================================================================================================
class ChainMambaCore(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 16, d_state: int = 8, d_conv: int = 4, n_layers: int = 2,
                 expand: int = 1, dt_rank: int = 1, dt_min: float = 0.06, dt_max: float = 0.5,
                 delta_floor: float = 0.05, a_min: float = 1.0, ffn_hidden: int = 24, bias: bool = True,
                 final_norm: bool = False, use_parallel_scan: bool = True, mixer: bool = True):
        super().__init__()

        # Step 1: settings
        self.use_parallel_scan = use_parallel_scan

        # Step 2: embedding: the input [x_t, u_t, mask] (4 numbers) becomes d_model features
        self.embed = nn.Linear(in_dim, d_model)

        # Step 3: the layers, built one after another (same order as before -> same random draws)
        layer_list = []
        for _ in range(n_layers):
            layer = ChainMambaLayer(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, delta_floor,
                                    a_min, ffn_hidden, bias, mixer)
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)

        # Step 4: optional final normalisation (switched off in this project: final_norm = false)
        if final_norm:
            self.norm_f = RMSNorm(d_model)
        else:
            self.norm_f = nn.Identity()          # does nothing

        # Step 5: how many features come out (the read-out head in dynamics.py needs this)
        self.out_dim = d_model

    def sequence(self, inp, state=None, parallel: bool | None = None):
        """Whole sequence at once. inp [B, T, in_dim]. Returns (features [B, T, d_model], list of layer states)."""

        # Step 1: use the setting from the config unless told otherwise
        if parallel is None:
            parallel = self.use_parallel_scan

        # Step 2: embed the input
        h = self.embed(inp)

        # Step 3: no state yet -> one empty state per layer
        if state is None:
            state = [None] * len(self.layers)

        # Step 4: run the layers one after another; each layer keeps its own state
        new_state = []
        for layer, layer_state in zip(self.layers, state):
            h, layer_state = layer.sequence(h, layer_state, parallel)
            new_state.append(layer_state)

        return self.norm_f(h), new_state

    def step(self, inp, state=None):
        """One time step. inp [B, in_dim]. Returns (features [B, d_model], list of layer states)."""

        # Step 1: embed the input
        h = self.embed(inp)

        # Step 2: no state yet -> one empty state per layer
        if state is None:
            state = [None] * len(self.layers)

        # Step 3: run the layers one after another; each layer takes its state in and gives the new one out
        new_state = []
        for layer, layer_state in zip(self.layers, state):
            h, layer_state = layer.step(h, layer_state)
            new_state.append(layer_state)

        return self.norm_f(h), new_state

    def decay_modules(self):
        """The decay-rate modules of every layer (used to check the certificate)."""
        modules = []
        for layer in self.layers:
            # a layer without memory has no decay rates to check
            if layer.mixer:
                modules.append(layer.decay)
        return modules

    def rho_max(self) -> float:
        """The largest guaranteed decay-factor bound over all layers."""
        bounds = []
        for layer in self.layers:
            # a layer without memory has no decay factor, so no bound
            if layer.mixer:
                bounds.append(layer.rho_max())

        # no memory anywhere -> nothing can grow -> report 0.0
        if len(bounds) == 0:
            return 0.0
        return max(bounds)