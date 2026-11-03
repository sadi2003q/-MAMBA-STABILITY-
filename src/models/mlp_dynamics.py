"""Memory-free multilayer perceptron core: the next increment depends only on the current step's input
[x_t, u_t, mask_t]. Control arm for "does Mamba's memory (or any memory) matter?". Same core interface as
the other cores; its state is always None, so context warm-up does nothing.
"""

from __future__ import annotations

import torch.nn as nn


class MLPCore(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 54, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*mods)
        self.out_dim = hidden

    def sequence(self, inp, state=None, parallel=None):
        return self.net(inp), None

    def step(self, inp, state=None):
        return self.net(inp), None

    def decay_modules(self):
        return []
