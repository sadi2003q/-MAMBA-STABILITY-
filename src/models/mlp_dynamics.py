"""Memory-free multilayer perceptron core.

The next change depends ONLY on the current step's input [position, velocity, push, mask]: there is no memory
carried from one step to the next. Brought over from the chain-mamba branch (same layers, same sizes), rewritten
one step per line.

It has the same plug-in shape as every other core in this folder:
    out_dim;  sequence(inp, state=None) -> (y, state);  step(inp, state=None) -> (y, state);  decay_modules()
Its state is always None, so the context warm-up does nothing.

The anchor and the increment scaling are NOT here: they live in src/models/dynamics.py and are switched on
from the config (readout: linear_skip / increment_normalisation).
"""

from __future__ import annotations

import torch.nn as nn


class MLPCore(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 54, layers: int = 2):
        super().__init__()

        # Step 1: build the list of layers one by one
        #   layer 1: in_dim (4)  -> hidden (54), then SiLU
        #   layer 2: hidden (54) -> hidden (54), then SiLU
        all_layers = []
        size_coming_in = in_dim
        for layer_number in range(layers):
            linear_layer = nn.Linear(size_coming_in, hidden)
            activation = nn.SiLU()
            all_layers.append(linear_layer)
            all_layers.append(activation)
            size_coming_in = hidden          # the next layer receives this layer's output

        # Step 2: chain the layers so they run one after another
        self.net = nn.Sequential(*all_layers)

        # Step 3: the read-out in dynamics.py needs to know how many features we give it
        self.out_dim = hidden

    def sequence(self, inp, state=None, parallel=None):
        """
        Run over a whole sequence.  inp [B, T, in_dim] -> features [B, T, hidden].
        Every time step is handled on its own (no memory), so all steps can go through at once.
        """
        features = self.net(inp)
        no_memory = None
        return features, no_memory

    def step(self, inp, state=None):
        """
        Run for ONE time step (used in chained prediction).  inp [B, in_dim] -> features [B, hidden].
        """
        features = self.net(inp)
        no_memory = None
        return features, no_memory

    def decay_modules(self):
        # No Mamba-style decay factors in this network, so nothing to report
        return []
