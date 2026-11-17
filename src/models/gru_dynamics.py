"""Gated recurrent unit core (no certificate). State = h, shape [1, B, hidden].

A second standard recurrent baseline next to the Long Short-Term Memory network.
It has the same plug-in shape as every other core in this folder:
    out_dim;  sequence(inp, state=None) -> (y, state);  step(inp, state=None) -> (y, state);  decay_modules()

The anchor and the increment scaling are NOT here: they live in src/models/dynamics.py and are switched on
from the experiment config (readout: linear_skip / increment_normalisation), exactly as for the
Long Short-Term Memory network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUCore(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int = 29):
        super().__init__()

        # Step 1: the gated recurrent unit itself (PyTorch's built-in layer)
        #   in_dim      = 4 numbers per step: [position, velocity, push, mask]
        #   hidden_size = size of the memory vector h
        self.gru = nn.GRU(in_dim, hidden_size, batch_first=True)

        # Step 2: the read-out in dynamics.py needs to know how many features we give it
        self.out_dim = hidden_size

    def sequence(self, inp, state=None, parallel: bool | None = None):
        """
        Run the network over a whole sequence.
            inp   [B, T, in_dim]
            state memory from before, or None at the very start
        Returns (features [B, T, hidden], memory after the last step).
        """
        # Option A: step-by-step reference path (used by some checks)
        if parallel is False:
            all_outputs = []
            number_of_steps = inp.shape[1]
            for t in range(number_of_steps):
                one_step_input = inp[:, t]
                output, state = self.step(one_step_input, state)
                all_outputs.append(output)
            stacked_outputs = torch.stack(all_outputs, dim=1)
            return stacked_outputs, state

        # Option B: normal fast path, the whole sequence in one call
        outputs, final_state = self.gru(inp, state)
        return outputs, final_state

    def step(self, inp, state=None):
        """
        Run the network for ONE time step (used in chained prediction).
            inp [B, in_dim]
        Returns (features [B, hidden], new memory).
        """
        # Step 1: nn.GRU expects a time axis, so add one of length 1
        inp_with_time_axis = inp.unsqueeze(1)          # [B, 1, in_dim]

        # Step 2: one step of the gated recurrent unit
        output, new_state = self.gru(inp_with_time_axis, state)

        # Step 3: remove the time axis again
        output_one_step = output[:, 0]                 # [B, hidden]

        return output_one_step, new_state

    def decay_modules(self):
        # No Mamba-style decay factors in this network, so nothing to report
        return []
