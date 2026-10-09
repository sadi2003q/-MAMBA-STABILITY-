"""Long Short-Term Memory network core (no certificate). State = (h, c), each [1, B, hidden]."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCore(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int = 25):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, batch_first=True)
        self.out_dim = hidden_size

    def sequence(self, inp, state=None, parallel: bool | None = None):
        if parallel is False:                     # step-by-step reference path
            ys = []
            for t in range(inp.shape[1]):
                y, state = self.step(inp[:, t], state)
                ys.append(y)
            return torch.stack(ys, dim=1), state
        return self.lstm(inp, state)

    def step(self, inp, state=None):
        y, state = self.lstm(inp.unsqueeze(1), state)
        return y[:, 0], state

    def decay_modules(self):
        return []
