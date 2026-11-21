"""
The three learned drone models, and one helper that runs any of them many steps ahead.

    A. chained predictor   = Chain-Mamba core with the memory removed (src/models/mamba_chain.py, mixer off)
                             + anchor (linear skip) + increment scaling (src/models/dynamics.py).
                             Trained with the chained-prediction recipe.
    B. same model, trained one step at a time (teacher forcing).
    C. plain one-step network: 2 hidden layers of 64 units, predicts the change of the state.

Every model works in NORMALISED units: each number is shifted by its average and divided by its
spread (standard deviation) from the training flights, so all numbers have a similar size.

Inputs at each step:
    state    12 numbers (position, velocity, angles, turning rates)
    commands 8 numbers  (the command applied during this step + the command applied during the step before,
                         because the motors lag and the thrust now still depends on the last command)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dynamics import DynamicsModel
from src.models.mamba_chain import ChainMambaCore


N_STATE = 12
N_COMMAND = 8


# ======================================================================================================
# Model A and B: the user's memory-free chained predictor
# ======================================================================================================
def build_chained_predictor():
    # Step 1: the core: two gated feed-forward layers, memory part switched OFF (mixer = False)
    input_size = DynamicsModel.in_dim(N_STATE, N_COMMAND)     # 12 + 8 + 1 = 21
    core = ChainMambaCore(
        in_dim=input_size,
        d_model=24,
        n_layers=2,
        ffn_hidden=48,
        mixer=False,
    )

    # Step 2: the read-out with the anchor (linear skip) and the increment scaling
    model = DynamicsModel(
        core,
        n_x=N_STATE,
        n_u=N_COMMAND,
        linear_skip=True,
        increment_normalisation=True,
    )
    return model


# ======================================================================================================
# Model C: a plain one-step network
# ======================================================================================================
class PlainOneStepNetwork(nn.Module):
    """next state = state + (network output) x (typical size of one step's change)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.layer1 = nn.Linear(N_STATE + N_COMMAND, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, N_STATE)
        self.register_buffer("increment_scale", torch.ones(N_STATE))

    def set_increment_scale(self, scale):
        self.increment_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def next_state(self, x, u):
        # Step 1: put state and commands side by side
        both = torch.cat([x, u], dim=-1)

        # Step 2: two hidden layers with the SiLU activation
        h = F.silu(self.layer1(both))
        h = F.silu(self.layer2(h))

        # Step 3: the change of the state, scaled to its typical size
        change = self.layer3(h) * self.increment_scale
        return x + change


def build_plain_network():
    return PlainOneStepNetwork(hidden=64)


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total


# ======================================================================================================
# One step and many steps, for any of the models (normalised units)
# ======================================================================================================
def one_step(model, x, u):
    """x [B, 12], u [B, 8] -> next state [B, 12]."""
    if isinstance(model, PlainOneStepNetwork):
        return model.next_state(x, u)

    # the chained predictor: same maths as DynamicsModel.chained() for one step (memory-free core)
    inp = model._inp(x, u, 0.0)
    y, _ = model.core.step(inp, None)
    return x + model._increment(y, inp)


def roll_out(model, x_start, u_sequence, disturbance=None):
    """
    Chained prediction: feed the model its own predictions.
        x_start    [B, 12]
        u_sequence [B, H, 8]
        disturbance [B, 12] or None: a fixed correction added after EVERY predicted step
    Returns predictions [B, H, 12].
    """
    predictions = []
    x = x_start
    horizon = u_sequence.shape[1]
    for k in range(horizon):
        x = one_step(model, x, u_sequence[:, k])
        if disturbance is not None:
            x = x + disturbance
        predictions.append(x)
    return torch.stack(predictions, dim=1)
