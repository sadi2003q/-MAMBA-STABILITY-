"""
Physics anchor: your chained predictor, with a cheap physics step added to its read-out (pilot B2).

Your model now:        next state = state + increment(network)                    (all normalised units)
With the physics anchor:  next state = state + cheap physics change + increment(network)

    cheap physics change = (cheap_step(state, push) - state) / state standard deviation
    increment(network)   = (head(features) + linear anchor) x remainder scale

The network (the same memory-removed core, head and linear anchor as before) only has to learn what the cheap
step gets wrong. The remainder scale is the standard deviation of that remainder in the training data (instead
of the standard deviation of the full step change), so the network's raw output still has unit size.

Everything else is unchanged: chained training, the recipe, chained prediction in the controller. The cheap step
has no trainable numbers, so it adds no parameters.
"""

from __future__ import annotations

import torch

from src.models.dynamics import DynamicsModel
from src.models.registry import build_core


class PhysicsAnchoredModel(DynamicsModel):
    def attach_physics(self, step_function, norm) -> None:
        """
        step_function(state [..., n_x], push [..., n_u]) -> next state [..., n_x], in physical units.
        norm: the normalisation numbers of the training data.
        """
        # Step 1: the cheap physics step (not a trainable part of the model)
        self.step_function = step_function

        # Step 2: normalisation numbers, kept with the model so they move to the same device.
        # persistent=False -> not saved in checkpoints (they come from the data every time)
        self.register_buffer("x_mean", torch.tensor(norm.x_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("x_std", torch.tensor(norm.x_std, dtype=torch.float32), persistent=False)
        self.register_buffer("u_mean", torch.tensor(norm.u_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("u_std", torch.tensor(norm.u_std, dtype=torch.float32), persistent=False)

    def _increment(self, y, inp):
        """Increment in normalised units = cheap physics change + the network's (scaled) increment."""
        # Step 1: the network's part, exactly as in your model (head + linear anchor, times the scale)
        learned_part = super()._increment(y, inp)

        # Step 2: take the state and the push out of the step input [x, u, mask], back to physical units
        x_normalised = inp[..., 0:self.n_x]
        u_normalised = inp[..., self.n_x:self.n_x + self.n_u]
        x_physical = x_normalised * self.x_std + self.x_mean
        u_physical = u_normalised * self.u_std + self.u_mean

        # Step 3: the cheap physics step, and its change in normalised units
        next_physical = self.step_function(x_physical, u_physical)
        physics_part = (next_physical - x_physical) / self.x_std

        # Step 4: both together
        return physics_part + learned_part


def build_physics_anchored_model(cfg: dict, n_x: int, n_u: int, step_function, norm) -> PhysicsAnchoredModel:
    """The same core and read-out settings as src/models/registry.build_model, plus the physics anchor."""
    core = build_core(cfg["model"], DynamicsModel.in_dim(n_x, n_u))
    readout = cfg.get("readout", {}) or {}
    model = PhysicsAnchoredModel(core, n_x, n_u, clamp=cfg.get("training", {}).get("clamp_state", 50.0),
                                 linear_skip=readout.get("linear_skip", False),
                                 increment_normalisation=readout.get("increment_normalisation", False))
    model.attach_physics(step_function, norm)
    return model
