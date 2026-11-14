"""
Predictors for model predictive control.

A predictor answers ONE question:
    "If the oscillator is now at state x_now, and I apply the pushes u_0, u_1, ..., u_(N-1),
     where will it be after each push?"

    predict(x_now, u_plan)  ->  predicted states
        x_now   [S, 2]      current state (position, velocity) of S oscillators
        u_plan  [S, N]      N planned pushes for each oscillator
        returns [S, N, 2]   predicted state after push 1, 2, ..., N
"""

from __future__ import annotations

import torch

class TrueModelPredictor:
    """Plans with the TRUE Van der Pol equations: the best possible predictor."""
    def __init__(self,
                 mu:float,
                 dt: float,
                 include_minus_x1:bool = True
    ):
        self.mu = mu
        self.dt = dt

        # the "- x1" term can be switched off in the config (we keep it on)
        if include_minus_x1: self.k = 1.0
        else: self.k = 0.0

    def predict(self,
                x_now:torch.Tensor, # shape : (S, 2)
                u_plan:torch.Tensor # shape : (s, N) --> N is how many push i want to set
    ) -> torch.Tensor:

        number_of_pushes = u_plan.shape[1]
        position = x_now[:,0]
        velocity = x_now[:,1]

        predicted_states = []
        for k in range(number_of_pushes):
            push = u_plan[:,k]
            position_change_rate = velocity
            velocity_change_rate = self.mu * (1.0 - position * position) * velocity - self.k * position + push

            new_position = position + self.dt * position_change_rate
            new_velocity = velocity + self.dt * velocity_change_rate

            position = new_position
            velocity = new_velocity

            state_after_push = torch.stack([position, velocity], dim=1)
            predicted_states.append(state_after_push)

        return torch.stack(predicted_states, dim=1)


class LearnedModelPredictor:
    """
    Plans with the trained model in CHAINED mode: it gets the true current state once,
    then it is fed its own prediction for every next step (exactly what it was trained for).

    The model works in NORMALISED units; the controller works in PHYSICAL units.
    This class converts in, runs the model step by step, and converts back.
    """
    def __init__(self,
                 model,
                 norm,
                 device: torch.device
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.x_mean = torch.tensor(norm.x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(norm.x_std, dtype=torch.float32, device=device)
        self.u_mean = torch.tensor(norm.u_mean, dtype=torch.float32, device=device)
        self.u_std = torch.tensor(norm.u_std, dtype=torch.float32, device=device)

        self.clamp = float(model.clamp)


        self.has_memory = False
        for layer in getattr(model.core, "layers", []):
            if getattr(layer, "mixer", False):
                self.has_memory = True


    def predict(self,
                x_now:torch.Tensor,
                u_plan:torch.Tensor
    ) -> torch.Tensor:

        number_of_systems = x_now.shape[0]
        number_of_pushes = u_plan.shape[1]

        x = (x_now - self.x_mean) / self.x_std
        mask = torch.zeros(number_of_systems, 1, dtype=torch.float32, device=self.device)
        memory = None

        # Step 5: chained prediction, one push at a time
        predicted_states = []
        for k in range(number_of_pushes):
            # 5a: this push for every oscillator, as a column [S, 1], in normalised units
            push = u_plan[:, k:k + 1]
            push_normalised = (push - self.u_mean) / self.u_std

            # 5b: the model's input: [state, push, mask] -> [S, 4]  (same order as in training)
            model_input = torch.cat([x, push_normalised, mask], dim=1)

            # 5c: one step of the core (for the memory-removed model `memory` stays None)
            features, memory = self.model.core.step(model_input, memory)

            # 5d: the read-out gives the CHANGE of the state (the increment), as in training
            increment = self.model._increment(features, model_input)

            # 5e: next state = current state + change, then the safety clamp
            x = x + increment
            x = torch.clamp(x, -self.clamp, self.clamp)

            # 5f: store the prediction in physical units
            x_physical = x * self.x_std + self.x_mean
            predicted_states.append(x_physical)

        # Step 6: [S, N, 2]
        return torch.stack(predicted_states, dim=1)































