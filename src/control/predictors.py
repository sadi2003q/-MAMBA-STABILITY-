"""
Predictors for model predictive control.

A predictor answers ONE question:
    "If the oscillator is now at state x_now, and I apply the pushes u_0, u_1, ..., u_(N-1),
     where will it be after each push?"

Both predictors in this file have the same method, so the controller can use either one:

    predict(x_now, u_plan)  ->  predicted states

        x_now   [S, 2]      current state (position, velocity) of S oscillators, physical units
        u_plan  [S, N]      N planned pushes for each oscillator, physical units
        returns [S, N, 2]   predicted state after push 1, after push 2, ..., after push N

S is the number of oscillators we control at the same time (100 starting points -> S = 100).
They never affect each other: row s of the answer depends only on row s of x_now and u_plan.
This is the same "batch" idea as in training (many windows at once), just used for control.

Everything is written with PyTorch, so the solver can ask:
    "if I change each planned push a tiny bit, how does the cost change?"  (the gradient)
PyTorch works this out automatically from the calculations below.
"""

from __future__ import annotations

import torch


# ==========================================================================================================
# 1. The perfect predictor: the true Van der Pol equations
# ==========================================================================================================
class TrueModelPredictor:
    """
    Plans with the TRUE equations (the same ones that generated the training data).
    A controller using this predictor is the best we can hope for: any gap between it and the
    learned controller is caused by prediction mistakes of the learned model.

        dx1/dt = x2
        dx2/dt = mu (1 - x1^2) x2 - x1 + u
        one step (forward Euler):  x_next = x + dt * (dx/dt)
    """

    def __init__(self, mu: float, dt: float, include_minus_x1: bool = True):
        self.mu = float(mu)
        self.dt = float(dt)

        # the "- x1" term can be switched off in the config (we keep it on, see vanderpol.yaml)
        if include_minus_x1:
            self.k = 1.0
        else:
            self.k = 0.0

    def predict(self, x_now: torch.Tensor, u_plan: torch.Tensor) -> torch.Tensor:
        # Step 1: how many pushes are planned
        number_of_pushes = u_plan.shape[1]

        # Step 2: split the state into position and velocity (each [S])
        position = x_now[:, 0]
        velocity = x_now[:, 1]

        # Step 3: apply the pushes one after another
        predicted_states = []
        for k in range(number_of_pushes):
            push = u_plan[:, k]                                            # [S]

            # 3a: how fast position and velocity change right now
            position_change_rate = velocity
            velocity_change_rate = self.mu * (1.0 - position * position) * velocity - self.k * position + push

            # 3b: one Euler step of size dt
            new_position = position + self.dt * position_change_rate
            new_velocity = velocity + self.dt * velocity_change_rate

            # 3c: the new state becomes the current state for the next push
            position = new_position
            velocity = new_velocity

            # 3d: store this predicted state as [S, 2]
            state_after_push = torch.stack([position, velocity], dim=1)
            predicted_states.append(state_after_push)

        # Step 4: stack the N states into one tensor [S, N, 2]
        return torch.stack(predicted_states, dim=1)


# ==========================================================================================================
# 2. The learned predictor: the trained chained model (memory removed)
# ==========================================================================================================
class LearnedModelPredictor:
    """
    Plans with the trained model, used in CHAINED mode: the model is given the true current state once,
    then it is fed its own prediction for every next step. This is exactly what it was trained for.

    The model works in NORMALISED units (each variable shifted and scaled to about mean 0, spread 1,
    using the training data). The controller works in PHYSICAL units. So this class:
        1. converts the current state and the pushes to normalised units,
        2. runs the model one step at a time (the same calculation as DynamicsModel.chained),
        3. converts every prediction back to physical units.

    About the memory: for the memory-removed model there is nothing to remember, so each step depends only
    on the current (predicted) state and push. For a model WITH memory we start every plan with an empty
    memory (Cevaal et al. do the same); a model with memory would do better with a warm-up on past states.
    """

    def __init__(self, model, norm, device: torch.device):
        # Step 1: keep the model, in evaluation mode, on the chosen device
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Step 2: the model's weights stay fixed; only the pushes are changed by the solver
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # Step 3: the normalisation numbers, as tensors
        #   x_mean, x_std: 2 numbers each (position, velocity)
        #   u_mean, u_std: 1 number each (push)
        self.x_mean = torch.tensor(norm.x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(norm.x_std, dtype=torch.float32, device=device)
        self.u_mean = torch.tensor(norm.u_mean, dtype=torch.float32, device=device)
        self.u_std = torch.tensor(norm.u_std, dtype=torch.float32, device=device)

        # Step 4: the same safety clamp as in training (normalised units)
        self.clamp = float(model.clamp)

        # Step 5: does the model have memory layers? (only used for a message in the log)
        self.has_memory = False
        for layer in getattr(model.core, "layers", []):
            if getattr(layer, "mixer", False):
                self.has_memory = True

    def predict(self, x_now: torch.Tensor, u_plan: torch.Tensor) -> torch.Tensor:
        # Step 1: sizes
        number_of_systems = x_now.shape[0]
        number_of_pushes = u_plan.shape[1]

        # Step 2: current state in normalised units  [S, 2]
        x = (x_now - self.x_mean) / self.x_std

        # Step 3: the "empty slot" flag is always 0 here: the state slot always holds a state
        mask = torch.zeros(number_of_systems, 1, dtype=torch.float32, device=self.device)

        # Step 4: empty memory at the start of every plan (None = nothing remembered)
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
