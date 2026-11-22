"""
Predictors that a model predictive controller can plan with. All of them have the same two functions:

    prepare(past_states, past_pushes)
        Called once per control step with the recent measured history (or None at the very first step).
        Returns (memory, predicted_now): the predictor's memory after the history (None for memory-free
        predictors) and its prediction of the CURRENT state from the last measured step. predicted_now is
        what the running correction (offset-free control) compares with the measurement.

    predict(x_now, u_plan, memory, correction)
        x_now [S, n_x], u_plan [S, N, n_u] (physical units)  ->  predicted states [S, N, n_x].
        `correction` [S, n_x] (or None) is added to every predicted step (offset-free control).

Kinds:
    PhysicsPredictor   any step function (true equations, a low-fidelity physics model, a linear model)
    LearnedPredictor   the trained model, in chained mode (works with memory on or off)
    HybridPredictor    Cena et al.'s switch: learned model when far from the target, linear model when near
"""

from __future__ import annotations

import torch


class PhysicsPredictor:
    """Plans with a physics step function: step_function(state [S, n_x], push [S, n_u]) -> next state."""

    def __init__(self, step_function):
        self.step_function = step_function

    def prepare(self, past_states, past_pushes):
        # Step 1: nothing measured yet -> nothing to prepare
        if past_states is None:
            return None, None

        # Step 2: the model's prediction of the current state, from the last measured state and push
        previous_state = past_states[:, -1]
        previous_push = past_pushes[:, -1]
        predicted_now = self.step_function(previous_state, previous_push)
        return None, predicted_now

    def predict(self, x_now, u_plan, memory, correction):
        state = x_now
        predictions = []
        for k in range(u_plan.shape[1]):
            state = self.step_function(state, u_plan[:, k])
            if correction is not None:
                state = state + correction
            predictions.append(state)
        return torch.stack(predictions, dim=1)


class LearnedPredictor:
    """Plans with the trained model, in chained mode (its own predictions fed back step after step)."""

    def __init__(self, model, norm, device):
        # Step 1: the model, frozen (only the pushes are changed by the solver)
        self.model = model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

        # Step 2: normalisation numbers as tensors
        self.x_mean = torch.tensor(norm.x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(norm.x_std, dtype=torch.float32, device=device)
        self.u_mean = torch.tensor(norm.u_mean, dtype=torch.float32, device=device)
        self.u_std = torch.tensor(norm.u_std, dtype=torch.float32, device=device)
        self.clamp = float(model.clamp)

    def _input(self, x_normalised, u_physical):
        """Model input [x, u, mask] for one step (mask = 0: the state slot holds a state)."""
        u_normalised = (u_physical - self.u_mean) / self.u_std
        mask = torch.zeros(x_normalised.shape[0], 1, device=self.device)
        return torch.cat([x_normalised, u_normalised, mask], dim=1)

    @torch.no_grad()
    def prepare(self, past_states, past_pushes):
        if past_states is None:
            return None, None

        # Step 1: inputs for every past step, in normalised units  [S, c, n_x + n_u + 1]
        x_normalised = (past_states - self.x_mean) / self.x_std
        u_normalised = (past_pushes - self.u_mean) / self.u_std
        mask = torch.zeros(x_normalised.shape[0], x_normalised.shape[1], 1, device=self.device)
        inputs = torch.cat([x_normalised, u_normalised, mask], dim=2)

        # Step 2: run the core over the history (fills the memory; with memory removed it returns no memory)
        features, memory = self.model.core.sequence(inputs)

        # Step 3: the last step's prediction = the model's prediction of the current state
        last_input = inputs[:, -1]
        increment = self.model._increment(features[:, -1], last_input)
        predicted_now = x_normalised[:, -1] + increment
        predicted_now_physical = predicted_now * self.x_std + self.x_mean
        return memory, predicted_now_physical

    def predict(self, x_now, u_plan, memory, correction):
        # Step 1: start from the measured state, in normalised units
        x = (x_now - self.x_mean) / self.x_std
        correction_normalised = None
        if correction is not None:
            correction_normalised = correction / self.x_std

        # Step 2: chain the model's own predictions over the plan
        predictions = []
        for k in range(u_plan.shape[1]):
            model_input = self._input(x, u_plan[:, k])
            features, memory = self.model.core.step(model_input, memory)
            x = x + self.model._increment(features, model_input)
            if correction_normalised is not None:
                x = x + correction_normalised
            x = torch.clamp(x, -self.clamp, self.clamp)
            predictions.append(x * self.x_std + self.x_mean)
        return torch.stack(predictions, dim=1)


class HybridPredictor:
    """
    Cena et al.'s hybrid controller (arXiv 2602.15954): plan with the LEARNED model while the pointing error is
    above `switch_angle_degrees`, and with the LINEAR model below it. The choice is made per spacecraft from
    the measured state at the start of each plan.
    """

    def __init__(self, learned: LearnedPredictor, linear: PhysicsPredictor, angle_function,
                 switch_angle_degrees: float):
        self.learned = learned
        self.linear = linear
        self.angle_function = angle_function
        self.switch_angle = float(switch_angle_degrees)
        self.use_linear = None              # [S] True = linear model in this control step

    def prepare(self, past_states, past_pushes):
        learned_memory, learned_now = self.learned.prepare(past_states, past_pushes)
        _, linear_now = self.linear.prepare(past_states, past_pushes)
        if learned_now is None:
            return (learned_memory, None), None
        # the prediction of "now" comes from whichever model planned the previous step
        if self.use_linear is None:
            predicted_now = learned_now
        else:
            predicted_now = torch.where(self.use_linear.unsqueeze(1), linear_now, learned_now)
        return (learned_memory, None), predicted_now

    def decide(self, x_now):
        """Choose the model for this control step: linear when the pointing error is small."""
        self.use_linear = self.angle_function(x_now) < self.switch_angle

    def predict(self, x_now, u_plan, memory, correction):
        learned_memory = memory[0] if memory is not None else None
        learned_path = self.learned.predict(x_now, u_plan, learned_memory, correction)
        linear_path = self.linear.predict(x_now, u_plan, None, correction)
        choose_linear = self.use_linear.unsqueeze(1).unsqueeze(2)       # [S, 1, 1]
        return torch.where(choose_linear, linear_path, learned_path)
