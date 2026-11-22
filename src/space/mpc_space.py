"""
Model predictive control for the spacecraft (the same design as src/control/mpc.py, extended to 6 states and
3 pushes, and to a predictor WITH memory that is warmed up on recent measurements).

Every control step (every dt seconds):
    1. WARM UP (learned predictor only): feed the last `context_len` measured states and pushes into Chain-Mamba.
       This fills its memory (so it can work out the current inertia) and gives its prediction of the
       CURRENT state.
    2. RUNNING CORRECTION (offset-free control, optional): mismatch = measured current state - predicted current
       state. correction <- (1 - filter) x correction + filter x mismatch. It is added to every predicted step.
    3. PLAN: Adam changes the N planned pushes (3 per step) to lower the cost; pushes are clipped to the limit.
       Warm start from last time's plan, shifted by one step.
    4. APPLY the first planned push.

Cost of one plan (the same shape as your Van der Pol controller):
    sum over steps 1..N-1 of  q_attitude |4 sigma_k|^2 + q_spin |omega_k|^2          (4 sigma ~ angle in radians)
    + at step N:              p_attitude |4 sigma_N|^2 + p_spin |omega_N|^2
    + sum over steps 0..N-1 of r |(u_k - u_(k-1)) / u_max|^2
"""

from __future__ import annotations

import torch

from src.systems.spacecraft import Spacecraft


# ==========================================================================================================
# Predictors
# ==========================================================================================================
class TrueSpacecraftPredictor:
    """Plans with the TRUE equations and the TRUE current inertia (the best possible reference)."""

    def __init__(self, system: Spacecraft):
        self.system = system
        self.inertia_now = None          # set by the closed loop before every planning step

    def prepare(self, past_states, past_pushes, device):
        """Nothing to warm up. The prediction of the current state uses the true equations from the last step."""
        if past_states is None:
            return None, None
        previous_state = past_states[:, -1]
        previous_push = past_pushes[:, -1]
        predicted_now = self.system.step(previous_state, previous_push, self.inertia_now)
        return None, predicted_now

    def predict(self, x_now, u_plan, memory, correction):
        """x_now [S, 6], u_plan [S, N, 3] -> predicted states [S, N, 6] (physical units)."""
        state = x_now
        predictions = []
        for k in range(u_plan.shape[1]):
            state = self.system.step(state, u_plan[:, k], self.inertia_now)
            if correction is not None:
                state = state + correction
            predictions.append(state)
        return torch.stack(predictions, dim=1)


class LearnedSpacecraftPredictor:
    """Plans with the trained Chain-Mamba (classical or quantum), in chained mode, after a warm-up."""

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
        u_normalised = (u_physical - self.u_mean) / self.u_std
        mask = torch.zeros(x_normalised.shape[0], 1, device=self.device)
        return torch.cat([x_normalised, u_normalised, mask], dim=1)

    @torch.no_grad()
    def prepare(self, past_states, past_pushes, device):
        """
        Warm-up on the recent measured history.
            past_states [S, c, 6], past_pushes [S, c, 3] = the last c measured states and the pushes applied
            from them (c = 0 at the very first step -> None).
        Returns (memory after the last past step, prediction of the CURRENT state in physical units).
        """
        if past_states is None:
            return None, None

        # Step 1: inputs for every past step, in normalised units
        x_normalised = (past_states - self.x_mean) / self.x_std
        u_normalised = (past_pushes - self.u_mean) / self.u_std
        mask = torch.zeros(x_normalised.shape[0], x_normalised.shape[1], 1, device=self.device)
        inputs = torch.cat([x_normalised, u_normalised, mask], dim=2)                   # [S, c, 10]

        # Step 2: run the core over the whole history at once (fills the memory)
        features, memory = self.model.core.sequence(inputs)

        # Step 3: the last step's prediction = the model's prediction of the current state
        last_input = inputs[:, -1]
        increment = self.model._increment(features[:, -1], last_input)
        predicted_now = x_normalised[:, -1] + increment
        predicted_now_physical = predicted_now * self.x_std + self.x_mean
        return memory, predicted_now_physical

    def predict(self, x_now, u_plan, memory, correction):
        """Chained prediction from x_now with the warmed-up memory. Returns [S, N, 6] in physical units."""
        x = (x_now - self.x_mean) / self.x_std
        correction_normalised = None
        if correction is not None:
            correction_normalised = correction / self.x_std

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


# ==========================================================================================================
# The controller
# ==========================================================================================================
class SpacecraftMPC:
    def __init__(self, predictor, controller_cfg: dict, solver_cfg: dict, u_max: float, offset_free: bool, device):
        self.predictor = predictor
        self.device = device
        self.horizon = int(controller_cfg["horizon"])
        self.q_attitude = float(controller_cfg["q_attitude"])
        self.q_spin = float(controller_cfg["q_spin"])
        self.p_attitude = float(controller_cfg["p_attitude"])
        self.p_spin = float(controller_cfg["p_spin"])
        self.r_change = float(controller_cfg["r_input_change"])
        self.u_max = float(u_max)
        self.iterations = int(solver_cfg["iterations"])
        self.first_iterations = int(solver_cfg["first_iterations"])
        self.learning_rate = float(solver_cfg["learning_rate"])       # in units of u_max
        self.offset_free = bool(offset_free)
        self.correction_filter = float(controller_cfg["correction_filter"])
        self.reset()

    def reset(self) -> None:
        self.previous_plan = None          # [S, N, 3] in units of u_max
        self.correction = None             # [S, 6] physical units

    def cost(self, x_now, plan_scaled, u_previous, memory):
        """One cost number per spacecraft [S]. plan_scaled [S, N, 3] is the plan divided by u_max."""
        u_plan = plan_scaled * self.u_max
        predicted = self.predictor.predict(x_now, u_plan, memory, self.correction)        # [S, N, 6]

        total = torch.zeros(x_now.shape[0], device=self.device)
        last_step = self.horizon - 1
        for k in range(self.horizon):
            angle_part = ((4.0 * predicted[:, k, 0:3]) ** 2).sum(dim=1)
            spin_part = (predicted[:, k, 3:6] ** 2).sum(dim=1)
            if k == last_step:
                total = total + self.p_attitude * angle_part + self.p_spin * spin_part
            else:
                total = total + self.q_attitude * angle_part + self.q_spin * spin_part

        push_before = u_previous / self.u_max
        for k in range(self.horizon):
            change = plan_scaled[:, k] - push_before
            total = total + self.r_change * (change ** 2).sum(dim=1)
            push_before = plan_scaled[:, k]
        return total

    def solve(self, x_now, u_previous, past_states, past_pushes):
        """
        x_now [S, 6], u_previous [S, 3], past_states / past_pushes: recent history (or None at the first step).
        Returns the push to apply now [S, 3].
        """
        number_of_systems = x_now.shape[0]

        # Step 1: warm-up and the prediction of the current state
        memory, predicted_now = self.predictor.prepare(past_states, past_pushes, self.device)

        # Step 2: running correction (offset-free control)
        if self.offset_free and predicted_now is not None:
            mismatch = (x_now - predicted_now).detach()
            if self.correction is None:
                self.correction = self.correction_filter * mismatch
            else:
                self.correction = (1.0 - self.correction_filter) * self.correction + self.correction_filter * mismatch

        # Step 3: starting guess (shifted old plan, or zeros) and number of rounds
        if self.previous_plan is None:
            plan = torch.zeros(number_of_systems, self.horizon, 3, device=self.device)
            rounds = self.first_iterations
        else:
            plan = torch.cat([self.previous_plan[:, 1:], self.previous_plan[:, -1:]], dim=1).clone()
            rounds = self.iterations
        plan.requires_grad_(True)
        optimizer = torch.optim.Adam([plan], lr=self.learning_rate)

        # Step 4: improvement rounds; remember each spacecraft's best plan
        best_plan = plan.detach().clone()
        best_cost = torch.full((number_of_systems,), float("inf"), device=self.device)
        for _ in range(rounds):
            optimizer.zero_grad()
            cost_each = self.cost(x_now, plan, u_previous, memory)
            with torch.no_grad():
                better = cost_each < best_cost
                best_cost = torch.where(better, cost_each, best_cost)
                best_plan[better] = plan.detach()[better]
            cost_each.sum().backward()
            optimizer.step()
            with torch.no_grad():
                plan.clamp_(-1.0, 1.0)

        # Step 5: check the plan after the last round too
        with torch.no_grad():
            cost_each = self.cost(x_now, plan, u_previous, memory)
            better = cost_each < best_cost
            best_plan[better] = plan.detach()[better]

        # Step 6: keep the plan for next time, return its first push in physical units
        self.previous_plan = best_plan
        return best_plan[:, 0] * self.u_max
