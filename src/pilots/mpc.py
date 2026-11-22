"""
Model predictive control shared by both space pilots (the same design as src/control/mpc.py and the
space-pilot branch's controller; only the cost is supplied by each pilot).

Every control step:
    1. PREPARE: the predictor looks at the last measured step and predicts the CURRENT state.
    2. RUNNING CORRECTION (offset-free control, optional, as in mpc01c):
           mismatch   = measured current state - predicted current state
           correction = (1 - filter) x correction + filter x mismatch
       The correction is added to every predicted step.
    3. PLAN: Adam changes the N planned pushes to lower the cost; pushes are clipped to the limit.
       Warm start from last time's plan, shifted by one step.
    4. APPLY the first planned push.

The pilot supplies   cost_function(predicted [S, N, n_x], plan_scaled [S, N, n_u], previous_scaled [S, n_u],
                                   reference [S, N, n_x] or None)  ->  cost per spacecraft [S]
where plan_scaled = planned pushes divided by the push limit (so every value lies in [-1, 1]).
"""

from __future__ import annotations

import time

import torch


class PilotMPC:
    def __init__(self, predictor, cost_function, horizon: int, n_u: int, u_max: float, solver_cfg: dict,
                 offset_free: bool, correction_filter: float, device):
        self.predictor = predictor
        self.cost_function = cost_function
        self.horizon = int(horizon)
        self.n_u = int(n_u)
        self.u_max = float(u_max)
        self.iterations = int(solver_cfg["iterations"])
        self.first_iterations = int(solver_cfg["first_iterations"])
        self.learning_rate = float(solver_cfg["learning_rate"])       # in units of the push limit
        self.offset_free = bool(offset_free)
        self.correction_filter = float(correction_filter)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.previous_plan = None          # [S, N, n_u] in units of the push limit
        self.correction = None             # [S, n_x] physical units
        self.planning_seconds = []         # time spent planning, one number per control step
        if hasattr(self.predictor, "use_linear"):
            self.predictor.use_linear = None   # the hybrid predictor forgets its last choice

    def total_cost(self, x_now, plan_scaled, u_previous, memory, reference):
        """Predict the path for this plan and score it. [S]"""
        u_plan = plan_scaled * self.u_max
        predicted = self.predictor.predict(x_now, u_plan, memory, self.correction)
        previous_scaled = u_previous / self.u_max
        return self.cost_function(predicted, plan_scaled, previous_scaled, reference)

    def solve(self, x_now, u_previous, past_states, past_pushes, reference=None):
        """
        x_now [S, n_x], u_previous [S, n_u], past_states / past_pushes: recent history (None at the first step),
        reference [S, N, n_x] or None. Returns the push to apply now [S, n_u] (physical units).
        """
        started = time.perf_counter()
        number_of_systems = x_now.shape[0]

        # Step 1: prepare (and, for the hybrid predictor, choose the model for this step)
        memory, predicted_now = self.predictor.prepare(past_states, past_pushes)
        if hasattr(self.predictor, "decide"):
            self.predictor.decide(x_now)

        # Step 2: running correction (offset-free control)
        if self.offset_free and predicted_now is not None:
            mismatch = (x_now - predicted_now).detach()
            if self.correction is None:
                self.correction = self.correction_filter * mismatch
            else:
                self.correction = (1.0 - self.correction_filter) * self.correction + self.correction_filter * mismatch

        # Step 3: starting guess (shifted old plan, or zeros) and number of rounds
        if self.previous_plan is None:
            plan = torch.zeros(number_of_systems, self.horizon, self.n_u, device=self.device)
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
            cost_each = self.total_cost(x_now, plan, u_previous, memory, reference)
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
            cost_each = self.total_cost(x_now, plan, u_previous, memory, reference)
            better = cost_each < best_cost
            best_plan[better] = plan.detach()[better]

        # Step 6: keep the plan for next time, return its first push in physical units
        self.previous_plan = best_plan
        self.planning_seconds.append(time.perf_counter() - started)
        return best_plan[:, 0] * self.u_max
