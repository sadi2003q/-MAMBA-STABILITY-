"""
Model predictive control (beginner-friendly version).

----------------------------------------------------------------------------------------------------------
The idea in four lines
----------------------------------------------------------------------------------------------------------
At every control step (every 0.1 seconds):
    1. PLAN:   find N future pushes that make the predicted future look best (lowest cost)
    2. APPLY:  send only the FIRST push to the real oscillator
    3. MEASURE: read the oscillator's new state
    4. REPEAT: plan again from the new state (the old plan, shifted by one step, is the starting guess)

----------------------------------------------------------------------------------------------------------
The cost of one plan (the same form as Cevaal, de Jong and Lazar, 2026)
----------------------------------------------------------------------------------------------------------
    cost =   sum over steps k = 1 .. N-1 of   Q_pos (position_k - target_pos)^2 + Q_vel (velocity_k - target_vel)^2
           + at the last step N:               P_pos (position_N - target_pos)^2 + P_vel (velocity_N - target_vel)^2
           + sum over steps k = 0 .. N-1 of   R (u_k - u_(k-1))^2          (u_(-1) = the push applied last time)

    Big Q, P  -> "get to the target quickly".
    Big R     -> "do not jerk the push around".

----------------------------------------------------------------------------------------------------------
How the plan is found (the "solver")
----------------------------------------------------------------------------------------------------------
Gradient descent with Adam, the same optimiser used for training, but here the numbers being changed are
the N planned PUSHES, not the model's weights. After every round each push is clipped back into the allowed
range [-u_max, u_max] (this is called "projected gradient descent").

Many oscillators at once: the total cost is the SUM of every oscillator's own cost. Oscillator s's pushes
only appear in oscillator s's cost, so its gradient only comes from its own cost; and Adam updates every
number on its own. So solving 100 plans together gives exactly the same plans as solving them one by one,
just faster.
"""

from __future__ import annotations

import torch


class MPCController:
    def __init__(self, predictor, controller_cfg: dict, solver_cfg: dict, device: torch.device):
        # Step 1: the predictor (true equations or learned model) used to imagine the future
        self.predictor = predictor
        self.device = device

        # Step 2: controller settings (see configs/control/mpc01_stabilise.yaml)
        self.horizon = int(controller_cfg["horizon"])
        self.q_position = float(controller_cfg["q_position"])
        self.q_velocity = float(controller_cfg["q_velocity"])
        self.p_position = float(controller_cfg["p_position"])
        self.p_velocity = float(controller_cfg["p_velocity"])
        self.r_input_change = float(controller_cfg["r_input_change"])
        self.u_max = float(controller_cfg["u_max"])

        # Step 3: the target state (position, velocity)
        target = controller_cfg["target"]
        self.target_position = float(target[0])
        self.target_velocity = float(target[1])

        # Step 4: solver settings
        self.iterations = int(solver_cfg["iterations"])
        self.learning_rate = float(solver_cfg["learning_rate"])

        # Step 5: the plan from the previous control step (used as the starting guess). None = no plan yet
        self.previous_plan = None

    # ------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the previous plan (call this before starting a new run)."""
        self.previous_plan = None

    # ------------------------------------------------------------------------------------------------------
    def cost(self, x_now: torch.Tensor, u_plan: torch.Tensor, u_previous: torch.Tensor) -> torch.Tensor:
        """
        Cost of each oscillator's plan.
            x_now      [S, 2]  current states
            u_plan     [S, N]  planned pushes
            u_previous [S]     the push applied at the last control step
        Returns [S]: one cost number per oscillator (smaller is better).
        """

        # Step 1: imagine the future with the predictor  -> [S, N, 2]
        predicted = self.predictor.predict(x_now, u_plan)

        number_of_systems = x_now.shape[0]
        total = torch.zeros(number_of_systems, dtype=torch.float32, device=self.device)

        # Step 2: state part of the cost, one planned step at a time
        last_step = self.horizon - 1
        for k in range(self.horizon):
            position_error = predicted[:, k, 0] - self.target_position
            velocity_error = predicted[:, k, 1] - self.target_velocity

            # the last planned step uses the terminal weights P, the others use Q
            if k == last_step:
                weight_position = self.p_position
                weight_velocity = self.p_velocity
            else:
                weight_position = self.q_position
                weight_velocity = self.q_velocity

            step_cost = weight_position * position_error * position_error \
                + weight_velocity * velocity_error * velocity_error
            total = total + step_cost

        # Step 3: input-change part of the cost: penalise jumps between consecutive pushes
        push_before = u_previous
        for k in range(self.horizon):
            push_now = u_plan[:, k]
            change = push_now - push_before
            total = total + self.r_input_change * change * change
            push_before = push_now

        return total

    # ------------------------------------------------------------------------------------------------------
    def _starting_guess(self, number_of_systems: int) -> torch.Tensor:
        """
        Warm start: last time's plan, moved one step forward in time.
            old plan:  [u0, u1, u2, ..., u9]     (u0 has just been applied)
            new guess: [u1, u2, ..., u9, u9]     (repeat the last push at the end)
        With no previous plan: all pushes 0.
        """
        if self.previous_plan is None:
            return torch.zeros(number_of_systems, self.horizon, dtype=torch.float32, device=self.device)

        without_first = self.previous_plan[:, 1:]          # [S, N-1]
        last_push = self.previous_plan[:, -1:]             # [S, 1]
        guess = torch.cat([without_first, last_push], dim=1)
        return guess.clone()

    # ------------------------------------------------------------------------------------------------------
    def solve(self, x_now: torch.Tensor, u_previous: torch.Tensor) -> torch.Tensor:
        """
        Find the best plan for every oscillator and return the FIRST push of each plan.
            x_now      [S, 2]  current states (physical units)
            u_previous [S]     the push applied at the last control step
        Returns [S]: the push to apply now.
        """
        number_of_systems = x_now.shape[0]

        # Step 1: the numbers the solver is allowed to change: the planned pushes
        u_plan = self._starting_guess(number_of_systems)
        u_plan.requires_grad_(True)

        # Step 2: Adam changes u_plan (NOT the model's weights)
        optimizer = torch.optim.Adam([u_plan], lr=self.learning_rate)

        # Step 3: remember the best plan found so far for each oscillator
        best_plan = u_plan.detach().clone()
        best_cost = [float("inf")] * number_of_systems

        # Step 4: the improvement rounds
        for _ in range(self.iterations):
            optimizer.zero_grad()

            # 4a: cost of the current plans, and their sum
            cost_each = self.cost(x_now, u_plan, u_previous)       # [S]
            total_cost = cost_each.sum()

            # 4b: keep the best plan of each oscillator (before changing it)
            cost_list = cost_each.detach().tolist()
            for s in range(number_of_systems):
                if cost_list[s] < best_cost[s]:
                    best_cost[s] = cost_list[s]
                    best_plan[s] = u_plan.detach()[s]

            # 4c: gradient: how the cost changes when each push changes a little
            total_cost.backward()

            # 4d: one Adam step on the pushes
            optimizer.step()

            # 4e: clip every push back into the allowed range [-u_max, u_max]
            with torch.no_grad():
                u_plan.clamp_(-self.u_max, self.u_max)

        # Step 5: the plan after the last round has not been checked yet: check it too
        with torch.no_grad():
            cost_each = self.cost(x_now, u_plan, u_previous)
        cost_list = cost_each.tolist()
        for s in range(number_of_systems):
            if cost_list[s] < best_cost[s]:
                best_cost[s] = cost_list[s]
                best_plan[s] = u_plan.detach()[s]

        # Step 6: keep the best plan for the warm start next time, return its first push
        self.previous_plan = best_plan
        first_push = best_plan[:, 0].clone()
        return first_push
