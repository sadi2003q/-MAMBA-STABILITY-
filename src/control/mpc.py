"""
Model predictive control (beginner-friendly version).

At every control step (every 0.1 seconds):
    1. PLAN:    find N future pushes that make the predicted future look best (lowest cost)
    2. APPLY:   send only the FIRST push to the real oscillator
    3. MEASURE: read the oscillator's new state
    4. REPEAT:  plan again from the new state

The cost of one plan (same form as Cevaal, de Jong and Lazar, 2026):
    cost =   sum over steps k = 1 .. N-1 of  Q_pos (position_k - target)^2 + Q_vel (velocity_k - target)^2
           + at the last step N:              P_pos (position_N - target)^2 + P_vel (velocity_N - target)^2
           + sum over steps k = 0 .. N-1 of  R (u_k - u_(k-1))^2      (u_(-1) = the push applied last time)
"""


from __future__ import annotations

import torch



class MPCController:
    def __init__(
            self,
            predictor,
            controller_cfg:dict,
            solver_cfg:dict,
            device:torch.device,
    ):
        self.predictor = predictor
        self.device = device

        self.horizon = int(controller_cfg['horizon'])
        self.q_position = float(controller_cfg['q_position'])
        self.q_velocity = float(controller_cfg['q_velocity'])
        self.p_position = float(controller_cfg['p_position'])
        self.p_velocity = float(controller_cfg['p_velocity'])
        self.r_input_change = float(controller_cfg['r_input_change'])
        self.u_max = float(controller_cfg['u_max'])

        target = controller_cfg['target']
        self.target_position = float(target[0])
        self.target_velocity = float(target[1])

        self.iterations = int(solver_cfg['iterations'])
        self.learning_rate = float(solver_cfg['learning_rate'])

        self.previous_plan = None

    def reset(self):
        self.previous_plan = None

    def cost(
            self,
            x_now:torch.Tensor,
            u_plan:torch.Tensor,
            u_previous:torch.Tensor,
    )-> torch.Tensor:
        """
        Cost of each oscillator's plan.
            x_now      [S, 2]  current states
            u_plan     [S, N]  planned pushes
            u_previous [S]     the push applied at the last control step
        Returns [S]: one cost number per oscillator (smaller is better).
        """

        predicted = self.predictor.predict(x_now, u_plan)
        number_of_systems = x_now.shape[0]
        total = torch.zeros(number_of_systems, dtype=torch.float, device=self.device)

        last_step = self.horizon - 1
        for k in range(self.horizon):
            position_error = predicted[:, k, 0] - self.target_position
            velocity_error = predicted[:, k, 1] - self.target_velocity


            if k == last_step:
                weight_position = self.p_position
                weight_velocity = self.p_velocity
            else :
                weight_position = self.q_position
                weight_velocity = self.q_velocity

            step_cost = weight_position * position_error * position_error \
                        + weight_velocity * velocity_error * velocity_error
            total = total + step_cost

        push_before = u_previous
        for k in range(self.horizon):
            push_now = u_plan[:, k]
            change = push_now - push_before
            total = total + self.r_input_change * change * change
            push_before = push_now

        return total


    def _starting_guess(self, number_of_systems: int) -> torch.Tensor:
        """
        Warm start: last time's plan, moved one step forward in time.
            old plan:  [u0, u1, u2, ..., u9]     (u0 has just been applied)
            new guess: [u1, u2, ..., u9, u9]     (repeat the last push at the end)
        With no previous plan: all pushes 0.
        """

        if self.previous_plan is None:
            return torch.zeros(number_of_systems, self.horizon, dtype=torch.float, device=self.device)

        without_first = self.previous_plan[:, 1:]
        last_push = self.previous_plan[:, -1:]
        guess = torch.cat([without_first, last_push], dim=1)
        return guess.clone()

    def solve(
            self,
            x_now:torch.Tensor,
            u_previous:torch.Tensor,
    )-> torch.Tensor:
        """
        Find the best plan for every oscillator and return the FIRST push of each plan.
            x_now      [S, 2]  current states (physical units)
            u_previous [S]     the push applied at the last control step
        Returns [S]: the push to apply now.
        """
        number_of_systems = x_now.shape[0]

        u_plan = self._starting_guess(number_of_systems)
        u_plan.requires_grad_(True)

        optimiser = torch.optim.Adam([u_plan], lr=self.learning_rate)
        best_plan = u_plan.detach().clone()
        best_cost = [float("inf")] * number_of_systems

        for _ in range(self.iterations):
            optimiser.zero_grad()

            cost_each = self.cost(x_now, u_plan, u_previous)
            total_cost = cost_each.sum()

            cost_list = cost_each.detach().tolist()
            for s in range(number_of_systems):
                if cost_list[s] < best_cost[s]:
                    best_cost[s] = cost_list[s]
                    best_plan[s] = u_plan.detach()[s]

            total_cost.backward()

            optimiser.step()

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












