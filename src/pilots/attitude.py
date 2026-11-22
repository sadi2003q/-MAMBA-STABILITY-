"""
Space pilot A: satellite pointing with a learned model, without the steady pointing error.

Problem (Cena, Martini, Chiaberge 2026, arXiv 2602.15954): model predictive control that plans with a
learned model alone ends with a steady pointing error (0.29 degrees in their test); their fix switches to
a linear model below 1 degree, which they say "leads to considerably higher torques than necessary", and
they trained without reaction-wheel friction.
This pilot: your memory-removed chained predictor + the running correction (mpc01c) instead of the switch.

This file holds the parts specific to pilot A:
    make_trajectories   training / validation / test data (NO friction, as in the paper)
    attitude_cost       the controller's cost
    make_test_starts    the same starting points for every controller
    run_closed_loop     controller + real satellite, step after step, with or without wheel friction
"""

from __future__ import annotations

import torch

from src.systems.cubesat_wheels import CubeSatWithWheels


# ==========================================================================================================
# Data
# ==========================================================================================================
def random_axis(number: int, generator: torch.Generator) -> torch.Tensor:
    """Random unit vectors [number, 3]."""
    axis = torch.randn(number, 3, generator=generator)
    size = torch.sqrt((axis * axis).sum(dim=1, keepdim=True))
    return axis / size


def random_wheel_speeds(number: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    """Wheel speeds with a random size in [low, high] and a random sign, per wheel. [number, 3]"""
    size = low + (high - low) * torch.rand(number, 3, generator=generator)
    sign = torch.where(torch.rand(number, 3, generator=generator) < 0.5, -1.0, 1.0)
    return size * sign


@torch.no_grad()
def make_trajectories(system: CubeSatWithWheels, data_cfg: dict, number: int, seed: int):
    """
    Returns states [number, steps + 1, 9] and pushes [number, steps, 3] (physical units).
    Each trajectory: random start attitude and spin, random wheel speeds, pushed by a weak pointing
    controller with random gains plus random piecewise-constant extra pushes. No friction (as in the paper).
    """
    generator = torch.Generator().manual_seed(seed)
    steps = int(data_cfg["trajectory_steps"])

    # Step 1: starting attitude, spin and wheel speeds
    axis = random_axis(number, generator)
    angle = torch.rand(number, generator=generator) * float(data_cfg["start_angle_max"])
    sigma = CubeSatWithWheels.sigma_from_angle(axis, angle)
    spin_max = float(data_cfg["start_spin_max"])
    omega = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * spin_max
    wheels = random_wheel_speeds(number, float(data_cfg["wheel_speed_low"]), float(data_cfg["wheel_speed_high"]),
                                 generator)
    state = torch.cat([sigma, omega, wheels], dim=1)

    # Step 2: gains of the weak pointing controller (different for every trajectory)
    gain_low = float(data_cfg["gain_attitude_low"])
    gain_high = float(data_cfg["gain_attitude_high"])
    gain_attitude = gain_low + (gain_high - gain_low) * torch.rand(number, 1, generator=generator)
    damp_low = float(data_cfg["gain_spin_low"])
    damp_high = float(data_cfg["gain_spin_high"])
    gain_spin = damp_low + (damp_high - damp_low) * torch.rand(number, 1, generator=generator)

    # Step 3: simulate step by step
    states = torch.zeros(number, steps + 1, 9)
    pushes = torch.zeros(number, steps, 3)
    states[:, 0] = state
    excitation = torch.zeros(number, 3)
    hold_left = torch.zeros(number, dtype=torch.long)
    excitation_size = float(data_cfg["excitation_fraction"]) * system.u_max

    for step in range(steps):
        # 3a: new random extra push when the old one has been held long enough (1 to 8 steps)
        new_excitation = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * excitation_size
        new_hold = torch.randint(1, 9, (number,), generator=generator)
        change = (hold_left <= 0).unsqueeze(1)
        excitation = torch.where(change, new_excitation, excitation)
        hold_left = torch.where(hold_left <= 0, new_hold, hold_left) - 1

        # 3b: weak pointing controller. The wheel pushes the body the OPPOSITE way, so to turn the body
        #     back towards the target the wheel push has the SAME sign as the error:
        #     push = gain_attitude x (4 sigma ~ angle in radians) + gain_spin x omega
        sigma_now = state[:, 0:3]
        omega_now = state[:, 3:6]
        push = gain_attitude * 4.0 * sigma_now + gain_spin * omega_now + excitation
        push = torch.clamp(push, -system.u_max, system.u_max)

        # 3c: the satellite moves one step (no friction in the training data)
        state = system.step(state, push)
        pushes[:, step] = push
        states[:, step + 1] = state

    return states, pushes


# ==========================================================================================================
# The controller's cost
# ==========================================================================================================
def make_attitude_cost(controller_cfg: dict):
    """
    Cost of one plan (the same shape as the Van der Pol controller and the space-pilot branch):
        sum over steps 1..N-1 of  q_attitude |4 sigma_k|^2 + q_spin |omega_k|^2      (4 sigma ~ angle in radians)
        + at step N:              p_attitude |4 sigma_N|^2 + p_spin |omega_N|^2
        + sum over steps of       r |push_k - push_(k-1)|^2        (pushes in units of the push limit)
    The wheel speeds are free (not in the cost).
    """
    q_attitude = float(controller_cfg["q_attitude"])
    q_spin = float(controller_cfg["q_spin"])
    p_attitude = float(controller_cfg["p_attitude"])
    p_spin = float(controller_cfg["p_spin"])
    r_change = float(controller_cfg["r_input_change"])

    def cost(predicted, plan_scaled, previous_scaled, reference):
        number = predicted.shape[0]
        horizon = predicted.shape[1]
        total = torch.zeros(number, device=predicted.device)

        # Part 1: pointing and spin along the plan
        for k in range(horizon):
            angle_part = ((4.0 * predicted[:, k, 0:3]) ** 2).sum(dim=1)
            spin_part = (predicted[:, k, 3:6] ** 2).sum(dim=1)
            if k == horizon - 1:
                total = total + p_attitude * angle_part + p_spin * spin_part
            else:
                total = total + q_attitude * angle_part + q_spin * spin_part

        # Part 2: changes of the push
        push_before = previous_scaled
        for k in range(horizon):
            change = plan_scaled[:, k] - push_before
            total = total + r_change * (change ** 2).sum(dim=1)
            push_before = plan_scaled[:, k]
        return total

    return cost


# ==========================================================================================================
# Test
# ==========================================================================================================
def make_test_starts(test_cfg: dict, device) -> torch.Tensor:
    """The same starting points for every controller. [starts, 9]"""
    generator = torch.Generator().manual_seed(int(test_cfg["start_seed"]))
    number = int(test_cfg["starts"])
    axis = random_axis(number, generator)
    low = float(test_cfg["start_angle_low"])
    high = float(test_cfg["start_angle_high"])
    angle = low + (high - low) * torch.rand(number, generator=generator)
    sigma = CubeSatWithWheels.sigma_from_angle(axis, angle)
    spin_max = float(test_cfg["start_spin_max"])
    omega = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * spin_max
    wheels = random_wheel_speeds(number, float(test_cfg["wheel_speed_low"]), float(test_cfg["wheel_speed_high"]),
                                 generator)
    return torch.cat([sigma, omega, wheels], dim=1).to(device)


def run_closed_loop(controller, system: CubeSatWithWheels, starts: torch.Tensor, test_cfg: dict,
                    friction: dict, context_len: int, device, bar=None) -> dict:
    """
    Run all satellites at once from `starts` [S, 9]. friction = {"viscous": ..., "coulomb": ...} for the
    REAL satellite (zeros = the same physics as the training data).
    Returns per-start metric lists.
    """
    steps = int(test_cfg["steps"])
    number = starts.shape[0]
    viscous = float(friction["viscous"])
    coulomb = float(friction["coulomb"])
    settle_angle = float(test_cfg["settle_angle"])
    steady_from = int(round(steps * (1.0 - float(test_cfg["steady_fraction"]))))
    lost_spin = float(test_cfg["lost_control_spin"])

    # Step 1: bookkeeping
    state = starts.clone()
    controller.reset()
    previous_push = torch.zeros(number, 3, device=device)
    history_states = []
    history_pushes = []
    angles = []
    effort = torch.zeros(number, device=device)
    holding_push = torch.zeros(number, device=device)
    lost = torch.zeros(number, dtype=torch.bool, device=device)

    for step in range(steps):
        # Step 2: the recent history (at most context_len steps; none at the very first step)
        window = min(step, context_len)
        if window > 0:
            past_states = torch.stack(history_states[-window:], dim=1)
            past_pushes = torch.stack(history_pushes[-window:], dim=1)
        else:
            past_states = None
            past_pushes = None

        # Step 3: plan and apply the first push
        push = controller.solve(state, previous_push, past_states, past_pushes).detach()
        history_states.append(state.clone())
        history_pushes.append(push.clone())

        # Step 4: the real satellite moves (satellites that lost control are frozen to keep numbers finite)
        new_state = system.step(state, push, viscous, coulomb)
        new_state = torch.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
        spin = CubeSatWithWheels.spin_rate(new_state)
        lost = lost | (spin > lost_spin) | (~torch.isfinite(spin))
        state = torch.where(lost.unsqueeze(1), state, new_state)

        # Step 5: running totals
        angle = CubeSatWithWheels.pointing_error_degrees(state)
        angles.append(angle)
        push_scaled = push / system.u_max
        effort = effort + (push_scaled ** 2).sum(dim=1) * system.dt
        if step >= steady_from:
            holding_push = holding_push + torch.sqrt((push_scaled ** 2).sum(dim=1))
        previous_push = push
        if bar is not None:
            bar.update(1)

    # Step 6: metrics per satellite
    angle_history = torch.stack(angles, dim=1)                         # [S, steps]
    steady_angle = angle_history[:, steady_from:].mean(dim=1)          # mean pointing error in the last part
    final_angle = angle_history[:, -1]
    holding_push = holding_push / max(1, steps - steady_from)

    # settling time: the first time after which the error stays below settle_angle until the end
    settle_time = []
    for s in range(number):
        last_above = -1
        for k in range(steps):
            if float(angle_history[s, k]) >= settle_angle:
                last_above = k
        if last_above == steps - 1:
            settle_time.append(float("inf"))                         # never settled
        else:
            settle_time.append((last_above + 1) * system.dt)

    return {
        "steady_angle": steady_angle.tolist(),
        "final_angle": final_angle.tolist(),
        "settle_time": settle_time,
        "effort": effort.tolist(),
        "holding_push": holding_push.tolist(),
        "lost_control": lost.tolist(),
    }
