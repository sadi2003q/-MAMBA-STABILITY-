"""
Space pilot B: keeping a spacecraft on a planned orbit near an irregular, spinning asteroid.

Problem (Cesarini and Silvestrini, Politecnico di Milano): a large Transformer surrogate (embedding 512,
8 heads) inside model predictive control, trained on a LOW-fidelity asteroid model; shown only near one
reference orbit ("future studies should extend the analysis to more complex reference trajectories and a
wider range of operational regimes"). Planning took 3.58 s per step on an embedded computer.
This pilot: your memory-removed chained predictor (about 3,300 parameters), trained on the low-fidelity
model over a WIDE range of orbits, + the running correction (mpc01c), tested on the HIGH-fidelity asteroid
on four different orbits.

This file holds the parts specific to pilot B:
    make_trajectories   data from a given asteroid model (low fidelity for training, high for checking)
    make_orbit_starts   the reference orbits and the starting points for every controller
    asteroid_cost       the controller's cost (follow the reference, use little fuel)
    run_closed_loop     controller + real (high-fidelity) asteroid, step after step
"""

from __future__ import annotations

import math

import torch

from src.systems.small_body import SmallBody, reference_orbit


# ==========================================================================================================
# Data
# ==========================================================================================================
def random_unit_vectors(number: int, generator: torch.Generator) -> torch.Tensor:
    vectors = torch.randn(number, 3, generator=generator)
    size = torch.sqrt((vectors * vectors).sum(dim=1, keepdim=True))
    return vectors / size


@torch.no_grad()
def random_start_states(number: int, data_cfg: dict, gm: float, true_spin_rate: float,
                        generator: torch.Generator) -> torch.Tensor:
    """
    Starting states spread over a shell around the asteroid, moving roughly like an orbit.
    Velocity = (circular speed x random factor) along a random direction at right angles to the position,
    seen from the spinning frame.  [number, 6]
    """
    # Step 1: position: random direction, random distance in [radius_low, radius_high]
    direction = random_unit_vectors(number, generator)
    low = float(data_cfg["radius_low"])
    high = float(data_cfg["radius_high"])
    radius = low + (high - low) * torch.rand(number, 1, generator=generator)
    position = direction * radius

    # Step 2: a random direction at right angles to the position
    other = random_unit_vectors(number, generator)
    tangent = torch.cross(direction, other, dim=1)
    tangent_size = torch.sqrt((tangent * tangent).sum(dim=1, keepdim=True))
    tangent = tangent / torch.clamp(tangent_size, min=1e-9)

    # Step 3: speed = circular speed x a random factor (not every start is a perfect circle)
    circular_speed = torch.sqrt(gm / radius)
    factor_low = float(data_cfg["speed_factor_low"])
    factor_high = float(data_cfg["speed_factor_high"])
    factor = factor_low + (factor_high - factor_low) * torch.rand(number, 1, generator=generator)
    velocity_fixed = tangent * circular_speed * factor

    # Step 4: seen from the spinning frame:  velocity - W x position   (W = [0, 0, true spin])
    spin = torch.tensor([0.0, 0.0, true_spin_rate]).expand_as(position)
    velocity = velocity_fixed - torch.cross(spin, position, dim=1)
    return torch.cat([position, velocity], dim=1)


@torch.no_grad()
def make_trajectories(body: SmallBody, data_cfg: dict, number: int, seed: int, gm_reference: float,
                      true_spin_rate: float):
    """
    Trajectories under the given asteroid model with random kicks. Trajectories that come closer than
    keep_radius_low or go further than keep_radius_high are thrown away (they crashed or escaped).
    Returns states [number, steps + 1, 6] and pushes [number, steps, 3] (km, km/s).
    """
    generator = torch.Generator().manual_seed(seed)
    steps = int(data_cfg["trajectory_steps"])
    keep_low = float(data_cfg["keep_radius_low"])
    keep_high = float(data_cfg["keep_radius_high"])
    kick_probability = float(data_cfg["kick_probability"])
    kept_states = []
    kept_pushes = []
    kept = 0
    attempts = 0

    while kept < number and attempts < 20:
        attempts = attempts + 1
        batch = 2 * (number - kept) + 16

        # Step 1: starting states
        state = random_start_states(batch, data_cfg, gm_reference, true_spin_rate, generator)
        states = torch.zeros(batch, steps + 1, 6)
        pushes = torch.zeros(batch, steps, 3)
        states[:, 0] = state
        good = torch.ones(batch, dtype=torch.bool)

        # Step 2: simulate with random kicks (each axis kicked with probability kick_probability)
        for step in range(steps):
            size = (torch.rand(batch, 3, generator=generator) * 2.0 - 1.0) * body.u_max
            switched_on = torch.rand(batch, 3, generator=generator) < kick_probability
            push = torch.where(switched_on, size, torch.zeros_like(size))
            state = body.step(state, push)
            distance = SmallBody.distance(state)
            good = good & (distance > keep_low) & (distance < keep_high) & torch.isfinite(distance)
            pushes[:, step] = push
            states[:, step + 1] = state

        # Step 3: keep the good ones
        kept_states.append(states[good])
        kept_pushes.append(pushes[good])
        kept = kept + int(good.sum())

    all_states = torch.cat(kept_states, dim=0)[:number]
    all_pushes = torch.cat(kept_pushes, dim=0)[:number]
    return all_states, all_pushes


# ==========================================================================================================
# Reference orbits and starting points
# ==========================================================================================================
def make_orbit_starts(test_cfg: dict, gm: float, true_spin_rate: float, dt: float, horizon: int, device):
    """
    For every orbit in the config, `starts_per_orbit` spacecraft with a random phase on the orbit and a
    small random starting offset.
    Returns starts [S, 6], references [S, steps + horizon + 1, 6], orbit names [S].
    """
    generator = torch.Generator().manual_seed(int(test_cfg["start_seed"]))
    steps = int(test_cfg["steps"])
    per_orbit = int(test_cfg["starts_per_orbit"])
    position_offset = float(test_cfg["start_position_offset"])
    velocity_offset = float(test_cfg["start_velocity_offset"])

    starts = []
    references = []
    names = []
    for orbit in test_cfg["orbits"]:
        for _ in range(per_orbit):
            phase = 360.0 * float(torch.rand(1, generator=generator))
            reference = reference_orbit(float(orbit["radius"]), float(orbit["inclination"]), float(orbit["node"]),
                                        phase, gm, true_spin_rate, steps + horizon, dt)
            offset_position = (torch.rand(3, generator=generator) * 2.0 - 1.0) * position_offset
            offset_velocity = (torch.rand(3, generator=generator) * 2.0 - 1.0) * velocity_offset
            start = reference[0].clone()
            start[0:3] = start[0:3] + offset_position
            start[3:6] = start[3:6] + offset_velocity
            starts.append(start)
            references.append(reference)
            names.append(orbit["name"])
    return torch.stack(starts).to(device), torch.stack(references).to(device), names


# ==========================================================================================================
# The controller's cost
# ==========================================================================================================
def make_asteroid_cost(controller_cfg: dict):
    """
    Cost of one plan:
        sum over steps 1..N-1 of  q_position |(r_k - r_ref_k) / position_scale|^2 + q_velocity |(v_k - v_ref_k) / velocity_scale|^2
        + at step N:              the same with p_position, p_velocity
        + sum over steps of       r_fuel |push_k|^2          (pushes in units of the push limit)
    """
    q_position = float(controller_cfg["q_position"])
    q_velocity = float(controller_cfg["q_velocity"])
    p_position = float(controller_cfg["p_position"])
    p_velocity = float(controller_cfg["p_velocity"])
    r_fuel = float(controller_cfg["r_fuel"])
    position_scale = float(controller_cfg["position_scale"])
    velocity_scale = float(controller_cfg["velocity_scale"])

    def cost(predicted, plan_scaled, previous_scaled, reference):
        number = predicted.shape[0]
        horizon = predicted.shape[1]
        total = torch.zeros(number, device=predicted.device)

        # Part 1: distance from the reference along the plan
        for k in range(horizon):
            position_error = (predicted[:, k, 0:3] - reference[:, k, 0:3]) / position_scale
            velocity_error = (predicted[:, k, 3:6] - reference[:, k, 3:6]) / velocity_scale
            position_part = (position_error ** 2).sum(dim=1)
            velocity_part = (velocity_error ** 2).sum(dim=1)
            if k == horizon - 1:
                total = total + p_position * position_part + p_velocity * velocity_part
            else:
                total = total + q_position * position_part + q_velocity * velocity_part

        # Part 2: fuel
        for k in range(horizon):
            total = total + r_fuel * (plan_scaled[:, k] ** 2).sum(dim=1)
        return total

    return cost


# ==========================================================================================================
# Test
# ==========================================================================================================
def run_closed_loop(controller, real_body: SmallBody, starts: torch.Tensor, references: torch.Tensor,
                    test_cfg: dict, horizon: int, context_len: int, device, bar=None) -> dict:
    """
    Run all spacecraft at once near the REAL (high-fidelity) asteroid.
    Returns per-start metric lists (position errors in metres, fuel in metres per second).
    """
    steps = int(test_cfg["steps"])
    number = starts.shape[0]
    crash_radius = float(test_cfg["crash_radius"])
    lost_error = float(test_cfg["lost_position_error"])

    # Step 1: bookkeeping
    state = starts.clone()
    controller.reset()
    previous_push = torch.zeros(number, 3, device=device)
    history_states = []
    history_pushes = []
    error_sum = torch.zeros(number, device=device)
    largest_error = torch.zeros(number, device=device)
    fuel = torch.zeros(number, device=device)
    lost = torch.zeros(number, dtype=torch.bool, device=device)

    for step in range(steps):
        # Step 2: recent history and the reference for the planned steps
        window = min(step, context_len)
        if window > 0:
            past_states = torch.stack(history_states[-window:], dim=1)
            past_pushes = torch.stack(history_pushes[-window:], dim=1)
        else:
            past_states = None
            past_pushes = None
        reference_ahead = references[:, step + 1:step + 1 + horizon]

        # Step 3: plan and apply the first push
        push = controller.solve(state, previous_push, past_states, past_pushes, reference_ahead).detach()
        history_states.append(state.clone())
        history_pushes.append(push.clone())

        # Step 4: the real spacecraft moves (lost ones are frozen to keep numbers finite)
        new_state = real_body.step(state, push)
        new_state = torch.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
        state = torch.where(lost.unsqueeze(1), state, new_state)

        # Step 5: distance from the reference, in metres
        difference = state[:, 0:3] - references[:, step + 1, 0:3]
        error_m = 1000.0 * torch.sqrt((difference * difference).sum(dim=1))
        distance = SmallBody.distance(state)
        lost = lost | (distance < crash_radius) | (error_m > 1000.0 * lost_error) | (~torch.isfinite(error_m))
        error_sum = error_sum + error_m
        largest_error = torch.maximum(largest_error, error_m)
        applied = torch.clamp(push, -real_body.u_max, real_body.u_max)
        fuel = fuel + 1000.0 * torch.sqrt((applied * applied).sum(dim=1))
        previous_push = push
        if bar is not None:
            bar.update(1)

    return {
        "mean_position_error_m": (error_sum / steps).tolist(),
        "largest_position_error_m": largest_error.tolist(),
        "fuel_m_per_s": fuel.tolist(),
        "lost": lost.tolist(),
    }


def spin_rate_from_config(cfg: dict) -> float:
    """True asteroid spin rate, radians per second."""
    return 2.0 * math.pi / (float(cfg["system"]["spin_period_hours"]) * 3600.0)
