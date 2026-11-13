"""
Closed loop: the controller and the REAL oscillator working together, step after step.

    real oscillator state  ->  controller plans  ->  first push  ->  real oscillator moves  ->  ...

The real oscillator is always the true Van der Pol equations (src/systems/vanderpol.py, NumPy).
The controller never sees the true equations unless it is the perfect-model controller.

Also here: the numbers that describe how well one run went (metrics).
"""

from __future__ import annotations

import time

import numpy as np
import torch


def run_closed_loop(controller, system, starts: np.ndarray, steps: int, device: torch.device,
                    blow_up_limit: float = 50.0, bar=None) -> dict:
    """
    Run S oscillators at the same time, each from its own starting point.

        controller  an MPCController, or None for "no control" (push always 0)
        system      the true oscillator (VanDerPol)
        starts      [S, 2] starting states (position, velocity)
        steps       number of control steps
        bar         optional progress bar, moved forward by 1 after every control step

    Returns a dictionary:
        states         [S, steps + 1, 2]  states over time (the first row is the start)
        pushes         [S, steps]         the push applied at each step
        plan_seconds   [steps]            time the controller needed to plan, at each step
        blew_up        [S]                True if that oscillator's state went beyond blow_up_limit
    """
    number_of_systems = starts.shape[0]

    # Step 1: arrays that will be filled in, step by step
    states = np.zeros((number_of_systems, steps + 1, 2))
    states[:, 0, :] = starts
    pushes = np.zeros((number_of_systems, steps))
    plan_seconds = np.zeros(steps)
    blew_up = np.zeros(number_of_systems, dtype=bool)

    # Step 2: no push yet before the first step
    previous_push = np.zeros(number_of_systems)

    # Step 3: forget any plan from an earlier run
    if controller is not None:
        controller.reset()

    # Step 4: the control loop
    for t in range(steps):
        current_states = states[:, t, :]

        # 4a: decide the push (and time how long the planning takes)
        if controller is None:
            push_now = np.zeros(number_of_systems)
        else:
            x_tensor = torch.tensor(current_states, dtype=torch.float32, device=device)
            previous_tensor = torch.tensor(previous_push, dtype=torch.float32, device=device)

            start_time = time.perf_counter()
            push_tensor = controller.solve(x_tensor, previous_tensor)
            end_time = time.perf_counter()

            plan_seconds[t] = end_time - start_time
            push_now = push_tensor.detach().cpu().numpy().astype(float)

        # 4b: the real oscillators move one step, one oscillator at a time
        for s in range(number_of_systems):
            # an oscillator that already blew up stays where it was (so nothing becomes NaN)
            if blew_up[s]:
                states[s, t + 1, :] = states[s, t, :]
                continue

            new_state = system.step(states[s, t, :], float(push_now[s]))

            # did this oscillator blow up? (not a number, or far outside the normal range)
            is_finite = bool(np.all(np.isfinite(new_state)))
            if is_finite:
                largest_size = np.max(np.abs(new_state))
                is_too_big = bool(largest_size > blow_up_limit)
            else:
                is_too_big = True

            if (not is_finite) or is_too_big:
                blew_up[s] = True
                states[s, t + 1, :] = states[s, t, :]
            else:
                states[s, t + 1, :] = new_state

        # 4c: remember the push for the next planning step
        pushes[:, t] = push_now
        previous_push = push_now

        if bar is not None:
            bar.update(1)

    result = {}
    result["states"] = states
    result["pushes"] = pushes
    result["plan_seconds"] = plan_seconds
    result["blew_up"] = blew_up
    return result


def run_metrics(states: np.ndarray, pushes: np.ndarray, blew_up: bool, target, dt: float,
                settle_radius: float) -> dict:
    """
    How well ONE run went (one oscillator from one starting point).

        states  [steps + 1, 2]   pushes  [steps]
    Returns:
        position_error   sum over time of (position - target position)^2     (Cevaal et al.'s ISE; smaller is better)
        position_abs     sum over time of |position - target position|       (Cevaal et al.'s IAE)
        push_energy      sum over time of push^2                             (how hard the controller worked)
        final_distance   distance to the target at the end
        at_rest          final_distance < settle_radius
        settle_time      seconds until the oscillator stays inside 2 x settle_radius for good (NaN if never)
        blew_up          the state went beyond the blow-up limit
    """
    # Step 1: errors from the target, skipping the starting state (the controller cannot change it)
    position_errors = states[1:, 0] - target[0]
    velocity_errors = states[1:, 1] - target[1]

    # Step 2: sums over time
    position_error = float(np.sum(position_errors * position_errors))
    position_abs = float(np.sum(np.abs(position_errors)))
    push_energy = float(np.sum(pushes * pushes))

    # Step 3: distance to the target at the last step
    last_position_error = states[-1, 0] - target[0]
    last_velocity_error = states[-1, 1] - target[1]
    final_distance = float(np.sqrt(last_position_error ** 2 + last_velocity_error ** 2))
    at_rest = bool(final_distance < settle_radius) and not bool(blew_up)

    # Step 4: settle time: after which step does the distance stay below 2 x settle_radius?
    distances = np.sqrt(position_errors ** 2 + velocity_errors ** 2)     # distance after step 1, 2, ...
    outside = distances >= 2.0 * settle_radius
    if blew_up or outside[-1]:
        settle_time = float("nan")                       # never settled
    elif not np.any(outside):
        settle_time = dt                                 # inside from the very first step
    else:
        last_outside_index = int(np.max(np.where(outside)[0]))
        settle_time = (last_outside_index + 2) * dt      # +1 for "next step", +1 because index 0 = step 1

    metrics = {}
    metrics["position_error"] = position_error
    metrics["position_abs"] = position_abs
    metrics["push_energy"] = push_energy
    metrics["final_distance"] = final_distance
    metrics["at_rest"] = at_rest
    metrics["settle_time"] = settle_time
    metrics["blew_up"] = bool(blew_up)
    return metrics
