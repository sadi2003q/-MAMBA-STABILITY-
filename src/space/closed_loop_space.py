"""
Closed loop for the spacecraft: the controller and the REAL spacecraft working together, step after step,
while the real inertia shrinks (fuel use) and the quantum hardware (if any) has the condition's defects.

Metrics for each spacecraft (one starting point):
    tracking_cost    sum over time of (pointing error in degrees)^2 x dt      (smaller is better)
    final_angle      pointing error at the end, degrees
    final_spin       spin rate at the end, radians per second
    at_rest          final_angle < at_rest_angle  and  final_spin < at_rest_spin
    lost_control     the spin rate ever went above lost_control_spin (the spacecraft started tumbling),
                     or a number stopped being finite
    effort           sum over time of |push / u_max|^2 x dt
"""

from __future__ import annotations

import torch

from src.space.data_space import inertia_at_step
from src.space.quantum import hardware_at_step, set_hardware
from src.systems.spacecraft import Spacecraft


def make_test_starts(test_cfg: dict, device) -> torch.Tensor:
    """The same starting points for every controller: random axis, angle in [angle_low, angle_high], small spin."""
    generator = torch.Generator().manual_seed(int(test_cfg["start_seed"]))
    number = int(test_cfg["starts"])
    axis = torch.randn(number, 3, generator=generator)
    axis = axis / torch.sqrt((axis * axis).sum(dim=1, keepdim=True))
    low = float(test_cfg["start_angle_low"])
    high = float(test_cfg["start_angle_high"])
    angle = low + (high - low) * torch.rand(number, generator=generator)
    sigma = Spacecraft.sigma_from_angle(axis, angle)
    spin_max = float(test_cfg["start_spin_max"])
    omega = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * spin_max
    return torch.cat([sigma, omega], dim=1).to(device)


def run_spacecraft_loop(controller, system: Spacecraft, starts: torch.Tensor, cfg: dict, condition: str,
                        quantum_model, context_len: int, device, bar=None) -> dict:
    """
    Run S spacecraft at once from `starts` [S, 6]. `quantum_model` is the model whose circuits get the
    condition's defects (None for classical predictors and the true-equations controller).
    Returns per-start metric lists.
    """
    test_cfg = cfg["test"]
    steps = int(test_cfg["steps"])
    number = starts.shape[0]

    # Step 1: the real inertia: nominal at the start, shrinking by `inertia_shrink` over the run
    nominal = torch.tensor(system.inertia_nominal, dtype=torch.float32, device=device)
    start_inertia = nominal.unsqueeze(0).repeat(number, 1)
    shrink = torch.full((number, 1), float(test_cfg["inertia_shrink"]), device=device)

    # Step 2: bookkeeping
    state = starts.clone()
    controller.reset()
    previous_push = torch.zeros(number, 3, device=device)
    history_states = []
    history_pushes = []
    tracking_cost = torch.zeros(number, device=device)
    effort = torch.zeros(number, device=device)
    lost = torch.zeros(number, dtype=torch.bool, device=device)
    switch_every = int(test_cfg["sun_shadow_switch_steps"])
    lost_spin = float(test_cfg["lost_control_spin"])

    for step in range(steps):
        # 2a: today's hardware defects (quantum models only) and today's true inertia
        if quantum_model is not None:
            set_hardware(quantum_model, **hardware_at_step(condition, step, steps, switch_every))
        inertia_now = inertia_at_step(start_inertia, shrink, step, steps)
        if hasattr(controller.predictor, "inertia_now"):
            controller.predictor.inertia_now = inertia_now

        # 2b: the recent history for the warm-up (at most context_len steps; none at the first step)
        window = min(step, context_len)
        if window > 0:
            past_states = torch.stack(history_states[-window:], dim=1)
            past_pushes = torch.stack(history_pushes[-window:], dim=1)
        else:
            past_states = None
            past_pushes = None

        # 2c: plan and apply the first push
        push = controller.solve(state, previous_push, past_states, past_pushes).detach()
        history_states.append(state.clone())
        history_pushes.append(push.clone())

        # 2d: the real spacecraft moves (spacecraft that already lost control are frozen to keep numbers finite)
        new_state = system.step(state, push, inertia_now)
        new_state = torch.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
        spin = Spacecraft.spin_rate(new_state)
        lost = lost | (spin > lost_spin) | (~torch.isfinite(spin))
        state = torch.where(lost.unsqueeze(1), state, new_state)

        # 2e: running totals
        angle = Spacecraft.pointing_error_degrees(state)
        tracking_cost = tracking_cost + angle * angle * system.dt
        effort = effort + ((push / system.u_max) ** 2).sum(dim=1) * system.dt
        previous_push = push
        if bar is not None:
            bar.update(1)

    # Step 3: hardware back to perfect
    if quantum_model is not None:
        set_hardware(quantum_model)

    final_angle = Spacecraft.pointing_error_degrees(state)
    final_spin = Spacecraft.spin_rate(state)
    at_rest = (final_angle < float(test_cfg["at_rest_angle"])) & (final_spin < float(test_cfg["at_rest_spin"])) & (~lost)
    return {
        "tracking_cost": tracking_cost.tolist(),
        "final_angle": final_angle.tolist(),
        "final_spin": final_spin.tolist(),
        "at_rest": at_rest.tolist(),
        "lost_control": lost.tolist(),
        "effort": effort.tolist(),
    }
