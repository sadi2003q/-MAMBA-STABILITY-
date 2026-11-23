"""
Prediction test for space pilot C (sloshing fuel): data in the style of a real mission, and the error check.

Only the spacecraft's angle and turn rate are recorded (the fuel is not measured), as on a real spacecraft.
"""

from __future__ import annotations

import math

import torch

from src.pilots.slosh import add_sensor_noise


def draw_training_fills(number: int, low: float, high: float, test_fills: list, gap: float,
                        generator: torch.Generator) -> torch.Tensor:
    """Fill levels drawn evenly from low..high, but never closer than `gap` to a test fill level (kept unseen)."""
    fills = []
    while len(fills) < number:
        candidate = low + (high - low) * float(torch.rand(1, generator=generator))
        too_close = False
        for test_fill in test_fills:
            if abs(candidate - test_fill) < gap:
                too_close = True
        if not too_close:
            fills.append(candidate)
    return torch.tensor(fills)


def make_mission_trajectories(system, cfg: dict, fills: torch.Tensor, seed: int, style: str):
    """
    Flight-like data at the given fill levels.
        style = "normal_operations": only normal attitude turns to new targets (no test kicks, no dither),
                                     as a real mission would log them
        style = "rich":              the same plus random held kicks and dither (easier for learning)
    Returns measured states [N, T + 1, 2], pushes [N, T, 1].
    """
    data_cfg = cfg["data"]
    steps = int(data_cfg["trajectory_steps"])
    number = fills.shape[0]
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high, size):
        return low + (high - low) * torch.rand(size, generator=generator)

    # Step 1: starting state of every trajectory (the fuel may still be moving from an earlier turn)
    start_angle = math.radians(float(data_cfg["start_angle_degrees"]))
    start_slosh = math.radians(float(data_cfg["start_slosh_degrees"]))
    state = torch.zeros(number, 4)
    state[:, 0] = uniform(-start_angle, start_angle, (number,))
    state[:, 1] = uniform(-0.01, 0.01, (number,))
    state[:, 2] = uniform(-start_slosh, start_slosh, (number,))
    state[:, 3] = uniform(-0.05, 0.05, (number,))

    # Step 2: each trajectory gets its own attitude-law tuning (different missions, different tunings)
    stiffness = uniform(0.3, 6.0, (number,))          # N m per rad
    damping = uniform(1.0, 15.0, (number,))           # N m per rad/s
    target = uniform(-start_angle, start_angle, (number,))
    if style == "rich":
        kick_share = uniform(0.0, 1.0, (number,))
        dither_size = 0.05 * system.u_max
    else:
        kick_share = torch.zeros(number)
        dither_size = 0.0
    kick = torch.zeros(number)
    kick_left = torch.zeros(number)

    measured_list = [add_sensor_noise(system.measure(state), cfg, generator)]
    push_list = []
    for step in range(steps):
        # 2a: a new pointing target now and then (about every 15 s): a normal attitude turn
        change = torch.rand(number, generator=generator) < 1.0 / 75.0
        target = torch.where(change, uniform(-start_angle, start_angle, (number,)), target)

        # 2b: random held kicks (only in the "rich" style)
        renew = kick_left <= 0
        kick = torch.where(renew, uniform(-system.u_max, system.u_max, (number,)), kick)
        kick_left = torch.where(renew, torch.randint(1, 25, (number,), generator=generator).float(), kick_left - 1)

        # 2c: the push the attitude law asks for, clipped to the actuator limit
        law = -stiffness * (state[:, 0] - target) - damping * state[:, 1]
        dither = dither_size * torch.randn(number, generator=generator)
        push = torch.clamp(law + kick_share * kick + dither, -system.u_max, system.u_max).unsqueeze(1)

        # 2d: the real spacecraft moves; the sensors record angle and turn rate only
        state = system.step(state, push, fills)
        measured_list.append(add_sensor_noise(system.measure(state), cfg, generator))
        push_list.append(push)

    measured = torch.stack(measured_list, dim=1)
    pushes = torch.stack(push_list, dim=1)
    return measured, pushes


@torch.no_grad()
def chained_errors(predictor, measured, pushes, context_len: int, horizon: int, every: int, device):
    """
    Root-mean-square error of the predicted ANGLE (degrees) and TURN RATE (degrees per second) after
    1..horizon chained steps. Each prediction starts after `context_len` measured steps were shown to the
    predictor (this fills the memory; a predictor without memory ignores them).
    measured [N, T + 1, 2], pushes [N, T, 1].
    """
    starts = []
    plans = []
    truths = []
    past_states = []
    past_pushes = []
    total = pushes.shape[1]
    start = context_len
    while start + horizon <= total:
        past_states.append(measured[:, start - context_len:start])
        past_pushes.append(pushes[:, start - context_len:start])
        starts.append(measured[:, start])
        plans.append(pushes[:, start:start + horizon])
        truths.append(measured[:, start + 1:start + 1 + horizon])
        start = start + every

    history_states = torch.cat(past_states).to(device)
    history_pushes = torch.cat(past_pushes).to(device)
    memory, _ = predictor.prepare(history_states, history_pushes)
    predicted = predictor.predict(torch.cat(starts).to(device), torch.cat(plans).to(device), memory, None)
    truth = torch.cat(truths).to(device)

    angle_difference = torch.rad2deg(predicted[:, :, 0] - truth[:, :, 0])
    rate_difference = torch.rad2deg(predicted[:, :, 1] - truth[:, :, 1])
    angle_error = torch.sqrt((angle_difference ** 2).mean(dim=0)).cpu()
    rate_error = torch.sqrt((rate_difference ** 2).mean(dim=0)).cpu()

    angle_list = []
    rate_list = []
    for k in range(horizon):
        angle_list.append(float(angle_error[k]))
        rate_list.append(float(rate_error[k]))
    return angle_list, rate_list
