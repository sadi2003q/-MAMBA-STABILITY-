"""
Training data for the spacecraft: many simulated trajectories, normalised and cut into windows.

Every trajectory has its OWN moment of inertia, and that inertia shrinks slowly during the trajectory
(fuel being used). The model is never told the inertia: it has to work it out from recent motion, which
is the job of Chain-Mamba's memory (warm-up on the last `context_len` measured steps).

How each trajectory is made:
    start attitude:  random axis, random angle between 0 and `start_angle_max` degrees
    start spin:      each axis random in [-start_spin_max, +start_spin_max]
    inertia:         nominal x (random factor per axis in [factor_low, factor_high]),
                     shrinking by a random fraction in [0, shrink_max] over the trajectory
    pushes:          a weak pointing controller with random gains (keeps the spacecraft from tumbling)
                     + random piecewise-constant excitation, clipped to the push limit
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.datasets import Normaliser
from src.systems.spacecraft import Spacecraft


def inertia_at_step(start_inertia: torch.Tensor, shrink: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
    """Inertia at a given step: shrinks linearly from start_inertia to start_inertia x (1 - shrink). [S, 3]"""
    fraction = step / max(1, total_steps - 1)
    factor = 1.0 - shrink * fraction                     # [S, 1]
    return start_inertia * factor


@torch.no_grad()
def make_trajectories(system: Spacecraft, data_cfg: dict, number: int, seed: int):
    """Returns states [number, steps + 1, 6] and pushes [number, steps, 3] (physical units, processor)."""
    generator = torch.Generator().manual_seed(seed)
    steps = int(data_cfg["trajectory_steps"])

    # Step 1: starting attitude: random axis, random angle
    axis = torch.randn(number, 3, generator=generator)
    axis = axis / torch.sqrt((axis * axis).sum(dim=1, keepdim=True))
    angle = torch.rand(number, generator=generator) * float(data_cfg["start_angle_max"])
    sigma = Spacecraft.sigma_from_angle(axis, angle)

    # Step 2: starting spin
    spin_max = float(data_cfg["start_spin_max"])
    omega = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * spin_max
    state = torch.cat([sigma, omega], dim=1)

    # Step 3: this trajectory's inertia and how much it shrinks
    nominal = torch.tensor(system.inertia_nominal, dtype=torch.float32)
    low = float(data_cfg["inertia_factor_low"])
    high = float(data_cfg["inertia_factor_high"])
    factor = low + (high - low) * torch.rand(number, 3, generator=generator)
    start_inertia = nominal * factor
    shrink = torch.rand(number, 1, generator=generator) * float(data_cfg["inertia_shrink_max"])

    # Step 4: gains of the weak pointing controller (different for every trajectory)
    gain_attitude = data_cfg["gain_attitude_low"] + (data_cfg["gain_attitude_high"] - data_cfg["gain_attitude_low"]) \
        * torch.rand(number, 1, generator=generator)
    gain_spin = data_cfg["gain_spin_low"] + (data_cfg["gain_spin_high"] - data_cfg["gain_spin_low"]) \
        * torch.rand(number, 1, generator=generator)

    # Step 5: simulate step by step
    states = torch.zeros(number, steps + 1, 6)
    pushes = torch.zeros(number, steps, 3)
    states[:, 0] = state
    excitation = torch.zeros(number, 3)
    hold_left = torch.zeros(number, dtype=torch.long)
    excitation_size = float(data_cfg["excitation_fraction"]) * system.u_max

    for step in range(steps):
        # 5a: new random excitation when the old one has been held long enough (1 to 8 steps)
        new_excitation = (torch.rand(number, 3, generator=generator) * 2.0 - 1.0) * excitation_size
        new_hold = torch.randint(1, 9, (number,), generator=generator)
        change = (hold_left <= 0).unsqueeze(1)
        excitation = torch.where(change, new_excitation, excitation)
        hold_left = torch.where(hold_left <= 0, new_hold, hold_left) - 1

        # 5b: weak pointing controller + excitation, clipped to the push limit
        sigma_now = state[:, 0:3]
        omega_now = state[:, 3:6]
        push = -gain_attitude * 4.0 * sigma_now - gain_spin * omega_now + excitation
        push = torch.clamp(push, -system.u_max, system.u_max)

        # 5c: the real spacecraft moves one step with today's inertia
        inertia_now = inertia_at_step(start_inertia, shrink, step, steps)
        state = system.step(state, push, inertia_now)

        pushes[:, step] = push
        states[:, step + 1] = state

    return states, pushes


def _windows(states: torch.Tensor, pushes: torch.Tensor, norm: Normaliser, length: int, stride: int):
    """Cut normalised windows of `length` states (and length - 1 pushes) from every trajectory."""
    x_mean = torch.tensor(norm.x_mean, dtype=torch.float32)
    x_std = torch.tensor(norm.x_std, dtype=torch.float32)
    u_mean = torch.tensor(norm.u_mean, dtype=torch.float32)
    u_std = torch.tensor(norm.u_std, dtype=torch.float32)

    state_windows = []
    push_windows = []
    total_states = states.shape[1]
    start = 0
    while start + length <= total_states:
        state_windows.append((states[:, start:start + length] - x_mean) / x_std)
        push_windows.append((pushes[:, start:start + length - 1] - u_mean) / u_std)
        start = start + stride
    return torch.cat(state_windows, dim=0), torch.cat(push_windows, dim=0)


def build_space_data(cfg: dict, system: Spacecraft) -> dict:
    """Training, validation and test windows + the normalisation numbers + the increment scale."""
    data_cfg = cfg["data"]
    seed = int(data_cfg["data_seed"])

    # Step 1: three separate sets of trajectories
    train_states, train_pushes = make_trajectories(system, data_cfg, int(data_cfg["train_trajectories"]), seed)
    val_states, val_pushes = make_trajectories(system, data_cfg, int(data_cfg["val_trajectories"]), seed + 1)
    test_states, test_pushes = make_trajectories(system, data_cfg, int(data_cfg["test_trajectories"]), seed + 2)

    # Step 2: normalisation numbers from the TRAINING data only
    all_train_states = train_states.reshape(-1, 6).numpy()
    all_train_pushes = train_pushes.reshape(-1, 3).numpy()
    norm = Normaliser(
        x_mean=np.mean(all_train_states, axis=0),
        x_std=np.std(all_train_states, axis=0) + 1e-8,
        u_mean=np.mean(all_train_pushes, axis=0),
        u_std=np.std(all_train_pushes, axis=0) + 1e-8,
    )

    # Step 3: windows
    length = int(data_cfg["context_len"]) + int(data_cfg["horizon"]) + 1
    stride = int(data_cfg["window_stride"])
    result = {"norm": norm}
    result["train_x"], result["train_u"] = _windows(train_states, train_pushes, norm, length, stride)
    result["val_x"], result["val_u"] = _windows(val_states, val_pushes, norm, length, stride)
    result["test_x"], result["test_u"] = _windows(test_states, test_pushes, norm, length, stride)

    # Step 4: typical size of one true step-to-step change (normalised units), for the anchor's increment scale
    x_mean = torch.tensor(norm.x_mean, dtype=torch.float32)
    x_std = torch.tensor(norm.x_std, dtype=torch.float32)
    normalised = (train_states - x_mean) / x_std
    changes = normalised[:, 1:] - normalised[:, :-1]
    result["increment_std"] = changes.reshape(-1, 6).std(dim=0).numpy()

    # Step 5: a few numbers for the log
    result["largest_spin"] = float(Spacecraft.spin_rate(train_states.reshape(-1, 6)).max())
    result["largest_angle"] = float(Spacecraft.pointing_error_degrees(train_states.reshape(-1, 6)).max())
    return result
