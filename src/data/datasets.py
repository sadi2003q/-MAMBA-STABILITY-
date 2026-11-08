"""Trajectory generation, normalisation and windowing.

A window holds true states x_0 ... x_{C+H} and inputs u_0 ... u_{C+H-1}
(C = context length, H = horizon). Models see x_0 ... x_C, then predict x_{C+1} ... x_{C+H}.
Everything a model sees is normalised with the training-set mean and standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.systems.vanderpol import VanDerPol


def multisine(n: int, dt: float, f_min: float, f_max: float, harmonics: int, peak: float,
              rng: np.random.Generator, spacing: str = "linear") -> np.ndarray:
    """
    Build a control input signal u by adding several sine waves together.

    Inputs:
        n          number of time steps in the signal
        dt         time between two steps
        f_min      frequency of the slowest wave
        f_max      frequency of the fastest wave
        harmonics  how many sine waves to add together
        peak       the largest absolute value the final signal may reach
        rng        random number generator (seeded, so results are repeatable)
        spacing    "linear" = equal gaps between frequencies
                   "log"    = equal ratios between frequencies

    Output:
        u          a NumPy array of length n
    """

    # Step 1: make the list of time points: 0, dt, 2*dt, ..., (n-1)*dt
    step_numbers = np.arange(n)
    t = step_numbers * dt

    # Step 2: choose the frequency of each sine wave
    if spacing == "log":
        # equal ratios, e.g. 0.05, 0.13, 0.32, 0.80, 2.00
        freqs = np.geomspace(f_min, f_max, harmonics)
    elif spacing == "linear":
        # equal gaps, e.g. 0.05, 0.54, 1.03, 1.51, 2.00
        freqs = np.linspace(f_min, f_max, harmonics)
    else:
        raise ValueError(f"Unknown multisine spacing '{spacing}'. Use 'linear' or 'log'.")

    # Step 3: choose a random starting point (phase) for each wave
    # One random angle between 0 and 2*pi per wave.
    # Drawn all at once, exactly like the old code,
    # so the same seed still gives the same data.
    phases = rng.uniform(0.0, 2.0 * np.pi, harmonics)

    # Step 4: make the sine waves one at a time and keep them in a list
    all_waves = []

    for i in range(harmonics):
        one_frequency = freqs[i]
        one_phase = phases[i]

        angle = 2.0 * np.pi * one_frequency * t + one_phase
        one_wave = np.sin(angle)

        all_waves.append(one_wave)

    # Step 5: put the waves side by side as columns -> shape (n, harmonics),
    # then add them together row by row (one row = one time step)
    #
    # Why np.sum and not a running "u = u + one_wave" loop:
    # on paper both give the same answer, but with 30 waves the computer adds them
    # in a different order and rounds slightly differently (about 1e-15).
    # np.sum adds them in exactly the same order as the original code,
    # so the data stays bit-for-bit identical to every earlier experiment.
    wave_table = np.stack(all_waves, axis=1)
    u = np.sum(wave_table, axis=1)

    # Step 6: find the largest absolute value in the signal
    largest_value = np.max(np.abs(u))

    # Step 7: rescale so the largest value becomes exactly `peak`
    u = u / largest_value
    u = u * peak

    return u


@dataclass
class Normaliser:
    x_mean: np.ndarray
    x_std: np.ndarray
    u_mean: np.ndarray
    u_std: np.ndarray

    def x(self, x):
        return (x - self.x_mean) / self.x_std

    def u(self, u):
        return (u - self.u_mean) / self.u_std

    def x_inverse(self, xn):
        if isinstance(xn, torch.Tensor):
            mean = torch.as_tensor(self.x_mean, dtype=xn.dtype, device=xn.device)
            std = torch.as_tensor(self.x_std, dtype=xn.dtype, device=xn.device)
            return xn * std + mean
        return xn * self.x_std + self.x_mean

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "u_mean", "u_std")}


@dataclass
class Split:
    x: np.ndarray          # [n + 1, 2] physical units
    u: np.ndarray          # [n, 1]
    xw: torch.Tensor       # [W, C + H + 1, 2] normalised windows
    uw: torch.Tensor       # [W, C + H, 1]
    starts: np.ndarray     # window start indices into x


def _trajectory(system: VanDerPol, n: int, data_cfg: dict, rng: np.random.Generator):
    """
    Make ONE long trajectory of the Van der Pol system.

    What it does:
        1. builds a wiggly control input u (a multisine)
        2. picks a random starting state x0
        3. simulates the true system pushed by u

    Returns:
        x          the simulated states (one row per time step)
        u_column   the control input as a column, shape (n, 1)
    """

    # Step 1: read the multisine settings from the config
    multisine_settings = data_cfg["multisine"]
    f_min = multisine_settings["f_min"]
    f_max = multisine_settings["f_max"]
    harmonics = multisine_settings["harmonics"]
    peak = multisine_settings["peak"]
    spacing = multisine_settings.get("spacing", "linear")   # "linear" if not given

    # Step 2: build the control input signal u (a flat list of length n)
    u = multisine(n, system.dt, f_min, f_max, harmonics, peak, rng, spacing)

    # Step 3: pick a random starting state between the low and high limits
    low_limit = data_cfg["initial_state_low"]
    high_limit = data_cfg["initial_state_high"]
    x0 = rng.uniform(low_limit, high_limit)

    # Step 4: simulate the true system, starting at x0, pushed by u
    x = system.simulate(x0, u)

    # Step 5: turn u from a flat list, shape (n,), into a column, shape (n, 1)
    # The model expects one column per input signal.
    number_of_steps = len(u)
    u_column = u.reshape(number_of_steps, 1)

    return x, u_column


def _windows(x: np.ndarray, u: np.ndarray, norm: Normaliser, length: int, stride: int,
             max_windows: int | None, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Cut one long trajectory into many short, overlapping pieces called "windows".

    Example with stride = 2:
        window 1 starts at step 0
        window 2 starts at step 2
        window 3 starts at step 4
        ...

    Each window contains:
        length + 1 states   (the starting state, plus one new state after each push)
        length     inputs   (one push per step)
    """

    # Step 1: list every possible starting step for a window
    # The last window must still fit completely inside the trajectory.
    last_possible_start = len(u) - length
    starts = np.arange(0, last_possible_start + 1, stride)

    # Step 2: if there are too many windows, keep only a random subset
    if max_windows is not None:
        if len(starts) > max_windows:
            chosen_starts = rng.choice(starts, size=max_windows, replace=False)
            starts = np.sort(chosen_starts)    # put them back in time order

    # Step 3: normalise the states and the inputs
    # (shift and scale them so they are centred near 0 with a spread of about 1)
    x_normalised = norm.x(x)
    u_normalised = norm.u(u)

    # Step 4: cut out each window, one at a time
    state_windows = []
    input_windows = []

    for start in starts:
        state_end = start + length + 1
        input_end = start + length

        one_state_window = x_normalised[start:state_end]
        one_input_window = u_normalised[start:input_end]

        state_windows.append(one_state_window)
        input_windows.append(one_input_window)

    # Step 5: stack the windows into one big array each
    # state_array shape: (number of windows, length + 1, number of state variables)
    # input_array shape: (number of windows, length, 1)
    state_array = np.stack(state_windows)
    input_array = np.stack(input_windows)

    # Step 6: convert to PyTorch tensors (32-bit numbers) so the model can use them
    state_tensor = torch.tensor(state_array, dtype=torch.float32)
    input_tensor = torch.tensor(input_array, dtype=torch.float32)

    return state_tensor, input_tensor, starts


def build_data(cfg: dict) -> dict:
    """
    Generate the training, validation and test data.

    The data depends only on data_seed, so every model seed
    is trained and tested on exactly the same data.
    """

    # Step 1: read the data settings and create the true system
    data_cfg = cfg["data"]
    system = VanDerPol.from_config(cfg)

    # Step 2: create the random generator used for the trajectories
    data_seed = data_cfg["data_seed"]
    rng = np.random.default_rng(data_seed)

    # Step 3: make one long trajectory for each split
    # The order (train, then val, then test) matters:
    # all three share one random generator, so a different order would give different data.
    trajectories = {}

    for split in ("train", "val", "test"):
        setting_name = f"{split}_samples"            # e.g. "train_samples"
        number_of_steps = int(data_cfg[setting_name])

        x, u = _trajectory(system, number_of_steps, data_cfg, rng)
        trajectories[split] = (x, u)

    # Step 4: compute the normalisation numbers from the TRAINING data only
    # (validation and test data must never influence training)
    x_train, u_train = trajectories["train"]

    x_mean = np.mean(x_train, axis=0)    # average of each state variable
    x_std = np.std(x_train, axis=0)      # spread of each state variable
    u_mean = np.mean(u_train, axis=0)    # average of the input
    u_std = np.std(u_train, axis=0)      # spread of the input

    norm = Normaliser(x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std)

    # Step 5: window settings
    context_len = int(data_cfg["context_len"])   # steps the model sees first
    horizon = int(data_cfg["horizon"])           # steps the model must predict
    length = context_len + horizon               # total steps in one window
    stride = int(data_cfg["window_stride"])      # gap between two window starts

    # Step 6: the maximum number of windows for each split (None means no limit)
    evaluation_cfg = cfg.get("evaluation", {})

    limits = {}
    limits["train"] = data_cfg.get("max_train_windows")
    limits["val"] = data_cfg.get("max_val_windows")
    limits["test"] = evaluation_cfg.get("max_eval_windows")

    # Step 7: a SEPARATE random generator for choosing windows
    # (seed + 1, so choosing windows never changes the trajectories above)
    window_rng = np.random.default_rng(data_seed + 1)

    # Step 8: cut each trajectory into windows
    splits = {}

    for split in ("train", "val", "test"):
        x, u = trajectories[split]
        split_limit = limits[split]

        xw, uw, starts = _windows(x, u, norm, length, stride, split_limit, window_rng)

        splits[split] = Split(x=x, u=u, xw=xw, uw=uw, starts=starts)

    # Step 9: typical size of one true step-to-step change, in normalised units
    # (training trajectory only). Used by the anchor's increment scale
    # ("increment_normalisation" in a model config); other models ignore it.
    x_train_normalised = norm.x(x_train)
    one_step_changes = np.diff(x_train_normalised, axis=0)    # x[t+1] - x[t], for every t
    increment_std = np.std(one_step_changes, axis=0)          # one number per state variable

    # Step 10: collect everything in one dictionary
    result = {}
    result["system"] = system
    result["norm"] = norm
    result["splits"] = splits
    result["increment_std"] = increment_std
    result["context_len"] = context_len
    result["horizon"] = horizon

    return result
