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
    """Sum of sinusoids with random phases, rescaled to the given peak amplitude."""
    t = np.arange(n) * dt
    if spacing == "log":
        freqs = np.geomspace(f_min, f_max, harmonics)
    elif spacing == "linear":
        freqs = np.linspace(f_min, f_max, harmonics)
    else:
        raise ValueError(f"Unknown multisine spacing '{spacing}'.")
    phases = rng.uniform(0.0, 2.0 * np.pi, harmonics)
    u = np.sin(2.0 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]).sum(axis=1)
    return u / np.abs(u).max() * peak


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
    ms = data_cfg["multisine"]
    u = multisine(n, system.dt, ms["f_min"], ms["f_max"], ms["harmonics"], ms["peak"], rng,
                  ms.get("spacing", "linear"))
    x0 = rng.uniform(data_cfg["initial_state_low"], data_cfg["initial_state_high"])
    x = system.simulate(x0, u)
    return x, u[:, None]


def _windows(x: np.ndarray, u: np.ndarray, norm: Normaliser, length: int, stride: int,
             max_windows: int | None, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    starts = np.arange(0, len(u) - length + 1, stride)
    if max_windows is not None and len(starts) > max_windows:
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    xn, un = norm.x(x), norm.u(u)
    xw = np.stack([xn[s:s + length + 1] for s in starts])
    uw = np.stack([un[s:s + length] for s in starts])
    return torch.tensor(xw, dtype=torch.float32), torch.tensor(uw, dtype=torch.float32), starts


def build_data(cfg: dict) -> dict:
    """Generate train / validation / test splits. Identical for every model seed (data_seed)."""
    data_cfg = cfg["data"]
    system = VanDerPol.from_config(cfg)
    rng = np.random.default_rng(data_cfg["data_seed"])
    trajectories = {}
    for split in ("train", "val", "test"):
        trajectories[split] = _trajectory(system, int(data_cfg[f"{split}_samples"]), data_cfg, rng)

    x_tr, u_tr = trajectories["train"]
    norm = Normaliser(x_mean=x_tr.mean(0), x_std=x_tr.std(0), u_mean=u_tr.mean(0), u_std=u_tr.std(0))

    length = int(data_cfg["context_len"]) + int(data_cfg["horizon"])
    stride = int(data_cfg["window_stride"])
    limits = {"train": data_cfg.get("max_train_windows"), "val": data_cfg.get("max_val_windows"),
              "test": cfg.get("evaluation", {}).get("max_eval_windows")}
    win_rng = np.random.default_rng(data_cfg["data_seed"] + 1)
    splits = {}
    for split, (x, u) in trajectories.items():
        xw, uw, starts = _windows(x, u, norm, length, stride, limits[split], win_rng)
        splits[split] = Split(x=x, u=u, xw=xw, uw=uw, starts=starts)
    return {"system": system, "norm": norm, "splits": splits,
            "context_len": int(data_cfg["context_len"]), "horizon": int(data_cfg["horizon"])}
