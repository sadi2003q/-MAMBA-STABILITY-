"""Controlled Van der Pol oscillator, forward Euler discretisation (NumPy; no learning code here).

    dx1/dt = x2
    dx2/dt = mu (1 - x1^2) x2 - x1 + u          (the "- x1" can be switched off for reference)
    x_{k+1} = x_k + dt * f(x_k, u_k)
"""

from __future__ import annotations

import numpy as np


class VanDerPol:
    n_x = 2
    n_u = 1

    def __init__(self, mu: float = 1.0, dt: float = 0.1, include_minus_x1: bool = True):
        self.mu = float(mu)
        self.dt = float(dt)
        self.k = 1.0 if include_minus_x1 else 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "VanDerPol":
        s = cfg["system"]
        return cls(mu=s["mu"], dt=s["dt"], include_minus_x1=s.get("include_minus_x1", True))

    def f(self, x: np.ndarray, u: np.ndarray | float) -> np.ndarray:
        """Continuous-time vector field. x [..., 2], u [...] or [..., 1]."""
        u = np.asarray(u, dtype=float)
        if u.ndim == x.ndim:
            u = u[..., 0]
        x1, x2 = x[..., 0], x[..., 1]
        return np.stack([x2, self.mu * (1.0 - x1 ** 2) * x2 - self.k * x1 + u], axis=-1)

    def step(self, x: np.ndarray, u) -> np.ndarray:
        return x + self.dt * self.f(x, u)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the one-step map, I + dt * Df(x). x [..., 2] -> [..., 2, 2]. Independent of u."""
        x1, x2 = x[..., 0], x[..., 1]
        J = np.zeros(x.shape[:-1] + (2, 2))
        J[..., 0, 0] = 1.0
        J[..., 0, 1] = self.dt
        J[..., 1, 0] = self.dt * (-2.0 * self.mu * x1 * x2 - self.k)
        J[..., 1, 1] = 1.0 + self.dt * self.mu * (1.0 - x1 ** 2)
        return J

    def simulate(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate from x0 with input sequence u [n] (or [n, 1]). Returns states [n + 1, 2]."""
        u = np.asarray(u, dtype=float).reshape(len(u))
        x = np.empty((len(u) + 1, 2))
        x[0] = x0
        for k in range(len(u)):
            x[k + 1] = self.step(x[k], u[k])
            if not np.all(np.isfinite(x[k + 1])) or np.abs(x[k + 1]).max() > 1e6:
                raise FloatingPointError(
                    f"Van der Pol simulation blew up at step {k + 1}. Reduce the input amplitude or "
                    "the time step (see configs/systems/vanderpol.yaml).")
        return x
