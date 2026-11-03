"""Four interconnected tanks (quadruple-tank process), as in Cevaal, de Jong and Lazar (2026), Equation 33 and
Table 5 (from Blaud et al., 2022). NumPy only.

    dx1/dt = -a1/Sc sqrt(2 g x1) + a3/Sc sqrt(2 g x3) + gamma_a / (3600 Sc) u1
    dx2/dt = -a2/Sc sqrt(2 g x2) + a4/Sc sqrt(2 g x4) + gamma_b / (3600 Sc) u2
    dx3/dt = -a3/Sc sqrt(2 g x3)                      + (1 - gamma_b) / (3600 Sc) u2
    dx4/dt = -a4/Sc sqrt(2 g x4)                      + (1 - gamma_a) / (3600 Sc) u1

x = water levels (metres), u = pump flows (cubic metres per hour, 0 to 4), all four levels measured.
Sampled every Ts = 5 seconds with the input held constant (zero-order hold); each sample is integrated with
`substeps` fourth-order Runge-Kutta steps. Levels are kept non-negative inside the square roots.
"""

from __future__ import annotations

import numpy as np


class FourTank:
    n_x = 4
    n_u = 2

    def __init__(self, dt: float = 5.0, Sc: float = 0.06, a=(1.31e-4, 1.51e-4, 9.27e-5, 8.82e-5),
                 gamma_a: float = 0.3, gamma_b: float = 0.4, g: float = 9.81, substeps: int = 10):
        self.dt = float(dt)
        self.Sc = float(Sc)
        self.a = np.asarray(a, dtype=float)
        self.gamma_a, self.gamma_b, self.g = float(gamma_a), float(gamma_b), float(g)
        self.substeps = int(substeps)

    @classmethod
    def from_config(cls, cfg: dict) -> "FourTank":
        s = cfg["system"]
        return cls(dt=s["dt"], Sc=s.get("Sc", 0.06), a=s.get("a", (1.31e-4, 1.51e-4, 9.27e-5, 8.82e-5)),
                   gamma_a=s.get("gamma_a", 0.3), gamma_b=s.get("gamma_b", 0.4), g=s.get("g", 9.81),
                   substeps=s.get("substeps", 10))

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time vector field. x [..., 4], u [..., 2]."""
        u = np.asarray(u, dtype=float)
        q = self.a * np.sqrt(2.0 * self.g * np.maximum(x, 0.0)) / self.Sc          # outflows, [..., 4]
        k = 1.0 / (3600.0 * self.Sc)
        u1, u2 = u[..., 0], u[..., 1]
        return np.stack([-q[..., 0] + q[..., 2] + self.gamma_a * k * u1,
                         -q[..., 1] + q[..., 3] + self.gamma_b * k * u2,
                         -q[..., 2] + (1.0 - self.gamma_b) * k * u2,
                         -q[..., 3] + (1.0 - self.gamma_a) * k * u1], axis=-1)

    def step(self, x: np.ndarray, u) -> np.ndarray:
        h = self.dt / self.substeps
        u = np.asarray(u, dtype=float)
        for _ in range(self.substeps):
            k1 = self.f(x, u)
            k2 = self.f(x + 0.5 * h * k1, u)
            k3 = self.f(x + 0.5 * h * k2, u)
            k4 = self.f(x + h * k3, u)
            x = np.maximum(x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        return x

    def jacobian(self, x: np.ndarray, u=None, eps: float = 1e-6) -> np.ndarray:
        """Jacobian of the one-sample map with respect to x (central differences). x [..., 4] -> [..., 4, 4].
        The input matters only through the operating point; a mid-range input (2, 2) is used when none is given."""
        x = np.asarray(x, dtype=float)
        u = np.full(x.shape[:-1] + (2,), 2.0) if u is None else np.broadcast_to(u, x.shape[:-1] + (2,))
        J = np.zeros(x.shape[:-1] + (4, 4))
        for j in range(4):
            d = np.zeros(4)
            d[j] = eps
            J[..., :, j] = (self.step(x + d, u) - self.step(np.maximum(x - d, 0.0), u)) / (2.0 * d[j])
        return J

    def simulate(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate from x0 with inputs u [n, 2]. Returns levels [n + 1, 4]."""
        u = np.asarray(u, dtype=float).reshape(len(u), 2)
        x = np.empty((len(u) + 1, 4))
        x[0] = x0
        for k in range(len(u)):
            x[k + 1] = self.step(x[k], u[k])
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Four-tank simulation produced non-finite levels.")
        return x
