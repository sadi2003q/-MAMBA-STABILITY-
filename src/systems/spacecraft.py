"""
Rigid spacecraft attitude (pointing) dynamics, with a moment of inertia that CHANGES over time.
Used by the space paper (configs/space/), not by the certificate paper.

----------------------------------------------------------------------------------------------------------
The state (6 numbers) and the push (3 numbers)
----------------------------------------------------------------------------------------------------------
    sigma  [3]   attitude as modified Rodrigues parameters. Rotation angle = 4 x atan(|sigma|).
                 sigma = 0 means "pointing exactly at the target".
    omega  [3]   angular velocity (radians per second), in the spacecraft's own axes
    torque [3]   push from the reaction wheels (newton metres), limited to |torque| <= u_max per axis

----------------------------------------------------------------------------------------------------------
The equations (standard rigid-body attitude model)
----------------------------------------------------------------------------------------------------------
    d(omega)/dt = ( torque - omega x (J omega) ) / J          (J = diagonal moment of inertia)
    d(sigma)/dt = 1/4 x [ (1 - sigma.sigma) omega + 2 (sigma x omega) + 2 (sigma.omega) sigma ]

    "x" is the cross product, "." the dot product.
    After every step: if |sigma| > 1, switch to the equivalent "shadow" set  sigma -> -sigma / |sigma|^2
    (the same physical attitude, described the short way round).

The moment of inertia J changes over time (fuel being used, parts moving). The caller passes the inertia
that is true at each step; this file only does the physics.
"""

from __future__ import annotations

import math

import torch


class Spacecraft:
    n_x = 6
    n_u = 3

    def __init__(self, dt: float, substeps: int, inertia_nominal, u_max: float):
        # Step 1: time between two control decisions, and how finely each step is simulated
        self.dt = float(dt)
        self.substeps = int(substeps)

        # Step 2: the nominal moment of inertia (one number per axis) and the push limit
        self.inertia_nominal = [float(value) for value in inertia_nominal]
        self.u_max = float(u_max)

    @classmethod
    def from_config(cls, cfg: dict) -> "Spacecraft":
        s = cfg["system"]
        return cls(dt=s["dt"], substeps=s["substeps"], inertia_nominal=s["inertia_nominal"], u_max=s["u_max"])

    # ------------------------------------------------------------------------------------------------------
    def derivative(self, state: torch.Tensor, torque: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
        """
        How fast the state changes right now.
            state   [S, 6]   (sigma, omega)
            torque  [S, 3]
            inertia [S, 3]   diagonal moment of inertia of each spacecraft
        Returns [S, 6].
        """
        # Step 1: split the state
        sigma = state[:, 0:3]
        omega = state[:, 3:6]

        # Step 2: rotation part:  d(omega)/dt = (torque - omega x (J omega)) / J
        angular_momentum = inertia * omega                                  # J omega  [S, 3]
        gyroscopic = torch.cross(omega, angular_momentum, dim=1)            # omega x (J omega)
        omega_change = (torque - gyroscopic) / inertia

        # Step 3: attitude part:  d(sigma)/dt = 1/4 [ (1 - s.s) w + 2 (s x w) + 2 (s.w) s ]
        sigma_squared = (sigma * sigma).sum(dim=1, keepdim=True)            # s.s     [S, 1]
        sigma_dot_omega = (sigma * omega).sum(dim=1, keepdim=True)          # s.w     [S, 1]
        sigma_cross_omega = torch.cross(sigma, omega, dim=1)                # s x w   [S, 3]
        term_1 = (1.0 - sigma_squared) * omega
        term_2 = 2.0 * sigma_cross_omega
        term_3 = 2.0 * sigma_dot_omega * sigma
        sigma_change = 0.25 * (term_1 + term_2 + term_3)

        # Step 4: put the two parts back together
        return torch.cat([sigma_change, omega_change], dim=1)

    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def shadow_switch(state: torch.Tensor) -> torch.Tensor:
        """If |sigma| > 1, use the equivalent short-way-round attitude  -sigma / |sigma|^2."""
        sigma = state[:, 0:3]
        omega = state[:, 3:6]
        sigma_squared = (sigma * sigma).sum(dim=1, keepdim=True)
        needs_switch = sigma_squared > 1.0
        switched = -sigma / torch.clamp(sigma_squared, min=1e-12)
        sigma = torch.where(needs_switch, switched, sigma)
        return torch.cat([sigma, omega], dim=1)

    def step(self, state: torch.Tensor, torque: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
        """
        One control step of length dt (Runge-Kutta, `substeps` small steps), push held constant.
            state [S, 6], torque [S, 3], inertia [S, 3]  ->  next state [S, 6]
        """
        # Step 1: the push can never be larger than the wheels allow
        torque = torch.clamp(torque, -self.u_max, self.u_max)

        # Step 2: small Runge-Kutta steps
        small = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.derivative(state, torque, inertia)
            k2 = self.derivative(state + 0.5 * small * k1, torque, inertia)
            k3 = self.derivative(state + 0.5 * small * k2, torque, inertia)
            k4 = self.derivative(state + small * k3, torque, inertia)
            state = state + (small / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state = self.shadow_switch(state)
        return state

    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def pointing_error_degrees(state: torch.Tensor) -> torch.Tensor:
        """Angle between where the spacecraft points and the target: 4 x atan(|sigma|), in degrees. [S]"""
        sigma = state[:, 0:3]
        size = torch.sqrt((sigma * sigma).sum(dim=1))
        return 4.0 * torch.atan(size) * 180.0 / math.pi

    @staticmethod
    def spin_rate(state: torch.Tensor) -> torch.Tensor:
        """How fast the spacecraft is turning, radians per second. [S]"""
        omega = state[:, 3:6]
        return torch.sqrt((omega * omega).sum(dim=1))

    @staticmethod
    def sigma_from_angle(axis: torch.Tensor, angle_degrees: torch.Tensor) -> torch.Tensor:
        """Attitude for a rotation of `angle_degrees` about the unit vector `axis`: sigma = axis x tan(angle / 4)."""
        angle_radians = angle_degrees * math.pi / 180.0
        size = torch.tan(angle_radians / 4.0)
        return axis * size.unsqueeze(1)
