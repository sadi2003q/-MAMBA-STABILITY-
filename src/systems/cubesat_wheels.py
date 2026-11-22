"""
A small satellite (CubeSat) turned by three reaction wheels. Used by space pilot A
(configs/pilots/pilot_a_attitude_offset.yaml), not by the certificate paper.

The same kind of spacecraft as Cena, Martini and Chiaberge (2026, arXiv 2602.15954): a CubeSat whose
pointing is controlled with reaction wheels. They generated data with the Basilisk simulator; this file is a
smaller simulator of the same physics, written here so the pilot needs no extra library (to disclose).

----------------------------------------------------------------------------------------------------------
The state (9 numbers) and the push (3 numbers)
----------------------------------------------------------------------------------------------------------
    sigma        [3]  attitude as modified Rodrigues parameters. Rotation angle = 4 x atan(|sigma|).
                      sigma = 0 means "pointing exactly at the target".
    omega        [3]  angular velocity of the satellite body (radians per second), in its own axes
    wheel_speed  [3]  speed of each reaction wheel relative to the body (radians per second)
    push         [3]  motor torque on each wheel (newton metres), limited to |push| <= u_max per axis.
                      The wheel pushes back on the body with the opposite torque.

----------------------------------------------------------------------------------------------------------
The equations (standard rigid body with three wheels on the body axes)
----------------------------------------------------------------------------------------------------------
    friction       f = c_viscous x wheel_speed + c_coulomb x tanh(wheel_speed / 1 rad/s)
    wheel          d(wheel_speed)/dt = (push - f) / J_wheel
    body           J d(omega)/dt = - omega x (J omega + J_wheel wheel_speed) - (push - f)
    attitude       d(sigma)/dt = 1/4 [ (1 - s.s) w + 2 (s x w) + 2 (s.w) s ]
    After every small step: if |sigma| > 1, use the equivalent short-way-round attitude -sigma / |sigma|^2.

Friction is the effect Cena et al. list as missing from their training data. In this pilot the training data
has NO friction (as theirs), and the test is run both without and with friction.
"""

from __future__ import annotations

import math

import torch


class CubeSatWithWheels:
    n_x = 9
    n_u = 3

    def __init__(self, dt: float, substeps: int, inertia, wheel_inertia: float, u_max: float):
        # Step 1: time between two control decisions, and how finely each step is simulated
        self.dt = float(dt)
        self.substeps = int(substeps)

        # Step 2: body moment of inertia (one number per axis), wheel inertia, push limit
        self.inertia = [float(value) for value in inertia]
        self.wheel_inertia = float(wheel_inertia)
        self.u_max = float(u_max)

    @classmethod
    def from_config(cls, cfg: dict) -> "CubeSatWithWheels":
        s = cfg["system"]
        return cls(dt=s["dt"], substeps=s["substeps"], inertia=s["inertia"], wheel_inertia=s["wheel_inertia"],
                   u_max=s["u_max"])

    # ------------------------------------------------------------------------------------------------------
    def friction_torque(self, wheel_speed: torch.Tensor, viscous: float, coulomb: float) -> torch.Tensor:
        """Wheel bearing friction [S, 3] (newton metres). Zero when both coefficients are zero."""
        viscous_part = viscous * wheel_speed
        coulomb_part = coulomb * torch.tanh(wheel_speed / 1.0)
        return viscous_part + coulomb_part

    def derivative(self, state: torch.Tensor, push: torch.Tensor, viscous: float, coulomb: float) -> torch.Tensor:
        """
        How fast the state changes right now.
            state [S, 9], push [S, 3]   ->   [S, 9]
        """
        # Step 1: split the state
        sigma = state[:, 0:3]
        omega = state[:, 3:6]
        wheel_speed = state[:, 6:9]

        # Step 2: the torque that really reaches each wheel (motor push minus friction)
        friction = self.friction_torque(wheel_speed, viscous, coulomb)
        wheel_torque = push - friction

        # Step 3: wheels speed up
        wheel_change = wheel_torque / self.wheel_inertia

        # Step 4: body:  J d(omega)/dt = - omega x (J omega + J_wheel wheel_speed) - wheel_torque
        inertia = torch.tensor(self.inertia, dtype=state.dtype, device=state.device)
        body_momentum = inertia * omega
        wheel_momentum = self.wheel_inertia * wheel_speed
        total_momentum = body_momentum + wheel_momentum
        gyroscopic = torch.cross(omega, total_momentum, dim=1)
        omega_change = (-gyroscopic - wheel_torque) / inertia

        # Step 5: attitude:  d(sigma)/dt = 1/4 [ (1 - s.s) w + 2 (s x w) + 2 (s.w) s ]
        sigma_squared = (sigma * sigma).sum(dim=1, keepdim=True)
        sigma_dot_omega = (sigma * omega).sum(dim=1, keepdim=True)
        sigma_cross_omega = torch.cross(sigma, omega, dim=1)
        term_1 = (1.0 - sigma_squared) * omega
        term_2 = 2.0 * sigma_cross_omega
        term_3 = 2.0 * sigma_dot_omega * sigma
        sigma_change = 0.25 * (term_1 + term_2 + term_3)

        # Step 6: put the three parts back together
        return torch.cat([sigma_change, omega_change, wheel_change], dim=1)

    @staticmethod
    def shadow_switch(state: torch.Tensor) -> torch.Tensor:
        """If |sigma| > 1, use the equivalent short-way-round attitude  -sigma / |sigma|^2."""
        sigma = state[:, 0:3]
        rest = state[:, 3:]
        sigma_squared = (sigma * sigma).sum(dim=1, keepdim=True)
        needs_switch = sigma_squared > 1.0
        switched = -sigma / torch.clamp(sigma_squared, min=1e-12)
        sigma = torch.where(needs_switch, switched, sigma)
        return torch.cat([sigma, rest], dim=1)

    def step(self, state: torch.Tensor, push: torch.Tensor, viscous: float = 0.0, coulomb: float = 0.0):
        """
        One control step of length dt (Runge-Kutta, `substeps` small steps), push held constant.
        viscous / coulomb = friction coefficients (0 = no friction, as in the training data).
        """
        # Step 1: the push can never be larger than the wheels allow
        push = torch.clamp(push, -self.u_max, self.u_max)

        # Step 2: small Runge-Kutta steps
        small = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.derivative(state, push, viscous, coulomb)
            k2 = self.derivative(state + 0.5 * small * k1, push, viscous, coulomb)
            k3 = self.derivative(state + 0.5 * small * k2, push, viscous, coulomb)
            k4 = self.derivative(state + small * k3, push, viscous, coulomb)
            state = state + (small / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state = self.shadow_switch(state)
        return state

    # ------------------------------------------------------------------------------------------------------
    def linear_step(self, state: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        """
        The simple LINEAR model of Cena et al.'s hybrid controller: the satellite near the target, no
        friction, no gyroscopic coupling, small angles (sigma ~ angle / 4). Exact for a constant push:
            omega_next  = omega - dt x push / J
            sigma_next  = sigma + (dt x omega - dt^2 / 2 x push / J) / 4
            wheel_next  = wheel + dt x push / J_wheel
        """
        # Step 1: the push can never be larger than the wheels allow
        push = torch.clamp(push, -self.u_max, self.u_max)

        # Step 2: split the state
        sigma = state[:, 0:3]
        omega = state[:, 3:6]
        wheel_speed = state[:, 6:9]
        inertia = torch.tensor(self.inertia, dtype=state.dtype, device=state.device)

        # Step 3: the three linear updates
        body_acceleration = -push / inertia
        new_omega = omega + self.dt * body_acceleration
        angle_change = self.dt * omega + 0.5 * self.dt * self.dt * body_acceleration
        new_sigma = sigma + 0.25 * angle_change
        new_wheel_speed = wheel_speed + self.dt * push / self.wheel_inertia
        return torch.cat([new_sigma, new_omega, new_wheel_speed], dim=1)

    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def pointing_error_degrees(state: torch.Tensor) -> torch.Tensor:
        """Angle between where the satellite points and the target: 4 x atan(|sigma|), in degrees. [S]"""
        sigma = state[:, 0:3]
        size = torch.sqrt((sigma * sigma).sum(dim=1))
        return 4.0 * torch.atan(size) * 180.0 / math.pi

    @staticmethod
    def spin_rate(state: torch.Tensor) -> torch.Tensor:
        """How fast the satellite body is turning, radians per second. [S]"""
        omega = state[:, 3:6]
        return torch.sqrt((omega * omega).sum(dim=1))

    @staticmethod
    def sigma_from_angle(axis: torch.Tensor, angle_degrees: torch.Tensor) -> torch.Tensor:
        """Attitude for a rotation of `angle_degrees` about the unit vector `axis`: sigma = axis x tan(angle / 4)."""
        angle_radians = angle_degrees * math.pi / 180.0
        size = torch.tan(angle_radians / 4.0)
        return axis * size.unsqueeze(1)
