"""
A spacecraft that turns (one axis, in a plane) during an engine burn, with liquid fuel sloshing in its tank.

The fuel slosh is modelled with the standard "equivalent mechanical model": the moving part of the fuel acts like
a pendulum hanging from a hinge behind the spacecraft's centre of mass. The engine's thrust pushes the
spacecraft forward, so inside the spacecraft the fuel feels a pull backwards (like gravity), and the pendulum
swings back and forth around it. Only the spacecraft is measured (angle and turn rate); the fuel is NOT.

State (4 numbers):  [angle (rad), turn rate (rad/s), slosh angle (rad), slosh rate (rad/s)]
Push (1 number):    turning torque from the attitude actuators (N m), limited to +-u_max

Equations (derived with Lagrange's method, check: /tmp/derive.py logic is repeated in analysis/p04):
    p = hinge-to-fuel position in the spacecraft frame = [-(b + l cos(slosh)), l sin(slosh)]
    mass matrix M = [[J + m b^2 + 2 m b l cos(slosh) + m l^2,   -m l (b cos(slosh) + l)],
                     [-m l (b cos(slosh) + l),                    m l^2               ]]
    right side  r = [ u + m a l sin(slosh) - m b l sin(slosh) slosh_rate^2 + 2 m b l sin(slosh) slosh_rate turn_rate,
                     -m a l sin(slosh) - m b l sin(slosh) turn_rate^2 - c slosh_rate ]
    M [angle_acceleration, slosh_acceleration] = r

Simplification (to disclose): the spacecraft's forward acceleration a is taken as constant (the pendulum's pull on
the forward motion is ignored). The fill level changes the slosh mass, pendulum length and inertia, as in
equivalent-mechanical-model tables: a fuller tank -> shorter pendulum -> faster slosh.
"""

from __future__ import annotations

import math

import torch


class SloshSpacecraft:
    def __init__(self, dt: float = 0.2, substeps: int = 4, u_max: float = 0.4, dry_inertia: float = 8.0,
                 hinge_offset: float = 0.3, thrust_acceleration: float = 0.5, tank_fuel_mass: float = 30.0,
                 damping_ratio: float = 0.01):
        self.dt = float(dt)                              # seconds between control decisions
        self.substeps = int(substeps)                    # Runge-Kutta steps inside one control step
        self.u_max = float(u_max)                        # largest turning torque, N m
        self.dry_inertia = float(dry_inertia)            # spacecraft without fuel, kg m^2
        self.hinge_offset = float(hinge_offset)          # b: hinge distance behind the centre of mass, m
        self.thrust_acceleration = float(thrust_acceleration)   # a: forward acceleration during the burn, m/s^2
        self.tank_fuel_mass = float(tank_fuel_mass)      # fuel mass when the tank is full, kg
        self.damping_ratio = float(damping_ratio)        # slosh damping ratio (light: about 1%)

    # ------------------------------------------------------------------------------------------------------
    # Fill-level-dependent numbers
    # ------------------------------------------------------------------------------------------------------
    def parameters(self, fill):
        """
        fill = fraction of the tank that is full (0.3 to 0.9), a float or a tensor [S].
        Returns J (spacecraft + fixed fuel inertia), m (sloshing mass), l (pendulum length), b, a, c (damping).
        """
        if not torch.is_tensor(fill):
            fill_value = float(fill)
            fuel = self.tank_fuel_mass * fill_value
            slosh_mass = fuel * (0.9 - 0.6 * fill_value)
            length = 0.18 * (1.1 - fill_value)
            fixed_fuel = fuel - slosh_mass
            inertia = self.dry_inertia + fixed_fuel * self.hinge_offset ** 2
            pendulum_frequency = math.sqrt(self.thrust_acceleration / length)
            damping = 2.0 * self.damping_ratio * pendulum_frequency * slosh_mass * length ** 2
            return {"J": inertia, "m": slosh_mass, "l": length, "b": self.hinge_offset,
                    "a": self.thrust_acceleration, "c": damping}

        fuel = self.tank_fuel_mass * fill
        slosh_mass = fuel * (0.9 - 0.6 * fill)
        length = 0.18 * (1.1 - fill)
        fixed_fuel = fuel - slosh_mass
        inertia = self.dry_inertia + fixed_fuel * self.hinge_offset ** 2
        pendulum_frequency = torch.sqrt(self.thrust_acceleration / length)
        damping = 2.0 * self.damping_ratio * pendulum_frequency * slosh_mass * length ** 2
        return {"J": inertia, "m": slosh_mass, "l": length, "b": torch.full_like(fill, self.hinge_offset),
                "a": torch.full_like(fill, self.thrust_acceleration), "c": damping}

    # ------------------------------------------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------------------------------------------
    def derivative(self, state: torch.Tensor, push: torch.Tensor, p: dict) -> torch.Tensor:
        """state [S, 4], push [S, 1], p = parameters for each spacecraft ([S] tensors)  ->  d state / dt [S, 4]."""
        turn_rate = state[:, 1]
        slosh = state[:, 2]
        slosh_rate = state[:, 3]
        torque = push[:, 0]
        J, m, l, b, a, c = p["J"], p["m"], p["l"], p["b"], p["a"], p["c"]

        cos_s = torch.cos(slosh)
        sin_s = torch.sin(slosh)

        # Step 1: mass matrix entries
        m11 = J + m * b ** 2 + 2.0 * m * b * l * cos_s + m * l ** 2
        m12 = -m * l * (b * cos_s + l)
        m22 = m * l ** 2

        # Step 2: right-hand side
        r1 = (torque + m * a * l * sin_s - m * b * l * sin_s * slosh_rate ** 2
              + 2.0 * m * b * l * sin_s * slosh_rate * turn_rate)
        r2 = -m * a * l * sin_s - m * b * l * sin_s * turn_rate ** 2 - c * slosh_rate

        # Step 3: solve the 2 x 2 system  M [angle_acc, slosh_acc] = r
        determinant = m11 * m22 - m12 * m12
        angle_acceleration = (m22 * r1 - m12 * r2) / determinant
        slosh_acceleration = (m11 * r2 - m12 * r1) / determinant

        return torch.stack([turn_rate, angle_acceleration, slosh_rate, slosh_acceleration], dim=1)

    def step(self, state: torch.Tensor, push: torch.Tensor, fill: torch.Tensor, substeps=None) -> torch.Tensor:
        """
        One control step (dt) with classic Runge-Kutta, the push held constant and clipped to the limit.
        substeps: Runge-Kutta steps inside dt (default: the simulator's own; planners use 1 to save time).
        """
        push = torch.clamp(push, -self.u_max, self.u_max)
        p = self.parameters(fill)
        if substeps is None:
            substeps = self.substeps
        h = self.dt / substeps
        x = state
        for _ in range(substeps):
            k1 = self.derivative(x, push, p)
            k2 = self.derivative(x + 0.5 * h * k1, push, p)
            k3 = self.derivative(x + 0.5 * h * k2, push, p)
            k4 = self.derivative(x + h * k3, push, p)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def rigid_step(self, measured: torch.Tensor, push: torch.Tensor, nominal_fill: float) -> torch.Tensor:
        """
        The usual model WITHOUT slosh: all fuel frozen in place at the nominal fill level.
        measured [S, 2] = [angle, turn rate] -> next [S, 2].
        """
        push = torch.clamp(push, -self.u_max, self.u_max)
        p = self.parameters(nominal_fill)
        frozen_inertia = p["J"] + p["m"] * (p["b"] + p["l"]) ** 2
        angle_acceleration = push[:, 0] / frozen_inertia
        angle = measured[:, 0] + measured[:, 1] * self.dt + 0.5 * angle_acceleration * self.dt ** 2
        turn_rate = measured[:, 1] + angle_acceleration * self.dt
        return torch.stack([angle, turn_rate], dim=1)

    @staticmethod
    def measure(state: torch.Tensor) -> torch.Tensor:
        """What the sensors see: angle and turn rate only."""
        return state[:, :2]
