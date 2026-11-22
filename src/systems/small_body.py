"""
A spacecraft moving near a small, irregular, spinning asteroid. Used by space pilot B
(configs/pilots/pilot_b_asteroid_proximity.yaml), not by the certificate paper.

The same problem as Cesarini and Silvestrini (Politecnico di Milano, "Transformer-based Neural Predictive
Control for Small-Body Proximity Operations"): translational motion near asteroid 433 Eros, written in the
asteroid's own spinning frame, controlled with small velocity kicks. They use a polyhedral gravity model
built from Eros's real shape (498 faces = "high fidelity", 98 faces = "low fidelity"). This pilot uses a
simpler gravity model of an Eros-LIKE shape (to disclose): the shape is filled with equal point masses
("mass concentrations"); fine spacing = high fidelity (the real asteroid), coarse spacing = low fidelity.
The low-fidelity model also gets the paper's errors: density +10%, spin rate +1%, spin axis tilted 3 degrees.

----------------------------------------------------------------------------------------------------------
The state (6 numbers) and the push (3 numbers), in kilometres and seconds
----------------------------------------------------------------------------------------------------------
    position  [3]  km, in the asteroid's spinning frame (origin = asteroid centre)
    velocity  [3]  km/s, in the same frame
    push      [3]  velocity kick (km/s) applied at the start of each step, |push| <= dv_max per axis

----------------------------------------------------------------------------------------------------------
The equations (motion in a spinning frame)
----------------------------------------------------------------------------------------------------------
    d(position)/dt = velocity
    d(velocity)/dt = gravity(position) - 2 W x velocity - W x (W x position)
                     (W = spin vector of the asteroid; the last two terms are the Coriolis and
                      centrifugal effects of the spinning frame)
    gravity(position) = sum over mass points of  GM_point x (point - position) / (distance^2 + soft^2)^(3/2)
"""

from __future__ import annotations

import math

import torch


def point_inside_shape(point, lobes) -> bool:
    """True if the point [x, y, z] (km) lies inside at least one of the ellipsoid lobes."""
    for lobe in lobes:
        total = 0.0
        for axis in range(3):
            distance = point[axis] - lobe["center"][axis]
            total = total + (distance / lobe["semi_axes"][axis]) ** 2
        if total <= 1.0:
            return True
    return False


def build_mass_points(lobes, spacing: float, half_size: float = 20.0) -> torch.Tensor:
    """Fill the shape with points on a grid of the given spacing (km). Returns [M, 3], centred on the mass centre."""
    # Step 1: the grid values along one axis
    values = []
    value = -half_size
    while value <= half_size + 1e-9:
        values.append(value)
        value = value + spacing

    # Step 2: keep every grid point that lies inside the shape
    points = []
    for x in values:
        for y in values:
            for z in values:
                if point_inside_shape([x, y, z], lobes):
                    points.append([x, y, z])
    points = torch.tensor(points, dtype=torch.float32)

    # Step 3: move the points so their centre (the centre of mass) is at the origin
    centre = points.mean(dim=0)
    points = points - centre
    return points


class SmallBody:
    n_x = 6
    n_u = 3

    def __init__(self, gm: float, spin_rate: float, spin_tilt_degrees: float, mass_points: torch.Tensor,
                 softening: float, dt: float, substeps: int, dv_max: float):
        # Step 1: total gravity strength, spread equally over the mass points
        self.gm = float(gm)
        self.mass_points = mass_points
        self.gm_each = self.gm / mass_points.shape[0]
        self.softening = float(softening)

        # Step 2: the spin vector. Tilt = angle between this model's spin axis and the true one (z axis)
        tilt = spin_tilt_degrees * math.pi / 180.0
        self.spin_vector = [spin_rate * math.sin(tilt), 0.0, spin_rate * math.cos(tilt)]

        # Step 3: time stepping and the push limit
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.u_max = float(dv_max)

    @classmethod
    def from_config(cls, cfg: dict, fidelity: str) -> "SmallBody":
        """fidelity = "high" (the real asteroid) or "low" (the model used for training and planning)."""
        s = cfg["system"]
        spin_rate = 2.0 * math.pi / (float(s["spin_period_hours"]) * 3600.0)
        if fidelity == "high":
            points = build_mass_points(s["lobes"], float(s["spacing_high"]))
            return cls(gm=s["gm"], spin_rate=spin_rate, spin_tilt_degrees=0.0, mass_points=points,
                       softening=s["softening"], dt=s["dt"], substeps=s["substeps"], dv_max=s["dv_max"])
        low = s["low_fidelity"]
        points = build_mass_points(s["lobes"], float(low["spacing"]))
        return cls(gm=float(s["gm"]) * float(low["density_factor"]),
                   spin_rate=spin_rate * float(low["spin_factor"]),
                   spin_tilt_degrees=float(low["spin_tilt_degrees"]), mass_points=points,
                   softening=s["softening"], dt=s["dt"], substeps=s["substeps"], dv_max=s["dv_max"])

    # ------------------------------------------------------------------------------------------------------
    def gravity(self, position: torch.Tensor) -> torch.Tensor:
        """
        Gravity at every position [S, 3] (km) from all mass points. Returns [S, 3] (km/s^2).
        A Python loop over hundreds of points would be far too slow, so all points are done at once:
            difference[s, m, :] = point m - position s
        """
        # Step 0: keep the mass points on the same device as the positions (copied only when it changes)
        if self.mass_points.device != position.device:
            self.mass_points = self.mass_points.to(position.device)
        points = self.mass_points

        # Step 1: vector from every spacecraft to every mass point   [S, M, 3]
        points_expanded = points.unsqueeze(0)               # [1, M, 3]
        positions_expanded = position.unsqueeze(1)          # [S, 1, 3]
        difference = points_expanded - positions_expanded   # [S, M, 3]

        # Step 2: softened distance cubed   [S, M]
        squared = (difference * difference).sum(dim=2)
        distance_cubed = (squared + self.softening ** 2) ** 1.5

        # Step 3: add up the pull of every point   [S, 3]
        pull = difference / distance_cubed.unsqueeze(2)
        return self.gm_each * pull.sum(dim=1)

    def derivative(self, state: torch.Tensor) -> torch.Tensor:
        """[S, 6] -> [S, 6]: velocity, and gravity + Coriolis + centrifugal acceleration."""
        # Step 1: split the state
        position = state[:, 0:3]
        velocity = state[:, 3:6]

        # Step 2: the spin vector as a tensor of the same shape as the positions
        spin = torch.tensor(self.spin_vector, dtype=state.dtype, device=state.device)
        spin = spin.expand_as(position)

        # Step 3: the three accelerations
        gravity = self.gravity(position)
        coriolis = -2.0 * torch.cross(spin, velocity, dim=1)
        centrifugal = -torch.cross(spin, torch.cross(spin, position, dim=1), dim=1)
        acceleration = gravity + coriolis + centrifugal
        return torch.cat([velocity, acceleration], dim=1)

    def step(self, state: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        """One step: kick the velocity by `push` (clipped), then coast for dt (Runge-Kutta substeps)."""
        # Step 1: the kick, never larger than the thrusters allow
        push = torch.clamp(push, -self.u_max, self.u_max)
        position = state[:, 0:3]
        velocity = state[:, 3:6] + push
        state = torch.cat([position, velocity], dim=1)

        # Step 2: coast
        small = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.derivative(state)
            k2 = self.derivative(state + 0.5 * small * k1)
            k3 = self.derivative(state + 0.5 * small * k2)
            k4 = self.derivative(state + small * k3)
            state = state + (small / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    @staticmethod
    def distance(state: torch.Tensor) -> torch.Tensor:
        """Distance from the asteroid centre, km. [S]"""
        position = state[:, 0:3]
        return torch.sqrt((position * position).sum(dim=1))


# ==========================================================================================================
# Reference orbits (what the controller must follow)
# ==========================================================================================================
def reference_orbit(radius: float, inclination_degrees: float, node_degrees: float, phase_degrees: float,
                    gm: float, spin_rate: float, steps: int, dt: float) -> torch.Tensor:
    """
    A circular orbit, fixed in space, as seen from the spinning asteroid frame (true spin about z).
    Returns [steps + 1, 6]: position and velocity at times 0, dt, 2 dt, ...
    """
    # Step 1: the orbit plane (two unit vectors e1, e2) from the inclination and the node angle
    inclination = inclination_degrees * math.pi / 180.0
    node = node_degrees * math.pi / 180.0
    e1 = [math.cos(node), math.sin(node), 0.0]
    e2 = [-math.cos(inclination) * math.sin(node), math.cos(inclination) * math.cos(node), math.sin(inclination)]

    # Step 2: how fast the orbit goes round
    orbit_rate = math.sqrt(gm / radius ** 3)
    phase = phase_degrees * math.pi / 180.0

    states = []
    for k in range(steps + 1):
        time = k * dt

        # Step 3: position and velocity in the non-spinning frame
        angle = orbit_rate * time + phase
        position_fixed = []
        velocity_fixed = []
        for axis in range(3):
            position_fixed.append(radius * (math.cos(angle) * e1[axis] + math.sin(angle) * e2[axis]))
            velocity_fixed.append(radius * orbit_rate * (-math.sin(angle) * e1[axis] + math.cos(angle) * e2[axis]))

        # Step 4: turn into the spinning frame (rotate back by the asteroid's angle about z)
        body_angle = spin_rate * time
        c = math.cos(body_angle)
        s = math.sin(body_angle)
        px = c * position_fixed[0] + s * position_fixed[1]
        py = -s * position_fixed[0] + c * position_fixed[1]
        pz = position_fixed[2]
        vx = c * velocity_fixed[0] + s * velocity_fixed[1]
        vy = -s * velocity_fixed[0] + c * velocity_fixed[1]
        vz = velocity_fixed[2]

        # Step 5: velocity seen from the spinning frame = rotated velocity - W x position  (W = [0, 0, spin])
        vx = vx + spin_rate * py
        vy = vy - spin_rate * px
        states.append([px, py, pz, vx, vy, vz])
    return torch.tensor(states, dtype=torch.float32)


# ==========================================================================================================
# The cheap physics step used by the physics anchor (pilot B2)
# ==========================================================================================================
class PointMassStep:
    """
    A CHEAP physics step: the asteroid as ONE point mass + the spinning frame's Coriolis and centrifugal terms,
    one Runge-Kutta step per control step. It uses only what the low-fidelity model knows (its total mass and
    its spin vector), never the real asteroid. No mass-point sum, so its cost does not depend on how detailed
    the gravity model is.

    Mathematical check (analysis/p03_physics_anchor_residual.py): on states like the training data, what this
    step gets wrong is 2,262 times smaller in position and 124 times smaller in velocity than the full step
    change, so the network only has to learn a small remainder (the irregular shape).
    """

    def __init__(self, gm: float, spin_vector, dt: float, substeps: int, dv_max: float):
        self.gm = float(gm)
        self.spin_vector = [float(value) for value in spin_vector]
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.u_max = float(dv_max)

    @classmethod
    def from_low_fidelity(cls, low_body: SmallBody, substeps: int) -> "PointMassStep":
        """The same total mass and spin vector as the low-fidelity model (what the planner is allowed to know)."""
        return cls(gm=low_body.gm, spin_vector=low_body.spin_vector, dt=low_body.dt, substeps=substeps,
                   dv_max=low_body.u_max)

    def derivative(self, state: torch.Tensor) -> torch.Tensor:
        # Step 1: split the state
        position = state[..., 0:3]
        velocity = state[..., 3:6]

        # Step 2: point-mass gravity  -GM r / |r|^3
        squared = (position * position).sum(dim=-1, keepdim=True)
        distance_cubed = torch.clamp(squared, min=1e-6) ** 1.5
        gravity = -self.gm * position / distance_cubed

        # Step 3: Coriolis and centrifugal terms of the spinning frame
        spin = torch.tensor(self.spin_vector, dtype=state.dtype, device=state.device)
        spin = spin.expand_as(position)
        coriolis = -2.0 * torch.cross(spin, velocity, dim=-1)
        centrifugal = -torch.cross(spin, torch.cross(spin, position, dim=-1), dim=-1)
        return torch.cat([velocity, gravity + coriolis + centrifugal], dim=-1)

    def step(self, state: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        """Kick the velocity by `push` (clipped), then coast for dt. Works for any leading shape [..., 6]."""
        push = torch.clamp(push, -self.u_max, self.u_max)
        position = state[..., 0:3]
        velocity = state[..., 3:6] + push
        state = torch.cat([position, velocity], dim=-1)
        small = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.derivative(state)
            k2 = self.derivative(state + 0.5 * small * k1)
            k3 = self.derivative(state + 0.5 * small * k2)
            k4 = self.derivative(state + small * k3)
            state = state + (small / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state
