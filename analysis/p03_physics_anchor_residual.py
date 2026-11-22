"""
Mathematical check for pilot B follow-up: how much smaller is the network's job if a CHEAP physics step
predicts most of each step and the network learns only the remainder ("physics anchor")?

Pilot B (seed 0): learned + running correction 103.5 m against low-fidelity physics + correction 66.0 m and
true physics 47.6 m. The learned model's own fitting error (vs the low-fidelity model it was trained on) was
524 m after 16 steps, about 1/5 of the low-fidelity model's drift from the real asteroid (2,661 m). The
correction removes most of a slowly changing error, but the fitting error changes with position, so more of it
stays.

Cheap physics step: the asteroid as ONE point mass (same total mass as the low-fidelity model) + the
low-fidelity spin vector's Coriolis and centrifugal terms, ONE Runge-Kutta step of 300 s (the low-fidelity model
uses 82 mass points and 4 Runge-Kutta steps).

Measured on states like the training data (shell 25-70 km, roughly orbital speeds, random kicks):
    full step change         = low-fidelity next state - current state          (what the network learns now)
    remainder after the cheap step = low-fidelity next state - cheap-step next state   (what it would learn)
If the network keeps the same RELATIVE accuracy, its fitting error shrinks by the ratio of the two sizes.

Run:  python analysis/p03_physics_anchor_residual.py      (NumPy only, under a minute)
"""

import math

import numpy as np

GM = 4.463e-4
SPIN_PERIOD_HOURS = 5.27
LOBES = [
    {"center": [-7.0, 0.0, 0.0], "semi_axes": [10.0, 6.0, 5.5]},
    {"center": [8.0, 1.0, 0.0], "semi_axes": [9.0, 5.0, 4.5]},
]
SPACING_LOW = 3.0
DENSITY_FACTOR_LOW = 1.10
SPIN_FACTOR_LOW = 1.01
SPIN_TILT_LOW_DEGREES = 3.0
SOFTENING = 0.5
DT = 300.0
SUBSTEPS_LOW = 4
PUSH_LIMIT = 5.0e-4
SAMPLES = 2000


def inside_shape(point):
    for lobe in LOBES:
        total = 0.0
        for axis in range(3):
            total = total + ((point[axis] - lobe["center"][axis]) / lobe["semi_axes"][axis]) ** 2
        if total <= 1.0:
            return True
    return False


def mass_points(spacing):
    points = []
    values = np.arange(-20.0, 20.0 + 1e-9, spacing)
    for x in values:
        for y in values:
            for z in values:
                if inside_shape([x, y, z]):
                    points.append([x, y, z])
    points = np.array(points)
    return points - points.mean(axis=0)


POINTS_LOW = mass_points(SPACING_LOW)
SPIN = 2.0 * math.pi / (SPIN_PERIOD_HOURS * 3600.0)
TILT = SPIN_TILT_LOW_DEGREES * math.pi / 180.0
SPIN_LOW = SPIN_FACTOR_LOW * SPIN * np.array([math.sin(TILT), 0.0, math.cos(TILT)])
GM_LOW = GM * DENSITY_FACTOR_LOW


def gravity_mass_points(position):
    difference = POINTS_LOW - position
    squared = (difference * difference).sum(axis=1)
    distance_cubed = (squared + SOFTENING ** 2) ** 1.5
    return (GM_LOW / len(POINTS_LOW)) * (difference / distance_cubed[:, None]).sum(axis=0)


def gravity_point_mass(position):
    distance = math.sqrt(float(position @ position))
    return -GM_LOW * position / distance ** 3


def derivative(state, gravity_function):
    position = state[0:3]
    velocity = state[3:6]
    acceleration = gravity_function(position)
    acceleration = acceleration - 2.0 * np.cross(SPIN_LOW, velocity)
    acceleration = acceleration - np.cross(SPIN_LOW, np.cross(SPIN_LOW, position))
    return np.concatenate([velocity, acceleration])


def step(state, push, gravity_function, substeps):
    state = state.copy()
    state[3:6] = state[3:6] + push
    small = DT / substeps
    for _ in range(substeps):
        k1 = derivative(state, gravity_function)
        k2 = derivative(state + 0.5 * small * k1, gravity_function)
        k3 = derivative(state + 0.5 * small * k2, gravity_function)
        k4 = derivative(state + small * k3, gravity_function)
        state = state + (small / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def main():
    rng = np.random.default_rng(0)
    full_changes = []
    remainders_one_step = []
    remainders_two_steps = []
    distances = []
    for _ in range(SAMPLES):
        # a start like the training data
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        radius = rng.uniform(25.0, 70.0)
        position = radius * direction
        tangent = np.cross(direction, rng.normal(size=3))
        tangent = tangent / np.linalg.norm(tangent)
        speed = math.sqrt(GM / radius) * rng.uniform(0.85, 1.15)
        velocity = speed * tangent - np.cross(np.array([0.0, 0.0, SPIN]), position)
        state = np.concatenate([position, velocity])
        push = np.where(rng.uniform(size=3) < 0.3, rng.uniform(-PUSH_LIMIT, PUSH_LIMIT, size=3), 0.0)

        low = step(state, push, gravity_mass_points, SUBSTEPS_LOW)
        cheap_one = step(state, push, gravity_point_mass, 1)
        cheap_two = step(state, push, gravity_point_mass, 2)
        full_changes.append(low - state)
        remainders_one_step.append(low - cheap_one)
        remainders_two_steps.append(low - cheap_two)
        distances.append(radius)

    full_changes = np.array(full_changes)
    remainders_one_step = np.array(remainders_one_step)
    remainders_two_steps = np.array(remainders_two_steps)

    def sizes(array):
        position_part = 1000.0 * np.sqrt((array[:, 0:3] ** 2).sum(axis=1)).std()
        velocity_part = 1000.0 * np.sqrt((array[:, 3:6] ** 2).sum(axis=1)).std()
        position_rms = 1000.0 * np.sqrt((array[:, 0:3] ** 2).sum(axis=1).mean())
        velocity_rms = 1000.0 * np.sqrt((array[:, 3:6] ** 2).sum(axis=1).mean())
        return position_rms, velocity_rms, position_part, velocity_part

    print("Pilot B follow-up — physics anchor check (low-fidelity asteroid, one 300 s step)")
    print(f"  {SAMPLES} states, distance 25-70 km; sizes are root-mean-square over the states")
    print()
    print("                                         position (m)   velocity (m/s)")
    full = sizes(full_changes)
    one = sizes(remainders_one_step)
    two = sizes(remainders_two_steps)
    print(f"  full step change (learned now)        {full[0]:12.1f}   {full[1]:12.4f}")
    print(f"  remainder after cheap step (1 RK)     {one[0]:12.1f}   {one[1]:12.4f}")
    print(f"  remainder after cheap step (2 RK)     {two[0]:12.1f}   {two[1]:12.4f}")
    print()
    print(f"  shrink factor, 1 Runge-Kutta step:  position {full[0] / one[0]:6.0f}x   velocity {full[1] / one[1]:6.0f}x")
    print(f"  shrink factor, 2 Runge-Kutta steps: position {full[0] / two[0]:6.0f}x   velocity {full[1] / two[1]:6.0f}x")

    # how the remainder depends on distance (the irregular shape matters most close in)
    print()
    print("  remainder (1 RK) by distance:   position (m)   velocity (mm/s)")
    distances = np.array(distances)
    for low_edge, high_edge in [(25, 35), (35, 50), (50, 70)]:
        chosen = (distances >= low_edge) & (distances < high_edge)
        part = remainders_one_step[chosen]
        p = 1000.0 * np.sqrt((part[:, 0:3] ** 2).sum(axis=1).mean())
        v = 1e6 * np.sqrt((part[:, 3:6] ** 2).sum(axis=1).mean())
        print(f"    {low_edge}-{high_edge} km                    {p:8.2f}        {v:8.3f}")


if __name__ == "__main__":
    main()
