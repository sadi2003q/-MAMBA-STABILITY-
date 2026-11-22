"""
Mathematical check for space pilot B (spacecraft near an asteroid, Cesarini and Silvestrini 2026).

Questions, answered with NumPy only (no training):
    1. How wrong is the LOW-fidelity asteroid model (coarse mass points, density +10%, spin +1%, spin axis
       tilted 3 degrees, as in the paper) compared with the HIGH-fidelity one, at orbit distances 30-60 km?
    2. How far off would a 16-step (80-minute) prediction drift because of that error?  drift ~ 1/2 a T^2
    3. Can the thrusters beat that error? (push limit per step / step time = largest acceleration we can make)

Both models use the same Eros-like two-lobe shape filled with equal point masses ("mass concentrations").

Run:  python analysis/p02_asteroid_model_mismatch.py      (NumPy only, a few seconds)
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------
# Settings (the same numbers as configs/pilots/pilot_b_asteroid_proximity.yaml)
# ---------------------------------------------------------------------------------------------------------
GM = 4.463e-4                    # km^3 / s^2  (433 Eros)
SPIN_PERIOD_HOURS = 5.27         # 433 Eros
LOBES = [
    {"center": [-7.0, 0.0, 0.0], "semi_axes": [10.0, 6.0, 5.5]},
    {"center": [8.0, 1.0, 0.0], "semi_axes": [9.0, 5.0, 4.5]},
]
SPACING_HIGH = 1.5               # km between mass points, high fidelity
SPACING_LOW = 3.0                # km, low fidelity
DENSITY_FACTOR_LOW = 1.10        # +10%
SPIN_FACTOR_LOW = 1.01           # +1%
SPIN_TILT_LOW_DEGREES = 3.0
DT = 300.0                       # seconds per step (5 minutes)
HORIZON = 16                     # steps (80 minutes)
PUSH_LIMIT = 5.0e-4              # km/s per axis per step (0.5 metres per second)


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
    # centre of mass at the origin (so the orbit centre is the asteroid's centre)
    points = points - points.mean(axis=0)
    return points


def gravity(position, points, gm_total):
    """Acceleration at `position` [3] from equal point masses (km/s^2)."""
    acceleration = np.zeros(3)
    gm_each = gm_total / len(points)
    for point in points:
        difference = point - position
        distance = math.sqrt(float(difference @ difference) + 0.25)   # 0.5 km softening
        acceleration = acceleration + gm_each * difference / distance ** 3
    return acceleration


def frame_terms(position, velocity, spin_vector):
    """Coriolis + centrifugal acceleration in the rotating (asteroid-fixed) frame."""
    coriolis = -2.0 * np.cross(spin_vector, velocity)
    centrifugal = -np.cross(spin_vector, np.cross(spin_vector, position))
    return coriolis + centrifugal


def main():
    high = mass_points(SPACING_HIGH)
    low = mass_points(SPACING_LOW)
    spin = 2.0 * math.pi / (SPIN_PERIOD_HOURS * 3600.0)
    spin_high = np.array([0.0, 0.0, spin])
    tilt = SPIN_TILT_LOW_DEGREES * math.pi / 180.0
    spin_low = SPIN_FACTOR_LOW * spin * np.array([math.sin(tilt), 0.0, math.cos(tilt)])

    print("Space pilot B — mathematical check (no training)")
    print(f"  mass points: high fidelity {len(high)}, low fidelity {len(low)}")
    largest_push_acceleration = PUSH_LIMIT / DT
    print(f"  largest acceleration the thrusters can make (one axis): {largest_push_acceleration:.2e} km/s^2")
    print()
    print("  distance | gravity size | gravity error | frame-term error | total error | 80-min drift | error / push limit")

    rng = np.random.default_rng(0)
    for radius in [30.0, 40.0, 50.0, 60.0]:
        errors = []
        gravity_sizes = []
        gravity_errors = []
        frame_errors = []
        for _ in range(40):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            position = radius * direction
            # velocity of a circular orbit, seen in the rotating frame
            speed = math.sqrt(GM / radius)
            tangent = np.cross(direction, rng.normal(size=3))
            tangent = tangent / np.linalg.norm(tangent)
            velocity = speed * tangent - np.cross(spin_high, position)

            g_high = gravity(position, high, GM)
            g_low = gravity(position, low, GM * DENSITY_FACTOR_LOW)
            f_high = frame_terms(position, velocity, spin_high)
            f_low = frame_terms(position, velocity, spin_low)
            gravity_sizes.append(np.linalg.norm(g_high))
            gravity_errors.append(np.linalg.norm(g_low - g_high))
            frame_errors.append(np.linalg.norm(f_low - f_high))
            errors.append(np.linalg.norm((g_low + f_low) - (g_high + f_high)))

        mean_error = float(np.mean(errors))
        drift_km = 0.5 * mean_error * (HORIZON * DT) ** 2
        print(f"  {radius:5.0f} km |   {np.mean(gravity_sizes):.2e}  |   {np.mean(gravity_errors):.2e}  |"
              f"     {np.mean(frame_errors):.2e}     |  {mean_error:.2e}  |  {1000 * drift_km:7.0f} m   |"
              f"   {mean_error / largest_push_acceleration:.2f}")
    print()
    print("Reading: the low-fidelity model is clearly wrong (a drift of kilometres in 80 minutes, against the")
    print("paper's tracking errors of about 13 m), mostly from the tilted spin axis (the 'frame-term' column),")
    print("so the problem is real. The error is only 12-17% of the push limit, so a controller can cancel it")
    print("once it knows it (the running correction).")


if __name__ == "__main__":
    main()
