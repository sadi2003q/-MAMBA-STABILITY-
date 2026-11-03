"""a07 — Can the hidden state be rebuilt from a SHORT window of measurements?

Why this matters (Phase 2 of the Chain-Mamba paper plan):
    Phase 2 hides some states, hoping that Mamba's memory will then be needed. The memory-free control gets a
    window of the last k = 4 measurements. If the hidden state can be rebuilt exactly from 2 or 3
    measurements, the memory-free control already sees the full state, and memory has nothing to add.

What it checks (no training, NumPy only, a few seconds):
    Part 1  Van der Pol, forward Euler, only x1 measured.
            Since x1(t+1) = x1(t) + dt * x2(t), the velocity is x2(t) = (x1(t+1) - x1(t)) / dt, exactly.
    Part 2  Four-tank, Runge-Kutta, only levels 1 and 2 measured.
            Level 3 is rebuilt from two samples of level 1 with a simple finite-difference formula.

Result (3 November 2026): Van der Pol error 2e-15 (exact); four-tank median error 3 mm against a level spread
of 430 mm. Mathematical signal: NEGATIVE for Phase 2 as designed (noise-free data). Memory can only help when a
short window cannot stand in for it: measurement noise plus an unknown parameter per trajectory.

Usage:
    python analysis/a07_partial_observability.py
"""

import numpy as np


def vanderpol_part():
    # Step 1: settings (same as configs/systems/vanderpol.yaml)
    dt = 0.1
    mu = 1.0
    number_of_steps = 20000
    rng = np.random.default_rng(0)

    # Step 2: simulate with forward Euler and a random input
    x = np.zeros((number_of_steps + 1, 2))
    x[0] = [1.0, 0.0]
    u = rng.uniform(-1.0, 1.0, number_of_steps)
    for k in range(number_of_steps):
        x1 = x[k, 0]
        x2 = x[k, 1]
        next_x1 = x1 + dt * x2
        next_x2 = x2 + dt * (mu * (1.0 - x1 ** 2) * x2 - x1 + u[k])
        x[k + 1] = [next_x1, next_x2]

    # Step 3: rebuild x2 from two consecutive measurements of x1
    position_now = x[:-1, 0]
    position_next = x[1:, 0]
    velocity_rebuilt = (position_next - position_now) / dt
    velocity_true = x[:-1, 1]

    # Step 4: report
    largest_error = np.abs(velocity_rebuilt - velocity_true).max()
    print("Part 1 — Van der Pol, only x1 measured")
    print(f"  Largest error rebuilding the velocity from two position samples: {largest_error:.2e}")
    print("  The hidden velocity is known exactly from 2 measurements.")
    print()


def four_tank_vector_field(x, u, a, Sc, g, gamma_a, gamma_b):
    """Same equations as src/systems/four_tank.py (Cevaal et al., Equation 33)."""
    k = 1.0 / (3600.0 * Sc)
    outflow = a * np.sqrt(2.0 * g * np.maximum(x, 0.0)) / Sc
    dx1 = -outflow[0] + outflow[2] + gamma_a * k * u[0]
    dx2 = -outflow[1] + outflow[3] + gamma_b * k * u[1]
    dx3 = -outflow[2] + (1.0 - gamma_b) * k * u[1]
    dx4 = -outflow[3] + (1.0 - gamma_a) * k * u[0]
    return np.array([dx1, dx2, dx3, dx4])


def four_tank_part():
    # Step 1: settings (same as configs/systems/four_tank.yaml)
    Sc = 0.06
    g = 9.81
    gamma_a = 0.3
    gamma_b = 0.4
    a = np.array([1.31e-4, 1.51e-4, 9.27e-5, 8.82e-5])
    sample_time = 5.0
    substeps = 10
    h = sample_time / substeps
    number_of_steps = 20000
    rng = np.random.default_rng(0)

    # Step 2: piecewise-constant pump inputs, held for 10 to 60 samples
    u = np.zeros((number_of_steps, 2))
    current_input = rng.uniform(0.0, 4.0, 2)
    samples_left = 0
    for k in range(number_of_steps):
        if samples_left == 0:
            current_input = rng.uniform(0.0, 4.0, 2)
            samples_left = int(rng.integers(10, 61))
        u[k] = current_input
        samples_left = samples_left - 1

    # Step 3: simulate with fourth-order Runge-Kutta
    x = np.zeros((number_of_steps + 1, 4))
    x[0] = [0.5, 0.5, 0.3, 0.3]
    for k in range(number_of_steps):
        state = x[k].copy()
        for _ in range(substeps):
            k1 = four_tank_vector_field(state, u[k], a, Sc, g, gamma_a, gamma_b)
            k2 = four_tank_vector_field(state + 0.5 * h * k1, u[k], a, Sc, g, gamma_a, gamma_b)
            k3 = four_tank_vector_field(state + 0.5 * h * k2, u[k], a, Sc, g, gamma_a, gamma_b)
            k4 = four_tank_vector_field(state + h * k3, u[k], a, Sc, g, gamma_a, gamma_b)
            state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state = np.maximum(state, 0.0)
        x[k + 1] = state

    # Step 4: rebuild level 3 from two samples of level 1
    #   d(level 1)/dt = -outflow_1 + outflow_3 + gamma_a * k * u1
    #   so outflow_3 is about  slope of level 1 + outflow_1 - gamma_a * k * u1
    k_input = 1.0 / (3600.0 * Sc)
    slope_level1 = (x[1:, 0] - x[:-1, 0]) / sample_time
    outflow_1 = a[0] * np.sqrt(2.0 * g * x[:-1, 0]) / Sc
    outflow_3 = slope_level1 + outflow_1 - gamma_a * k_input * u[:, 0]
    outflow_3 = np.maximum(outflow_3, 0.0)
    level3_rebuilt = (outflow_3 * Sc / a[2]) ** 2 / (2.0 * g)
    level3_true = x[:-1, 2]

    # Step 5: report
    errors = np.abs(level3_rebuilt - level3_true)
    print("Part 2 — Four-tank, only levels 1 and 2 measured")
    print(f"  Spread (standard deviation) of level 3: {x[:, 2].std():.3f} m")
    print(f"  Median error rebuilding level 3 from two samples of level 1: {np.median(errors):.4f} m")
    print(f"  95th percentile of that error: {np.percentile(errors, 95):.4f} m")
    print("  A crude formula already gets the hidden level to about 1% of its spread; a learned")
    print("  network with a 4-sample window can do better.")
    print()


def main():
    vanderpol_part()
    four_tank_part()
    print("Mathematical signal: NEGATIVE for Phase 2 as designed (noise-free, fixed parameters).")
    print("Memory has a mathematical reason to help only when a short window cannot replace it:")
    print("measurement noise plus an unknown parameter that differs between trajectories.")


if __name__ == "__main__":
    main()
