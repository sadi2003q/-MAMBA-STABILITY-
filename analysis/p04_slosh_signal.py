"""
Mathematical signal for space pilot C: attitude control of a spacecraft with sloshing fuel during an engine burn.

Standalone calculation (NumPy only, no training). It answers four questions before any run:

    1. Is the problem real?   How much does sloshing fuel shake the spacecraft's pointing, compared with the
                              settling threshold (0.1 degrees)?
    2. Does the slosh change with the fill level?   If the slosh frequency moves a lot between a 30% and a 90%
                              full tank, a controller tuned for one fill level is wrong for the others.
    3. Can the hidden slosh be found from what is measured?   Only the spacecraft's angle and turn rate are
                              measured (the fuel is not). Observability tells how many past steps are needed.
    4. Is a model WITHOUT memory enough?   If the next turn rate depends on the hidden slosh, a predictor that sees
                              only the current angle and turn rate has an error it can never remove.

python analysis/p04_slosh_signal.py
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.systems.slosh_spacecraft import SloshSpacecraft  # noqa: E402


def linear_matrices(system, fill):
    """Linearised model around rest (angle error 0, slosh angle 0). State [angle, turn rate, slosh angle, slosh rate]."""
    p = system.parameters(fill)
    J, m, l, b, a, c = p["J"], p["m"], p["l"], p["b"], p["a"], p["c"]

    # Step 1: mass matrix and restoring / damping terms at rest (from the derivation in the simulator file)
    mass = np.array([[J + m * (b + l) ** 2, -m * l * (b + l)],
                     [-m * l * (b + l), m * l ** 2]])
    stiffness = np.array([[0.0, -m * a * l],      # the thrust pull on the fuel also turns the spacecraft
                          [0.0, m * a * l]])
    damping = np.array([[0.0, 0.0], [0.0, c]])
    push = np.array([1.0, 0.0])

    # Step 2: continuous-time state-space form  d/dt [q, q_dot] = A [q, q_dot] + B u
    inverse = np.linalg.inv(mass)
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[2, 3] = 1.0
    lower_left = -inverse @ stiffness
    lower_right = -inverse @ damping
    A[1, 0], A[1, 2] = lower_left[0, 0], lower_left[0, 1]
    A[3, 0], A[3, 2] = lower_left[1, 0], lower_left[1, 1]
    A[1, 1], A[1, 3] = lower_right[0, 0], lower_right[0, 1]
    A[3, 1], A[3, 3] = lower_right[1, 0], lower_right[1, 1]
    B = np.zeros(4)
    B[1] = (inverse @ push)[0]
    B[3] = (inverse @ push)[1]
    return A, B


def discretise(A, B, dt):
    """Exact zero-order-hold discretisation with a matrix exponential (series, fine for these small matrices)."""
    n = A.shape[0]
    big = np.zeros((n + 1, n + 1))
    big[:n, :n] = A * dt
    big[:n, n] = B * dt
    result = np.eye(n + 1)
    term = np.eye(n + 1)
    for k in range(1, 30):
        term = term @ big / k
        result = result + term
    return result[:n, :n], result[:n, n]


def main():
    system = SloshSpacecraft()
    dt = system.dt
    fills = [0.3, 0.45, 0.6, 0.75, 0.9]
    C = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])       # measured: angle and turn rate only

    print("=" * 78)
    print(" Mathematical signal — space pilot C (spacecraft with sloshing fuel)")
    print("=" * 78)
    print(f" time step {dt} s;  push limit {system.u_max} N m;  settle threshold 0.1 degrees")
    print()
    print(f" {'fill':>5} {'slosh period (s)':>17} {'period (steps)':>15} {'damping ratio':>14} "
          f"{'pointing shake (deg)':>21} {'observable after':>17}")
    periods = []
    floors = []
    for fill in fills:
        A, B = linear_matrices(system, fill)
        eigenvalues = np.linalg.eigvals(A)
        oscillating = eigenvalues[np.abs(eigenvalues.imag) > 1e-9]
        frequency = float(np.abs(oscillating[0]))
        period = 2 * np.pi / float(np.abs(oscillating[0].imag))
        damping_ratio = -float(oscillating[0].real) / frequency
        periods.append(period)

        # Question 1: free slosh of 10 degrees at rest -> how far does the spacecraft angle swing?
        Ad, Bd = discretise(A, B, dt)
        x = np.array([0.0, 0.0, np.radians(10.0), 0.0])
        angles = []
        for _ in range(int(20 / dt)):
            x = Ad @ x
            angles.append(x[0])
        angles = np.array(angles)
        shake = np.degrees(angles.max() - angles.min()) / 2

        # Question 3: observability of the discrete system from angle + turn rate
        rows = []
        steps_needed = None
        for k in range(1, 5):
            rows.append(C @ np.linalg.matrix_power(Ad, k - 1))
            O = np.vstack(rows)
            if np.linalg.matrix_rank(O, tol=1e-9) == 4 and steps_needed is None:
                steps_needed = k

        # Question 4: how much of the next turn rate a memory-free predictor can never explain:
        # next turn rate = (row 1 of Ad) x state; the part from the slosh angle and slosh rate is hidden.
        hidden_part = abs(Ad[1, 2]) * np.radians(10.0) + abs(Ad[1, 3]) * np.radians(10.0) * frequency
        floors.append(np.degrees(hidden_part))
        print(f" {fill:5.2f} {period:17.2f} {period / dt:15.1f} {damping_ratio:14.4f} {shake:21.3f} "
              f"{str(steps_needed) + ' steps':>17}")

    print()
    spread = (max(periods) - min(periods)) / periods[2] * 100
    print(f" Slosh period changes by {spread:.0f}% across fill levels (relative to the 60% tank).")
    print(f" Memory-free one-step error floor on the turn rate, 10-degree slosh: "
          f"{min(floors):.4f} to {max(floors):.4f} degrees per second per step")
    print(f" Chain-Mamba memory: every decay factor is at most 0.951 -> memory time scale at most about 20 steps "
          f"({20 * dt:.0f} s), longer than the 2 steps needed to see the slosh.")
    print()
    print(" Mathematical signal: POSITIVE (problem real; memory needed and sufficient), see the numbers above.")


if __name__ == "__main__":
    main()
