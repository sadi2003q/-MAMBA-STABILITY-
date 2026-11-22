"""
Space paper, mathematical signal for space01_quantum_hardware (standalone, no training).

Question: how wrong can the predictor's belief about "how much a push turns the spacecraft" be before the
predictive controller makes the loop UNSTABLE?

Model (one axis, small angles): sigma ~ angle / 4,
    sigma_next = sigma + dt / 4 x omega + dt^2 / (8 J) x torque
    omega_next = omega + dt / J x torque
The controller is approximated by its infinite-horizon linear-quadratic regulator, designed with the
predictor's (possibly wrong) push effect  k x B_true, and applied to the real spacecraft (B_true).
The loop is stable when the largest eigenvalue size (spectral radius) of  A - B_true K  is below 1.

Run:  python analysis/s01_gain_margin.py
"""

import numpy as np
from scipy.linalg import solve_discrete_are

# Step 1: the linearised spacecraft (same dt as configs/space/space01_quantum_hardware.yaml)
dt = 0.5
inertia = 1.0
A = np.array([[1.0, dt / 4.0],
              [0.0, 1.0]])
B_true = np.array([[dt * dt / (8.0 * inertia)],
                   [dt / inertia]])

# Step 2: controller weights: q_attitude 50 on (4 sigma)^2 -> 50 x 16 on sigma^2; spin weight 1; push weight 0.5
Q = np.diag([50.0 * 16.0, 1.0])
R = np.array([[0.5]])


def spectral_radius(believed_over_true: float) -> float:
    """Closed-loop spectral radius when the predictor believes pushes are `believed_over_true` x reality."""
    B_model = believed_over_true * B_true
    P = solve_discrete_are(A, B_model, Q, R)
    K = np.linalg.solve(R + B_model.T @ P @ B_model, B_model.T @ P @ A)
    closed_loop = A - B_true @ K
    return float(max(abs(np.linalg.eigvals(closed_loop))))


# Step 3: a table of believed / true push effect
print("believed / true push effect -> closed-loop spectral radius")
for ratio in [0.25, 0.4, 0.5, 0.6, 0.7, 1.0, 1.43, 2.0, 3.0, 4.0]:
    radius = spectral_radius(ratio)
    print(f"  {ratio:4.2f} -> {radius:.3f}   {'stable' if radius < 1 else 'UNSTABLE'}")

# Step 4: the exact threshold, by halving the interval
low, high = 0.4, 0.7
for _ in range(40):
    middle = (low + high) / 2
    if spectral_radius(middle) >= 1.0:
        low = middle
    else:
        high = middle
print(f"Unstable below believed / true = {high:.3f}")

# Step 5: the test scenario: inertia shrinks 30% -> pushes 1/0.7 = 1.43x stronger than a non-adapting model thinks
print("Inertia -30%, model does not adapt: believed / true = 0.70 ->", round(spectral_radius(0.70), 3))
for extra in [0.95, 0.9, 0.85, 0.8]:
    ratio = 0.70 * extra
    print(f"  plus a further x{extra} under-estimate from hardware errors: {ratio:.3f} -> {spectral_radius(ratio):.3f}")
