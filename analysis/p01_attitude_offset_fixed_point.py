"""
Mathematical check for space pilot A (satellite pointing, Cena et al. 2026).

Question: when the controller's model does not know about a small constant torque on the spacecraft
(here: reaction-wheel friction), does model predictive control leave a steady pointing error? And does the
running correction (offset-free control, as in mpc01c) remove it?

Model of the check (one axis, small angles, no push limit, so the controller has an exact formula):
    true spacecraft:      angle_next  = angle + dt x spin + dt^2 / 2 x (-push + friction) / inertia
                          spin_next   = spin  + dt x (-push + friction) / inertia
    controller's model:   the same, but with friction = 0
    controller's cost:    sum over k = 1..N of  q_angle x angle_k^2 + q_spin x spin_k^2   (step N uses p_...)
                          + r x sum over k of ((push_k - push_(k-1)) / push_limit)^2
    The best plan of a quadratic cost with a linear model is found exactly with one linear solve.

Run:  python analysis/p01_attitude_offset_fixed_point.py      (NumPy only, a few seconds)
"""

import numpy as np


# ---------------------------------------------------------------------------------------------------------
# Settings (the same numbers as configs/pilots/pilot_a_attitude_offset.yaml)
# ---------------------------------------------------------------------------------------------------------
INERTIA = 0.05          # kilogram square metres (one axis)
DT = 0.5                # seconds between control decisions
HORIZON = 10            # planned steps (5 seconds ahead)
PUSH_LIMIT = 2.0e-3     # newton metres
Q_ANGLE = 50.0
Q_SPIN = 5.0
P_ANGLE = 100.0
P_SPIN = 10.0
R_CHANGE = 0.5
FRICTION_TORQUE = 7.0e-5        # newton metres: viscous + Coulomb friction at about 200 rad/s wheel speed
CORRECTION_FILTER = 0.5         # running correction: half old value, half newest mismatch
STEPS = 2000                    # closed-loop steps (500 seconds, long enough to reach the steady state)


def true_step(angle, spin, push, friction):
    """One step of the real spacecraft (push held constant for dt)."""
    acceleration = (-push + friction) / INERTIA
    new_angle = angle + DT * spin + 0.5 * DT * DT * acceleration
    new_spin = spin + DT * acceleration
    return new_angle, new_spin


def predicted_path_matrices():
    """
    The model's predicted angles and spins over the horizon are LINEAR in the planned pushes:
        angles = a0 + A_push @ pushes,   spins = s0 + S_push @ pushes
    This builds the push matrices A_push and S_push (the start parts are added later).
    """
    A_push = np.zeros((HORIZON, HORIZON))
    S_push = np.zeros((HORIZON, HORIZON))
    for k in range(HORIZON):               # predicted step k + 1
        for j in range(k + 1):             # push j (applied during step j)
            steps_after = k - j            # how many full steps after push j ends
            # push j changes the spin by -dt / inertia, and the angle by -dt^2/2/inertia during its own step
            # plus -dt / inertia x dt for every later step
            S_push[k, j] = -DT / INERTIA
            A_push[k, j] = -(0.5 * DT * DT) / INERTIA - (DT / INERTIA) * DT * steps_after
    return A_push, S_push


def start_parts(angle, spin, correction_angle, correction_spin):
    """Predicted angles / spins over the horizon when every planned push is zero (plus the correction)."""
    angles = np.zeros(HORIZON)
    spins = np.zeros(HORIZON)
    a = angle
    s = spin
    for k in range(HORIZON):
        a, s = true_step(a, s, 0.0, 0.0)       # the MODEL: no friction
        a = a + correction_angle                # running correction added to every predicted step
        s = s + correction_spin
        angles[k] = a
        spins[k] = s
    return angles, spins


def best_plan(angle, spin, previous_push, correction_angle, correction_spin, A_push, S_push):
    """Exact minimiser of the quadratic cost (unconstrained). Returns the planned pushes [HORIZON]."""
    a0, s0 = start_parts(angle, spin, correction_angle, correction_spin)

    # Step 1: weights per predicted step (the last step uses the terminal weights)
    weight_angle = np.full(HORIZON, Q_ANGLE)
    weight_spin = np.full(HORIZON, Q_SPIN)
    weight_angle[-1] = P_ANGLE
    weight_spin[-1] = P_SPIN

    # Step 2: the push-change matrix D: (D @ pushes)_k = push_k - push_(k-1), with push_(-1) = previous push
    D = np.eye(HORIZON)
    for k in range(1, HORIZON):
        D[k, k - 1] = -1.0
    d0 = np.zeros(HORIZON)
    d0[0] = -previous_push

    # Step 3: normal equations of the least-squares problem
    H = A_push.T @ np.diag(weight_angle) @ A_push
    H = H + S_push.T @ np.diag(weight_spin) @ S_push
    H = H + (R_CHANGE / PUSH_LIMIT ** 2) * (D.T @ D)
    g = A_push.T @ (weight_angle * a0)
    g = g + S_push.T @ (weight_spin * s0)
    g = g + (R_CHANGE / PUSH_LIMIT ** 2) * (D.T @ d0)
    return np.linalg.solve(H, -g)


def run_closed_loop(friction, offset_free):
    """Closed loop from 10 degrees off target. Returns the final angle in degrees and the final push."""
    A_push, S_push = predicted_path_matrices()
    angle = 10.0 * np.pi / 180.0
    spin = 0.0
    previous_push = 0.0
    previous_angle = None
    previous_spin = None
    correction_angle = 0.0
    correction_spin = 0.0

    for _ in range(STEPS):
        # Step 1: running correction from the newest mismatch (measured now - model's prediction of now)
        if offset_free and previous_angle is not None:
            predicted_angle, predicted_spin = true_step(previous_angle, previous_spin, previous_push, 0.0)
            mismatch_angle = angle - predicted_angle
            mismatch_spin = spin - predicted_spin
            correction_angle = (1.0 - CORRECTION_FILTER) * correction_angle + CORRECTION_FILTER * mismatch_angle
            correction_spin = (1.0 - CORRECTION_FILTER) * correction_spin + CORRECTION_FILTER * mismatch_spin

        # Step 2: plan, apply the first push
        plan = best_plan(angle, spin, previous_push, correction_angle, correction_spin, A_push, S_push)
        push = plan[0]
        previous_angle = angle
        previous_spin = spin
        angle, spin = true_step(angle, spin, push, friction)
        previous_push = push

    return angle * 180.0 / np.pi, previous_push


def main():
    print("Space pilot A — mathematical check (one axis, small angles, exact unconstrained controller)")
    print(f"  unmodelled friction torque: {FRICTION_TORQUE:.1e} N m  ({100 * FRICTION_TORQUE / PUSH_LIMIT:.1f}% of the push limit)")
    print()
    for friction in [0.0, FRICTION_TORQUE]:
        for offset_free in [False, True]:
            final_angle, final_push = run_closed_loop(friction, offset_free)
            label_friction = "friction" if friction > 0 else "no friction"
            label_fix = "with running correction" if offset_free else "no correction"
            print(f"  {label_friction:12s} {label_fix:24s} final pointing error {abs(final_angle):.5f} degrees"
                  f"   final push {final_push:+.2e} N m")
    print()
    print("Reading: a non-zero final error = steady offset. The correction should bring it to (almost) zero,")
    print("because at a steady state the measured mismatch equals the missing torque's effect exactly.")


if __name__ == "__main__":
    main()
