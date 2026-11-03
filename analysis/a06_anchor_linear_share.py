"""a06 — How much of the Van der Pol one-step increment can a plain linear anchor represent?

No training. Uses the same data recipe as the repository (forward Euler, time step 0.1, mu = 1,
multisine input with peak 15 and 30 linearly spaced harmonics, data seed 12345).

Why this matters:
  Chain-Mamba's read-out has a linear "anchor":
      increment = head(features) + W_skip [x1, x2, u]
  If most of the true increment is linear in [x1, x2, u], then the anchor gives that part away for free,
  and the recurrent network only has to learn what is left. Any network given the anchor gets this help,
  so a comparison where only one network has it is not fair.

Run:
    python analysis/a06_anchor_linear_share.py
"""

import numpy as np


def make_multisine(n_steps, dt, rng):
    # Step 1: time axis
    t = np.arange(n_steps) * dt

    # Step 2: 30 frequencies, evenly spaced between 0.0049 and 4.88 hertz
    frequencies = np.linspace(0.0049, 4.88, 30)

    # Step 3: a random phase for every frequency
    phases = rng.uniform(0.0, 2.0 * np.pi, 30)

    # Step 4: add all the sine waves together
    angles = 2.0 * np.pi * frequencies[None, :] * t[:, None] + phases[None, :]
    signal = np.sin(angles).sum(axis=1)

    # Step 5: rescale so the largest value is 15
    largest = np.abs(signal).max()
    signal = signal / largest * 15.0
    return signal


def simulate_vanderpol(x_start, u, dt, mu):
    n_steps = len(u)
    x = np.zeros((n_steps + 1, 2))
    x[0] = x_start

    for k in range(n_steps):
        position = x[k, 0]
        velocity = x[k, 1]

        # Van der Pol right-hand side
        d_position = velocity
        d_velocity = mu * (1.0 - position * position) * velocity - position + u[k]

        # forward Euler step
        x[k + 1, 0] = position + dt * d_position
        x[k + 1, 1] = velocity + dt * d_velocity
    return x


def main():
    # Step 1: settings (same as configs/systems/vanderpol.yaml)
    n_steps = 40000
    dt = 0.1
    mu = 1.0
    rng = np.random.default_rng(12345)

    # Step 2: make the training trajectory
    u = make_multisine(n_steps, dt, rng)
    x_start = rng.uniform([-2.5, -2.0], [2.5, 2.0])
    x = simulate_vanderpol(x_start, u, dt, mu)

    # Step 3: normalise with the trajectory's own mean and standard deviation (as the repository does)
    x_normalised = (x - x.mean(axis=0)) / x.std(axis=0)
    u_normalised = (u - u.mean()) / u.std()

    # Step 4: the targets are the one-step increments x_{k+1} - x_k
    increments = np.diff(x_normalised, axis=0)

    # Step 5: the anchor's input is [x1, x2, u] (the mask is always 0 in chained prediction)
    anchor_input = np.column_stack([x_normalised[:-1], u_normalised])

    # Step 6: best possible linear anchor (least squares, no bias, like W_skip)
    weights, _, _, _ = np.linalg.lstsq(anchor_input, increments, rcond=None)
    linear_part = anchor_input @ weights
    left_over = increments - linear_part

    # Step 7: report how much of each increment the anchor explains
    print("a06 — share of the true one-step increment a linear anchor can represent")
    names = ["position", "velocity"]
    for i in range(2):
        total_variance = increments[:, i].var()
        left_variance = left_over[:, i].var()
        explained = 1.0 - left_variance / total_variance
        print(f"  {names[i]:8s}: anchor explains {100 * explained:6.2f}%, "
              f"left for the recurrent network {100 * (1 - explained):5.2f}%")
    print()
    print("  Meaning: a network WITH the anchor only has to learn the leftover part.")
    print("  A network WITHOUT it (the Long Short-Term Memory network in exp03) must learn everything.")


if __name__ == "__main__":
    main()
