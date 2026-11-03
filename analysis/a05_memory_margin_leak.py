"""a05 — Why the latent certificate needs a MARGIN for chained prediction (NumPy only, no training).

Linearised chained-prediction error with one memory channel (scalar version of the augmented Jacobian, a01):

    e_x(t+1) = j * e_x(t) + c * e_h(t) + eps(t)      prediction error (j = how the plant step treats an error)
    e_h(t+1) = rho * e_h(t) + c * e_x(t)             memory error (rho = decay factor, the certificate: rho < 1)

c is the coupling between the fed-back prediction and the memory (both directions, same size for simplicity).
Teacher forcing never feeds a wrong state, so it never sees c; nothing in that training keeps c small.

Exact growth factor per step: the largest eigenvalue of [[j, c], [c, rho]]
    lambda = (j + rho) / 2 + sqrt(((j - rho) / 2)^2 + c^2)
With j = 1:
    rho -> 1 (certificate with no margin):             lambda = 1 + c            (coupling enters LINEARLY)
    margin m = 1 - rho much larger than c:             lambda ~ 1 + c^2 / m      (coupling enters QUADRATICALLY)
So a guaranteed margin turns a first-order leak into a second-order one. The trained exp01d full Mamba has
largest decay factors 0.996-0.999 (no margin); Chain-Mamba guarantees rho <= exp(-0.05 * 1.0) = 0.951.

Also printed: the 50-step accumulation of a constant one-step mistake eps = 1 (sum of the error recursion),
the quantity behind the "mistakes line up" finding of Stage 2b.

Run: python analysis/a05_memory_margin_leak.py
"""

import numpy as np


def growth(j: float, rho: float, c: float) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.array([[j, c], [c, rho]])))))


def accumulate(j: float, rho: float, c: float, steps: int = 50) -> float:
    """|e_x| after `steps` steps with a constant one-step mistake eps = 1 and zero initial errors."""
    ex, eh = 0.0, 0.0
    for _ in range(steps):
        ex, eh = j * ex + c * eh + 1.0, rho * eh + c * ex
    return abs(ex)


def main():
    rhos = [0.999, 0.996, 0.951, 0.80, 0.50]
    print("Growth factor per step (largest eigenvalue), plant neutral (j = 1)")
    print(f"{'coupling c':>11s}" + "".join(f"{'rho=' + str(r):>12s}" for r in rhos))
    for c in [0.005, 0.02, 0.05, 0.10]:
        print(f"{c:>11.3f}" + "".join(f"{growth(1.0, r, c):>12.4f}" for r in rhos))

    print("\nExcess growth (lambda - 1) against the two approximations, c = 0.02")
    for r in rhos:
        lam = growth(1.0, r, 0.02)
        print(f"  rho={r:<6} exact {lam - 1:.5f}   linear c = {0.02:.5f}   quadratic c^2/(1-rho) = {0.02 ** 2 / (1 - r):.5f}")

    print("\n50-step accumulation of a constant one-step mistake (1.0 each step); no memory leak would give 50")
    print(f"{'coupling c':>11s}" + "".join(f"{'rho=' + str(r):>12s}" for r in rhos))
    for c in [0.005, 0.02, 0.05]:
        print(f"{c:>11.3f}" + "".join(f"{accumulate(1.0, r, c):>12.1f}" for r in rhos))

    print("\nMemory steady-state gain 1/(1 - rho):  " + ", ".join(f"rho={r}: {1 / (1 - r):.0f}" for r in rhos))
    print("Chain-Mamba bound: rho <= exp(-delta_floor * a_min) = exp(-0.05) = %.4f -> gain <= %.1f"
          % (np.exp(-0.05), 1 / (1 - np.exp(-0.05))))


if __name__ == "__main__":
    main()
