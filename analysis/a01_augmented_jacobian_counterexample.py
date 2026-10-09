"""
a01 — The latent stability certificate does not bound chained-prediction error.

Standalone mathematical check (NumPy only, no training, no project code).

What this script shows
----------------------
When a dynamics model is used in chained prediction (its own prediction is fed
back as the next input), the quantity that decides whether errors grow is the
Jacobian (matrix of first derivatives) of the AUGMENTED state
    z_t = (physical prediction x_t, hidden state h_t).
The latent certificate of Halloran et al. (2025) only says that the hidden-state
decay factors lie strictly between 0 and 1. This script shows, in four parts,
that this is not enough:

  Part 1  The two-by-two example from the research brief.
  Part 2  A closed-form result for the scalar selective state-space model with a
          residual readout and no skip term: the chained prediction is stable
          IF AND ONLY IF
              -2 (1 + a) < c b < 0,
          where a is the certified decay factor, b the input gate and c the
          readout gate. With a skip term d, a necessary condition is
              c b + d (1 - a) < 0.
          The certificate (0 < a < 1) never fixes the sign of the loop gain,
          so any positive loop gain gives growing errors.
  Part 3  The multi-dimensional version: a necessary condition for stability is
              det( -(D + C (I - A)^(-1) B) ) > 0,
          i.e. a condition on the hidden state's steady-state gain, which the
          certificate does not constrain. Checked numerically on random draws.
  Part 4  Monte Carlo: how often a certified model (all decay factors in (0, 1))
          with random, plausible gates gives an unstable augmented Jacobian.

Model being linearised (minimal selective state-space cell, residual readout):
    h_{t+1}       = A_bar h_t + B x_t                 (A_bar diagonal, entries in (0, 1))
    x_{t+1}       = x_t + C h_{t+1} + D x_t
Linearised at the fixed point x = 0, h = 0. At this point the input dependence of
A_bar, B and C (the "selective" part) contributes nothing to the Jacobian, because
each such derivative multiplies h = 0 or x = 0. So the linearisation is exact here.

Assumptions: linearisation at the origin; input embedding and output projection
absorbed into B, C and D; control input held fixed.

Run:  python analysis/a01_augmented_jacobian_counterexample.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HORIZON = 50
RESULTS_DIR = Path(os.environ.get("ANALYSIS_OUT", "outputs/analysis"))


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def augmented_jacobian(A_bar: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Jacobian of z_{t+1} with respect to z_t, for z = (x, h).

    x_{t+1} = (I + D + C B) x_t + C A_bar h_t
    h_{t+1} = B x_t + A_bar h_t
    """
    n = B.shape[1]
    top = np.hstack([np.eye(n) + D + C @ B, C @ A_bar])
    bottom = np.hstack([B, A_bar])
    return np.vstack([top, bottom])


def error_growth(J: np.ndarray, horizon: int) -> float:
    """Largest amplification of an initial error after `horizon` chained steps."""
    return float(np.linalg.norm(np.linalg.matrix_power(J, horizon), 2))


# ---------------------------------------------------------------- Part 1
def part1_brief_example() -> dict:
    J = np.array([[1.0, 1.0], [1.0, 0.5]])
    eig = np.sort(np.linalg.eigvals(J).real)[::-1]
    closed_form = ((1.5 + np.sqrt(4.25)) / 2, (1.5 - np.sqrt(4.25)) / 2)
    out = {
        "latent_block": 0.5,
        "eigenvalues": eig.tolist(),
        "closed_form_eigenvalues": list(closed_form),
        "error_amplification_after_50_steps": error_growth(J, HORIZON),
    }
    print("Part 1 — the two-by-two example from the research brief")
    print(f"  Latent block = 0.5, so the certificate holds (0 < 0.5 < 1).")
    print(f"  Eigenvalues of the augmented Jacobian: {eig[0]:.4f} and {eig[1]:.4f}")
    print(f"  An initial error is amplified about {out['error_amplification_after_50_steps']:.3g} times after 50 chained steps.")
    print()
    return out


# ---------------------------------------------------------------- Part 2
def check_scalar_family(rng: np.random.Generator, draws: int = 200_000) -> dict:
    """Closed form for the scalar cell, verified numerically.

    Augmented Jacobian:  J = [[1 + d + c b, c a], [b, a]]
        trace = 1 + a + c b + d,     det = a (1 + d)      (the c b terms cancel)
    Characteristic polynomial p(lam) = lam^2 - trace * lam + det.
    Jury test: all eigenvalues inside the unit circle  <=>
        p(1) > 0,  p(-1) > 0,  |det| < 1.
    Here p(1) = -(c b + d (1 - a)).  With d = 0:  p(1) = -c b,
    p(-1) = 2 (1 + a) + c b,  det = a, so stable  <=>  -2 (1 + a) < c b < 0.
    The certificate (0 < a < 1) only secures |det| < 1.
    """
    a = rng.uniform(0.0, 1.0, draws)            # certified decay factor
    b = rng.normal(0.0, 1.0, draws)
    c = rng.normal(0.0, 1.0, draws)
    d = rng.normal(0.0, 0.3, draws)

    trace = 1.0 + a + c * b + d
    det = a * (1.0 + d)
    disc = trace**2 - 4.0 * det
    sqrt_disc = np.sqrt(disc.astype(complex))
    lam1 = (trace + sqrt_disc) / 2.0
    lam2 = (trace - sqrt_disc) / 2.0
    rho = np.maximum(np.abs(lam1), np.abs(lam2))
    stable_numeric = rho < 1.0

    # Jury test (exact, general d): p(1) > 0, p(-1) > 0, |det| < 1
    p1 = 1.0 - trace + det           # = -(c b) - d + a d = -(c b + d (1 - a))
    pm1 = 1.0 + trace + det
    stable_jury = (p1 > 0) & (pm1 > 0) & (np.abs(det) < 1.0)
    agree = float(np.mean(stable_numeric == stable_jury))

    # Key fact: p(1) = -(c b + d (1 - a)). Sign is set by the gates, not by a.
    positive_loop = (c * b + d * (1.0 - a)) > 0
    unstable_when_positive_loop = float(np.mean(~stable_numeric[positive_loop]))

    out = {
        "draws": draws,
        "jury_test_agrees_with_eigenvalues": agree,
        "fraction_with_positive_loop_gain": float(np.mean(positive_loop)),
        "fraction_unstable_given_positive_loop_gain": unstable_when_positive_loop,
        "fraction_unstable_overall": float(np.mean(~stable_numeric)),
    }
    print("Part 2 — scalar selective state-space cell with residual readout")
    print("  Model:  h' = a h + b x,   x' = x + c h' + d x,   certified decay 0 < a < 1")
    print("  Closed form (d = 0): chained prediction is stable  if and only if  -2 (1 + a) < c b < 0.")
    print("  General d:           a necessary condition is  c b + d (1 - a) < 0.")
    print(f"  Jury test agrees with direct eigenvalues on {100*agree:.2f}% of {draws:,} random draws.")
    print(f"  Draws with positive loop gain c b + d (1 - a) > 0: {100*out['fraction_with_positive_loop_gain']:.1f}%")
    print(f"    of these, unstable: {100*unstable_when_positive_loop:.2f}%  (the theory says exactly 100%)")
    print(f"  Overall unstable despite the certificate: {100*out['fraction_unstable_overall']:.1f}%")
    print()
    return out


# ---------------------------------------------------------------- Part 3
def part3_matrix_condition(rng: np.random.Generator, draws: int = 20_000,
                           n: int = 2, m: int = 16) -> dict:
    """Necessary condition in many dimensions.

    det(J - I) = det(A_bar - I) * det(D + G),   G = C (I - A_bar)^(-1) B
    For a stable J (all eigenvalues inside the unit circle), the sign of
    det(J - I) is (-1)^(n + m). Since sign det(A_bar - I) = (-1)^m, stability
    requires  sign det(D + G) = (-1)^n,  i.e.  det(-(D + G)) > 0.
    """
    identity_ok = 0
    violated = 0
    violated_and_unstable = 0
    stable = 0
    stable_and_condition = 0
    for _ in range(draws):
        a = rng.uniform(0.0, 1.0, m)
        A_bar = np.diag(a)
        B = rng.normal(0.0, 1.0 / np.sqrt(n), (m, n))
        C = rng.normal(0.0, 0.3 / np.sqrt(m), (n, m))
        D = rng.normal(0.0, 0.1, (n, n))
        J = augmented_jacobian(A_bar, B, C, D)
        G = C @ np.diag(1.0 / (1.0 - a)) @ B
        lhs = np.linalg.det(J - np.eye(n + m))
        rhs = np.linalg.det(A_bar - np.eye(m)) * np.linalg.det(D + G)
        if np.isclose(lhs, rhs, rtol=1e-6, atol=1e-10):
            identity_ok += 1
        cond = np.linalg.det(-(D + G)) > 0
        is_stable = spectral_radius(J) < 1.0
        if not cond:
            violated += 1
            violated_and_unstable += int(not is_stable)
        if is_stable:
            stable += 1
            stable_and_condition += int(cond)
    out = {
        "draws": draws, "physical_dim": n, "hidden_dim": m,
        "determinant_identity_holds_fraction": identity_ok / draws,
        "fraction_violating_condition": violated / draws,
        "violating_and_unstable_fraction": violated_and_unstable / max(violated, 1),
        "stable_draws_satisfying_condition_fraction": stable_and_condition / max(stable, 1),
    }
    print(f"Part 3 — multi-dimensional case (physical size {n}, hidden size {m})")
    print("  Identity:  det(J - I) = det(A_bar - I) * det(D + C (I - A_bar)^(-1) B)")
    print(f"  Identity verified on {100*out['determinant_identity_holds_fraction']:.2f}% of {draws:,} random draws.")
    print("  Necessary condition for stable chained prediction:  det( -(D + C (I - A_bar)^(-1) B) ) > 0")
    print(f"  Draws violating it: {100*out['fraction_violating_condition']:.1f}%;"
          f" of these, unstable: {100*out['violating_and_unstable_fraction']:.2f}% (theory: 100%)")
    print(f"  Stable draws that satisfy it: {100*out['stable_draws_satisfying_condition_fraction']:.2f}% (theory: 100%)")
    print("  The certificate (A_bar entries in (0, 1)) does not fix the sign of this determinant.")
    print()
    return out


# ---------------------------------------------------------------- Part 4
def part4_how_fast(rng: np.random.Generator) -> dict:
    """For the scalar cell, how fast does error grow for small positive loop gain?"""
    rows = []
    print("Part 4 — size of the growth for small positive loop gain (scalar cell, d = 0)")
    print("  decay a   loop gain c b   largest eigenvalue   amplification after 50 steps")
    for a in (0.5, 0.9, 0.99):
        for k in (0.001, 0.01, 0.05):
            c, b = 1.0, k
            J = np.array([[1.0 + c * b, c * a], [b, a]])
            rho = spectral_radius(J)
            amp = error_growth(J, HORIZON)
            rows.append({"a": a, "loop_gain": k, "spectral_radius": rho, "amplification_50": amp})
            print(f"  {a:7.2f}   {k:13.3f}   {rho:18.4f}   {amp:12.3g}")
    print("  Analytic rule: largest eigenvalue is about 1 + c b / (1 - a) when c b is much smaller")
    print("  than 1 - a, and about 1 + square root of (c b) when a is close to 1.")
    print("  So a slowly decaying memory (a near 1) magnifies a tiny loop gain into fast growth.")
    print()
    return {"rows": rows}


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("a01 — Does the latent certificate bound chained-prediction error?")
    print("=" * 78)
    results = {
        "part1": part1_brief_example(),
        "part2": check_scalar_family(rng),
        "part3": part3_matrix_condition(rng),
        "part4": part4_how_fast(rng),
    }
    print("Conclusion")
    print("  The certificate fixes only the determinant-like part of the augmented Jacobian.")
    print("  Stability of chained prediction depends on the steady-state gain of the hidden")
    print("  state, D + C (I - A_bar)^(-1) B, which no part of the certificate constrains.")
    print("  Mathematical signal for 'the certificate does not transfer': POSITIVE.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "a01_augmented_jacobian_counterexample.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved numbers to {path}")


if __name__ == "__main__":
    main()
