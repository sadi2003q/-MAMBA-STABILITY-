"""
a03 — The latent certificate places no constraint on closed-loop growth or timing.

Standalone mathematical check (NumPy only; no training, no project code). Runs in about a second.

The question
------------
The latent certificate of Halloran et al. says: every memory decay factor a_bar is between 0 and 1.
Stage 1 showed that removing it changes nothing, and that the full Mamba's free-running oscillation
is about 24% too slow (one loop takes 9.02 time units instead of 7.27).

Could the certificate have stopped that? This script shows it cannot: for ANY certified decay
factors, the rest of the model (input gate, readout, skip term) can give the chained-prediction loop
ANY eigenvalues we like for the physical state: any growth rate and any oscillation speed.

The model, linearised (same form as analysis/a01)
-------------------------------------------------
    memory:      h_{t+1} = A_bar h_t + B x_t             (A_bar diagonal, every entry in (0, 1))
    prediction:  x_{t+1} = x_t + C h_{t+1} + D x_t

Chained prediction feeds x_{t+1} back in, so the closed loop for z = (x, h) has the matrix

    J = [ I + D + C B     C A_bar ]
        [ B               A_bar   ]

Part 1 — one physical state, one memory (scalars a, b, c, d)
-----------------------------------------------------------
    J = [ 1 + d + c b    c a ]
        [ b              a   ]

    trace       = 1 + d + c b + a
    determinant = (1 + d + c b) a - c a b = (1 + d) a          (the c b terms cancel)

To get two target eigenvalues lam1, lam2 (sum S, product P; both real numbers, also for a
complex pair), solve two equations:

    (1 + d) a     = P        ->  d   = P / a - 1
    1 + d + cb + a = S       ->  c b = S - P / a - a

a > 0 always (that is the certificate), so a solution always exists. Pick b = 1, c = c b.

Part 2 — any number of physical states and memories
---------------------------------------------------
With n physical states and m >= n memories, pair physical state i with memory i and leave the
other memories unconnected (their B rows and C columns are zero). The matrix J then splits into
n separate 2 x 2 blocks (each solved as in Part 1, with its own certified a_i) plus the unused
decay factors themselves. So any n target pairs can be placed, whatever the certified A_bar is.

What this means for timing
--------------------------
For a linear oscillator, an eigenvalue pair r * exp(+/- i theta) turns the state by theta per step,
so one loop takes 2 pi / theta steps. Part 1 can set theta (the clock) and r (growth or decay)
freely, with the certificate satisfied. The certificate is blind to the clock.

Assumptions
-----------
- Linearisation: J describes small errors only.
- The selective part (A_bar, B, C changing with the input) is held fixed. Selection only adds more
  terms to the Jacobian, so it gives the rest of the model more freedom, not less.
- The full Mamba contains this minimal cell as a special case (other layers and the convolution
  can pass the signal straight through), so the result carries over to it.

Run:  python analysis/a03_certificate_places_no_constraint.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(os.environ.get("ANALYSIS_OUT", "outputs/analysis"))

# Settings of the Van der Pol experiments (configs/systems/vanderpol.yaml and Stage 1 results)
TIME_STEP = 0.1
TRUE_PERIOD = 7.27            # time units per loop of the true (forward Euler) system, zero input
MAMBA_PERIOD = 9.02           # full Mamba, chained training, seed 0 (exp01d)


# ------------------------------------------------------------------ Part 1 helpers
def build_two_by_two(a, lam1, lam2):
    """Return the 2 x 2 closed-loop matrix with certified decay a and eigenvalues lam1, lam2."""
    # Step 1: sum and product of the target eigenvalues (real numbers, also for a complex pair)
    S = (lam1 + lam2).real
    P = (lam1 * lam2).real

    # Step 2: solve the two equations from the docstring
    d = P / a - 1.0
    cb = S - P / a - a

    # Step 3: pick b = 1, so c = c b
    b = 1.0
    c = cb

    # Step 4: write out the matrix
    J = np.array([[1.0 + d + c * b, c * a],
                  [b, a]])
    return J, {"a": a, "b": b, "c": c, "d": d}


def eigenvalue_mismatch(found, wanted):
    """Largest distance between two lists of eigenvalues, after pairing each wanted value
    with its nearest unused found value."""
    remaining = list(found)
    worst = 0.0
    for w in wanted:
        distances = [abs(f - w) for f in remaining]
        nearest = int(np.argmin(distances))
        worst = max(worst, distances[nearest])
        remaining.pop(nearest)
    return worst


def random_target_pair(rng):
    """Half the time two real eigenvalues, half the time a complex pair."""
    if rng.uniform() < 0.5:
        lam1 = rng.uniform(-2.0, 2.0)
        lam2 = rng.uniform(-2.0, 2.0)
        return complex(lam1), complex(lam2)
    radius = rng.uniform(0.0, 1.5)
    angle = rng.uniform(0.0, np.pi)
    lam1 = radius * np.exp(1j * angle)
    lam2 = np.conj(lam1)
    return lam1, lam2


# ------------------------------------------------------------------ Part 1
def part1_examples():
    """Named examples: the same certified decay a gives any clock and any growth."""
    # Step 1: how much the true system turns per step, and how much the slow model turns
    true_angle = 2.0 * np.pi * TIME_STEP / TRUE_PERIOD
    slow_angle = 2.0 * np.pi * TIME_STEP / MAMBA_PERIOD

    # Step 2: the target eigenvalue pairs
    examples = [
        ("true clock, neutral (radius 1)", np.exp(1j * true_angle)),
        ("24% slow clock, neutral", np.exp(1j * slow_angle)),
        ("true clock, growing 5% per step", 1.05 * np.exp(1j * true_angle)),
        ("true clock, decaying 5% per step", 0.95 * np.exp(1j * true_angle)),
    ]

    rows = []
    for a in (0.1, 0.5, 0.9, 0.99):
        for label, lam in examples:
            # Step 3: build the matrix and compute its eigenvalues
            J, params = build_two_by_two(a, lam, np.conj(lam))
            found = np.linalg.eigvals(J)
            mismatch = eigenvalue_mismatch(found, [lam, np.conj(lam)])

            # Step 4: read off the clock (steps per loop) and the growth per step
            biggest = found[np.argmax(np.abs(found))]
            turn_per_step = abs(np.angle(biggest))
            if turn_per_step > 0:
                period_time = 2.0 * np.pi / turn_per_step * TIME_STEP
            else:
                period_time = float("inf")
            rows.append({"certified_decay_a": a, "target": label,
                         "loop_time_units": float(period_time),
                         "growth_per_step": float(abs(biggest)),
                         "eigenvalue_mismatch": float(mismatch), **params})

    # Step 5: the 2 x 2 example of the research brief (latent block 0.5, eigenvalues 1.78 and -0.28)
    S = 1.5
    P = -0.5
    root = np.sqrt(S * S - 4.0 * P)
    brief_lam1 = (S + root) / 2.0
    brief_lam2 = (S - root) / 2.0
    J_brief, _ = build_two_by_two(0.5, complex(brief_lam1), complex(brief_lam2))
    return rows, J_brief


def part1_random(n_draws, rng):
    worst = 0.0
    passed = 0
    for _ in range(n_draws):
        # Step 1: a random certified decay factor, strictly between 0 and 1
        a = rng.uniform(0.01, 0.999)
        # Step 2: random target eigenvalues
        lam1, lam2 = random_target_pair(rng)
        # Step 3: build and check
        J, _ = build_two_by_two(a, lam1, lam2)
        mismatch = eigenvalue_mismatch(np.linalg.eigvals(J), [lam1, lam2])
        scale = max(1.0, abs(lam1), abs(lam2))
        relative = mismatch / scale
        worst = max(worst, relative)
        if relative < 1e-6:
            passed = passed + 1
    return {"draws": n_draws, "passed": passed, "worst_relative_mismatch": worst}


# ------------------------------------------------------------------ Part 2
def build_general(a_bar, target_pairs):
    """n physical states, m memories (m >= n). Physical state i is paired with memory i."""
    n = len(target_pairs)
    m = len(a_bar)
    B = np.zeros((m, n))
    C = np.zeros((n, m))
    D = np.zeros((n, n))
    for i in range(n):
        lam1, lam2 = target_pairs[i]
        _, params = build_two_by_two(a_bar[i], lam1, lam2)
        B[i, i] = params["b"]
        C[i, i] = params["c"]
        D[i, i] = params["d"]
    A = np.diag(a_bar)
    top = np.hstack([np.eye(n) + D + C @ B, C @ A])
    bottom = np.hstack([B, A])
    return np.vstack([top, bottom])


def part2_random(n_draws, n, m, rng):
    worst = 0.0
    passed = 0
    for _ in range(n_draws):
        # Step 1: random certified decay factors, all strictly between 0 and 1
        a_bar = rng.uniform(0.01, 0.999, size=m)
        # Step 2: one random target pair per physical state
        pairs = [random_target_pair(rng) for _ in range(n)]
        # Step 3: what the eigenvalues should be: every target, plus the unused decay factors
        wanted = []
        for lam1, lam2 in pairs:
            wanted.append(lam1)
            wanted.append(lam2)
        for j in range(n, m):
            wanted.append(complex(a_bar[j]))
        # Step 4: build and compare
        J = build_general(a_bar, pairs)
        mismatch = eigenvalue_mismatch(np.linalg.eigvals(J), wanted)
        scale = max(1.0, max(abs(w) for w in wanted))
        relative = mismatch / scale
        worst = max(worst, relative)
        if relative < 1e-6:
            passed = passed + 1
    return {"draws": n_draws, "physical_states": n, "memories": m,
            "passed": passed, "worst_relative_mismatch": worst}


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("a03 — The latent certificate places no constraint on closed-loop growth or timing")
    print("=" * 78)

    # Part 1: named examples
    rows, J_brief = part1_examples()
    print("Part 1 — one physical state, one memory")
    print("  Same certified decay a, different clocks and growth rates (loop time in time units):")
    for row in rows:
        if row["certified_decay_a"] != 0.9:
            continue
        print(f"    a = 0.9  {row['target']:<36} loop {row['loop_time_units']:6.2f}"
              f"   growth per step {row['growth_per_step']:.3f}   mismatch {row['eigenvalue_mismatch']:.1e}")
    worst_named = max(row["eigenvalue_mismatch"] for row in rows)
    print(f"  All {len(rows)} named examples (a = 0.1, 0.5, 0.9, 0.99): largest eigenvalue mismatch {worst_named:.1e}")
    print(f"  Research-brief example rebuilt: J = {J_brief.round(6).tolist()} (the brief's matrix is [[1, 1], [1, 0.5]])")

    random1 = part1_random(20000, rng)
    print(f"  Random targets: {random1['passed']} of {random1['draws']} placed exactly"
          f" (worst relative mismatch {random1['worst_relative_mismatch']:.1e})")
    print()

    # Part 2: many physical states and memories
    random2 = part2_random(5000, n=2, m=16, rng=rng)
    print("Part 2 — two physical states, sixteen memories (the simplified Mamba's memory size per channel)")
    print(f"  Random targets: {random2['passed']} of {random2['draws']} placed exactly"
          f" (worst relative mismatch {random2['worst_relative_mismatch']:.1e})")
    print()

    print("Conclusion")
    print("  For every certified set of decay factors, the input gate, readout and skip term can give the")
    print("  chained-prediction loop any eigenvalues: any clock speed and any growth rate.")
    print("  So the certificate cannot keep the model's clock right; a guarantee that does must constrain")
    print("  the whole loop (prediction plus memory), not the memory alone.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"time_step": TIME_STEP, "true_period": TRUE_PERIOD, "mamba_period": MAMBA_PERIOD,
           "part1_named_examples": rows, "part1_brief_matrix": J_brief.tolist(),
           "part1_random": random1, "part2_random": random2}
    path = RESULTS_DIR / "a03_certificate_places_no_constraint.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved numbers to {path}")


if __name__ == "__main__":
    main()
