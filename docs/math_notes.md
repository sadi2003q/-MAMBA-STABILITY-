# Mathematical notes

Derivations and mathematical signals, newest last. Each entry names the script in `analysis/` that reproduces it.

---

## 9 October 2026 — a01: the latent certificate does not bound chained-prediction error

**Script:** `analysis/a01_augmented_jacobian_counterexample.py` (NumPy only, runs in under a second).

**Model linearised.** Minimal selective state-space cell with a residual readout, used in chained prediction:

- hidden state: h_{t+1} = A_bar h_t + B x_t, with A_bar diagonal and every entry in (0, 1) — this is the latent certificate;
- physical prediction: x_{t+1} = x_t + C h_{t+1} + D x_t.

Linearised at x = 0, h = 0. There the input dependence of A_bar, B and C (the "selective" part) drops out of the Jacobian exactly, because each such derivative multiplies x = 0 or h = 0.

**Augmented Jacobian** (state z = (x, h)):

    J = [ I + D + C B    C A_bar ]
        [ B              A_bar   ]

**Result 1 — scalar cell, exact.** With scalar a, b, c and no skip term: trace = 1 + a + c b, determinant = a (the c b terms cancel). By the Jury test, chained prediction is stable if and only if

    -2 (1 + a) < c b < 0.

The certificate (0 < a < 1) secures only the determinant condition. The sign of the loop gain c b, set by the input gate and the readout gate, is left completely free. Any positive loop gain gives an eigenvalue above 1. With a skip term d, a necessary condition is c b + d (1 - a) < 0. Checked on 200,000 random draws: Jury test and direct eigenvalues agree on 100%; every draw with positive loop gain was unstable.

**Result 2 — any dimension.** By the Schur complement,

    det(J - I) = det(A_bar - I) · det(D + G),    G = C (I - A_bar)^(-1) B.

G is the steady-state gain of the hidden-state pathway. If J is stable, det(J - I) has sign (-1)^(n+m) (n physical, m hidden dimensions), so a **necessary condition for stable chained prediction is det(-(D + G)) > 0**. The certificate does not constrain this sign. Checked on 20,000 random draws (n = 2, m = 16): identity holds on 100%; every draw violating the condition was unstable; every stable draw satisfied it. About half of certified random models violate it.

**Result 3 — slow memory amplifies a small loop gain.** For small positive loop gain, the largest eigenvalue is about 1 + c b / (1 - a), and about 1 + square root of (c b) when a is near 1. Example: a = 0.9, c b = 0.01 gives about 96 times error amplification over 50 steps, with the certificate satisfied.

```
Mathematical signal: POSITIVE (for "the latent certificate does not transfer")
What was checked:    augmented Jacobian of a linearised selective state-space cell in chained prediction
Result:              stability needs det(-(D + C (I - A_bar)^(-1) B)) > 0; the certificate does not fix it
Assumptions:         linearisation at the origin; embedding and output projection absorbed into B, C, D
Implication:         the negative half of the paper (H2) has a clean proof sketch; the leak is through
                     the hidden state's steady-state gain, matching the first attempt's diagnostic
```

**Caution for the design phase.** The Van der Pol origin is itself unstable, so a good model *must* be unstable at the origin. The condition above therefore cannot be imposed as "make the model stable". The useful version is relative: the model's augmented Jacobian, seen from the physical state, should match the true system's Jacobian, and the certificate should bound the difference. This is the entry point for H4.

---

## 9 October 2026 — a02: how much error amplification the true Van der Pol system allows

**Script:** `analysis/a02_vanderpol_jacobian_along_orbit.py` (NumPy, SciPy, Matplotlib; a few seconds). Arguments: `--mu`, `--dts`, `--horizon`.

**Correction to the research brief.** Section 3.1 of the brief writes dx2/dt = mu (1 - x1^2) x2 + u. The standard Van der Pol oscillator, and the only form with a limit cycle, is dx2/dt = mu (1 - x1^2) x2 - x1 + u. Without "- x1", every point with x2 = 0 is an equilibrium and nothing oscillates. The scripts use the standard form; the brief and the system config should be fixed to match. *(Update, same day: the brief is fixed. Cevaal et al.'s own Equation 31 prints the same form without "- x1"; under their multisine input, peak 15, that form blows up within 211 to 950 Euler steps in 10 of 10 trials, so it must be a typo. Details in `docs/decisions.md`.)*

**Continuous cycle (mu = 1).** Period 6.6633, amplitude about 2.009. Floquet multipliers (how a perturbation changes after one full period): 1.000 and 8.6e-4. Timing errors are kept forever; shape errors die out. This is transverse contraction.

**Forward Euler, along the cycle (50-step horizon):**

| Time step | One-step largest singular value (max / median) | Part of cycle above 1 | 50-step amplification, any direction (max / median) | Along the orbit (max) | Across the orbit (max / median) |
|---|---|---|---|---|---|
| 0.01 | 1.029 / 1.010 | 100% | 2.10 / 1.46 | 1.82 | 1.53 / 0.55 |
| 0.05 | 1.174 / 1.049 | 100% | 8.38 / 1.08 | 8.15 | 0.47 / 0.058 |
| 0.1 | 1.430 / 1.094 | 100% | 5.23 / 2.25 | 4.17 | 0.25 / 0.020 |

Readings:

- In ordinary Euclidean distance the true one-step map stretches some direction **everywhere** on the cycle, not only on parts. Part of this is structural (the symmetric part of the Van der Pol Jacobian always has a non-negative eigenvalue), part is forward Euler (which slightly expands any rotation). The brief's "above 1 on parts of the orbit" understates it.
- A perfect model would still amplify errors by up to about 2 to 8 times over 50 steps, almost all of it along the orbit (timing). A certificate demanding less than that would force the model to be wrong.
- Across-orbit (shape) errors shrink over long windows but can grow over short ones (up to 1.53 over 0.5 time units at time step 0.01). So a contraction certificate needs a state-dependent way of measuring distance (a contraction metric), not plain Euclidean distance.
- The worst-case geometric bound using the per-step maximum is extremely loose (111 to 1.35e8), so per-step bounds are useless here; finite-horizon products are the right measure.

```
Mathematical signal: NEGATIVE for global contraction; POSITIVE for a relative / transverse certificate
What was checked:    singular values of one-step and 50-step Jacobian products along the limit cycle;
                     Floquet multipliers of the continuous cycle
Result:              one-step stretch above 1 on 100% of the cycle; perfect-model 50-step amplification
                     up to 2.1 (time step 0.01), 8.4 (0.05), 5.2 (0.1); shape-error multiplier 8.6e-4 per period
Assumptions:         control input zero; forward Euler; small errors
Implication:         a usable certificate must (1) be relative to the true system's amplification,
                     (2) leave timing errors alone, (3) contract shape errors in a state-dependent metric.
                     Time step choice matters: record it in configs/systems/vanderpol.yaml and match Cevaal et al.
```

---

## 19 October 2026 — a03: the latent certificate places no constraint on closed-loop growth or timing

**Script:** `analysis/a03_certificate_places_no_constraint.py` (NumPy only, about a second).

**Question.** Stage 1 found the full Mamba's free-running oscillation about 24% too slow (9.02 against 7.27 time units per loop), with the certificate holding in every run. Could a certificate on the memory decay ever prevent that?

**Model linearised** (same form as a01): memory h_{t+1} = A_bar h_t + B x_t with A_bar diagonal and every entry in (0, 1); prediction x_{t+1} = x_t + C h_{t+1} + D x_t. In chained prediction the loop for (x, h) has the matrix

    J = [ I + D + C B    C A_bar ]
        [ B              A_bar   ]

**Result 1 — one physical state, one memory, exact.** With scalars, trace = 1 + d + c b + a and determinant = (1 + d) a (the c b terms cancel, as in a01). For any target eigenvalues with sum S and product P (real, or a complex pair):

    d = P / a - 1,     c b = S - P / a - a.

A solution exists for every certified a, because a > 0. With a = 0.5 and the targets 1.78 and -0.28 this rebuilds the research brief's matrix [[1, 1], [1, 0.5]] exactly.

**Result 2 — any size.** With n physical states and m >= n memories, pairing physical state i with memory i (other memories unconnected) splits J into n blocks of size 2 x 2, each solved as in Result 1, plus the unused decay factors. So any n target pairs can be placed for any certified A_bar.

**Link to timing.** An eigenvalue pair r exp(+/- i theta) turns the state by theta per step, so one loop takes 2 pi / theta steps; r sets growth or decay. With the same certified a = 0.9, the script builds loops of 7.27 (true clock) and 9.02 (the full Mamba's slow clock) time units, and growth rates of 0.95, 1.00 and 1.05 per step.

**Checks.** 16 named examples: largest eigenvalue mismatch 8.0e-15. 20,000 random targets (one state, one memory): all placed, worst relative mismatch 1.3e-11. 5,000 random targets with two physical states and sixteen memories: all placed, worst relative mismatch 1.9e-11.

```
Mathematical signal: POSITIVE (for "the certificate cannot constrain timing or growth")
What was checked:    closed-loop eigenvalues of the linearised certified cell in chained prediction
Result:              any eigenvalues (any clock speed, any growth) can be placed for any certified A_bar
Assumptions:         linearisation; selection (input-dependent A_bar, B, C) held fixed, which only adds
                     freedom; the full Mamba contains the minimal cell as a special case
Implication:         the slow clock is not something the certificate could have prevented. A guarantee
                     that keeps the clock right must constrain the whole prediction-plus-memory loop
                     (Stage 3), e.g. keep the along-orbit multiplier at 1 and contract across the orbit
```

**Limits.** This is a possibility result about the linearised loop: it shows the certificate leaves the clock free, not that training will pick a wrong clock. Whether the full Mamba's slow clock comes from feedback amplification or from consistent one-step timing errors is what Stage 2 (Q2, Q3) measures on the trained models.
