"""
a02 — How much error amplification does the true Van der Pol system itself allow?

Standalone mathematical check (NumPy, SciPy, Matplotlib; no training, no project code).

Why this matters
----------------
Any certificate we put on the learned model must not forbid what the true system
does. If the true system amplifies small errors on parts of its limit cycle, then
"global contraction" (every one-step Jacobian shrinks every direction) is
impossible for an accurate model. This script measures, along the limit cycle:

  1. Per-step:  the largest singular value of the one-step Jacobian
                J_k = I + dt * Df(x_k)   (forward Euler, as in the research brief).
                Singular value = how much the step stretches the worst direction.
  2. 50-step:   the largest singular value of the product J_{k+49} ... J_k,
                for every starting point on the cycle. This is the true system's
                own finite-horizon error amplification: the most a perfect model
                would amplify an initial error over the planning window.
  3. Split into "along the orbit" (a timing / phase error, which the flow neither
                damps nor corrects) and "across the orbit" (a shape error, which an
                orbitally stable limit cycle damps).
  4. Floquet multipliers of the continuous-time cycle (how a perturbation changes
                after one full period): one equals 1 (phase), the other is below 1
                (shape), which is exactly transverse contraction.

System (standard controlled Van der Pol oscillator, control input u = 0 here):
    dx1/dt = x2
    dx2/dt = mu (1 - x1^2) x2 - x1 + u
NOTE: the research brief (Section 3.1) writes dx2/dt = mu (1 - x1^2) x2 + u,
without the "- x1" term. Without it there is no oscillation and no limit cycle
(every point with x2 = 0 is an equilibrium). This script uses the standard form.

Assumptions: control input zero; forward Euler discretisation; errors small enough
for the linearisation to hold over the horizon.

Run:  python analysis/a02_vanderpol_jacobian_along_orbit.py
      python analysis/a02_vanderpol_jacobian_along_orbit.py --mu 1.0 --dts 0.01 0.05 0.1 --horizon 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

RESULTS_DIR = Path(os.environ.get("ANALYSIS_OUT", "outputs/analysis"))


# ---------------------------------------------------------------- system
def vdp(x: np.ndarray, mu: float) -> np.ndarray:
    return np.array([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]])


def vdp_jacobian(x: np.ndarray, mu: float) -> np.ndarray:
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * x[0] * x[1] - 1.0, mu * (1.0 - x[0] ** 2)]])


def euler_step(x: np.ndarray, mu: float, dt: float) -> np.ndarray:
    return x + dt * vdp(x, mu)


# ---------------------------------------------------------------- continuous cycle
def continuous_cycle(mu: float) -> dict:
    """Period and Floquet multipliers of the continuous-time limit cycle."""
    def rhs(t, z):
        x = z[:2]
        phi = z[2:].reshape(2, 2)
        dphi = vdp_jacobian(x, mu) @ phi
        return np.concatenate([vdp(x, mu), dphi.ravel()])

    # settle onto the cycle
    settle = solve_ivp(lambda t, x: vdp(x, mu), (0, 200), [2.0, 0.0], rtol=1e-11, atol=1e-12)
    x0 = settle.y[:, -1]

    # find period with a section x2 = 0, x2 decreasing, x1 > 0
    def section(t, x):
        return x[1]
    section.direction = -1
    sol = solve_ivp(lambda t, x: vdp(x, mu), (0, 60), x0, rtol=1e-11, atol=1e-12,
                    events=section, dense_output=True)
    hits = [t for t, y in zip(sol.t_events[0], sol.y_events[0]) if y[0] > 0]
    start_t = hits[0]
    period = hits[1] - hits[0]
    x_start = sol.sol(start_t)

    var = solve_ivp(rhs, (0, period), np.concatenate([x_start, np.eye(2).ravel()]),
                    rtol=1e-11, atol=1e-12)
    monodromy = var.y[2:, -1].reshape(2, 2)
    multipliers = np.sort(np.abs(np.linalg.eigvals(monodromy)))[::-1]

    # Liouville check: product of multipliers = exp(integral of divergence)
    grid = np.linspace(0, period, 20001)
    path = solve_ivp(lambda t, x: vdp(x, mu), (0, period), x_start, t_eval=grid,
                     rtol=1e-11, atol=1e-12).y
    divergence = mu * (1.0 - path[0] ** 2)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 1.x has only trapz
    liouville = float(np.exp(trapezoid(divergence, grid)))

    return {
        "period": float(period),
        "floquet_multipliers": multipliers.tolist(),
        "product_check_liouville": liouville,
        "amplitude_x1": float(np.max(np.abs(path[0]))),
    }


# ---------------------------------------------------------------- discrete analysis
def euler_cycle(mu: float, dt: float, settle_time: float = 300.0) -> np.ndarray:
    """Points of the forward-Euler map's own cycle (one period, plus margin)."""
    x = np.array([2.0, 0.0])
    for _ in range(int(settle_time / dt)):
        x = euler_step(x, mu, dt)
    # one period worth of steps, found by return to the section x2 = 0 (falling, x1 > 0)
    pts = [x]
    crossings = 0
    for _ in range(int(40.0 / dt)):
        x_new = euler_step(x, mu, dt)
        if x[1] > 0 >= x_new[1] and x_new[0] > 0:
            crossings += 1
            if crossings == 2:
                break
            pts = [x_new]
        else:
            pts.append(x_new)
        x = x_new
    return np.array(pts)


def analyse_dt(mu: float, dt: float, horizon: int) -> dict:
    orbit = euler_cycle(mu, dt)
    n = len(orbit)
    jacs = np.array([np.eye(2) + dt * vdp_jacobian(p, mu) for p in orbit])
    sv_step = np.array([np.linalg.svd(J, compute_uv=False) for J in jacs])
    sigma_max_step = sv_step[:, 0]

    # finite-horizon products, wrapping around the cycle
    amp_total = np.empty(n)
    amp_along = np.empty(n)
    amp_across = np.empty(n)
    for k in range(n):
        phi = np.eye(2)
        for j in range(horizon):
            phi = jacs[(k + j) % n] @ phi
        amp_total[k] = np.linalg.svd(phi, compute_uv=False)[0]
        # direction along the flow at the start and end of the window
        f_start = vdp(orbit[k], mu)
        f_end = vdp(orbit[(k + horizon) % n], mu)
        t_start = f_start / np.linalg.norm(f_start)
        n_end = np.array([-f_end[1], f_end[0]]) / np.linalg.norm(f_end)
        n_start = np.array([-f_start[1], f_start[0]]) / np.linalg.norm(f_start)
        # along-orbit (phase) error: how much a shift along the orbit is stretched
        amp_along[k] = np.linalg.norm(phi @ t_start)
        # across-orbit (shape) error: component of the image of a normal kick that
        # stays normal at the end of the window
        amp_across[k] = abs(n_end @ (phi @ n_start))

    # continuous-time stretching rate: largest eigenvalue of the symmetric part of Df
    # ("logarithmic norm"). Positive = the true flow itself stretches some direction
    # there. Forward Euler adds extra stretching on top (it slightly expands any rotation).
    lognorm = np.array([np.linalg.eigvalsh(0.5 * (vdp_jacobian(p, mu) + vdp_jacobian(p, mu).T))[-1]
                        for p in orbit])

    gamma = float(sigma_max_step.max())
    geometric = float((gamma ** horizon - 1.0) / (gamma - 1.0)) if gamma != 1.0 else float(horizon)

    return {
        "dt": dt,
        "horizon_steps": horizon,
        "horizon_time": dt * horizon,
        "steps_per_period": n,
        "per_step_largest_singular_value": {
            "max": float(sigma_max_step.max()),
            "median": float(np.median(sigma_max_step)),
            "min": float(sigma_max_step.min()),
            "fraction_of_orbit_above_1": float(np.mean(sigma_max_step > 1.0)),
        },
        "continuous_stretching_rate": {
            "max": float(lognorm.max()), "min": float(lognorm.min()),
            "fraction_of_orbit_positive": float(np.mean(lognorm > 0.0)),
        },
        "horizon_amplification_total": {
            "max": float(amp_total.max()), "median": float(np.median(amp_total)),
            "min": float(amp_total.min()),
        },
        "horizon_amplification_along_orbit": {
            "max": float(amp_along.max()), "median": float(np.median(amp_along)),
            "min": float(amp_along.min()),
        },
        "horizon_amplification_across_orbit": {
            "max": float(amp_across.max()), "median": float(np.median(amp_across)),
            "min": float(amp_across.min()),
        },
        "worst_case_geometric_bound_using_max_per_step": geometric,
        "_series": {"orbit": orbit.tolist(), "sigma_max_step": sigma_max_step.tolist(),
                    "amp_total": amp_total.tolist(), "amp_along": amp_along.tolist(),
                    "amp_across": amp_across.tolist()},
    }


def make_figure(results: list[dict], mu: float, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for r in results:
        s = r["_series"]
        phase = np.linspace(0, 1, len(s["sigma_max_step"]), endpoint=False)
        axes[0].plot(phase, s["sigma_max_step"], label=f"time step {r['dt']}")
        axes[1].plot(phase, s["amp_total"], label=f"time step {r['dt']}")
    axes[0].axhline(1.0, color="grey", ls="--", lw=1)
    axes[0].set_title("One step: largest stretch of an error")
    axes[0].set_xlabel("position along the limit cycle (fraction of one period)")
    axes[0].set_ylabel("largest singular value")
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title(f"{results[0]['horizon_steps']} steps: true system's own error amplification")
    axes[1].set_xlabel("starting position along the cycle (fraction of one period)")
    axes[1].set_ylabel("largest singular value of the product")
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    s = results[-1]["_series"]
    orbit = np.array(s["orbit"])
    sc = axes[2].scatter(orbit[:, 0], orbit[:, 1], c=s["sigma_max_step"], s=6, cmap="coolwarm")
    fig.colorbar(sc, ax=axes[2], label="one-step largest singular value")
    axes[2].set_title(f"Where the cycle stretches errors (time step {results[-1]['dt']})")
    axes[2].set_xlabel("position x1")
    axes[2].set_ylabel("velocity x2")
    fig.suptitle(f"Van der Pol oscillator, mu = {mu}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "a02_vanderpol_jacobian_along_orbit.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--dts", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()

    print("=" * 78)
    print(f"a02 — Error amplification of the true Van der Pol system (mu = {args.mu})")
    print("=" * 78)

    cont = continuous_cycle(args.mu)
    print("Continuous-time limit cycle")
    print(f"  Period: {cont['period']:.4f} time units; largest |x1| on the cycle: {cont['amplitude_x1']:.4f}")
    m = cont["floquet_multipliers"]
    print(f"  Floquet multipliers (change of a perturbation after one full period): {m[0]:.6f} and {m[1]:.3e}")
    print(f"  Check: their product should equal exp(integral of divergence) = {cont['product_check_liouville']:.3e}")
    print("  Reading: 1 = timing errors are kept, never corrected; the small one = shape errors die out.")
    print("  This is transverse contraction: the right kind of stability to aim for.")
    print()

    results = []
    for dt in args.dts:
        r = analyse_dt(args.mu, dt, args.horizon)
        results.append(r)
        p = r["per_step_largest_singular_value"]
        t = r["horizon_amplification_total"]
        a = r["horizon_amplification_along_orbit"]
        c = r["horizon_amplification_across_orbit"]
        print(f"Time step {dt}  (forward Euler; {args.horizon} steps = {r['horizon_time']:.2f} time units;"
              f" {r['steps_per_period']} steps per period)")
        print(f"  One step, largest singular value: max {p['max']:.4f}, median {p['median']:.4f},"
              f" min {p['min']:.4f}")
        print(f"    above 1 (stretches some direction) on {100*p['fraction_of_orbit_above_1']:.1f}% of the cycle")
        s = r["continuous_stretching_rate"]
        print(f"    the continuous flow itself stretches on {100*s['fraction_of_orbit_positive']:.1f}% of the cycle"
              f" (rate from {s['min']:.3f} to {s['max']:.3f}); the rest of the stretch is from forward Euler")
        print(f"  Over {args.horizon} steps, true system's own amplification of an error:")
        print(f"    any direction:     max {t['max']:.3g}, median {t['median']:.3g}, min {t['min']:.3g}")
        print(f"    along the orbit:   max {a['max']:.3g}, median {a['median']:.3g}, min {a['min']:.3g}")
        print(f"    across the orbit:  max {c['max']:.3g}, median {c['median']:.3g}, min {c['min']:.3g}")
        print(f"  Worst-case bound if every step stretched by the maximum: {r['worst_case_geometric_bound_using_max_per_step']:.3g}"
              " (times the one-step error)")
        print()

    worst = max(r["per_step_largest_singular_value"]["fraction_of_orbit_above_1"] for r in results)
    across_max = max(r["horizon_amplification_across_orbit"]["max"] for r in results)
    print("Conclusion")
    print(f"  The true one-step map stretches some direction on up to {100*worst:.0f}% of the cycle,")
    print("  so a model required to contract everywhere cannot match the true system.")
    print(f"  The {args.horizon}-step numbers above are what a PERFECT model would also show; a learned")
    print("  model's error-amplification certificate should be stated RELATIVE to them.")
    print("  Only shape errors (across the orbit) die out, and only over long enough windows")
    print(f"  in ordinary distance (short windows reach {across_max:.3g}); a contraction certificate")
    print("  will need a state-dependent way of measuring distance, not plain Euclidean distance.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = make_figure(results, args.mu, RESULTS_DIR / "figures")
    summary = {"mu": args.mu, "continuous_cycle": cont,
               "per_time_step": [{k: v for k, v in r.items() if k != "_series"} for r in results]}
    path = RESULTS_DIR / "a02_vanderpol_jacobian_along_orbit.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved numbers to {path}")
    if fig_path:
        print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
