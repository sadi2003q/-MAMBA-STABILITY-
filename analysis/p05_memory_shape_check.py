"""
Why did the long short-term memory network beat Chain-Mamba on the sloshing-fuel test?

Standalone calculation (NumPy only, no training). The hidden fuel slosh shows up in the spacecraft's turn rate as a
slowly fading wave. Over the 25 predicted steps a memory has to carry that wave forward. We ask: how well can each
KIND of memory reproduce the true wave, in the best case?

    memory type A  (Chain-Mamba now):   sums of real, fading numbers a^k with 0 < a <= 0.951
    memory type B  (longer memory):     sums of real, fading numbers a^k with 0 < a <= 0.9975
    memory type C  (rotating memory):   sums of fading rotations r^k cos(w k), r^k sin(w k) with r <= 0.951
    memory type D  (rotating + longer): the same with r <= 0.9975

The long short-term memory network's memory is mixed by a full 25 x 25 matrix and its forget gate can sit near 1,
so it can hold a fading rotation: it behaves like type D.

Best case = least-squares fit with a large dictionary of allowed decay values (an upper bound on what training can reach).
Assumption: the memory runs on its own over the 25 steps (in reality it also receives the model's own predictions,
which can partly refresh it), so this measures what the memory can carry by itself.

python analysis/p05_memory_shape_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p04_slosh_signal import discretise, linear_matrices  # noqa: E402
from src.systems.slosh_spacecraft import SloshSpacecraft  # noqa: E402

STEPS = 25


def true_waves(system, fill):
    """Turn-rate response over 25 steps to a hidden slosh started at 4 different phases (no push)."""
    A, B = linear_matrices(system, fill)
    Ad, _ = discretise(A, B, system.dt)
    waves = []
    for phase in np.linspace(0, np.pi, 4, endpoint=False):
        x = np.array([0.0, 0.0, np.cos(phase), np.sin(phase) * 2.0])
        wave = []
        for _ in range(STEPS):
            x = Ad @ x
            wave.append(x[1])
        waves.append(np.array(wave))
    return waves


def dictionary(kind, largest):
    k = np.arange(1, STEPS + 1)
    columns = []
    radii = np.linspace(0.05, largest, 25)
    if kind == "real":
        for r in radii:
            columns.append(r ** k)
    else:
        for r in radii:
            for w in np.linspace(0.05, np.pi, 24):
                columns.append(r ** k * np.cos(w * k))
                columns.append(r ** k * np.sin(w * k))
    return np.array(columns).T


MEMORY_NUMBERS = 8      # Chain-Mamba keeps d_state = 8 numbers per channel
RIDGE = 1e-3            # small penalty on huge, cancelling coefficients (relative to the wave size)


def ridge_fit(columns, wave):
    """Least squares with a small penalty on coefficient size; returns the relative error left over."""
    gram = columns.T @ columns + RIDGE * np.eye(columns.shape[1]) * float(np.mean(np.sum(columns ** 2, axis=0)))
    coefficients = np.linalg.solve(gram, columns.T @ wave)
    rest = wave - columns @ coefficients
    return np.linalg.norm(rest) / np.linalg.norm(wave)


def best_fit_error(wave, D, kind):
    """
    Greedy choice of the best memory components, using exactly MEMORY_NUMBERS numbers:
    real memory -> 8 decaying numbers; rotating memory -> 4 decaying rotations (each uses 2 numbers).
    """
    if kind == "real":
        groups = [[i] for i in range(D.shape[1])]
        picks = MEMORY_NUMBERS
    else:
        groups = [[i, i + 1] for i in range(0, D.shape[1], 2)]
        picks = MEMORY_NUMBERS // 2
    chosen = []
    for _ in range(picks):
        best_group, best_error = None, None
        for g in range(len(groups)):
            if g in chosen:
                continue
            columns = []
            for c in chosen + [g]:
                columns.extend(groups[c])
            error = ridge_fit(D[:, columns], wave)
            if best_error is None or error < best_error:
                best_group, best_error = g, error
        chosen.append(best_group)
    return best_error


def main():
    system = SloshSpacecraft()
    kinds = [("A  real, decay <= 0.951 (Chain-Mamba now)", "real", 0.951),
             ("B  real, decay <= 0.9975 (longer memory)", "real", 0.9975),
             ("C  rotating, decay <= 0.951", "rotate", 0.951),
             ("D  rotating, decay <= 0.9975 (like LSTM)", "rotate", 0.9975)]
    fills = [0.35, 0.5, 0.7, 0.85]
    print("=" * 78)
    print(" Best-case error of each memory type at carrying the hidden slosh wave 25 steps")
    print(" (0% = perfect, 100% = no better than predicting nothing). Smaller is better.")
    print("=" * 78)
    header = f" {'memory type':<44}"
    for fill in fills:
        header = header + f"{str(int(fill * 100)) + '% full':>9}"
    print(header)
    for label, kind, largest in kinds:
        D = dictionary(kind, largest)
        row = f" {label:<44}"
        for fill in fills:
            errors = [best_fit_error(w / np.linalg.norm(w), D, kind) for w in true_waves(system, fill)]
            row = row + f"{100 * max(errors):>8.2f}%"
        print(row)
    print()
    print(" Memory left after 25 steps:  0.951^25 = %.2f   0.9975^25 = %.2f" % (0.951 ** 25, 0.9975 ** 25))


if __name__ == "__main__":
    main()
