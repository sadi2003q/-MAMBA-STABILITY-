"""Small helpers shared by the two space-pilot run tools: result files, medians, plain-language labels,
and the chained-prediction check."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch


# ==========================================================================================================
# results.csv (one row per finished run; used to resume)
# ==========================================================================================================
def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict, columns: list) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ==========================================================================================================
# Numbers and words
# ==========================================================================================================
def median(values) -> float:
    """Median of a list (infinite values allowed, e.g. 'never settled')."""
    ordered = sorted(float(value) for value in values)
    number = len(ordered)
    if number == 0:
        return float("nan")
    middle = number // 2
    if number % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def ratio(new: float, base: float) -> float:
    """new / base, with care for zero and infinity."""
    if math.isinf(new) and math.isinf(base):
        return 1.0
    if base == 0.0:
        if new == 0.0:
            return 1.0
        return float("inf")
    return new / base


def change_words(new: float, base: float) -> str:
    """'+35% (clearly worse)' style label; smaller is better for every metric used here."""
    r = ratio(new, base)
    if math.isinf(r):
        return "far worse"
    percent = 100.0 * (r - 1.0)
    if abs(percent) <= 10.0:
        word = "about the same"
    elif abs(percent) <= 20.0:
        word = "slightly better" if percent < 0 else "slightly worse"
    else:
        word = "clearly better" if percent < 0 else "clearly worse"
    return f"{percent:+.0f}% ({word})"


def fmt(value: float, digits: int = 3) -> str:
    if math.isinf(value):
        return "never"
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


# ==========================================================================================================
# Chained-prediction check (the same predict() the controller uses)
# ==========================================================================================================
@torch.no_grad()
def chained_prediction_errors(predictor, states: torch.Tensor, pushes: torch.Tensor, horizon: int,
                              every: int, error_function, device) -> list:
    """
    Start a chained prediction every `every` steps of every test trajectory, run it `horizon` steps with the
    true pushes, and compare with the true states.
        states [N, T + 1, n_x], pushes [N, T, n_u] (physical units)
        error_function(predicted [W, n_x], true [W, n_x]) -> error per window [W]
    Returns the root-mean-square error after 1, 2, ..., horizon steps (a list of `horizon` numbers).
    """
    starting_states = []
    planned_pushes = []
    true_paths = []
    total_steps = pushes.shape[1]
    start = 0
    while start + horizon <= total_steps:
        starting_states.append(states[:, start])
        planned_pushes.append(pushes[:, start:start + horizon])
        true_paths.append(states[:, start + 1:start + 1 + horizon])
        start = start + every

    x_now = torch.cat(starting_states, dim=0).to(device)
    u_plan = torch.cat(planned_pushes, dim=0).to(device)
    truth = torch.cat(true_paths, dim=0).to(device)
    predicted = predictor.predict(x_now, u_plan, None, None)

    errors = []
    for k in range(horizon):
        per_window = error_function(predicted[:, k], truth[:, k])
        errors.append(float(torch.sqrt((per_window ** 2).mean())))
    return errors
