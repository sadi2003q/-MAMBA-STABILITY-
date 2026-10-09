"""Small file helpers: JSON, results table with resume, training log."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

RESULT_COLUMNS = [
    "arm", "seed", "group", "role", "training_regime", "model", "decay_mode", "parameters",
    "diverged", "one_step_rmse", "rollout_rmse", "error_at_10", "error_at_50", "growth_1_to_50",
    "other_regime_rollout_rmse", "certificate_holds_at_end", "fraction_decay_positive",
    "model_amplification_median", "true_amplification_median", "amplification_ratio_median",
    "limit_cycle_amplitude_error", "limit_cycle_period_error", "limit_cycle_status",
    "nonfinite_steps", "epochs_run", "train_minutes",
]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_clean(obj), indent=2))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def completed_runs(results_csv: Path) -> set[tuple[str, int]]:
    if not results_csv.exists():
        return set()
    with open(results_csv) as fh:
        return {(row["arm"], int(row["seed"])) for row in csv.DictReader(fh)}


def append_result(results_csv: Path, row: dict) -> None:
    new = not results_csv.exists()
    with open(results_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: _fmt(row.get(k)) for k in RESULT_COLUMNS})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if math.isfinite(v) else ""
    return v


def write_train_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
