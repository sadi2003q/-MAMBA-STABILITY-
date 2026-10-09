"""Statistics for Stage 2 (8 or more seeds): bootstrap intervals, Mann-Whitney test, Cliff's delta."""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def bootstrap_median_ci(values, n_boot: int = 10000, seed: int = 0, level: float = 0.95):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.quantile(meds, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.median(v)), float(lo), float(hi)


def bootstrap_ratio_ci(a, b, n_boot: int = 10000, seed: int = 0, level: float = 0.95):
    """Interval for median(a) / median(b), resampling each group independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ra = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    rb = np.median(rng.choice(b, size=(n_boot, len(b)), replace=True), axis=1)
    ratio = ra / rb
    lo, hi = np.quantile(ratio, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.median(a) / np.median(b)), float(lo), float(hi)


def cliffs_delta(a, b) -> float:
    """Share of pairs where a > b minus share where a < b. Range -1 to 1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def compare(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else float("nan")
    d = cliffs_delta(a, b)
    return {"p_value": p, "cliffs_delta": d, "significant": bool(p < 0.05 and abs(d) > 0.33)}
