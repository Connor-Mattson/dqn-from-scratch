"""Across-seed statistics for run comparisons (provided scaffold).

A single training run is one sample of a very noisy process, so one seed per arm cannot
tell you whether a change helped. This module turns several seeds per arm into a mean
curve with a standard-error band, and runs Welch's t-test on the arms' final scores.

Self-contained on purpose: the project depends on numpy but not scipy, so the Student-t
tail is computed here from the regularised incomplete beta function. ``tests/test_stats.py``
pins it against published t-table values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

GRID_POINTS = 240  # resolution of the shared x-axis the seeds are averaged on


# --------------------------------------------------------------------------------------
# Student-t tail
# --------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        # Even step, then odd step: each iteration advances the fraction by two terms.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_lead = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    lead = math.exp(log_lead)
    # The fraction converges fast only on one side of this point; reflect onto the other.
    if x < (a + 1.0) / (a + b + 2.0):
        return lead * _betacf(a, b, x) / a
    return 1.0 - lead * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a Student-t statistic with ``df`` degrees of freedom."""
    if df <= 0 or not math.isfinite(t) or not math.isfinite(df):
        return float("nan")
    return regularised_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


# --------------------------------------------------------------------------------------
# Welch's t-test
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TTest:
    """Welch's unequal-variance t-test of ``b`` against ``a``."""

    difference: float  # mean(b) - mean(a)
    stderr: float  # standard error of that difference
    t: float
    df: float
    p: float
    n_a: int
    n_b: int

    @property
    def significant(self) -> bool:
        return math.isfinite(self.p) and self.p < 0.05


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> TTest | None:
    """Compare two independent samples without assuming equal variance.

    Returns ``None`` when either arm has fewer than two seeds — with one sample there is
    no within-arm variance to compare against, and any "significance" would be invented.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    n_a, n_b = first.size, second.size
    if n_a < 2 or n_b < 2:
        return None

    var_a, var_b = float(first.var(ddof=1)), float(second.var(ddof=1))
    se_a, se_b = var_a / n_a, var_b / n_b
    stderr = math.sqrt(se_a + se_b)
    difference = float(second.mean() - first.mean())
    if stderr == 0.0:
        # Two arms of identical constants: no difference and no spread to test it against.
        return TTest(difference, 0.0, float("nan"), float("nan"), float("nan"), n_a, n_b)

    t = difference / stderr
    # Welch-Satterthwaite: the effective df is dragged toward the noisier, smaller arm.
    df = (se_a + se_b) ** 2 / (se_a**2 / (n_a - 1) + se_b**2 / (n_b - 1))
    return TTest(difference, stderr, t, df, t_two_sided_p(t, df), n_a, n_b)


# --------------------------------------------------------------------------------------
# Averaging seeds onto a shared axis
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedBand:
    """Several seeds of one arm, averaged onto a shared x-axis."""

    x: np.ndarray  # shared grid the seeds were resampled onto
    mean: np.ndarray  # mean across seeds at each grid point
    stderr: np.ndarray  # standard error of that mean (zeros when n_seeds == 1)
    n_seeds: int

    @property
    def has_band(self) -> bool:
        """A band is only meaningful once there is between-seed variance to show."""
        return self.n_seeds > 1


def aggregate_seeds(curves: Sequence[tuple[np.ndarray, np.ndarray]]) -> SeedBand | None:
    """Average ``(x, y)`` curves of differing lengths onto one grid.

    Seeds do not share an x-axis: each run plays a different number of episodes, ending
    on different environment steps. Each curve is therefore resampled by interpolation
    onto a common grid before averaging, rather than averaged by episode index — which
    would silently compare episode 300 of a good run against episode 300 of a bad one.

    The grid spans the range *every* seed covers (the latest start, the earliest end), so
    the number of seeds behind the mean is constant across the whole width. Returns
    ``None`` when no curve has two usable points.
    """
    usable = []
    for x, y in curves:
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        if x_array.size >= 2 and x_array.size == y_array.size:
            usable.append((x_array, y_array))
    if not usable:
        return None

    lo = max(float(x[0]) for x, _ in usable)
    hi = min(float(x[-1]) for x, _ in usable)
    if not hi > lo:
        return None

    grid = np.linspace(lo, hi, GRID_POINTS)
    # np.interp needs ascending x; episode end-steps are already monotonic by construction.
    resampled = np.vstack([np.interp(grid, x, y) for x, y in usable])
    mean = resampled.mean(axis=0)
    n = resampled.shape[0]
    stderr = resampled.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
    return SeedBand(grid, mean, stderr, n)


def mean_stderr(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, standard error)`` of a sample; the error is 0.0 for a single value."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))
