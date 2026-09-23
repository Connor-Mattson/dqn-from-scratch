"""Reward-curve plotting (provided scaffold).

Uses matplotlib's object-oriented ``Figure`` API directly, so no GUI backend is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from dqn.stats import SeedBand

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
# Categorical slots for multi-run comparisons, assigned in this fixed order. Blue/orange
# clears colour-vision-deficiency separation on this surface by a wide margin (worst-pair
# protanopia dE 24.7, normal vision 33.6, against a floor of 8 and 15).
#
# The fourth slot exists for the 2x2 target-update cross (vanilla/double x hard/soft) and
# is indigo rather than the more obvious yellow: with four ribbons on one axes any two can
# end up adjacent, and under that stricter all-pairs test yellow collides with the orange
# in slot 2 (normal-vision dE 13.7, below the floor of 15). Indigo is the only addition
# that clears both gates all-pairs — worst normal-vision dE 16.3, worst CVD dE 9.2. Each
# arm is also direct-labelled with its final value, which is what makes four series legal
# at all: identity never rests on colour alone.
SERIES_PALETTE = (SERIES, "#eb6834", "#1baf7a", "#4a3aa7")
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Trailing mean over up to ``window`` most recent values; same length as ``values``."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    ends = np.arange(1, array.size + 1)
    starts = np.maximum(ends - window, 0)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def save_reward_curve(
    episode_returns: Sequence[float],
    path: str | Path,
    window: int = 20,
    title: str = "DQN on CartPole-v1: episode return",
    max_return: float | None = 500.0,
) -> Path:
    """Save raw episode returns plus their trailing mean as a PNG and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    returns = np.asarray(episode_returns, dtype=np.float64)
    episodes = np.arange(1, returns.size + 1)

    fig = Figure(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    if max_return is not None:
        ax.axhline(max_return, color=BASELINE, linewidth=1, zorder=1)
        ax.annotate(
            f"max return ({max_return:g})",
            xy=(0, max_return),
            xycoords=("axes fraction", "data"),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8,
        )
    if returns.size:
        smoothed = moving_average(returns, window)
        ax.plot(episodes, returns, color=SERIES, alpha=0.3, linewidth=1, label="Episode return", zorder=2)
        ax.plot(
            episodes,
            smoothed,
            color=SERIES,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"Mean of last {window} episodes",
            zorder=3,
        )
        ax.annotate(
            f"{smoothed[-1]:.0f}",
            xy=(episodes[-1], smoothed[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
        ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No completed episodes", transform=ax.transAxes, ha="center", color=INK_MUTED)

    ax.set_title(title, loc="left", color=INK_SECONDARY, fontsize=11)
    ax.set_xlabel("Episode", color=INK_MUTED)
    ax.set_ylabel("Return", color=INK_MUTED)
    # Returns start at 0 on envs like CartPole, but can be negative (LunarLander), so only
    # pin the floor to 0 when nothing falls below it.
    lowest = float(returns.min()) if returns.size else 0.0
    if lowest < 0:
        # Headroom above the data so the upper-left legend clears the curve.
        highest = max(float(returns.max()), max_return or 0.0)
        span = highest - lowest
        ax.set_ylim(lowest - 0.05 * span, highest + 0.22 * span)
        ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


MAX_RETURN_VISIBLE_FRACTION = 0.5


def _visible_max_return(highest: float, max_return: float | None) -> float | None:
    """Whether the ``max return`` ceiling earns the vertical space it costs.

    Forcing the axis up to 500 for a run that peaks near 150 spends most of the plot on
    empty space and squashes the curves into the bottom fifth, which is the difference
    between two arms being legible and not. Below half the ceiling, drop the rule and let
    the data set the scale.
    """
    if max_return is None or highest < MAX_RETURN_VISIBLE_FRACTION * max_return:
        return None
    return max_return


def _style_axes(
    ax, title: str, xlabel: str, lowest: float, highest: float, max_return: float | None, ylabel: str = "Return"
) -> None:
    """Shared chrome for the comparison charts: recessive grid, no box, headroom.

    ``lowest`` is 0 unless some return dipped below it, in which case the axis extends down
    to fit it and a zero rule is drawn.
    """
    ax.set_title(title, loc="left", color=INK_SECONDARY, fontsize=11)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    # Headroom above the tallest mark, so the legend sits in empty space rather than on
    # the curves or the max-return rule.
    ceiling = max(highest, max_return or 0.0)
    span = ceiling - lowest
    if span > 0:
        floor = lowest - 0.05 * span if lowest < 0 else 0.0
        ax.set_ylim(floor, ceiling + span * 0.22)
    else:
        ax.set_ylim(bottom=0)
    if lowest < 0:
        ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def _annotate_endpoints(
    ax, endpoints: Sequence[tuple[float, float, str]], span: float, value_format: str = ".0f"
) -> None:
    """Direct-label each series' final value, nudged apart where several finish together.

    At four arms these labels are not decoration: they are what keeps identity from resting
    on colour alone, so two arms ending a few points apart must not overprint each other.
    Working from the lowest value up, each label is pushed to clear the one below it by a
    fixed gap in data coordinates. Only the *text* moves — the number printed is always the
    series' true final value, so the nudge costs a little vertical precision and no accuracy.
    """
    gap = 0.045 * span if span > 0 else 0.0
    placed: list[tuple[float, float, float, str]] = []
    for x, value, color in sorted(endpoints, key=lambda end: end[1]):
        y = value if not placed else max(value, placed[-1][1] + gap)
        placed.append((x, y, value, color))
    for x, y, value, _ in placed:
        ax.annotate(
            f"{value:{value_format}}",
            xy=(x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
            annotation_clip=False,
        )


def _draw_max_return_rule(ax, max_return: float | None) -> None:
    """The 500 ceiling, annotated on the right so it clears the upper-left legend."""
    if max_return is None:
        return
    ax.axhline(max_return, color=BASELINE, linewidth=1, zorder=1)
    ax.annotate(
        f"max return ({max_return:g})",
        xy=(1.0, max_return),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )


@dataclass(frozen=True)
class BandCurve:
    """One arm of an experiment: several seeds averaged, ready to plot against another arm.

    ``color`` pins the arm's hue by role rather than argument order, exactly as
    ``RunCurve`` does, so "hard" stays blue however the arms are passed in.
    """

    label: str
    band: SeedBand
    color: str | None = None


def save_band_comparison_curve(
    arms: Sequence[BandCurve],
    path: str | Path,
    title: str = "DQN on CartPole-v1: hard vs soft target updates",
    legend_title: str = "Mean across seeds, shaded ±1 s.e.",
    max_return: float | None = 500.0,
    ylabel: str = "Return",
    value_format: str = ".0f",
) -> Path:
    """Save one PNG overlaying several arms' across-seed mean curves, each with an error band.

    This replaces the single-run chart's faint raw trace with a standard-error ribbon: with
    several seeds per arm the raw per-episode traces overlap into noise, and the question
    the chart has to answer is whether the arms are separated by more than run-to-run
    variation — which is what a ±1 s.e. band shows directly. Arms whose ribbons overlap
    along their whole length have not been told apart by the seeds run so far.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not arms:
        raise ValueError("save_band_comparison_curve needs at least one run")
    if len(arms) > len(SERIES_PALETTE):
        raise ValueError(f"save_band_comparison_curve supports at most {len(SERIES_PALETTE)} runs, got {len(arms)}")

    fig = Figure(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    lowest, highest = 0.0, 0.0
    endpoints = []
    for slot, arm in enumerate(arms):
        band = arm.band
        color = arm.color or SERIES_PALETTE[slot]
        upper = band.mean + band.stderr
        highest = max(highest, float(upper.max()))
        lowest = min(lowest, float((band.mean - band.stderr).min()))
        if band.has_band:
            # No edge line on the ribbon: the mean is the mark, the band is context.
            ax.fill_between(band.x, band.mean - band.stderr, upper, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            band.x,
            band.mean,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=arm.label,
            zorder=3,
        )
        endpoints.append((float(band.x[-1]), float(band.mean[-1]), color))

    legend = ax.legend(title=legend_title, loc="upper left", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    legend.get_title().set_color(INK_MUTED)
    legend.get_title().set_fontsize(8)
    # After the loop: `highest` decides whether the ceiling rule is worth drawing at all.
    visible_max = _visible_max_return(highest, max_return)
    _draw_max_return_rule(ax, visible_max)
    _style_axes(ax, title, "Environment step", lowest, highest, visible_max, ylabel)
    _annotate_endpoints(ax, endpoints, max(highest, visible_max or 0.0) - lowest, value_format)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


@dataclass(frozen=True)
class RunCurve:
    """One training run's episode returns, ready to plot against another's.

    ``steps`` is the environment step each episode ended on; when every run supplies it
    the comparison is drawn on a shared environment-step axis rather than episode index.
    ``color`` pins a series to a hex colour so a run keeps its identity regardless of the
    order runs are passed in; ``None`` takes the next categorical slot.
    """

    label: str
    returns: Sequence[float]
    steps: Sequence[int] | None = None
    color: str | None = None


def save_comparison_curve(
    runs: Sequence[RunCurve],
    path: str | Path,
    window: int = 20,
    title: str = "DQN on CartPole-v1: hard vs soft target updates",
    max_return: float | None = 500.0,
) -> Path:
    """Save one PNG overlaying several runs' learning curves and return the path.

    Each run contributes a faint raw-return trace and a bold trailing mean, and is direct-
    labelled with its final smoothed value so identity never rests on colour alone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not runs:
        raise ValueError("save_comparison_curve needs at least one run")
    if len(runs) > len(SERIES_PALETTE):
        raise ValueError(f"save_comparison_curve supports at most {len(SERIES_PALETTE)} runs, got {len(runs)}")

    # Episode index is only a fair shared clock when no run reports environment steps:
    # the better agent plays fewer, longer episodes, which compresses its curve leftward.
    on_steps = all(run.steps is not None and len(run.steps) == len(run.returns) for run in runs)

    fig = Figure(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    lowest, highest = 0.0, 0.0
    plotted = False
    endpoints = []
    for slot, run in enumerate(runs):
        returns = np.asarray(run.returns, dtype=np.float64)
        if returns.size == 0:
            continue
        plotted = True
        highest = max(highest, float(returns.max()))
        lowest = min(lowest, float(returns.min()))
        color = run.color or SERIES_PALETTE[slot]
        x = np.asarray(run.steps, dtype=np.float64) if on_steps else np.arange(1, returns.size + 1)
        smoothed = moving_average(returns, window)
        ax.plot(x, returns, color=color, alpha=0.18, linewidth=1, zorder=2)
        ax.plot(
            x,
            smoothed,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=run.label,
            zorder=3,
        )
        endpoints.append((float(x[-1]), float(smoothed[-1]), color))

    if plotted:
        legend = ax.legend(
            title=f"Trailing mean of {window} episodes",
            loc="upper left",
            frameon=False,
            labelcolor=INK_SECONDARY,
            fontsize=9,
        )
        legend.get_title().set_color(INK_MUTED)
        legend.get_title().set_fontsize(8)
    else:
        ax.text(0.5, 0.5, "No completed episodes", transform=ax.transAxes, ha="center", color=INK_MUTED)

    visible_max = _visible_max_return(highest, max_return)
    _draw_max_return_rule(ax, visible_max)
    _style_axes(ax, title, "Environment step" if on_steps else "Episode", lowest, highest, visible_max)
    _annotate_endpoints(ax, endpoints, max(highest, visible_max or 0.0) - lowest)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path
