"""Headline results figure for the README: one row, three panels.

    python -m dqn.results_figure [--output results/results.png]

(a) CartPole-v1 and (b) LunarLander-v3: hard vs soft (Polyak) target updates, 5 seeds each,
trailing mean of 20 episodes resampled onto a shared environment-step grid, ±1 s.e.
(c) MaxBiasCasino-v0: maximization bias of vanilla vs Double DQN as the number of slot
machines grows (σ = 2), read from the casino sweep.

Reads the run directories under ``outputs/`` and ``results/casino/sweep.csv``; trains nothing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.artifacts import discover_returns_csvs, load_returns_csv
from dqn.plotting import moving_average
from dqn.stats import aggregate_seeds

WINDOW = 20

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e4e3df"
AXIS = "#b9b8b2"

HARD = "#2a78d6"
SOFT = "#eb6834"
VANILLA = "#4a3aa7"
DOUBLE = "#1baf7a"


def load_band(run_dir: Path):
    curves = []
    for _, path in discover_returns_csvs(run_dir):
        returns, steps = load_returns_csv(path)
        curves.append((steps, moving_average(returns, WINDOW)))
    return aggregate_seeds(curves)


def style_axes(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", color=TEXT_PRIMARY, pad=22)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)


def reference_line(ax, y: float, label: str, x_right: float) -> None:
    ax.axhline(y, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x_right, y, label, ha="right", va="bottom", fontsize=8.5, color=TEXT_MUTED)


def end_label(ax, x: float, y: float, text: str, color: str, dy: float = 0) -> None:
    ax.plot([x], [y], marker="o", markersize=5, color=color, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=5)
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", va="center",
                fontsize=9.5, fontweight="bold", color=TEXT_PRIMARY)


def learning_panel(ax, env: str, arms, reference: tuple[float, str, float], ylim, label_offsets) -> None:
    style_axes(ax, env, f"Trailing mean of {WINDOW} episodes · 5 seeds · ±1 s.e.")
    for (label, run_dir, color), dy in zip(arms, label_offsets):
        band = load_band(run_dir)
        ax.fill_between(band.x, band.mean - band.stderr, band.mean + band.stderr,
                        color=color, alpha=0.14, linewidth=0)
        ax.plot(band.x, band.mean, color=color, linewidth=2, label=label, solid_capstyle="round")
        end_label(ax, band.x[-1], band.mean[-1], f"{band.mean[-1]:.0f}", color, dy)
    reference_line(ax, *reference)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 100_000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Environment steps", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylabel("Episode return", fontsize=9.5, color=TEXT_SECONDARY)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=TEXT_PRIMARY, handlelength=1.6)


def casino_panel(ax, sweep_csv: Path, sigma: float = 2.0) -> None:
    rows = [r for r in csv.DictReader(sweep_csv.open()) if float(r["sigma"]) == sigma]
    style_axes(ax, "MaxBiasCasino-v0 (custom env)",
               f"Q-value overestimation vs. ground truth · σ = {sigma:g} · 5 seeds · ±1 s.e.")
    for arm, color, marker in (("Vanilla DQN", VANILLA, "o"), ("Double DQN", DOUBLE, "D")):
        arm_rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: int(r["n_actions"]))
        n = np.array([int(r["n_actions"]) for r in arm_rows])
        bias = np.array([float(r["bias"]) for r in arm_rows])
        se = np.array([float(r["bias_se"]) for r in arm_rows])
        ax.fill_between(n, bias - se, bias + se, color=color, alpha=0.14, linewidth=0)
        ax.plot(n, bias, color=color, linewidth=2, marker=marker, markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm)
        ax.annotate(f"+{bias[-1]:.2f}", (n[-1], bias[-1]), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9.5, fontweight="bold", color=TEXT_PRIMARY)
    reference_line(ax, 0.0, "unbiased", 50)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(1.8, 70)
    ax.set_ylim(-0.06, 0.42)
    ax.set_xlabel("Slot machines (noisy actions)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylabel("Bootstrap bias (Q − Q*)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=TEXT_PRIMARY, handlelength=1.6)


def make_figure(outputs: Path, sweep_csv: Path, output_path: Path) -> Path:
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.28})
    learning_panel(
        axes[0], "CartPole-v1",
        [("Hard target sync", outputs / "hard", HARD), ("Soft (Polyak) τ = 0.005", outputs / "soft", SOFT)],
        (500, "max return 500", 70_000), (0, 600), label_offsets=(-8, 8),
    )
    learning_panel(
        axes[1], "LunarLander-v3",
        [("Hard target sync", outputs / "lunar", HARD), ("Soft (Polyak) τ = 0.005", outputs / "lunar-soft", SOFT)],
        (200, "solved = 200", 50_000), (-200, 330), label_offsets=(-8, 8),
    )
    casino_panel(axes[2], sweep_csv)
    fig.subplots_adjust(left=0.05, right=0.965, top=0.83, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--sweep", type=Path, default=Path("results/casino/sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/results.png"))
    args = parser.parse_args(argv)
    print(f"saved {make_figure(args.outputs, args.sweep, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
