"""Compare training arms' learning curves.

    python -m dqn.compare --hard outputs/hard --soft outputs/soft
    python -m dqn.compare --a outputs/vanilla --b outputs/double \\
        --labels "vanilla DQN" "Double DQN" --output results/double_vs_vanilla.png
    python -m dqn.compare --arm "ddqn hard=outputs/ddqn-hard" \\
        --arm "ddqn soft=outputs/ddqn-soft" --output results/target_update_cross.png

The arms were originally hard and soft target updates, and those remain the flag names and
default labels. Nothing here is specific to them: ``--a``/``--b`` are aliases for the same
two positions and ``--labels`` renames them, so any pair of runs that differ in one setting
— Double DQN against vanilla, say — is compared the same way.

``--arm LABEL=PATH`` adds further arms beyond those two, up to four in total, which is what
puts the whole 2x2 cross of ``--double`` against ``--target-update-mode`` on one chart. Every
extra arm is tested against the *first* arm, so with two arms this reads exactly as it always
did. Four is the ceiling because it is the last series count the palette separates safely.

Each argument is one *arm* of the experiment: either a directory of per-seed runs (what
``python -m dqn.train --seeds 0 1 2 3 4 --output-dir outputs/hard`` writes) or a single
run's ``episode_returns.csv``. Seeds within an arm are averaged onto a shared
environment-step axis and drawn with a ±1 standard-error band, and the arms' final
scores are compared with Welch's t-test.

Nothing here retrains anything: run the experiments first, then point this at their
directories.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dqn.artifacts import COMPARISON_NAME, RESULTS_DIR, discover_returns_csvs, load_returns_csv
from dqn.plotting import SERIES_PALETTE, BandCurve, RunCurve, moving_average, save_band_comparison_curve
from dqn.plotting import save_comparison_curve
from dqn.stats import TTest, aggregate_seeds, mean_stderr, welch_ttest

HARD_LABEL = "hard targets"
SOFT_LABEL = "soft targets"
SOLVED_THRESHOLD = 195.0  # the classic CartPole bar: a trailing mean of 195 over 100 episodes
MAX_ARMS = len(SERIES_PALETTE)  # one categorical colour per arm; the palette is the ceiling
LABEL_COLUMN = 14  # minimum width of the table's arm column, widened to fit longer labels


@dataclass
class SeedRun:
    """One seed's episode returns."""

    seed: int | None
    path: Path
    returns: np.ndarray
    steps: np.ndarray | None


@dataclass
class RunSummary:
    """Headline numbers for one arm, aggregated over its seeds.

    With a single seed every ``*_stderr`` is 0.0 and the means are that run's own values,
    so this reads the same for a one-seed arm as the original single-run summary did.
    """

    label: str
    paths: list[Path]
    seeds: list[int | None]
    n_seeds: int
    episodes: int  # mean across seeds, rounded
    total_steps: int | None
    final_mean: float
    final_stderr: float
    best_mean: float
    best_stderr: float
    episodes_to_threshold: int | None  # median across the seeds that reached it
    steps_to_threshold: int | None
    seeds_reaching: int
    per_seed_final: list[float]


@dataclass
class CompareResult:
    plot_path: Path
    summaries: list[RunSummary]
    # One entry per arm after the first, each testing that arm against the first. An entry
    # is None unless both of its arms have >= 2 seeds.
    ttests: list[TTest | None] = field(default_factory=list)

    @property
    def ttest(self) -> TTest | None:
        """The second arm against the first — the whole story when there are only two."""
        return self.ttests[0] if self.ttests else None


def parse_arm_spec(spec: str) -> tuple[str, Path]:
    """Read a ``--arm`` value into ``(label, path)``.

    ``"ddqn soft=outputs/ddqn-soft"`` names the arm explicitly. A bare path is also
    accepted and labels itself from the directory name, so ``outputs/ddqn-soft`` becomes
    "ddqn soft" — the hyphen reads as a word break in a legend, not as punctuation.
    """
    label, sep, path = spec.partition("=")
    if sep and label.strip() and path.strip():
        return label.strip(), Path(path.strip())
    candidate = Path(spec.strip())
    if not spec.strip():
        raise ValueError("--arm needs a path, e.g. --arm outputs/ddqn-soft or --arm 'label=path'")
    stem = candidate.name if candidate.name else str(candidate)
    return stem.replace("-", " ").replace("_", " "), candidate


def load_arm(path: str | Path) -> list[SeedRun]:
    """Read every seed belonging to one arm of the experiment."""
    runs = []
    for seed, csv_path in discover_returns_csvs(path):
        returns, steps = load_returns_csv(csv_path)
        runs.append(SeedRun(seed, csv_path, returns, steps))
    return runs


def summarise_arm(label: str, runs: list[SeedRun], window: int, threshold: float) -> RunSummary:
    """Reduce one arm to the numbers worth putting next to another arm's.

    ``*_to_threshold`` answers "how much experience did this arm need to get good?" — the
    median across seeds of the first episode whose trailing mean reached ``threshold``, and
    the environment step it landed on. The median is taken over the seeds that *reached* it,
    with ``seeds_reaching`` reporting how many did; averaging in a never-reached seed as if
    it were a large number would quietly invent data.
    """
    paths = [run.path for run in runs]
    seeds = [run.seed for run in runs]
    scored = [run for run in runs if run.returns.size]
    if not scored:
        return RunSummary(
            label, paths, seeds, len(runs), 0, None, float("nan"), float("nan"),
            float("nan"), float("nan"), None, None, 0, [],
        )

    finals, bests, episode_hits, step_hits, episodes, last_steps = [], [], [], [], [], []
    for run in scored:
        smoothed = moving_average(run.returns, window)
        finals.append(float(smoothed[-1]))
        bests.append(float(smoothed.max()))
        episodes.append(int(run.returns.size))
        if run.steps is not None and run.steps.size == run.returns.size:
            last_steps.append(int(run.steps[-1]))
        reached = np.flatnonzero(smoothed >= threshold)
        if reached.size:
            episode_hits.append(int(reached[0]) + 1)
            if run.steps is not None and run.steps.size == run.returns.size:
                step_hits.append(int(run.steps[reached[0]]))

    final_mean, final_stderr = mean_stderr(finals)
    best_mean, best_stderr = mean_stderr(bests)
    return RunSummary(
        label=label,
        paths=paths,
        seeds=seeds,
        n_seeds=len(scored),
        episodes=int(round(float(np.mean(episodes)))),
        total_steps=int(round(float(np.mean(last_steps)))) if last_steps else None,
        final_mean=final_mean,
        final_stderr=final_stderr,
        best_mean=best_mean,
        best_stderr=best_stderr,
        episodes_to_threshold=int(np.median(episode_hits)) if episode_hits else None,
        steps_to_threshold=int(np.median(step_hits)) if step_hits else None,
        seeds_reaching=len(episode_hits),
        per_seed_final=finals,
    )


def _arm_label(label: str, n_seeds: int) -> str:
    """Put the seed count in the series name, so the legend carries it too."""
    return label if n_seeds <= 1 else f"{label} ({n_seeds} seeds)"


def _default_title(labels: Sequence[str]) -> str:
    """Name the arms in the title; the legend repeats them, so keep the same order."""
    if len(labels) == 2:
        return f"DQN on CartPole-v1: {labels[0]} vs {labels[1]}"
    return f"DQN on CartPole-v1: {', '.join(labels[:-1])} vs {labels[-1]}"


def compare_arms(
    arms: Sequence[tuple[str, str | Path]],
    output_path: str | Path,
    window: int = 20,
    threshold: float = SOLVED_THRESHOLD,
    title: str | None = None,
) -> CompareResult:
    """Plot every ``(label, path)`` arm into ``output_path`` and summarise each one.

    Every arm after the first is tested against the first, which makes the first arm the
    baseline the others are read against. With exactly two arms that is the single
    hard-vs-soft test this module has always printed.
    """
    if not arms:
        raise ValueError("compare_arms needs at least one arm")
    if len(arms) > MAX_ARMS:
        raise ValueError(
            f"compare_arms supports at most {MAX_ARMS} arms (one per palette colour), got {len(arms)}"
        )

    # Pin each arm's colour by position rather than argument order, so the first arm stays
    # blue and the second orange no matter how the arms are passed in.
    loaded = [
        (label, load_arm(path), color) for (label, path), color in zip(arms, SERIES_PALETTE)
    ]
    summaries = [summarise_arm(label, runs, window, threshold) for label, runs, _ in loaded]
    plot_title = title if title is not None else _default_title([label for label, _ in arms])

    # Averaging seeds needs a shared clock, which only the `step` column provides. Runs from
    # before that column existed fall back to the original per-episode overlay.
    on_steps = all(
        run.steps is not None and run.steps.size == run.returns.size and run.returns.size
        for _, runs, _ in loaded
        for run in runs
    )
    bands = []
    if on_steps:
        for (label, runs, color), summary in zip(loaded, summaries):
            band = aggregate_seeds([(run.steps, moving_average(run.returns, window)) for run in runs])
            if band is not None:
                bands.append(BandCurve(_arm_label(label, summary.n_seeds), band, color))

    if bands and len(bands) == len(loaded):
        seed_counts = {summary.n_seeds for summary in summaries}
        spread = "shaded ±1 s.e. across seeds" if max(seed_counts) > 1 else "single seed per arm"
        plot_path = save_band_comparison_curve(
            bands,
            output_path,
            title=plot_title,
            legend_title=f"Trailing mean of {window} episodes, {spread}",
        )
    else:
        curves = [
            RunCurve(label=label, returns=runs[0].returns, steps=runs[0].steps, color=color)
            for label, runs, color in loaded
            if runs
        ]
        plot_path = save_comparison_curve(curves, output_path, window=window, title=plot_title)

    baseline = summaries[0].per_seed_final
    ttests = [welch_ttest(baseline, summary.per_seed_final) for summary in summaries[1:]]
    return CompareResult(plot_path, summaries, ttests)


def compare(
    hard_path: str | Path,
    soft_path: str | Path,
    output_path: str | Path,
    window: int = 20,
    threshold: float = SOLVED_THRESHOLD,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    extra_arms: Sequence[tuple[str, str | Path]] | None = None,
) -> CompareResult:
    """Compare two arms, plus any ``extra_arms`` beyond them.

    ``labels`` renames the two named arms in the table, the legend and the default title,
    for experiments other than hard vs soft. The first path is always the first label.
    ``extra_arms`` carries its own labels, since there is no flag pair to name them.
    """
    first_label, second_label = labels if labels is not None else (HARD_LABEL, SOFT_LABEL)
    arms: list[tuple[str, str | Path]] = [
        (first_label, Path(hard_path)),
        (second_label, Path(soft_path)),
    ]
    arms.extend(extra_arms or [])
    if title is None and labels is None and not arms[2:]:
        title = "DQN on CartPole-v1: hard vs soft target updates"
    return compare_arms(arms, output_path, window, threshold, title)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two or more training arms.")
    parser.add_argument(
        "--hard",
        "--a",
        type=Path,
        default=Path("outputs") / "hard",
        help="run directory (or a single episode_returns CSV) for the first arm "
        "(default: the --target-update-mode hard arm)",
    )
    parser.add_argument(
        "--soft",
        "--b",
        type=Path,
        default=Path("outputs") / "soft",
        help="run directory (or a single episode_returns CSV) for the second arm "
        "(default: the --target-update-mode soft arm)",
    )
    parser.add_argument(
        "--arm",
        action="append",
        metavar="LABEL=PATH",
        default=None,
        dest="arms",
        help="add a further arm beyond --hard/--soft, repeatable, up to "
        f"{MAX_ARMS} arms in total, e.g. --arm 'ddqn soft=outputs/ddqn-soft'. "
        "A bare path labels itself from its directory name. Each extra arm is "
        "tested against the first arm",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        default=None,
        help=f'rename the arms, e.g. --labels "vanilla DQN" "Double DQN" '
        f"(default: {HARD_LABEL!r} and {SOFT_LABEL!r})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(RESULTS_DIR) / COMPARISON_NAME,
        help=f"where to save the comparison PNG (default: {RESULTS_DIR}/{COMPARISON_NAME})",
    )
    parser.add_argument("--window", type=int, default=20, help="episodes in the trailing mean")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SOLVED_THRESHOLD,
        help="trailing-mean return counted as 'good' when reporting how fast each arm got there",
    )
    parser.add_argument("--title", default=None, help="override the plot title")
    return parser


def _plus_minus(mean: float, stderr: float) -> str:
    return f"{mean:.1f}" if stderr == 0.0 else f"{mean:.1f} ± {stderr:.1f}"


def _label_width(summaries: Sequence[RunSummary]) -> int:
    """Widen the arm column to fit the longest label, never narrower than the original."""
    return max(LABEL_COLUMN, *(len(summary.label) for summary in summaries))


def format_summary(summary: RunSummary, width: int = LABEL_COLUMN) -> str:
    steps = "n/a" if summary.total_steps is None else f"{summary.total_steps:,}"
    if summary.episodes_to_threshold is None:
        reached = f"not reached (0/{summary.n_seeds} seeds)"
    else:
        where = (
            f"episode {summary.episodes_to_threshold}"
            if summary.steps_to_threshold is None
            else f"step {summary.steps_to_threshold:,}"
        )
        reached = f"{where} ({summary.seeds_reaching}/{summary.n_seeds} seeds)"
    return (
        f"{summary.label:<{width}} | {summary.n_seeds:5d} | {summary.episodes:8d} | {steps:>11} | "
        f"{_plus_minus(summary.final_mean, summary.final_stderr):>16} | "
        f"{_plus_minus(summary.best_mean, summary.best_stderr):>16} | {reached}"
    )


def format_ttest(ttest: TTest | None, summaries: Sequence[RunSummary]) -> str:
    """One line on whether two arms are actually distinguishable."""
    if ttest is None:
        return (
            "Significance: not testable — it needs at least 2 seeds per arm.\n"
            "  Re-run both arms with e.g. --seeds 0 1 2 3 4 to get an error bar on this comparison."
        )
    direction = f"{summaries[1].label} - {summaries[0].label}"
    if not np.isfinite(ttest.t):
        return f"Welch's t-test on final trailing mean ({direction}): no variance across seeds; nothing to test."
    verdict = "distinguishable at p < 0.05" if ttest.significant else "NOT distinguishable at p < 0.05"
    return (
        f"Welch's t-test on final trailing mean ({direction}): "
        f"{ttest.difference:+.1f} ± {ttest.stderr:.1f}, "
        f"t = {ttest.t:.2f}, df = {ttest.df:.1f}, p = {ttest.p:.4f} — {verdict}."
    )


def format_ttests(result: CompareResult) -> str:
    """Every arm against the first, one line each.

    Only the baseline pairings are printed. Testing all six pairs of a four-arm cross
    would spend most of its power on multiple comparisons without anyone having declared
    which contrast the experiment was actually about.
    """
    baseline = result.summaries[0]
    lines = [
        format_ttest(ttest, (baseline, summary))
        for ttest, summary in zip(result.ttests, result.summaries[1:])
    ]
    if len(lines) > 1:
        lines.insert(0, f"Each arm against {baseline.label}:")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        extra_arms = [parse_arm_spec(spec) for spec in (args.arms or [])]
        result = compare(
            args.hard, args.soft, args.output, args.window, args.threshold, args.title,
            args.labels, extra_arms,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print(
            "\nRun both arms first, for example:\n"
            "  python -m dqn.train --target-update-mode hard             --seeds 0 1 2 3 4 --output-dir outputs/hard\n"
            "  python -m dqn.train --target-update-mode soft --tau 0.005 --seeds 0 1 2 3 4 --output-dir outputs/soft",
            file=sys.stderr,
        )
        return 1

    width = _label_width(result.summaries)
    print(
        f"{'arm':<{width}} | {'seeds':>5} | {'episodes':>8} | {'env steps':>11} | "
        f"{f'final mean{args.window}':>16} | {'best mean':>16} | first trailing mean >= {args.threshold:g}"
    )
    for summary in result.summaries:
        print(format_summary(summary, width))
    print()
    print(format_ttests(result))
    print(f"\nComparison plot: {result.plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
