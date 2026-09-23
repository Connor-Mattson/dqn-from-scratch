"""Measure Q-value overestimation during training, and compare it across arms.

    python -m dqn.train --overestimation-every 2500 --seeds 0 1 2 3 4 --output-dir outputs/oe-vanilla
    python -m dqn.train --overestimation-every 2500 --double --seeds 0 1 2 3 4 --output-dir outputs/oe-double
    python -m dqn.overestimation --a outputs/oe-vanilla --b outputs/oe-double

Two numbers answer two different questions:

**Max-operator gap** — ``max_a' Q_target(s', a') - Q_target(s', argmax_a' Q_online(s', a'))``,
averaged over the non-terminal rows of every training minibatch. It is the vanilla bootstrap
value minus the Double DQN one on the *same* networks and states, so it is never negative
and is exactly how much Double DQN lowers the target at that moment. It says how much the
two networks disagree about the best next action and what that disagreement is worth. It
does *not* say whether either target is too high: a large gap could mean vanilla is
overestimating or Double DQN is underestimating.

**Overestimation bias** — ``max_a Q_online(s_t, a)`` minus the discounted return the greedy
policy actually collects from ``s_t``, averaged over the states of a few greedy episodes.
This is the measurement in van Hasselt et al. (2016), Figure 3. It compares against the
ground truth, so a positive value really is overestimation, and it is the number to use
when asking whether Double DQN helped.

Both are computed without touching any RNG the training loop uses (forward passes only, on a
separate evaluation env), so switching diagnostics on does not change the training run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dqn.artifacts import RESULTS_DIR, seed_from_name
from dqn.env import make_env
from dqn.plotting import SERIES_PALETTE, BandCurve, save_band_comparison_curve
from dqn.replay_buffer import Batch
from dqn.stats import aggregate_seeds, mean_stderr, welch_ttest

OVERESTIMATION_NAME = "overestimation.csv"
SEED_OVERESTIMATION_GLOB = "overestimation_seed*.csv"
BIAS_PLOT_NAME = "overestimation_bias.png"
GAP_PLOT_NAME = "overestimation_gap.png"
# Evaluation episodes use their own seeds, far from the training seeds, and the same seeds
# at every checkpoint so successive checkpoints are scored from the same start states.
EVAL_SEED_OFFSET = 20_000
# A truncated episode's return-to-go is missing its tail. A state is only scored when the
# discount weight of that missing tail, gamma ** (steps remaining), is at most this much —
# for gamma = 0.99 that is the first T - 299 states of a truncated episode. Episodes that
# really terminate have no tail and every state counts.
TAIL_TOLERANCE = 0.05
CSV_FIELDS = (
    "step",
    "gradient_steps",
    "target_gap",
    "argmax_disagreement",
    "q_predicted",
    "return_realised",
    "bias",
    "eval_return",
    "states_scored",
)


def max_operator_gap(
    online_network: nn.Module,
    target_network: nn.Module,
    next_states: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[float, float, int]:
    """Vanilla minus Double DQN bootstrap value on one minibatch.

    Returns ``(gap_sum, disagreements, rows)`` over the non-terminal rows — sums rather than
    means, so a caller can pool many batches into one exact mean.
    """
    with torch.no_grad():
        live = ~dones.bool()
        q_target = target_network(next_states[live])
        greedy_online = online_network(next_states[live]).argmax(dim=1, keepdim=True)
        vanilla = q_target.max(dim=1)
        double = q_target.gather(dim=1, index=greedy_online).squeeze(1)
        gap = vanilla.values - double
        disagree = vanilla.indices != greedy_online.squeeze(1)
        return float(gap.sum()), float(disagree.sum()), int(live.sum())


def discounted_returns_to_go(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """``G_t = r_t + gamma * G_{t+1}`` for every step of one episode."""
    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


@dataclass
class RolloutScore:
    q_predicted: float  # mean max_a Q(s_t, a) over the scored states
    return_realised: float  # mean discounted return-to-go over the same states
    eval_return: float  # mean undiscounted episode return, the familiar score
    states_scored: int

    @property
    def bias(self) -> float:
        return self.q_predicted - self.return_realised


def score_greedy_rollouts(
    online_network: nn.Module,
    env_id: str,
    gamma: float,
    episodes: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> RolloutScore:
    """Play ``episodes`` greedy episodes and set the predicted Q against the realised return."""
    env = make_env(env_id)
    predicted, realised, episode_returns = [], [], []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            q_values, rewards = [], []
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad():
                    q = online_network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                best = q.max(dim=1)
                q_values.append(float(best.values))
                state, reward, terminated, truncated, _ = env.step(int(best.indices))
                rewards.append(float(reward))

            returns = discounted_returns_to_go(rewards, gamma)
            steps_left = len(rewards) - np.arange(len(rewards))
            keep = np.ones(len(rewards), dtype=bool) if terminated else gamma**steps_left <= TAIL_TOLERANCE
            predicted.extend(np.asarray(q_values)[keep])
            realised.extend(returns[keep])
            episode_returns.append(sum(rewards))
    finally:
        env.close()

    if not predicted:
        nan = float("nan")
        return RolloutScore(nan, nan, float(np.mean(episode_returns)), 0)
    return RolloutScore(float(np.mean(predicted)), float(np.mean(realised)), float(np.mean(episode_returns)), len(predicted))


@dataclass
class OverestimationLog:
    """Pools the max-operator gap between checkpoints and scores rollouts at each one."""

    env_id: str
    gamma: float
    episodes: int
    seed: int
    device: torch.device | str = "cpu"
    rows: list[dict[str, float]] = field(default_factory=list)
    _gap_sum: float = 0.0
    _disagreements: float = 0.0
    _rows_seen: int = 0

    def record_batch(self, online_network: nn.Module, target_network: nn.Module, batch: Batch) -> None:
        gap_sum, disagreements, rows = max_operator_gap(online_network, target_network, batch.next_states, batch.dones)
        self._gap_sum += gap_sum
        self._disagreements += disagreements
        self._rows_seen += rows

    def checkpoint(self, step: int, gradient_steps: int, online_network: nn.Module) -> dict[str, float]:
        score = score_greedy_rollouts(
            online_network, self.env_id, self.gamma, self.episodes, self.seed + EVAL_SEED_OFFSET, self.device
        )
        seen = self._rows_seen
        row = {
            "step": step,
            "gradient_steps": gradient_steps,
            "target_gap": self._gap_sum / seen if seen else float("nan"),
            "argmax_disagreement": self._disagreements / seen if seen else float("nan"),
            "q_predicted": score.q_predicted,
            "return_realised": score.return_realised,
            "bias": score.bias,
            "eval_return": score.eval_return,
            "states_scored": score.states_scored,
        }
        self.rows.append(row)
        self._gap_sum = self._disagreements = 0.0
        self._rows_seen = 0
        return row


def save_overestimation_csv(path: str | Path, rows: Sequence[dict[str, float]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:g}" if isinstance(value, float) else value) for key, value in row.items()})
    return path


def load_overestimation_csv(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(row[key]) for row in rows], dtype=np.float64) for key in CSV_FIELDS}


def discover_overestimation_csvs(path: str | Path) -> list[tuple[int | None, Path]]:
    path = Path(path)
    if path.is_file():
        return [(seed_from_name(path), path)]
    seeded = sorted(path.glob(SEED_OVERESTIMATION_GLOB), key=lambda p: seed_from_name(p) or 0)
    if seeded:
        return [(seed_from_name(p), p) for p in seeded]
    if (path / OVERESTIMATION_NAME).is_file():
        return [(None, path / OVERESTIMATION_NAME)]
    raise FileNotFoundError(
        f"No overestimation logs in {path}. Train with diagnostics on first, e.g.\n"
        f"  python -m dqn.train --overestimation-every 2500 --seeds 0 1 2 3 4 --output-dir {path}"
    )


@dataclass
class ArmOverestimation:
    label: str
    runs: list[dict[str, np.ndarray]]
    # Per seed, averaged over the second half of its checkpoints: early checkpoints score a
    # near-random policy whose short episodes say little about the learned values.
    late_bias: list[float]
    late_relative: list[float]
    late_gap: list[float]
    late_disagreement: list[float]


def summarise_arm(label: str, path: str | Path) -> ArmOverestimation:
    runs = [load_overestimation_csv(csv_path) for _, csv_path in discover_overestimation_csvs(path)]
    late_bias, late_relative, late_gap, late_disagreement = [], [], [], []
    for run in runs:
        late = slice(len(run["step"]) // 2, None)
        late_bias.append(float(np.nanmean(run["bias"][late])))
        late_relative.append(float(np.nanmean(run["bias"][late] / run["return_realised"][late])))
        late_gap.append(float(np.nanmean(run["target_gap"][late])))
        late_disagreement.append(float(np.nanmean(run["argmax_disagreement"][late])))
    return ArmOverestimation(label, runs, late_bias, late_relative, late_gap, late_disagreement)


def _band(arm: ArmOverestimation, column: str, color: str) -> BandCurve | None:
    curves = []
    for run in arm.runs:
        ok = np.isfinite(run[column])
        curves.append((run["step"][ok], run[column][ok]))
    band = aggregate_seeds(curves)
    return None if band is None else BandCurve(f"{arm.label} ({len(arm.runs)} seeds)", band, color)


def plot_arms(arms: Sequence[ArmOverestimation], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    charts = [
        ("bias", BIAS_PLOT_NAME, "Overestimation: greedy max Q minus realised discounted return",
         "Predicted − realised value"),
        ("target_gap", GAP_PLOT_NAME, "Max-operator gap: vanilla minus Double DQN bootstrap value",
         "max Q_target − Q_target(argmax Q_online)"),
    ]
    paths = []
    for column, name, title, ylabel in charts:
        bands = [_band(arm, column, color) for arm, color in zip(arms, SERIES_PALETTE)]
        bands = [band for band in bands if band is not None]
        if bands:
            paths.append(
                save_band_comparison_curve(
                    bands, output_dir / name, title=title, legend_title="Mean across seeds, shaded ±1 s.e.",
                    max_return=None, ylabel=ylabel, value_format=".2f",
                )
            )
    return paths


def _pm(values: Sequence[float], fmt: str = ".2f") -> str:
    mean, stderr = mean_stderr(values)
    return f"{mean:{fmt}} ± {stderr:{fmt}}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Q-value overestimation across training arms.")
    parser.add_argument("--a", type=Path, default=Path("outputs") / "oe-vanilla", help="first arm (baseline)")
    parser.add_argument("--b", type=Path, default=Path("outputs") / "oe-double", help="second arm")
    parser.add_argument("--labels", nargs=2, default=("vanilla DQN", "Double DQN"), metavar=("FIRST", "SECOND"))
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR))
    args = parser.parse_args(argv)

    try:
        arms = [summarise_arm(label, path) for label, path in zip(args.labels, (args.a, args.b))]
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    width = max(14, *(len(arm.label) for arm in arms))
    print("Second half of training, mean ± s.e. across seeds:")
    print(f"{'arm':<{width}} | seeds | {'bias (Q - G)':>16} | {'bias / G':>16} | {'max-op gap':>14} | argmax disagree")
    for arm in arms:
        print(
            f"{arm.label:<{width}} | {len(arm.runs):5d} | {_pm(arm.late_bias):>16} | "
            f"{_pm([100 * r for r in arm.late_relative], '.1f') + ' %':>16} | "
            f"{_pm(arm.late_gap, '.3f'):>14} | {_pm([100 * d for d in arm.late_disagreement], '.1f')} %"
        )

    ttest = welch_ttest(arms[0].late_bias, arms[1].late_bias)
    print()
    if ttest is None:
        print("Significance: not testable — needs at least 2 seeds per arm.")
    else:
        verdict = "distinguishable" if ttest.significant else "NOT distinguishable"
        print(
            f"Welch's t-test on bias ({arms[1].label} - {arms[0].label}): {ttest.difference:+.2f} ± {ttest.stderr:.2f}, "
            f"t = {ttest.t:.2f}, df = {ttest.df:.1f}, p = {ttest.p:.4f} — {verdict} at p < 0.05."
        )
    for path in plot_arms(arms, args.output_dir):
        print(f"Plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
