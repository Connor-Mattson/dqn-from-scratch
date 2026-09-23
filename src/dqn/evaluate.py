"""Evaluate a trained DQN checkpoint with a greedy policy.

    python -m dqn.evaluate --checkpoint outputs/dqn.pt [--episodes N] [--render]

Prints the greedy policy's average return next to a uniformly random policy's baseline, and
saves a GIF replay of the median-scoring episode.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium.error import DependencyNotInstalled

from dqn.agent import select_action
from dqn.artifacts import CHECKPOINT_NAME, MEDIAN_GIF_NAME, RESULTS_DIR, load_checkpoint, save_episode_gif
from dqn.config import resolve_device
from dqn.env import env_dimensions, make_env


@dataclass
class EvalResult:
    env_id: str
    greedy_returns: list[float]
    random_returns: list[float] | None
    median_episode: int | None = None  # 1-based index into greedy_returns
    gif_path: Path | None = None


def run_episodes(
    env_id: str,
    policy: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
    render_mode: str | None = None,
) -> list[float]:
    """Run ``policy`` for ``episodes`` full episodes and return each episode's total reward."""
    env = make_env(env_id, render_mode=render_mode)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total, done = 0.0, False
            while not done:
                state, reward, terminated, truncated, _ = env.step(policy(state))
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return returns


def median_episode_index(returns: Sequence[float]) -> int:
    """Index of the episode holding the lower median return (a real episode, not an average)."""
    order = np.argsort(np.asarray(returns, dtype=float), kind="stable")
    return int(order[(len(order) - 1) // 2])


def record_episode(
    env_id: str,
    policy: Callable[[np.ndarray], int],
    seed: int,
) -> tuple[list[np.ndarray], int]:
    """Replay one episode with ``rgb_array`` rendering. Returns ``(frames, render_fps)``."""
    env = make_env(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        state, _ = env.reset(seed=seed)
        frames.append(env.render())
        done = False
        while not done:
            state, _, terminated, truncated, _ = env.step(policy(state))
            frames.append(env.render())
            done = terminated or truncated
        fps = int(env.metadata.get("render_fps", 30))
    finally:
        env.close()
    return frames, fps


def evaluate(
    checkpoint_path: str | Path,
    episodes: int = 20,
    seed: int = 10_000,
    render: bool = False,
    device: torch.device | str = "cpu",
    random_baseline: bool = True,
    gif_path: str | Path | None = None,
    gif_scale: float = 0.5,
) -> EvalResult:
    device = resolve_device(device)
    network, metadata = load_checkpoint(checkpoint_path, device)
    network.eval()
    env_id = metadata["env_id"]
    rng = np.random.default_rng(seed)

    def greedy_policy(state: np.ndarray) -> int:
        return select_action(network, state, 0.0, network.num_actions, rng, device)

    greedy_returns = run_episodes(env_id, greedy_policy, episodes, seed, "human" if render else None)

    median_episode, saved_gif = None, None
    if greedy_returns:
        index = median_episode_index(greedy_returns)
        median_episode = index + 1
        if gif_path is not None:
            # epsilon=0 makes the policy a pure argmax, so replaying episode `index` from its
            # own reset seed reproduces the run we just scored.
            try:
                frames, fps = record_episode(env_id, greedy_policy, seed + index)
            except (ImportError, DependencyNotInstalled) as exc:
                warnings.warn(
                    f"Skipping the median-episode GIF ({exc}). "
                    "Install the render extra with: pip install -e '.[render]'",
                    stacklevel=2,
                )
            else:
                saved_gif = save_episode_gif(frames, gif_path, fps=fps, scale=gif_scale)

    random_returns = None
    if random_baseline:
        probe = make_env(env_id)
        _, num_actions = env_dimensions(probe)
        probe.close()
        random_returns = run_episodes(env_id, lambda _: int(rng.integers(num_actions)), episodes, seed)
    return EvalResult(env_id, greedy_returns, random_returns, median_episode, saved_gif)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint greedily.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs") / CHECKPOINT_NAME)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000, help="episode i is reset with seed + i")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true", help="show a window (needs pip install -e '.[render]')")
    parser.add_argument("--no-random-baseline", action="store_true", help="skip the random-policy comparison")
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help=f"where to save the median episode's GIF (default: {RESULTS_DIR}/{MEDIAN_GIF_NAME})",
    )
    parser.add_argument("--no-gif", action="store_true", help="skip the median-episode GIF")
    parser.add_argument("--gif-scale", type=float, default=0.5, help="scale applied to GIF frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gif_path = None if args.no_gif else (args.gif or Path(RESULTS_DIR) / MEDIAN_GIF_NAME)
    try:
        result = evaluate(
            args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            device=args.device,
            random_baseline=not args.no_random_baseline,
            gif_path=gif_path,
            gif_scale=args.gif_scale,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        if "TODO(learner)" not in str(exc):
            raise
        print(f"Stopped at an unfinished learner task: {exc}", file=sys.stderr)
        return 1

    for episode, value in enumerate(result.greedy_returns, start=1):
        print(f"episode {episode:3d} | return {value:6.1f}")
    greedy = np.asarray(result.greedy_returns)
    print(f"\n{result.env_id} greedy policy: mean return {greedy.mean():.1f} ± {greedy.std():.1f} over {len(greedy)} episodes")
    if result.random_returns is not None:
        baseline = np.asarray(result.random_returns)
        print(f"{result.env_id} random policy: mean return {baseline.mean():.1f} ± {baseline.std():.1f} (baseline)")
    if result.gif_path is not None:
        median = result.greedy_returns[result.median_episode - 1]
        print(f"median episode {result.median_episode} (return {median:.1f}) saved to {result.gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
