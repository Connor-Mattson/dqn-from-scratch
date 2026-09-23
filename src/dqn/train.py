"""Train a DQN agent on a Gymnasium env (CartPole-v1 by default).

    python -m dqn.train [--env ENV_ID] [--total-steps N] [--seed S] [--output-dir DIR] ...

Provided plumbing: the environment loop, logging, and checkpoint/plot saving. The DQN
mechanics it calls (``ReplayBuffer``, ``select_action``, ``training_step``,
``should_sync_target``, ``sync_target_network``) live in ``dqn.replay_buffer``,
``dqn.agent`` and ``dqn.learning``.

``--double`` switches the bootstrap target to Double DQN; everything else about the run
is unchanged, which is what makes a vanilla arm and a double arm comparable.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from dqn.agent import select_action, sync_target_network, update_target_network
from dqn.artifacts import (
    CHECKPOINT_NAME,
    CURVE_NAME,
    RETURNS_NAME,
    save_checkpoint,
    save_returns_csv,
    seeded_name,
)
from dqn.config import TARGET_UPDATE_MODES, DQNConfig, resolve_device
from dqn.env import MAX_RETURNS, env_dimensions, make_env
from dqn.learning import training_step
from dqn.network import QNetwork
from dqn.overestimation import OVERESTIMATION_NAME, OverestimationLog, save_overestimation_csv
from dqn.plotting import save_reward_curve
from dqn.replay_buffer import ReplayBuffer
from dqn.schedule import linear_epsilon
from dqn.stats import mean_stderr

RECENT_EPISODES = 20


@dataclass
class TrainResult:
    episode_returns: list[float]
    episode_end_steps: list[int]  # environment step each episode finished on
    checkpoint_path: Path
    curve_path: Path
    returns_path: Path
    seed: int = 0
    overestimation_path: Path | None = None


def train(config: DQNConfig, artifact_seed: int | None = None) -> TrainResult:
    config.validate()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)

    env = make_env(config.env_id)
    obs_dim, num_actions = env_dimensions(env)

    online_network = QNetwork(obs_dim, num_actions, config.hidden_sizes).to(device)
    target_network = QNetwork(obs_dim, num_actions, config.hidden_sizes).to(device)
    sync_target_network(online_network, target_network)  # both networks start from the same weights
    optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_capacity, seed=config.seed)

    if config.target_update_mode == "soft":
        update_text = f"soft target updates, tau={config.tau:g}"
    else:
        update_text = f"hard target syncs every {config.target_update_interval:,} gradient steps"
    target_text = "Double DQN targets" if config.double_dqn else "vanilla DQN targets"
    print(
        f"Training on {config.env_id} for {config.total_steps:,} steps "
        f"(device={device}, {update_text}, {target_text})"
    )
    episode_returns: list[float] = []
    episode_end_steps: list[int] = []
    episode_return = 0.0
    gradient_steps = 0
    last_loss: float | None = None
    start_time = time.perf_counter()
    overestimation = (
        OverestimationLog(config.env_id, config.gamma, config.overestimation_episodes, config.seed, device)
        if config.overestimation_every
        else None
    )

    state, _ = env.reset(seed=config.seed)
    for step in range(1, config.total_steps + 1):
        epsilon = linear_epsilon(step, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
        action = select_action(online_network, state, epsilon, num_actions, rng, device)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Store `terminated`, not `truncated`: hitting CartPole's 500-step time limit is not a
        # true terminal state, so that transition should still count as having future value.
        buffer.push(state, action, float(reward), next_state, bool(terminated))
        episode_return += float(reward)
        state = next_state

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_end_steps.append(step)
            if len(episode_returns) % config.log_every_episodes == 0:
                recent_mean = float(np.mean(episode_returns[-RECENT_EPISODES:]))
                loss_text = "   n/a" if last_loss is None else f"{last_loss:.4f}"
                print(
                    f"episode {len(episode_returns):5d} | step {step:7d} | return {episode_return:6.1f} | "
                    f"mean(last {RECENT_EPISODES}) {recent_mean:6.1f} | epsilon {epsilon:.3f} | "
                    f"loss {loss_text} | {time.perf_counter() - start_time:6.1f}s"
                )
            episode_return = 0.0
            state, _ = env.reset()

        if step >= config.learning_starts and step % config.train_every == 0:
            batch = buffer.sample(config.batch_size).to(device)
            last_loss = training_step(
                online_network,
                target_network,
                optimizer,
                batch,
                config.gamma,
                config.max_grad_norm,
                config.double_dqn,
            )
            gradient_steps += 1
            update_target_network(
                online_network,
                target_network,
                gradient_steps,
                config.target_update_mode,
                config.target_update_interval,
                config.tau,
            )
            if overestimation is not None:
                overestimation.record_batch(online_network, target_network, batch)

        if overestimation is not None and gradient_steps and step % config.overestimation_every == 0:
            row = overestimation.checkpoint(step, gradient_steps, online_network)
            print(
                f"  overestimation @ step {step:7d} | Q {row['q_predicted']:7.2f} vs G {row['return_realised']:7.2f} "
                f"(bias {row['bias']:+7.2f}) | max-op gap {row['target_gap']:.3f} | "
                f"argmax disagree {100 * row['argmax_disagreement']:4.1f}%"
            )

    env.close()

    # With `artifact_seed` set, several seeds share one output directory and are told apart
    # by filename; without it the single-run names are unchanged.
    def artifact(name: str) -> Path:
        return output_dir / (name if artifact_seed is None else seeded_name(name, artifact_seed))

    checkpoint_path = save_checkpoint(artifact(CHECKPOINT_NAME), online_network, config, episode_returns)
    curve_path = save_reward_curve(
        episode_returns,
        artifact(CURVE_NAME),
        window=RECENT_EPISODES,
        title=f"DQN on {config.env_id}: episode return",
        max_return=MAX_RETURNS.get(config.env_id),
    )
    returns_path = save_returns_csv(artifact(RETURNS_NAME), episode_returns, episode_end_steps)
    overestimation_path = (
        save_overestimation_csv(artifact(OVERESTIMATION_NAME), overestimation.rows) if overestimation else None
    )

    print(f"\nFinished {len(episode_returns)} episodes and {gradient_steps:,} gradient steps.")
    if episode_returns:
        print(f"Mean return over the last {RECENT_EPISODES} episodes: {np.mean(episode_returns[-RECENT_EPISODES:]):.1f}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Reward curve: {curve_path}")
    print(f"Returns CSV:  {returns_path}")
    if overestimation_path is not None:
        print(f"Overestimation CSV: {overestimation_path}")
    return TrainResult(
        episode_returns, episode_end_steps, checkpoint_path, curve_path, returns_path, config.seed,
        overestimation_path,
    )


def train_seeds(config: DQNConfig, seeds: Sequence[int]) -> list[TrainResult]:
    """Train one run per seed into the same output directory.

    Every run is identical apart from its seed, so the spread across the returned results
    *is* the run-to-run noise that a single-seed comparison has no way to see. Artifacts
    are suffixed per seed, which is what lets `dqn.compare` pick the whole arm up from one
    directory.
    """
    if not seeds:
        raise ValueError("train_seeds needs at least one seed")
    results = []
    for index, seed in enumerate(seeds, start=1):
        print(f"\n=== seed {seed} ({index}/{len(seeds)}) ===")
        results.append(train(replace(config, seed=seed), artifact_seed=seed))
    return results


def summarise_seeds(results: Sequence[TrainResult], window: int = RECENT_EPISODES) -> None:
    """Print the across-seed mean and standard error of the final trailing return."""
    finals = [float(np.mean(r.episode_returns[-window:])) for r in results if r.episode_returns]
    if not finals:
        print("\nNo completed episodes in any seed.")
        return
    mean, stderr = mean_stderr(finals)
    print(f"\n=== {len(results)} seeds ===")
    for result in results:
        final = float(np.mean(result.episode_returns[-window:])) if result.episode_returns else float("nan")
        print(f"seed {result.seed:4d} | {len(result.episode_returns):5d} episodes | final mean{window} {final:6.1f}")
    print(f"across seeds: {mean:.1f} ± {stderr:.1f} (mean ± s.e. of final mean{window})")


def build_parser() -> argparse.ArgumentParser:
    defaults = DQNConfig()
    parser = argparse.ArgumentParser(description="Train a DQN agent on a Gymnasium environment.")
    parser.add_argument(
        "--env",
        default=defaults.env_id,
        help="Gymnasium env id with a flat Box observation and Discrete actions (e.g. LunarLander-v3)",
    )
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps, help="environment steps")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="train one run per seed into --output-dir (e.g. --seeds 0 1 2 3 4). Each run's "
        "artifacts get a _seed{N} suffix so the whole arm shares one directory. Overrides --seed.",
    )
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    parser.add_argument(
        "--target-update-mode",
        choices=TARGET_UPDATE_MODES,
        default=defaults.target_update_mode,
        help='"hard" copies the online weights on an interval; "soft" blends tau of them in every step',
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=defaults.target_update_interval,
        help="gradient steps between hard target-network syncs (hard mode only)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=defaults.tau,
        help="Polyak coefficient in (0, 1] for soft updates (soft mode only)",
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="use Double DQN bootstrap targets: the online network picks the next action, "
        "the target network values it (default: vanilla DQN)",
    )
    parser.add_argument(
        "--overestimation-every",
        type=int,
        default=defaults.overestimation_every,
        metavar="STEPS",
        help="every STEPS env steps, log predicted Q against realised greedy return and the "
        "vanilla-minus-double target gap (0 = off; see dqn.overestimation)",
    )
    parser.add_argument(
        "--overestimation-episodes",
        type=int,
        default=defaults.overestimation_episodes,
        help="greedy episodes played at each overestimation checkpoint",
    )
    parser.add_argument("--log-every-episodes", type=int, default=defaults.log_every_episodes)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--device", default=defaults.device, help='"auto", "cpu", "cuda", ...')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DQNConfig(
        env_id=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_mode=args.target_update_mode,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        double_dqn=args.double,
        overestimation_every=args.overestimation_every,
        overestimation_episodes=args.overestimation_episodes,
        log_every_episodes=args.log_every_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )
    try:
        if args.seeds:
            summarise_seeds(train_seeds(config, args.seeds))
            print("\nCompare the two arms with: python -m dqn.compare")
        else:
            train(config)
    except NotImplementedError as exc:
        if "TODO(learner)" not in str(exc):
            raise
        print(f"\nStopped at an unfinished learner task: {exc}", file=sys.stderr)
        print("Run `pytest -m learner` to see which tasks still fail.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
