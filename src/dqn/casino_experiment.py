"""Vanilla vs Double DQN on the maximization-bias casino, where the true Q-values are known.

    python -m dqn.casino_experiment                          # 10 seeds per arm, plot + GIF
    python -m dqn.casino_experiment --seeds 0 1 2 --no-gif   # quick look
    python -m dqn.casino_experiment --sweep                  # bias vs number of machines and sigma

Because ``Q*`` is known in closed form (see ``dqn.casino_env``), no rollouts are needed to
measure overestimation: every checkpoint reads the networks' values in both rooms and
subtracts the truth. Three numbers per checkpoint:

- **bootstrap bias** — the value the arm's own target rule assigns to the casino floor,
  minus the true -0.1. Vanilla uses ``max_a Q_target(floor, a)``; Double DQN uses
  ``Q_target(floor, argmax_a Q_online(floor, a))``. This is the number the doors inherit.
- **door bias** — ``max over doors of Q_online(lobby, door)`` minus ``gamma * bet_mean``.
- **gambles** — 1 when the greedy lobby action is a door rather than the exit. Averaged over
  seeds, it is the probability that the trained agent walks into a losing casino.

The loop reuses the learner-built pieces (``ReplayBuffer``, ``select_action``,
``training_step``, ``update_target_network``); only the diagnostics are casino-specific.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from dqn.agent import select_action, sync_target_network, update_target_network
from dqn.artifacts import RESULTS_DIR, save_episode_gif
from dqn.casino_env import EXIT, FLOOR, LOBBY, MaxBiasCasinoEnv
from dqn.config import resolve_device
from dqn.learning import training_step
from dqn.network import QNetwork
from dqn.plotting import BASELINE, GRIDLINE, INK_MUTED, INK_SECONDARY, SERIES_PALETTE, SURFACE
from dqn.replay_buffer import ReplayBuffer
from dqn.stats import aggregate_seeds, mean_stderr

OUTPUT_DIR = Path(RESULTS_DIR) / "casino"
ARMS = (("Vanilla DQN", False), ("Double DQN", True))


@dataclass
class CasinoConfig:
    n_actions: int = 20
    sigma: float = 2.0
    bet_mean: float = -0.1
    total_steps: int = 10_000
    learning_starts: int = 500
    buffer_capacity: int = 20_000  # holds the whole run: the replay noise is then shared by both networks
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 10.0
    # One hidden layer. With two one-hot inputs, a deeper ReLU net at this learning rate can
    # lose every unit in a layer and answer both rooms with the same vector; see `collapsed`.
    hidden_sizes: tuple[int, ...] = (64,)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 4_000
    target_update_mode: str = "hard"
    # Long enough that online and target errors partly decorrelate, which is what Double DQN relies on.
    target_update_interval: int = 2_000
    tau: float = 0.005
    checkpoint_every: int = 250
    device: str = "cpu"


@dataclass
class RunLog:
    steps: list[int] = field(default_factory=list)
    bootstrap_bias: list[float] = field(default_factory=list)
    door_bias: list[float] = field(default_factory=list)
    gambles: list[float] = field(default_factory=list)
    walkout_rate: list[float] = field(default_factory=list)  # fraction of recent episodes that walked out
    q_floor: list[np.ndarray] = field(default_factory=list)
    q_lobby: list[np.ndarray] = field(default_factory=list)
    # True when some hidden layer has no active ReLU unit for one of the rooms. The network can
    # then no longer tell the rooms apart, and its "bias" is an artifact, not the max operator.
    collapsed: bool = False


def room_obs(room: int, device: torch.device | str = "cpu") -> torch.Tensor:
    obs = torch.zeros(1, 2, device=device)
    obs[0, room] = 1.0
    return obs


def read_values(online: QNetwork, target: QNetwork, double: bool, device) -> dict[str, np.ndarray | float]:
    """Q-values in both rooms plus the floor value this arm's target rule would bootstrap from."""
    with torch.no_grad():
        q_lobby = online(room_obs(LOBBY, device))[0].cpu().numpy()
        q_floor = online(room_obs(FLOOR, device))[0].cpu().numpy()
        q_floor_target = target(room_obs(FLOOR, device))[0].cpu().numpy()
    bootstrap = q_floor_target[int(q_floor.argmax())] if double else q_floor_target.max()
    return {"q_lobby": q_lobby, "q_floor": q_floor, "bootstrap": float(bootstrap)}


def has_collapsed(network: QNetwork, device: torch.device | str = "cpu") -> bool:
    """Whether any hidden ReLU layer is fully dead for the lobby or the floor observation."""
    with torch.no_grad():
        h = torch.eye(2, device=device)
        for layer in network.net[:-1]:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU) and bool(((h > 0).sum(dim=1) == 0).any()):
                return True
    return False


def train_casino(config: CasinoConfig, seed: int, double: bool, verbose: bool = False) -> RunLog:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(config.device)
    env = MaxBiasCasinoEnv(config.n_actions, config.sigma, config.bet_mean)

    online = QNetwork(2, config.n_actions, config.hidden_sizes).to(device)
    target = QNetwork(2, config.n_actions, config.hidden_sizes).to(device)
    sync_target_network(online, target)
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_capacity, seed=seed)

    log = RunLog()
    walked_out: list[bool] = []
    gradient_steps = 0
    state, _ = env.reset(seed=seed)
    for step in range(1, config.total_steps + 1):
        fraction = min(1.0, step / config.epsilon_decay_steps)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
        action = select_action(online, state, epsilon, config.n_actions, rng, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.push(state, action, float(reward), next_state, bool(terminated))
        state = next_state
        if terminated or truncated:
            walked_out.append(info["walked_out"])
            state, _ = env.reset()

        if step >= config.learning_starts:
            batch = buffer.sample(config.batch_size).to(device)
            training_step(online, target, optimizer, batch, config.gamma, config.max_grad_norm, double)
            gradient_steps += 1
            update_target_network(
                online, target, gradient_steps, config.target_update_mode, config.target_update_interval, config.tau
            )

        if step % config.checkpoint_every == 0:
            values = read_values(online, target, double, device)
            q_lobby, q_floor = values["q_lobby"], values["q_floor"]
            log.steps.append(step)
            log.bootstrap_bias.append(values["bootstrap"] - config.bet_mean)
            log.door_bias.append(float(q_lobby[1:].max()) - config.gamma * config.bet_mean)
            log.gambles.append(float(int(q_lobby.argmax()) != EXIT))
            recent = walked_out[-200:]
            log.walkout_rate.append(float(np.mean(recent)) if recent else float("nan"))
            log.q_lobby.append(q_lobby)
            log.q_floor.append(q_floor)
            if verbose:
                print(
                    f"  step {step:6d} | bootstrap bias {log.bootstrap_bias[-1]:+.3f} | "
                    f"door bias {log.door_bias[-1]:+.3f} | greedy {'GAMBLES' if log.gambles[-1] else 'walks out'}"
                )
    env.close()
    log.collapsed = has_collapsed(online, device)
    return log


@dataclass
class ArmResult:
    label: str
    double: bool
    runs: list[RunLog]

    def late(self, column: str) -> list[float]:
        """Per-seed mean over the second half of the checkpoints."""
        return [float(np.nanmean(getattr(run, column)[len(run.steps) // 2 :])) for run in self.runs]


def _train_job(job: tuple[CasinoConfig, int, bool]) -> RunLog:
    torch.set_num_threads(1)  # many tiny runs side by side beat one run using every core
    return train_casino(*job)


def run_arms(config: CasinoConfig, seeds: Sequence[int], workers: int | None = None) -> list[ArmResult]:
    """Train every (arm, seed) pair, in parallel processes. Each run is seeded, so the result
    does not depend on scheduling."""
    jobs = [(config, seed, double) for _, double in ARMS for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        logs = list(pool.map(_train_job, jobs))
    return [ArmResult(label, double, logs[i * len(seeds) : (i + 1) * len(seeds)]) for i, (label, double) in enumerate(ARMS)]


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color="#1f1e1c", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Environment step", color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def save_summary_plot(arms: Sequence[ArmResult], config: CasinoConfig, path: Path) -> Path:
    """Three panels sharing the step axis: bootstrap bias, door bias, P(greedy agent gambles)."""
    panels = [
        ("bootstrap_bias", "Floor value the target bootstraps from, minus truth", "Overestimation"),
        ("door_bias", "Best casino door Q(lobby, door), minus truth", "Overestimation"),
        ("gambles", "Greedy agent walks into the casino", "Fraction of seeds"),
    ]
    fig = Figure(figsize=(14, 4.4), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(1, 3)
    for ax, (column, title, ylabel) in zip(axes, panels):
        for arm, color in zip(arms, SERIES_PALETTE):
            band = aggregate_seeds([(np.asarray(run.steps), np.asarray(getattr(run, column))) for run in arm.runs])
            if band is None:
                continue
            ax.fill_between(band.x, band.mean - band.stderr, band.mean + band.stderr, color=color, alpha=0.18, lw=0)
            ax.plot(band.x, band.mean, color=color, lw=2, label=arm.label)
            ax.annotate(
                f"{band.mean[-1]:.2f}", (band.x[-1], band.mean[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, va="center",
            )
        ax.axhline(0.0, color=INK_SECONDARY, lw=1, ls="--")
        _style(ax, title, ylabel)
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(
        f"Maximization-bias casino: {config.n_actions} machines, reward ~ N({config.bet_mean}, {config.sigma}²), "
        f"{len(arms[0].runs)} seeds, ±1 s.e.  Dashed line = truth / optimal.\n"
        f"Vanilla's bootstrap value only moves when the target is re-synced (every {config.target_update_interval:,} "
        "gradient steps); Double DQN's moves with the online network's pick.",
        x=0.01, ha="left", fontsize=10, color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


def run_sweep(base: CasinoConfig, seeds: Sequence[int], n_values: Sequence[int], sigmas: Sequence[float], path: Path):
    """Late-training bootstrap bias and gamble rate for every (n_actions, sigma) cell, both arms."""
    rows = []
    for n in n_values:
        for sigma in sigmas:
            config = replace(base, n_actions=n, sigma=sigma)
            for arm in run_arms(config, seeds):
                label = arm.label
                bias, bias_se = mean_stderr(arm.late("bootstrap_bias"))
                gamble, _ = mean_stderr(arm.late("gambles"))
                collapsed = sum(run.collapsed for run in arm.runs)
                rows.append({"n_actions": n, "sigma": sigma, "arm": label, "bias": bias, "bias_se": bias_se,
                             "gamble": gamble, "collapsed": collapsed})
                print(f"n={n:3d} sigma={sigma:4.2f} {label:<12} bias {bias:+.3f} ± {bias_se:.3f} | "
                      f"gambles {100 * gamble:5.1f}% | collapsed {collapsed}/{len(arm.runs)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.with_suffix(".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig = Figure(figsize=(12, 4.4), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (label, _) in zip(axes, ARMS):
        grid = np.array([[next(r["bias"] for r in rows if r["n_actions"] == n and r["sigma"] == s and r["arm"] == label)
                          for s in sigmas] for n in n_values])
        limit = max(0.05, max(abs(r["bias"]) for r in rows))
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        for i in range(len(n_values)):
            for j in range(len(sigmas)):
                ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(grid[i, j]) > 0.6 * limit else "#1f1e1c")
        ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        ax.set_yticks(range(len(n_values)), [str(n) for n in n_values])
        ax.set_xlabel("Reward noise σ", color=INK_SECONDARY, fontsize=9)
        ax.set_title(label, loc="left", fontsize=11)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    axes[0].set_ylabel("Machines (actions)", color=INK_SECONDARY, fontsize=9)
    fig.colorbar(image, ax=axes, label="Bootstrap bias (late training)", shrink=0.9)
    fig.suptitle("Where the max operator hurts: bias grows with action count and reward noise",
                 x=0.01, ha="left", fontsize=10, color=INK_SECONDARY)
    fig.savefig(path, facecolor=SURFACE)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vanilla vs Double DQN on the maximization-bias casino.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-actions", type=int, default=CasinoConfig.n_actions)
    parser.add_argument("--sigma", type=float, default=CasinoConfig.sigma)
    parser.add_argument("--total-steps", type=int, default=CasinoConfig.total_steps)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-gif", action="store_true", help="skip the side-by-side GIF")
    parser.add_argument("--sweep", action="store_true", help="run the (n_actions, sigma) sweep instead")
    args = parser.parse_args(argv)
    config = CasinoConfig(n_actions=args.n_actions, sigma=args.sigma, total_steps=args.total_steps)

    if args.sweep:
        path = run_sweep(config, args.seeds, (2, 5, 10, 20, 50), (0.0, 0.25, 0.5, 1.0, 2.0),
                         args.output_dir / "sweep.png")
        print(f"Plot: {path}\nCSV:  {path.with_suffix('.csv')}")
        return 0

    print(f"Training {len(ARMS)} arms x {len(args.seeds)} seeds on {config.n_actions} machines, sigma={config.sigma:g}")
    arms = run_arms(config, args.seeds)
    print("\nSecond half of training, mean ± s.e. across seeds:")
    for arm in arms:
        bias = mean_stderr(arm.late("bootstrap_bias"))
        door = mean_stderr(arm.late("door_bias"))
        gamble = mean_stderr(arm.late("gambles"))
        print(
            f"{arm.label:<12} | bootstrap bias {bias[0]:+.3f} ± {bias[1]:.3f} | door bias {door[0]:+.3f} ± {door[1]:.3f} "
            f"| greedy agent gambles {100 * gamble[0]:.0f}% ± {100 * gamble[1]:.0f}%"
        )
    collapsed = [sum(run.collapsed for run in arm.runs) for arm in arms]
    if any(collapsed):
        print(f"WARNING: collapsed networks (a dead ReLU layer; rooms indistinguishable): "
              + ", ".join(f"{arm.label} {n}/{len(arm.runs)}" for arm, n in zip(arms, collapsed)))
    else:
        print("No collapsed networks: every run still tells the two rooms apart.")
    print(f"Plot: {save_summary_plot(arms, config, args.output_dir / 'casino_bias.png')}")

    if not args.no_gif:
        from dqn.casino_render import render_duel

        frames = render_duel(arms, config)
        print(f"GIF:  {save_episode_gif(frames, args.output_dir / 'casino_duel.gif', fps=12, scale=1.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
