"""Hyperparameters and run settings (provided scaffold)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

TARGET_UPDATE_MODES = ("hard", "soft")


@dataclass
class DQNConfig:
    # Environment
    env_id: str = "CartPole-v1"
    seed: int = 0

    # Training budget
    total_steps: int = 50_000  # environment steps
    learning_starts: int = 1_000  # environment steps collected before the first gradient step
    train_every: int = 1  # environment steps between gradient steps

    # Replay
    buffer_capacity: int = 50_000
    batch_size: int = 64

    # Optimisation
    gamma: float = 0.99
    learning_rate: float = 5e-4
    max_grad_norm: float | None = 10.0
    hidden_sizes: tuple[int, ...] = (128, 128)

    # Exploration: epsilon decays linearly over environment steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000

    # Target network. "hard": copy the online weights every `target_update_interval`
    # gradient steps. "soft": after *every* gradient step, move the target a fraction
    # `tau` of the way toward the online weights (Polyak averaging). A soft update with
    # tau ~ 1 / target_update_interval tracks the online network at a broadly similar
    # rate, but continuously rather than in jumps.
    target_update_mode: str = "hard"
    target_update_interval: int = 500  # gradient steps between hard syncs ("hard" only)
    tau: float = 0.005  # Polyak coefficient in (0, 1] ("soft" only); 1.0 == a hard copy every step

    # Double DQN. Vanilla DQN's bootstrap term `max_a' Q_target(s', a')` both *picks* the
    # next action and *prices* it with the same noisy network, so any action whose value is
    # overestimated by noise is exactly the one the max selects — the bias is systematic and
    # upward. Double DQN splits the two roles between the online and target networks.
    double_dqn: bool = False

    # Overestimation diagnostics (dqn.overestimation). Every `overestimation_every` env steps,
    # play `overestimation_episodes` greedy episodes and compare predicted Q with the realised
    # discounted return. 0 turns it off. Measurement only: the training run is unchanged.
    overestimation_every: int = 0
    overestimation_episodes: int = 3

    # Logging / artifacts
    log_every_episodes: int = 10
    output_dir: str = "outputs"
    device: str = "auto"

    def validate(self) -> None:
        """Raise ValueError for settings the training loop cannot run with."""
        positive = {
            "total_steps": self.total_steps,
            "train_every": self.train_every,
            "buffer_capacity": self.buffer_capacity,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "log_every_episodes": self.log_every_episodes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be >= batch_size so a first minibatch can be sampled")
        if self.buffer_capacity < self.batch_size:
            raise ValueError("buffer_capacity must be >= batch_size")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("expected 0 <= epsilon_end <= epsilon_start <= 1")
        if self.target_update_mode not in TARGET_UPDATE_MODES:
            raise ValueError(
                f"target_update_mode must be one of {sorted(TARGET_UPDATE_MODES)}, got {self.target_update_mode!r}"
            )
        if self.overestimation_every < 0:
            raise ValueError(f"overestimation_every must be >= 0, got {self.overestimation_every}")
        if self.overestimation_episodes <= 0:
            raise ValueError(f"overestimation_episodes must be positive, got {self.overestimation_episodes}")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(name: str | torch.device) -> torch.device:
    """Map "auto" to CUDA when available, otherwise CPU (a tiny MLP gains nothing from MPS)."""
    if isinstance(name, str) and name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
