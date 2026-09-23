"""Checkpoint and log-file helpers (provided scaffold)."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from dqn.config import DQNConfig
from dqn.network import QNetwork

CHECKPOINT_NAME = "dqn.pt"
CURVE_NAME = "reward_curve.png"
RETURNS_NAME = "episode_returns.csv"
MEDIAN_GIF_NAME = "median_episode.gif"
COMPARISON_NAME = "hard_vs_soft.png"
RESULTS_DIR = "results"  # evaluation artifacts land here; training writes to DQNConfig.output_dir

# A multi-seed run puts every seed in one output directory, with the seed in each
# filename, so one arm of an experiment is a directory rather than a file.
SEED_RETURNS_GLOB = "episode_returns_seed*.csv"
SEED_SUFFIX_PATTERN = re.compile(r"_seed(-?\d+)(?=\.[^.]*$|$)")


def seeded_name(name: str, seed: int) -> str:
    """Insert ``_seed{N}`` before a filename's extension: ``a.csv`` -> ``a_seed3.csv``."""
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return f"{name}_seed{seed}"
    return f"{stem}_seed{seed}{dot}{suffix}"


def seed_from_name(path: str | Path) -> int | None:
    """Recover the seed a ``seeded_name`` filename encodes, or ``None`` if it has none."""
    match = SEED_SUFFIX_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def discover_returns_csvs(path: str | Path) -> list[tuple[int | None, Path]]:
    """Resolve one arm of an experiment into its per-seed returns CSVs.

    Accepts either a run directory or a single CSV, so all three layouts work:
    a ``--seeds`` directory holding ``episode_returns_seed{N}.csv`` per seed (sorted by
    seed), a single-seed directory holding a plain ``episode_returns.csv``, or an explicit
    path to one CSV. Returns ``(seed, path)`` pairs; ``seed`` is ``None`` when the
    filename does not encode one.
    """
    path = Path(path)
    if path.is_file():
        return [(seed_from_name(path), path)]
    if path.is_dir():
        seeded = [(seed_from_name(p), p) for p in path.glob(SEED_RETURNS_GLOB)]
        if seeded:
            return sorted(seeded, key=lambda pair: (pair[0] is None, pair[0] or 0))
        plain = path / RETURNS_NAME
        if plain.is_file():
            return [(None, plain)]
        raise FileNotFoundError(
            f"No episode returns in {path}{'/'} (looked for {SEED_RETURNS_GLOB} and {RETURNS_NAME}). "
            "Train a run first with: python -m dqn.train"
        )
    raise FileNotFoundError(f"No episode returns at {path}. Train a run first with: python -m dqn.train")


def save_checkpoint(
    path: str | Path,
    network: QNetwork,
    config: DQNConfig,
    episode_returns: Sequence[float],
) -> Path:
    """Save the online network plus enough metadata to rebuild it for evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": network.state_dict(),
        "obs_dim": network.obs_dim,
        "num_actions": network.num_actions,
        "hidden_sizes": list(network.hidden_sizes),
        "env_id": config.env_id,
        "config": config.to_dict(),
        "episode_returns": [float(r) for r in episode_returns],
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[QNetwork, dict[str, Any]]:
    """Rebuild a ``QNetwork`` from a checkpoint. Returns ``(network, metadata)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}. Train one first with: python -m dqn.train")
    payload = torch.load(path, map_location=device, weights_only=True)
    network = QNetwork(payload["obs_dim"], payload["num_actions"], payload["hidden_sizes"]).to(device)
    network.load_state_dict(payload["model_state_dict"])
    return network, payload


def save_episode_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 30,
    scale: float = 0.5,
) -> Path:
    """Write RGB ``frames`` (each ``[H, W, 3]`` uint8) to a looping animated GIF."""
    if len(frames) == 0:
        raise ValueError("save_episode_gif needs at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    if scale != 1.0:
        size = (max(1, round(images[0].width * scale)), max(1, round(images[0].height * scale)))
        images = [image.resize(size, Image.BILINEAR) for image in images]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return path


def save_returns_csv(
    path: str | Path,
    episode_returns: Sequence[float],
    episode_steps: Sequence[int] | None = None,
) -> Path:
    """Write one row per episode.

    ``episode_steps`` is the environment step each episode *ended* on. Passing it adds a
    ``step`` column, which is what lets two runs be compared on a common x-axis: a better
    agent plays fewer, longer episodes, so episode index is not a fair shared clock.
    Omitting it keeps the original two-column ``episode,return`` form.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if episode_steps is not None and len(episode_steps) != len(episode_returns):
        raise ValueError(
            "episode_steps must be the same length as episode_returns, "
            f"got {len(episode_steps)} and {len(episode_returns)}"
        )
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"] if episode_steps is None else ["episode", "step", "return"])
        for index, value in enumerate(episode_returns):
            row = [index + 1, f"{float(value):g}"]
            if episode_steps is not None:
                row.insert(1, int(episode_steps[index]))
            writer.writerow(row)
    return path


def load_returns_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a returns CSV back as ``(returns, steps)``.

    ``steps`` is ``None`` for files written without the optional ``step`` column.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No episode returns at {path}. Train a run first with: python -m dqn.train")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.empty(0, dtype=np.float64), None
    returns = np.array([float(row["return"]) for row in rows], dtype=np.float64)
    if "step" not in rows[0]:
        return returns, None
    return returns, np.array([int(row["step"]) for row in rows], dtype=np.int64)
