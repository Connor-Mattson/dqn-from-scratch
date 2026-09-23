"""Gymnasium environment creation (provided scaffold)."""

from __future__ import annotations

import gymnasium as gym

# Envs whose per-episode return has a known ceiling, drawn as a reference line on reward curves.
MAX_RETURNS = {"CartPole-v0": 200.0, "CartPole-v1": 500.0}


def make_env(env_id: str = "CartPole-v1", render_mode: str | None = None) -> gym.Env:
    """Create a Gymnasium environment. CartPole ships with Gymnasium; Box2D envs such as
    LunarLander-v3 need ``pip install -e '.[box2d]'``."""
    return gym.make(env_id, render_mode=render_mode)


def env_dimensions(env: gym.Env) -> tuple[int, int]:
    """Return ``(obs_dim, num_actions)`` for a flat Box-observation, Discrete-action env."""
    obs_space, action_space = env.observation_space, env.action_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"DQN needs a Discrete action space, got {action_space}")
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
        raise TypeError(f"expected a flat Box observation space, got {obs_space}")
    return int(obs_space.shape[0]), int(action_space.n)
