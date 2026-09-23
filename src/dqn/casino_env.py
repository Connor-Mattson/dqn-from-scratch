"""A two-room casino that isolates DQN's maximization bias (Sutton & Barto, Example 6.7).

    import gymnasium as gym
    import dqn.casino_env  # registers the id
    env = gym.make("MaxBiasCasino-v0", n_actions=20, sigma=2.0, render_mode="rgb_array")

The agent starts in the **lobby**. Action 0 walks out through the green exit: reward 0,
episode over. Actions ``1 .. n_actions-1`` are casino doors: reward 0, and the agent lands on
the **casino floor**. There, every action pulls one of ``n_actions`` slot machines, pays a
reward drawn from ``Normal(bet_mean, sigma**2)``, and ends the episode.

Every machine loses on average (``bet_mean < 0``), so the answer is known exactly:

    Q*(floor, any)  = bet_mean                 (-0.1)
    Q*(lobby, exit) = 0
    Q*(lobby, door) = gamma * bet_mean         (-0.099 for gamma = 0.99)

and the only optimal policy is to walk out. Vanilla DQN bootstraps the doors from
``max_a Q_target(floor, a)``: the maximum of ``n_actions`` noisy estimates of -0.1, which is
biased upward, and more so as ``n_actions`` and ``sigma`` grow. Once that max drifts above
zero, a casino door looks better than the exit and the greedy agent gambles. Double DQN picks
the machine with the online network and prices it with the target network, which removes the
part of that bias where the two networks' errors differ. Error they share, such as the replay
buffer's finite-sample estimate of each machine's mean, survives (see ``dqn.casino_experiment``).

Observations are one-hot room indicators, ``[1, 0]`` in the lobby and ``[0, 1]`` on the floor,
so the flat-MLP ``QNetwork`` works unchanged.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_ID = "MaxBiasCasino-v0"
LOBBY, FLOOR = 0, 1
EXIT = 0  # the lobby action that walks away; every other lobby action is a casino door


def optimal_q_values(n_actions: int, bet_mean: float, gamma: float) -> dict[str, np.ndarray]:
    """``Q*`` for both rooms, one entry per action."""
    lobby = np.full(n_actions, gamma * bet_mean)
    lobby[EXIT] = 0.0
    return {"lobby": lobby, "floor": np.full(n_actions, bet_mean)}


class MaxBiasCasinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        n_actions: int = 20,
        sigma: float = 2.0,
        bet_mean: float = -0.1,
        render_mode: str | None = None,
    ) -> None:
        if n_actions < 2:
            raise ValueError(f"n_actions must be >= 2 (an exit and at least one door), got {n_actions}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}")
        self.n_actions = n_actions
        self.sigma = float(sigma)
        self.bet_mean = float(bet_mean)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

        self.room = LOBBY
        self.done = False
        # What the renderer needs to draw the last transition.
        self.last_action: int | None = None
        self.last_reward: float | None = None
        self.last_room: int | None = None
        self.bankroll = 0.0  # running winnings across episodes, for the HUD only
        self._renderer = None

    def _obs(self) -> np.ndarray:
        obs = np.zeros(2, dtype=np.float32)
        obs[self.room] = 1.0
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.room = LOBBY
        self.done = False
        self.last_action = self.last_reward = self.last_room = None
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"room": "lobby"}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")
        action = int(action)
        self.last_room, self.last_action = self.room, action

        if self.room == LOBBY:
            reward = 0.0
            terminated = action == EXIT
            if not terminated:
                self.room = FLOOR
        else:
            reward = float(self.np_random.normal(self.bet_mean, self.sigma))
            terminated = True

        self.last_reward = reward
        self.bankroll += reward
        self.done = terminated
        info = {"room": "lobby" if self.room == LOBBY else "floor", "walked_out": self.last_room == LOBBY and action == EXIT}
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; pass render_mode='rgb_array' or 'human'")
            return None
        from dqn.casino_render import CasinoRenderer  # pygame is only needed when rendering

        if self._renderer is None:
            self._renderer = CasinoRenderer(self.n_actions, self.bet_mean, human=self.render_mode == "human")
        frame = self._renderer.draw_env(self)
        if self.render_mode == "human":
            self._renderer.show(frame, self.metadata["render_fps"])
            return None
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="dqn.casino_env:MaxBiasCasinoEnv")
