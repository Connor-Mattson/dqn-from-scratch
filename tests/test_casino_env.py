"""The maximization-bias casino: src/dqn/casino_env.py and the casino experiment's diagnostics."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from dqn.casino_env import ENV_ID, EXIT, MaxBiasCasinoEnv, optimal_q_values
from dqn.casino_experiment import CasinoConfig, read_values, train_casino

pytestmark = pytest.mark.scaffold

LOBBY_OBS = [1.0, 0.0]
FLOOR_OBS = [0.0, 1.0]


def test_registered_with_gymnasium_and_accepts_kwargs():
    env = gym.make(ENV_ID, n_actions=7, sigma=0.5)
    assert env.action_space.n == 7
    assert env.observation_space.shape == (2,)
    env.close()


def test_walking_out_ends_the_episode_with_nothing():
    env = MaxBiasCasinoEnv()
    obs, _ = env.reset(seed=0)
    assert obs.tolist() == LOBBY_OBS and obs.dtype == np.float32
    obs, reward, terminated, truncated, info = env.step(EXIT)
    assert (reward, terminated, truncated, info["walked_out"]) == (0.0, True, False, True)


def test_every_door_leads_to_the_floor_and_every_machine_ends_it():
    env = MaxBiasCasinoEnv(n_actions=5)
    for door in range(1, 5):
        env.reset(seed=door)
        obs, reward, terminated, _, _ = env.step(door)
        assert obs.tolist() == FLOOR_OBS and reward == 0.0 and not terminated
        _, _, terminated, _, info = env.step(door)
        assert terminated and not info["walked_out"]


def test_payouts_have_the_advertised_mean_and_spread():
    env = MaxBiasCasinoEnv(n_actions=3, sigma=2.0, bet_mean=-0.1)
    env.reset(seed=0)
    rewards = []
    for machine in range(6000):
        env.reset()
        env.step(1)
        rewards.append(env.step(machine % 3)[1])
    assert np.mean(rewards) == pytest.approx(-0.1, abs=0.08)
    assert np.std(rewards) == pytest.approx(2.0, rel=0.05)


def test_zero_sigma_pays_exactly_the_mean():
    env = MaxBiasCasinoEnv(sigma=0.0)
    env.reset(seed=0)
    env.step(1)
    assert env.step(4)[1] == pytest.approx(-0.1)


def test_step_after_the_end_raises():
    env = MaxBiasCasinoEnv()
    env.reset(seed=0)
    env.step(EXIT)
    with pytest.raises(RuntimeError):
        env.step(EXIT)


def test_optimal_q_values_make_walking_out_the_only_optimal_action():
    q = optimal_q_values(n_actions=4, bet_mean=-0.1, gamma=0.99)
    assert q["floor"].tolist() == pytest.approx([-0.1] * 4)
    assert q["lobby"].tolist() == pytest.approx([0.0, -0.099, -0.099, -0.099])
    assert int(q["lobby"].argmax()) == EXIT


class RoomTable(nn.Module):
    """Returns a fixed row of Q-values per room, keyed on the one-hot observation."""

    def __init__(self, lobby: list[float], floor: list[float]) -> None:
        super().__init__()
        self.table = torch.tensor([lobby, floor])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.table[obs.argmax(dim=1)]


def test_read_values_bootstraps_like_each_arm():
    online = RoomTable([0.0, 0.1, 0.2], floor=[0.5, -1.0, 0.0])  # online favours machine 0
    target = RoomTable([0.0, 0.0, 0.0], floor=[-0.3, 0.9, 0.1])  # target's max is machine 1
    vanilla = read_values(online, target, double=False, device="cpu")
    double = read_values(online, target, double=True, device="cpu")
    assert vanilla["bootstrap"] == pytest.approx(0.9)  # max_a Q_target(floor, a)
    assert double["bootstrap"] == pytest.approx(-0.3)  # Q_target(floor, argmax_a Q_online(floor, a))
    assert vanilla["q_lobby"].tolist() == pytest.approx([0.0, 0.1, 0.2])


def test_short_training_run_logs_every_checkpoint():
    config = CasinoConfig(n_actions=4, total_steps=600, learning_starts=100, checkpoint_every=200,
                          target_update_interval=50, hidden_sizes=(16,))
    log = train_casino(config, seed=0, double=True)
    assert log.steps == [200, 400, 600]
    assert all(np.isfinite(log.bootstrap_bias)) and set(log.gambles) <= {0.0, 1.0}
    assert log.q_floor[0].shape == (4,)


def test_render_returns_an_rgb_frame():
    pytest.importorskip("pygame")
    env = MaxBiasCasinoEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(2)
    env.step(3)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    env.close()


def test_has_collapsed_spots_a_dead_layer():
    from dqn.casino_experiment import has_collapsed
    from dqn.network import QNetwork

    network = QNetwork(2, 3, (4,))
    with torch.no_grad():
        network.net[0].weight.fill_(1.0)
        network.net[0].bias.fill_(0.0)
    assert not has_collapsed(network)
    with torch.no_grad():
        network.net[0].weight[:, 1] = -5.0  # every unit is dead for the floor observation
    assert has_collapsed(network)
