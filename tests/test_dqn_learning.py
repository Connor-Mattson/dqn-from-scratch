"""Learner tasks 3-4: Bellman targets, DQN loss, and the training step (src/dqn/learning.py)."""

import copy

import pytest
import torch

from dqn.learning import compute_dqn_targets, dqn_loss, training_step
from dqn.network import QNetwork
from dqn.replay_buffer import Batch
from helpers import TableQNetwork

pytestmark = pytest.mark.learner

OBS_DIM = 4


def make_batch(actions, rewards, dones) -> Batch:
    size = len(actions)
    return Batch(
        states=torch.zeros(size, OBS_DIM),
        actions=torch.tensor(actions, dtype=torch.int64),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        next_states=torch.zeros(size, OBS_DIM),
        dones=torch.tensor(dones, dtype=torch.float32),
    )


# --- compute_dqn_targets -------------------------------------------------------------------


def test_targets_bootstrap_from_max_target_q_unless_terminal():
    target_network = TableQNetwork([[1.0, 3.0], [5.0, 2.0], [-2.0, -4.0]])
    targets = compute_dqn_targets(
        rewards=torch.tensor([1.0, 0.5, -1.0]),
        next_states=torch.zeros(3, OBS_DIM),
        dones=torch.tensor([0.0, 1.0, 0.0]),
        gamma=0.9,
        target_network=target_network,
    )
    assert targets.shape == (3,)
    assert targets.dtype == torch.float32
    torch.testing.assert_close(targets, torch.tensor([3.7, 0.5, -2.8]))


def test_terminal_targets_equal_rewards():
    target_network = TableQNetwork([[1e6, 1e6], [1e6, 1e6]])
    rewards = torch.tensor([1.0, -2.0])
    targets = compute_dqn_targets(rewards, torch.zeros(2, OBS_DIM), torch.ones(2), 0.99, target_network)
    torch.testing.assert_close(targets, rewards)


def test_targets_are_detached():
    target_network = TableQNetwork([[1.0, 2.0]])
    targets = compute_dqn_targets(torch.tensor([1.0]), torch.zeros(1, OBS_DIM), torch.zeros(1), 0.9, target_network)
    assert not targets.requires_grad


# --- dqn_loss ------------------------------------------------------------------------------


def test_loss_uses_taken_actions_target_network_and_huber():
    online = TableQNetwork([[1.0, 2.0], [3.0, 4.0]])
    target = TableQNetwork([[0.0, 10.0], [4.0, 1.0]])
    batch = make_batch(actions=[1, 0], rewards=[1.0, 1.0], dones=[0.0, 0.0])

    loss = dqn_loss(online, target, batch, gamma=0.5)

    assert loss.ndim == 0
    # Q_online(s, a) = [2, 3]; targets = [1 + 0.5 * 10, 1 + 0.5 * 4] = [6, 3]; Huber(-4, 0) mean = 1.75.
    torch.testing.assert_close(loss.detach(), torch.tensor(1.75))


def test_loss_gradients_reach_only_sampled_online_q_values():
    online = TableQNetwork([[1.0, 2.0], [3.0, 4.0]])
    target = TableQNetwork([[0.0, 10.0], [4.0, 1.0]])
    batch = make_batch(actions=[1, 0], rewards=[1.0, 1.0], dones=[0.0, 0.0])

    dqn_loss(online, target, batch, gamma=0.5).backward()

    torch.testing.assert_close(online.q_values.grad, torch.tensor([[0.0, -0.5], [0.0, 0.0]]))
    assert target.q_values.grad is None


# --- training_step -------------------------------------------------------------------------


def test_training_step_updates_online_network_only():
    torch.manual_seed(0)
    online = QNetwork(OBS_DIM, 2, hidden_sizes=(16,))
    target = copy.deepcopy(online)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(0.1)  # make the networks differ so a target bug changes the result
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
    batch = Batch(
        states=torch.randn(8, OBS_DIM),
        actions=torch.randint(0, 2, (8,)),
        rewards=torch.ones(8),
        next_states=torch.randn(8, OBS_DIM),
        dones=torch.tensor([0.0, 1.0] * 4),
    )
    online_before = copy.deepcopy(online.state_dict())
    target_before = copy.deepcopy(target.state_dict())

    loss = training_step(online, target, optimizer, batch, gamma=0.99)

    assert type(loss) is float
    assert any(not torch.equal(online_before[k], v) for k, v in online.state_dict().items())
    assert all(torch.equal(target_before[k], v) for k, v in target.state_dict().items())
    assert all(p.grad is None for p in target.parameters())


def test_training_step_ignores_stale_gradients():
    online = TableQNetwork([[0.0, 0.0]])
    target = TableQNetwork([[100.0, 100.0]])
    online.q_values.grad = torch.full((1, 2), 1000.0)  # leftover gradient from "a previous step"
    optimizer = torch.optim.SGD(online.parameters(), lr=1.0)
    batch = make_batch(actions=[0], rewards=[0.0], dones=[0.0])

    training_step(online, target, optimizer, batch, gamma=1.0)

    # Fresh Huber gradient is [[-1, 0]], so one SGD step with lr=1 gives [[1, 0]].
    torch.testing.assert_close(online.q_values.detach(), torch.tensor([[1.0, 0.0]]))


def test_training_step_clips_gradient_norm():
    online = TableQNetwork([[0.0, 0.0]])
    target = TableQNetwork([[100.0, 100.0]])
    optimizer = torch.optim.SGD(online.parameters(), lr=1.0)
    batch = make_batch(actions=[0], rewards=[0.0], dones=[0.0])
    before = online.q_values.detach().clone()

    training_step(online, target, optimizer, batch, gamma=1.0, max_grad_norm=0.1)

    step_size = torch.linalg.norm(online.q_values.detach() - before).item()
    assert step_size == pytest.approx(0.1, rel=1e-4)
