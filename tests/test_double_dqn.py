"""Double DQN targets: ``compute_double_dqn_targets`` in src/dqn/learning.py.

The tables below are chosen so the two networks *disagree* about the best next action —
the only situation in which Double DQN differs from vanilla DQN at all.
"""

import copy

import pytest
import torch

from dqn.config import DQNConfig
from dqn.learning import compute_dqn_targets, compute_double_dqn_targets, dqn_loss, training_step
from dqn.network import QNetwork
from dqn.replay_buffer import Batch
from dqn.train import train
from helpers import TableQNetwork

pytestmark = pytest.mark.learner

OBS_DIM = 4

# Online argmax per row: 1, 0, 0. The target network ranks rows 0 and 2 the other way round,
# so a vanilla max would take 10.0 and 2.0 where Double DQN takes -10.0 and 1.0.
ONLINE_TABLE = [[1.0, 3.0], [5.0, 2.0], [-2.0, -4.0]]
TARGET_TABLE = [[10.0, -10.0], [0.0, 7.0], [1.0, 2.0]]


def make_networks() -> tuple[TableQNetwork, TableQNetwork]:
    return TableQNetwork(ONLINE_TABLE), TableQNetwork(TARGET_TABLE)


def make_batch(actions, rewards, dones) -> Batch:
    size = len(actions)
    return Batch(
        states=torch.zeros(size, OBS_DIM),
        actions=torch.tensor(actions, dtype=torch.int64),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        next_states=torch.zeros(size, OBS_DIM),
        dones=torch.tensor(dones, dtype=torch.float32),
    )


# --- compute_double_dqn_targets -------------------------------------------------------------


def test_double_targets_price_the_online_argmax_with_the_target_network():
    online, target = make_networks()

    targets = compute_double_dqn_targets(
        rewards=torch.tensor([1.0, 0.5, -1.0]),
        next_states=torch.zeros(3, OBS_DIM),
        dones=torch.tensor([0.0, 1.0, 0.0]),
        gamma=0.9,
        online_network=online,
        target_network=target,
    )

    assert targets.shape == (3,)
    assert targets.dtype == torch.float32
    # Row 0: a* = 1, Q_target = -10 -> 1 + 0.9 * -10. Row 1 is terminal. Row 2: a* = 0, Q_target = 1.
    torch.testing.assert_close(targets, torch.tensor([-8.0, 0.5, -0.1]))


def test_double_targets_differ_from_vanilla_where_the_networks_disagree():
    online, target = make_networks()
    rewards = torch.tensor([1.0, 0.5, -1.0])
    next_states = torch.zeros(3, OBS_DIM)
    dones = torch.zeros(3)

    double = compute_double_dqn_targets(rewards, next_states, dones, 0.9, online, target)
    vanilla = compute_dqn_targets(rewards, next_states, dones, 0.9, target)

    # Vanilla takes each row's largest target value; double takes the one the online net named.
    torch.testing.assert_close(vanilla, torch.tensor([10.0, 6.8, 0.8]))
    assert not torch.allclose(double, vanilla)
    # The whole point: the decoupled estimate is never the inflated one.
    assert bool((double <= vanilla + 1e-6).all())


def test_double_targets_match_vanilla_when_both_networks_rank_actions_alike():
    online = TableQNetwork([[1.0, 3.0], [5.0, 2.0]])
    target = TableQNetwork([[2.0, 9.0], [8.0, 1.0]])
    rewards = torch.tensor([0.5, 0.5])
    next_states = torch.zeros(2, OBS_DIM)
    dones = torch.zeros(2)

    double = compute_double_dqn_targets(rewards, next_states, dones, 0.9, online, target)

    torch.testing.assert_close(double, torch.tensor([8.6, 7.7]))
    torch.testing.assert_close(double, compute_dqn_targets(rewards, next_states, dones, 0.9, target))


def test_terminal_double_targets_equal_rewards():
    online = TableQNetwork([[1.0, -1.0], [-1.0, 1.0]])
    target = TableQNetwork([[1e6, 1e6], [1e6, 1e6]])
    rewards = torch.tensor([1.0, -2.0])

    targets = compute_double_dqn_targets(
        rewards, torch.zeros(2, OBS_DIM), torch.ones(2), 0.99, online, target
    )

    torch.testing.assert_close(targets, rewards)


def test_double_targets_are_detached_from_both_networks():
    online, target = make_networks()

    targets = compute_double_dqn_targets(
        torch.tensor([1.0, 1.0, 1.0]), torch.zeros(3, OBS_DIM), torch.zeros(3), 0.9, online, target
    )

    assert not targets.requires_grad
    assert targets.grad_fn is None
    assert online.q_values.grad is None
    assert target.q_values.grad is None


def test_double_targets_query_each_network_once_with_a_batched_input():
    online, target = make_networks()

    compute_double_dqn_targets(
        torch.tensor([1.0, 1.0, 1.0]), torch.zeros(3, OBS_DIM), torch.zeros(3), 0.9, online, target
    )

    assert [shape for shape, _ in online.calls] == [(3, OBS_DIM)]
    assert [shape for shape, _ in target.calls] == [(3, OBS_DIM)]


# --- the flag that reaches the training loop ------------------------------------------------


def test_dqn_loss_uses_double_targets_when_asked():
    online = TableQNetwork([[1.0, 2.0], [3.0, 4.0]])
    target = TableQNetwork([[0.0, 10.0], [4.0, 1.0]])
    batch = make_batch(actions=[1, 0], rewards=[1.0, 1.0], dones=[0.0, 0.0])

    loss = dqn_loss(online, target, batch, gamma=0.5, double=True)

    assert loss.ndim == 0
    # Online argmax is action 1 in both rows; Q_target at action 1 = [10, 1], so the targets
    # are [6, 1.5] against predictions [2, 3]. Huber(-4) = 3.5, Huber(1.5) = 1.0, mean = 2.25.
    # The vanilla arm of this same batch scores 1.75 (tests/test_dqn_learning.py).
    torch.testing.assert_close(loss.detach(), torch.tensor(2.25))


def test_training_step_with_double_updates_only_the_online_network():
    torch.manual_seed(0)
    online = QNetwork(OBS_DIM, 2, hidden_sizes=(16,))
    target = copy.deepcopy(online)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(0.1)
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

    loss = training_step(online, target, optimizer, batch, gamma=0.99, double=True)

    assert type(loss) is float
    assert any(not torch.equal(online_before[k], v) for k, v in online.state_dict().items())
    assert all(torch.equal(target_before[k], v) for k, v in target.state_dict().items())
    assert all(p.grad is None for p in target.parameters())


def test_a_double_dqn_run_trains_and_saves_artifacts(tmp_path):
    config = DQNConfig(
        double_dqn=True,
        total_steps=300,
        learning_starts=64,
        batch_size=32,
        buffer_capacity=1_000,
        epsilon_decay_steps=200,
        target_update_interval=50,
        log_every_episodes=1_000,
        output_dir=str(tmp_path),
        device="cpu",
    )

    result = train(config)

    assert result.episode_returns
    for path in (result.checkpoint_path, result.curve_path, result.returns_path):
        assert path.is_file() and path.stat().st_size > 0
