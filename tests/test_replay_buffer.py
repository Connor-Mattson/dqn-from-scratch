"""Learner task 1: ReplayBuffer (src/dqn/replay_buffer.py)."""

import numpy as np
import pytest
import torch

from dqn.replay_buffer import Batch, ReplayBuffer

pytestmark = pytest.mark.learner

OBS_DIM = 4


def push_transition(buffer: ReplayBuffer, index: int, done: bool = False) -> None:
    """Push a transition whose reward and observation values encode `index`."""
    state = np.full(OBS_DIM, index, dtype=np.float32)
    next_state = np.full(OBS_DIM, index + 0.5, dtype=np.float32)
    buffer.push(state, index % 2, float(index), next_state, done)


def test_len_counts_stored_transitions_up_to_capacity():
    buffer = ReplayBuffer(capacity=3, seed=0)
    assert len(buffer) == 0
    for i in range(5):
        push_transition(buffer, i)
        assert len(buffer) == min(i + 1, 3)


def test_full_buffer_replaces_oldest_transitions():
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        push_transition(buffer, i)
    batch = buffer.sample(3)
    assert sorted(batch.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_sample_returns_batch_with_contract_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=10, seed=0)
    for i in range(6):
        push_transition(buffer, i, done=(i == 5))
    batch = buffer.sample(4)

    assert isinstance(batch, Batch)
    assert batch.states.shape == (4, OBS_DIM) and batch.states.dtype == torch.float32
    assert batch.actions.shape == (4,) and batch.actions.dtype == torch.int64
    assert batch.rewards.shape == (4,) and batch.rewards.dtype == torch.float32
    assert batch.next_states.shape == (4, OBS_DIM) and batch.next_states.dtype == torch.float32
    assert batch.dones.shape == (4,) and batch.dones.dtype == torch.float32
    assert all(t.device.type == "cpu" for t in batch)


def test_sampled_fields_stay_aligned_within_each_transition():
    buffer = ReplayBuffer(capacity=10, seed=0)
    for i in range(10):
        push_transition(buffer, i, done=(i % 3 == 0))
    batch = buffer.sample(10)
    for state, action, reward, next_state, done in zip(*batch):
        index = int(reward.item())
        assert torch.all(state == index)
        assert torch.all(next_state == index + 0.5)
        assert action.item() == index % 2
        assert done.item() == (1.0 if index % 3 == 0 else 0.0)


def test_dones_are_zero_one_masks():
    buffer = ReplayBuffer(capacity=10, seed=0)
    push_transition(buffer, 0, done=True)
    push_transition(buffer, 1, done=False)
    batch = buffer.sample(2)
    by_reward = dict(zip(batch.rewards.tolist(), batch.dones.tolist()))
    assert by_reward == {0.0: 1.0, 1.0: 0.0}


def test_sample_has_no_duplicate_transitions():
    buffer = ReplayBuffer(capacity=50, seed=0)
    for i in range(50):
        push_transition(buffer, i)
    rewards = buffer.sample(50).rewards.tolist()
    assert sorted(rewards) == [float(i) for i in range(50)]


def test_sampling_is_random():
    buffer = ReplayBuffer(capacity=100, seed=0)
    for i in range(100):
        push_transition(buffer, i)
    draws = {tuple(buffer.sample(10).rewards.tolist()) for _ in range(5)}
    assert len(draws) > 1, "repeated samples were identical; minibatches should be random"


def test_sampling_is_reproducible_with_the_same_seed():
    first, second = ReplayBuffer(capacity=100, seed=7), ReplayBuffer(capacity=100, seed=7)
    for i in range(100):
        push_transition(first, i)
        push_transition(second, i)
    assert torch.equal(first.sample(16).rewards, second.sample(16).rewards)


def test_sampling_more_than_stored_raises_value_error():
    buffer = ReplayBuffer(capacity=10, seed=0)
    for i in range(3):
        push_transition(buffer, i)
    with pytest.raises(ValueError):
        buffer.sample(4)
