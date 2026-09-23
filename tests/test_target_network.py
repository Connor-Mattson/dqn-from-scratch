"""Learner task 5: target-network sync schedule and hard update (src/dqn/agent.py)."""

import copy

import pytest
import torch

from dqn.agent import should_sync_target, sync_target_network
from dqn.network import QNetwork

pytestmark = pytest.mark.learner


def make_networks() -> tuple[QNetwork, QNetwork]:
    torch.manual_seed(0)
    online = QNetwork(4, 2, hidden_sizes=(8,))
    torch.manual_seed(1)
    target = QNetwork(4, 2, hidden_sizes=(8,))
    return online, target


def test_sync_copies_every_online_weight():
    online, target = make_networks()
    assert not all(torch.equal(a, b) for a, b in zip(online.state_dict().values(), target.state_dict().values()))

    sync_target_network(online, target)

    for (name, online_tensor), target_tensor in zip(online.state_dict().items(), target.state_dict().values()):
        assert torch.equal(online_tensor, target_tensor), f"{name} was not copied"


def test_sync_copies_values_without_sharing_storage():
    online, target = make_networks()
    sync_target_network(online, target)
    target_after_sync = copy.deepcopy(target.state_dict())

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, target_after_sync[name]), f"changing the online {name} changed the target"


def test_should_sync_on_positive_multiples_of_interval():
    assert [step for step in range(1, 1601) if should_sync_target(step, 500)] == [500, 1000, 1500]
    assert all(should_sync_target(step, 1) for step in range(1, 20))


def test_should_sync_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        should_sync_target(10, 0)
