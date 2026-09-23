"""Soft (Polyak) target updates: ``soft_update_target_network`` in src/dqn/agent.py."""

import copy

import pytest
import torch

from dqn.agent import soft_update_target_network, sync_target_network, update_target_network
from dqn.network import QNetwork

pytestmark = pytest.mark.learner


def make_networks() -> tuple[QNetwork, QNetwork]:
    """Two same-shaped networks with deliberately different weights."""
    torch.manual_seed(0)
    online = QNetwork(4, 2, hidden_sizes=(8,))
    torch.manual_seed(1)
    target = QNetwork(4, 2, hidden_sizes=(8,))
    return online, target


def largest_gap(online: QNetwork, target: QNetwork) -> float:
    """Largest absolute elementwise difference between the two networks' weights."""
    return max(
        float((a - b).abs().max())
        for a, b in zip(online.state_dict().values(), target.state_dict().values())
    )


def test_soft_update_blends_every_parameter():
    online, target = make_networks()
    online_before = copy.deepcopy(online.state_dict())
    target_before = copy.deepcopy(target.state_dict())

    soft_update_target_network(online, target, tau=0.25)

    for name, after in target.state_dict().items():
        expected = 0.25 * online_before[name] + 0.75 * target_before[name]
        torch.testing.assert_close(after, expected, msg=f"{name} is not the tau=0.25 blend")


def test_soft_update_leaves_the_online_network_untouched():
    online, target = make_networks()
    online_before = copy.deepcopy(online.state_dict())

    soft_update_target_network(online, target, tau=0.5)

    for name, tensor in online.state_dict().items():
        assert torch.equal(tensor, online_before[name]), f"the online {name} was modified"


def test_tau_of_one_matches_a_hard_sync():
    online, target = make_networks()
    hard_online, hard_target = make_networks()
    sync_target_network(hard_online, hard_target)

    soft_update_target_network(online, target, tau=1.0)

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, hard_target.state_dict()[name], msg=f"{name} != a hard copy")


def test_soft_update_copies_values_without_sharing_storage():
    online, target = make_networks()
    soft_update_target_network(online, target, tau=1.0)
    target_after_update = copy.deepcopy(target.state_dict())

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, target_after_update[name]), f"changing the online {name} changed the target"


def test_soft_update_keeps_the_target_out_of_the_autograd_graph():
    online, target = make_networks()

    soft_update_target_network(online, target, tau=0.1)

    for name, parameter in target.named_parameters():
        assert parameter.is_leaf, f"target {name} is no longer a leaf tensor"
        assert parameter.grad_fn is None, f"target {name} carries a grad_fn"
        assert parameter.grad is None, f"target {name} picked up a gradient"


def test_repeated_updates_converge_toward_the_online_network():
    online, target = make_networks()
    gap_before = largest_gap(online, target)

    for _ in range(100):
        soft_update_target_network(online, target, tau=0.05)

    # The remaining gap decays by (1 - tau) per update: 0.95 ** 100 is about 0.006.
    assert largest_gap(online, target) < 0.05 * gap_before


def test_soft_update_rejects_tau_outside_the_unit_interval():
    online, target = make_networks()
    for tau in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            soft_update_target_network(online, target, tau)


def test_soft_mode_updates_the_target_after_every_gradient_step():
    online, target = make_networks()
    target_before = copy.deepcopy(target.state_dict())

    for gradient_step in (1, 2, 3):
        assert update_target_network(online, target, gradient_step, "soft", interval=500, tau=0.5) == "soft"

    for name, tensor in target.state_dict().items():
        assert not torch.equal(tensor, target_before[name]), f"target {name} never moved in soft mode"
