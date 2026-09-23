"""Learner task 2: epsilon-greedy select_action (src/dqn/agent.py)."""

import numpy as np
import pytest

from dqn.agent import select_action
from helpers import TableQNetwork

pytestmark = pytest.mark.learner

STATE = np.zeros(4, dtype=np.float32)


def test_greedy_picks_highest_q_value_as_python_int():
    rng = np.random.default_rng(0)
    assert select_action(TableQNetwork([[0.1, 0.9]]), STATE, 0.0, 2, rng) == 1
    action = select_action(TableQNetwork([[2.0, -1.0, 0.5]]), STATE, 0.0, 3, rng)
    assert type(action) is int
    assert action == 0


def test_greedy_query_is_batched_and_gradient_free():
    network = TableQNetwork([[0.0, 1.0]])
    select_action(network, STATE, 0.0, 2, np.random.default_rng(0))
    assert network.calls == [((1, 4), False)], "expected one [1, obs_dim] forward pass with grad disabled"


def test_epsilon_one_explores_every_valid_action():
    network = TableQNetwork([[0.0, 10.0, 0.0]])
    rng = np.random.default_rng(123)
    actions = [select_action(network, STATE, 1.0, 3, rng) for _ in range(300)]
    assert all(type(a) is int for a in actions)
    assert set(actions) == {0, 1, 2}


def test_intermediate_epsilon_mixes_random_and_greedy_actions():
    network = TableQNetwork([[0.0, 1.0]])
    rng = np.random.default_rng(0)
    actions = [select_action(network, STATE, 0.5, 2, rng) for _ in range(2000)]
    # Greedy action is 1; action 0 only appears via the random branch (probability 0.5 * 1/2).
    fraction_non_greedy = actions.count(0) / len(actions)
    assert 0.18 < fraction_non_greedy < 0.32
