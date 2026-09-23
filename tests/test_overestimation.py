"""Overestimation diagnostics: src/dqn/overestimation.py and its wiring into train.py."""

import numpy as np
import pytest
import torch

from dqn import overestimation as overestimation_cli
from dqn.config import DQNConfig
from dqn.network import QNetwork
from dqn.overestimation import (
    CSV_FIELDS,
    discounted_returns_to_go,
    load_overestimation_csv,
    max_operator_gap,
    save_overestimation_csv,
    score_greedy_rollouts,
)
from dqn.train import train
from helpers import TableQNetwork

pytestmark = pytest.mark.scaffold

# Same tables as test_double_dqn.py: the networks disagree on every row. Vanilla takes
# 10, 7, 2 where Double DQN takes -10, 0, 1, so the per-row gaps are 20, 7 and 1.
ONLINE_TABLE = [[1.0, 3.0], [5.0, 2.0], [-2.0, -4.0]]
TARGET_TABLE = [[10.0, -10.0], [0.0, 7.0], [1.0, 2.0]]


def test_max_operator_gap_is_vanilla_minus_double_per_row():
    gap_sum, disagreements, rows = max_operator_gap(
        TableQNetwork(ONLINE_TABLE), TableQNetwork(TARGET_TABLE), torch.zeros(3, 4), torch.zeros(3)
    )
    assert (gap_sum, disagreements, rows) == (28.0, 3.0, 3)


def test_max_operator_gap_skips_terminal_rows():
    # TableQNetwork returns its first B rows, so the terminal row goes last.
    gap_sum, disagreements, rows = max_operator_gap(
        TableQNetwork(ONLINE_TABLE), TableQNetwork(TARGET_TABLE), torch.zeros(3, 4), torch.tensor([0.0, 0.0, 1.0])
    )
    assert (gap_sum, disagreements, rows) == (27.0, 2.0, 2)


def test_max_operator_gap_is_zero_when_networks_agree():
    torch.manual_seed(0)
    network = QNetwork(4, 2)
    gap_sum, disagreements, rows = max_operator_gap(network, network, torch.randn(32, 4), torch.zeros(32))
    assert (gap_sum, disagreements, rows) == (0.0, 0.0, 32)


def test_discounted_returns_to_go():
    np.testing.assert_allclose(discounted_returns_to_go([1.0, 1.0, 1.0], 0.5), [1.75, 1.5, 1.0])


def test_rollout_score_is_finite_and_bias_is_predicted_minus_realised():
    torch.manual_seed(0)
    score = score_greedy_rollouts(QNetwork(4, 2), "CartPole-v1", 0.99, episodes=2, seed=0)
    assert score.states_scored > 0
    assert score.bias == pytest.approx(score.q_predicted - score.return_realised)


def test_csv_round_trip(tmp_path):
    row = dict.fromkeys(CSV_FIELDS, 1.5) | {"step": 100, "gradient_steps": 50, "states_scored": 7}
    loaded = load_overestimation_csv(save_overestimation_csv(tmp_path / "o.csv", [row, row]))
    assert loaded["bias"].tolist() == [1.5, 1.5]
    assert loaded["step"].tolist() == [100, 100]


def test_config_rejects_negative_interval():
    with pytest.raises(ValueError, match="overestimation_every"):
        DQNConfig(overestimation_every=-1).validate()


def _config(tmp_path, **overrides):
    return DQNConfig(
        total_steps=400, learning_starts=100, batch_size=32, buffer_capacity=1_000,
        epsilon_decay_steps=200, log_every_episodes=1_000, output_dir=str(tmp_path), device="cpu",
        **overrides,
    )


@pytest.mark.parametrize("double", [False, True])
def test_diagnostics_do_not_change_training(tmp_path, double):
    off = train(_config(tmp_path / "off", double_dqn=double))
    on = train(_config(tmp_path / "on", double_dqn=double, overestimation_every=100, overestimation_episodes=1))
    assert off.episode_returns == on.episode_returns
    assert off.overestimation_path is None
    rows = load_overestimation_csv(on.overestimation_path)
    assert rows["step"].tolist() == [100, 200, 300, 400]
    assert np.all(rows["target_gap"] >= 0)


def test_compare_cli_writes_both_plots(tmp_path, capsys):
    for arm, bias in (("a", 5.0), ("b", 1.0)):
        for seed in range(2):
            rows = [
                dict.fromkeys(CSV_FIELDS, 0.1) | {"step": step, "bias": bias + seed, "return_realised": 50.0}
                for step in (1000, 2000, 3000, 4000)
            ]
            save_overestimation_csv(tmp_path / arm / f"overestimation_seed{seed}.csv", rows)
    assert overestimation_cli.main(
        ["--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"), "--output-dir", str(tmp_path / "out")]
    ) == 0
    assert (tmp_path / "out" / "overestimation_bias.png").is_file()
    assert (tmp_path / "out" / "overestimation_gap.png").is_file()
    assert "-4.00" in capsys.readouterr().out
