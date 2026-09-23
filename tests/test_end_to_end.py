"""End-to-end smoke check: passes once every TODO(learner) task is implemented.

This only proves the pieces fit together on a tiny budget. To see real learning, run the
default `python -m dqn.train` and then `python -m dqn.evaluate`.
"""

import pytest

from dqn.artifacts import discover_returns_csvs
from dqn.config import DQNConfig
from dqn.evaluate import evaluate
from dqn.train import train, train_seeds

pytestmark = pytest.mark.learner


@pytest.mark.parametrize("target_update_mode", ["hard", "soft"])
def test_short_training_run_saves_artifacts_that_evaluate_can_load(tmp_path, target_update_mode):
    config = DQNConfig(
        target_update_mode=target_update_mode,
        total_steps=400,
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
    assert len(result.episode_end_steps) == len(result.episode_returns)
    assert result.episode_end_steps == sorted(result.episode_end_steps)
    for path in (result.checkpoint_path, result.curve_path, result.returns_path):
        assert path.is_file() and path.stat().st_size > 0

    evaluation = evaluate(result.checkpoint_path, episodes=2, random_baseline=True)
    assert len(evaluation.greedy_returns) == 2
    assert all(1.0 <= r <= 500.0 for r in evaluation.greedy_returns)
    assert evaluation.random_returns is not None and len(evaluation.random_returns) == 2


def test_training_several_seeds_writes_one_file_set_per_seed(tmp_path):
    config = DQNConfig(
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

    results = train_seeds(config, [0, 1])

    assert [result.seed for result in results] == [0, 1]
    assert len({result.returns_path for result in results}) == 2
    for result in results:
        for path in (result.checkpoint_path, result.curve_path, result.returns_path):
            assert path.is_file() and path.stat().st_size > 0
            assert "seed" in path.name
    # Both seeds share one directory, and compare picks the whole arm up from it.
    assert [seed for seed, _ in discover_returns_csvs(tmp_path)] == [0, 1]
    # Different seeds must actually produce different runs, or the sweep proves nothing.
    assert results[0].episode_returns != results[1].episode_returns
