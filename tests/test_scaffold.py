"""Checks for the provided plumbing. These should pass before any TODO(learner) work."""

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from dqn import compare as compare_cli
from dqn import evaluate as evaluate_cli
from dqn import train as train_cli
from dqn.agent import update_target_network
from dqn.artifacts import (
    discover_returns_csvs,
    load_checkpoint,
    load_returns_csv,
    save_checkpoint,
    save_returns_csv,
    seed_from_name,
    seeded_name,
)
from dqn.compare import MAX_ARMS, compare, compare_arms, parse_arm_spec
from dqn.config import DQNConfig, resolve_device
from dqn.env import env_dimensions, make_env
from dqn.network import QNetwork
from dqn.plotting import (
    BandCurve,
    RunCurve,
    moving_average,
    save_band_comparison_curve,
    save_comparison_curve,
    save_reward_curve,
)
from dqn.replay_buffer import Batch, ReplayBuffer
from dqn.schedule import linear_epsilon
from dqn.stats import aggregate_seeds

pytestmark = pytest.mark.scaffold


@pytest.mark.parametrize(
    "module",
    ["dqn", "dqn.agent", "dqn.artifacts", "dqn.compare", "dqn.config", "dqn.env", "dqn.evaluate",
     "dqn.learning", "dqn.network", "dqn.plotting", "dqn.replay_buffer", "dqn.schedule", "dqn.stats",
     "dqn.train"],
)
def test_package_modules_import(module):
    importlib.import_module(module)


def test_cartpole_env_dimensions_and_reset():
    env = make_env("CartPole-v1")
    try:
        assert env_dimensions(env) == (4, 2)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4,)
    finally:
        env.close()


def test_q_network_outputs_one_value_per_action():
    q_values = QNetwork(4, 2, hidden_sizes=(32, 32))(torch.zeros(5, 4))
    assert q_values.shape == (5, 2)
    assert q_values.dtype == torch.float32


def test_linear_epsilon_schedule():
    assert linear_epsilon(0, 1.0, 0.1, 100) == pytest.approx(1.0)
    assert linear_epsilon(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert linear_epsilon(100, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert linear_epsilon(10_000, 1.0, 0.1, 100) == pytest.approx(0.1)


def test_batch_to_device_preserves_fields():
    batch = Batch(torch.zeros(2, 4), torch.tensor([0, 1]), torch.ones(2), torch.ones(2, 4), torch.zeros(2))
    moved = batch.to("cpu")
    assert isinstance(moved, Batch)
    assert all(torch.equal(a, b) for a, b in zip(batch, moved))


def test_replay_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_default_config_is_valid_and_bad_values_are_rejected():
    DQNConfig().validate()
    with pytest.raises(ValueError):
        DQNConfig(learning_starts=10, batch_size=64).validate()
    with pytest.raises(ValueError):
        DQNConfig(target_update_interval=0).validate()


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    network = QNetwork(4, 2, hidden_sizes=(8,))
    path = save_checkpoint(tmp_path / "model.pt", network, DQNConfig(), [10.0, 20.0])

    loaded, metadata = load_checkpoint(path)

    obs = torch.randn(3, 4)
    assert torch.equal(network(obs), loaded(obs))
    assert metadata["env_id"] == "CartPole-v1"
    assert metadata["episode_returns"] == [10.0, 20.0]


def test_missing_checkpoint_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m dqn.train"):
        load_checkpoint(tmp_path / "missing.pt")


def test_returns_csv(tmp_path):
    path = save_returns_csv(tmp_path / "returns.csv", [9.0, 12.5])
    with path.open() as f:
        assert list(csv.reader(f)) == [["episode", "return"], ["1", "9"], ["2", "12.5"]]


def test_moving_average_is_trailing_mean():
    np.testing.assert_allclose(moving_average([1, 2, 3, 4], window=2), [1.0, 1.5, 2.5, 3.5])
    assert moving_average([], window=5).size == 0


@pytest.mark.parametrize("returns", [[], [12.0, 30.0, 25.0, 60.0, 120.0]])
def test_reward_curve_saves_png(tmp_path, returns):
    path = save_reward_curve(returns, tmp_path / "curve.png", window=3)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_target_update_mode_and_tau_are_validated():
    DQNConfig(target_update_mode="soft", tau=0.01).validate()
    with pytest.raises(ValueError, match="target_update_mode"):
        DQNConfig(target_update_mode="polyak").validate()
    with pytest.raises(ValueError, match="tau"):
        DQNConfig(target_update_mode="soft", tau=0.0).validate()
    with pytest.raises(ValueError, match="tau"):
        DQNConfig(tau=1.5).validate()


def test_hard_mode_dispatch_syncs_only_on_the_interval():
    torch.manual_seed(0)
    online = QNetwork(4, 2, hidden_sizes=(8,))
    torch.manual_seed(1)
    target = QNetwork(4, 2, hidden_sizes=(8,))

    assert update_target_network(online, target, 3, "hard", interval=5, tau=0.5) is None
    assert any(not torch.equal(a, b) for a, b in zip(online.state_dict().values(), target.state_dict().values()))

    assert update_target_network(online, target, 5, "hard", interval=5, tau=0.5) == "hard"
    assert all(torch.equal(a, b) for a, b in zip(online.state_dict().values(), target.state_dict().values()))


def test_unknown_target_update_mode_is_rejected():
    network = QNetwork(4, 2, hidden_sizes=(8,))
    with pytest.raises(ValueError, match="mode"):
        update_target_network(network, network, 1, "polyak", interval=5, tau=0.5)


def test_returns_csv_records_steps_when_given(tmp_path):
    path = save_returns_csv(tmp_path / "returns.csv", [9.0, 12.5], [14, 31])
    with path.open() as f:
        assert list(csv.reader(f)) == [["episode", "step", "return"], ["1", "14", "9"], ["2", "31", "12.5"]]

    returns, steps = load_returns_csv(path)
    np.testing.assert_allclose(returns, [9.0, 12.5])
    np.testing.assert_array_equal(steps, [14, 31])


def test_returns_csv_without_steps_round_trips(tmp_path):
    returns, steps = load_returns_csv(save_returns_csv(tmp_path / "returns.csv", [9.0, 12.5]))
    np.testing.assert_allclose(returns, [9.0, 12.5])
    assert steps is None


def test_returns_csv_rejects_mismatched_steps(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        save_returns_csv(tmp_path / "returns.csv", [9.0, 12.5], [14])


@pytest.mark.parametrize("steps", [None, [5, 12, 30]])
def test_comparison_curve_saves_png(tmp_path, steps):
    path = save_comparison_curve(
        [RunCurve("hard", [10.0, 20.0, 30.0], steps), RunCurve("soft", [12.0, 40.0, 90.0], steps)],
        tmp_path / "compare.png",
        window=2,
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def test_comparison_curve_rejects_an_empty_run_list(tmp_path):
    with pytest.raises(ValueError, match="at least one run"):
        save_comparison_curve([], tmp_path / "compare.png")


def test_compare_plots_both_runs_and_reports_when_each_got_good(tmp_path):
    steps = list(range(10, 101, 10))
    hard = save_returns_csv(tmp_path / "hard.csv", [10.0] * 5 + [200.0] * 5, steps)
    soft = save_returns_csv(tmp_path / "soft.csv", [10.0] * 5 + [400.0] * 5, steps)

    result = compare(hard, soft, tmp_path / "compare.png", window=2, threshold=195.0)

    assert result.plot_path.read_bytes().startswith(b"\x89PNG")
    assert [summary.episodes for summary in result.summaries] == [10, 10]
    # Trailing mean of 2: hard clears 195 one episode later than soft, at step 70 vs 60.
    assert [summary.steps_to_threshold for summary in result.summaries] == [70, 60]
    assert result.summaries[1].best_mean > result.summaries[0].best_mean


def test_compare_reports_a_run_that_never_reached_the_threshold(tmp_path):
    flat = save_returns_csv(tmp_path / "flat.csv", [10.0] * 4, [10, 20, 30, 40])

    summary = compare(flat, flat, tmp_path / "compare.png", window=2).summaries[0]

    assert summary.episodes_to_threshold is None and summary.steps_to_threshold is None
    assert summary.total_steps == 40


def test_seeded_name_puts_the_seed_before_the_extension():
    assert seeded_name("episode_returns.csv", 3) == "episode_returns_seed3.csv"
    assert seeded_name("dqn.pt", 0) == "dqn_seed0.pt"
    assert seeded_name("noextension", 7) == "noextension_seed7"
    assert seed_from_name("outputs/hard/episode_returns_seed12.csv") == 12
    assert seed_from_name("episode_returns.csv") is None


def test_discover_returns_csvs_finds_every_seed_in_a_directory(tmp_path):
    for seed in (2, 0, 10):
        save_returns_csv(tmp_path / seeded_name("episode_returns.csv", seed), [1.0], [1])

    found = discover_returns_csvs(tmp_path)

    # Ordered by seed numerically, not lexically: seed 10 sorts after seed 2, not after seed 1.
    assert [seed for seed, _ in found] == [0, 2, 10]


def test_discover_returns_csvs_falls_back_to_a_single_seed_run(tmp_path):
    plain = save_returns_csv(tmp_path / "episode_returns.csv", [1.0], [1])

    assert discover_returns_csvs(tmp_path) == [(None, plain)]
    assert discover_returns_csvs(plain) == [(None, plain)]


def test_discover_returns_csvs_ignores_a_stale_single_seed_file(tmp_path):
    # Re-running an arm with --seeds leaves the earlier single-run CSV in the directory.
    # The seeded files win outright rather than being quietly averaged in with it.
    save_returns_csv(tmp_path / "episode_returns.csv", [999.0], [1])
    for seed in (0, 1):
        save_returns_csv(tmp_path / seeded_name("episode_returns.csv", seed), [2.0], [1])

    found = discover_returns_csvs(tmp_path)

    assert [seed for seed, _ in found] == [0, 1]
    assert all("seed" in path.name for _, path in found)


def test_discover_returns_csvs_reports_a_directory_with_no_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m dqn.train"):
        discover_returns_csvs(tmp_path)
    with pytest.raises(FileNotFoundError, match="python -m dqn.train"):
        discover_returns_csvs(tmp_path / "never-trained")


def test_band_comparison_curve_saves_png(tmp_path):
    hard = aggregate_seeds([(np.array([10, 50, 100]), np.array([5.0, 40.0, 80.0])),
                            (np.array([12, 60, 110]), np.array([8.0, 30.0, 95.0]))])
    soft = aggregate_seeds([(np.array([10, 50, 100]), np.array([6.0, 60.0, 150.0])),
                            (np.array([11, 55, 105]), np.array([4.0, 70.0, 130.0]))])

    path = save_band_comparison_curve(
        [BandCurve("hard", hard), BandCurve("soft", soft)], tmp_path / "band.png"
    )

    assert path.read_bytes().startswith(b"\x89PNG")


def test_band_comparison_curve_rejects_an_empty_arm_list(tmp_path):
    with pytest.raises(ValueError, match="at least one run"):
        save_band_comparison_curve([], tmp_path / "band.png")


def test_compare_averages_seeds_and_tests_significance(tmp_path):
    steps = list(range(100, 2001, 100))
    hard_dir, soft_dir = tmp_path / "hard", tmp_path / "soft"
    for seed in range(4):
        save_returns_csv(hard_dir / seeded_name("episode_returns.csv", seed), [50.0 + seed] * 20, steps)
        save_returns_csv(soft_dir / seeded_name("episode_returns.csv", seed), [300.0 + seed] * 20, steps)

    result = compare(hard_dir, soft_dir, tmp_path / "compare.png", window=5, threshold=195.0)

    assert result.plot_path.read_bytes().startswith(b"\x89PNG")
    assert [summary.n_seeds for summary in result.summaries] == [4, 4]
    assert result.summaries[0].final_mean == pytest.approx(51.5)  # mean of 50, 51, 52, 53
    assert result.summaries[0].final_stderr > 0
    # Only the soft arm ever clears 195, and every one of its seeds does.
    assert result.summaries[0].seeds_reaching == 0
    assert result.summaries[1].seeds_reaching == 4
    assert result.ttest is not None
    assert result.ttest.difference == pytest.approx(250.0)
    assert result.ttest.significant


def test_compare_reports_no_significance_test_for_single_seed_arms(tmp_path):
    steps = list(range(10, 101, 10))
    hard = save_returns_csv(tmp_path / "hard.csv", [10.0] * 10, steps)
    soft = save_returns_csv(tmp_path / "soft.csv", [400.0] * 10, steps)

    result = compare(hard, soft, tmp_path / "compare.png", window=2)

    # A large gap on one seed each is still not evidence: there is no within-arm variance.
    assert result.ttest is None
    assert [summary.n_seeds for summary in result.summaries] == [1, 1]
    assert result.summaries[0].final_stderr == 0.0


def test_double_dqn_is_off_by_default_and_switched_on_by_a_flag():
    assert DQNConfig().double_dqn is False
    DQNConfig(double_dqn=True).validate()
    assert train_cli.build_parser().parse_args([]).double is False
    assert train_cli.build_parser().parse_args(["--double"]).double is True


def test_compare_arms_can_be_renamed_for_other_experiments(tmp_path):
    steps = list(range(10, 101, 10))
    vanilla = save_returns_csv(tmp_path / "vanilla.csv", [10.0] * 10, steps)
    double = save_returns_csv(tmp_path / "double.csv", [20.0] * 10, steps)

    result = compare(
        vanilla, double, tmp_path / "compare.png", window=2, labels=("vanilla DQN", "Double DQN")
    )

    assert [summary.label for summary in result.summaries] == ["vanilla DQN", "Double DQN"]
    args = compare_cli.build_parser().parse_args(
        ["--a", "outputs/vanilla", "--b", "outputs/double", "--labels", "vanilla DQN", "Double DQN"]
    )
    assert (args.hard, args.soft) == (Path("outputs/vanilla"), Path("outputs/double"))
    assert args.labels == ["vanilla DQN", "Double DQN"]


def test_arm_spec_takes_an_explicit_label_or_names_itself_from_the_directory():
    assert parse_arm_spec("ddqn soft=outputs/ddqn-soft") == ("ddqn soft", Path("outputs/ddqn-soft"))
    # A bare path labels itself, with the hyphen read as a word break for the legend.
    assert parse_arm_spec("outputs/ddqn-hard") == ("ddqn hard", Path("outputs/ddqn-hard"))
    with pytest.raises(ValueError, match="--arm needs a path"):
        parse_arm_spec("   ")


def test_compare_adds_extra_arms_and_tests_each_against_the_first(tmp_path):
    steps = list(range(100, 2001, 100))
    dirs = {}
    for name, base in [("hard", 50.0), ("soft", 60.0), ("ddqn-hard", 300.0), ("ddqn-soft", 310.0)]:
        dirs[name] = tmp_path / name
        for seed in range(4):
            save_returns_csv(
                dirs[name] / seeded_name("episode_returns.csv", seed), [base + seed] * 20, steps
            )

    result = compare(
        dirs["hard"], dirs["soft"], tmp_path / "cross.png", window=5, threshold=195.0,
        extra_arms=[("ddqn hard", dirs["ddqn-hard"]), ("ddqn soft", dirs["ddqn-soft"])],
    )

    assert result.plot_path.read_bytes().startswith(b"\x89PNG")
    assert [s.label for s in result.summaries] == ["hard targets", "soft targets", "ddqn hard", "ddqn soft"]
    # One test per arm after the first, each against the first — not all six pairings.
    assert len(result.ttests) == 3
    assert [round(t.difference) for t in result.ttests] == [10, 250, 260]
    # `.ttest` stays the second-against-first story the two-arm callers already print.
    assert result.ttest is result.ttests[0]


def test_compare_arms_refuses_more_arms_than_the_palette_separates(tmp_path):
    csv_path = save_returns_csv(tmp_path / "run.csv", [10.0] * 4, [10, 20, 30, 40])

    with pytest.raises(ValueError, match=f"at most {MAX_ARMS} arms"):
        compare_arms([(f"arm{i}", csv_path) for i in range(MAX_ARMS + 1)], tmp_path / "too-many.png")
    with pytest.raises(ValueError, match="at least one arm"):
        compare_arms([], tmp_path / "none.png")


def test_cli_defaults_match_documented_commands():
    train_args = train_cli.build_parser().parse_args([])
    assert train_args.total_steps == DQNConfig().total_steps
    assert train_args.output_dir == "outputs"
    assert train_args.target_update_mode == "hard"
    assert train_args.tau == DQNConfig().tau
    assert train_args.seeds is None  # --seed alone keeps the original single-run layout
    assert train_cli.build_parser().parse_args(["--seeds", "0", "1", "2"]).seeds == [0, 1, 2]
    compare_args = compare_cli.build_parser().parse_args([])
    assert compare_args.hard == Path("outputs/hard")
    assert compare_args.soft == Path("outputs/soft")
    eval_args = evaluate_cli.build_parser().parse_args(["--checkpoint", "outputs/dqn.pt"])
    assert eval_args.checkpoint == Path("outputs/dqn.pt")
    assert evaluate_cli.build_parser().parse_args([]).checkpoint == Path("outputs/dqn.pt")
